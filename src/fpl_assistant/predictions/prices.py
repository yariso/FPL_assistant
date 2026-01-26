"""
Price Change Prediction for FPL.

Predicts player price changes based on transfer activity.
FPL uses a threshold-based system for price changes.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..data.models import Player
from ..data.storage import Database

logger = logging.getLogger(__name__)


@dataclass
class PriceChangeThreshold:
    """
    Price change threshold data.

    FPL uses a formula based on:
    - Net transfers (in - out)
    - Current ownership
    - Time since last change
    """

    player_id: int
    current_price: float
    net_transfers: int
    threshold_estimate: float  # Estimated transfers needed for change
    threshold_progress: float  # 0-1 scale, 1 = change imminent
    direction: str  # "rise", "fall", "stable"
    confidence: float  # 0-1


@dataclass
class PricePrediction:
    """Price change prediction for a player."""

    player_id: int
    player_name: str
    team_name: str
    current_price: float
    predicted_change: float  # +0.1, -0.1, or 0
    probability: float  # 0-1
    expected_time: str  # "tonight", "tomorrow", "this week", "unlikely"
    net_transfers_week: int
    ownership_percent: float
    recommendation: str  # "Transfer IN now", "Transfer OUT now", "Hold", "Watch"


class PricePredictor:
    """
    Predicts FPL price changes based on transfer activity.

    FPL Price Change Rules (approximate):
    - Price changes happen overnight (typically around 2:30 AM UK)
    - Threshold depends on ownership % and net transfers
    - Higher ownership = more transfers needed
    - ~0.2% of ownership needed as net transfers for rise
    - ~0.4% of ownership needed as net transfers for fall
    - Maximum 2 price changes per gameweek per player
    """

    # Approximate threshold multipliers (from community research)
    RISE_THRESHOLD_BASE = 0.002  # ~0.2% of ownership as net transfers
    FALL_THRESHOLD_BASE = 0.004  # ~0.4% of ownership as net transfers (harder to fall)

    # Total managers estimate (update each season)
    TOTAL_MANAGERS = 11_000_000  # ~11 million managers in 2024/25

    def __init__(self, db: Database):
        """Initialize the price predictor."""
        self.db = db

    def estimate_threshold(
        self,
        player: Player,
        direction: str = "rise",
    ) -> float:
        """
        Estimate the transfer threshold needed for a price change.

        Args:
            player: Player object
            direction: "rise" or "fall"

        Returns:
            Estimated number of net transfers needed
        """
        # Current owners estimate
        owners = int(self.TOTAL_MANAGERS * player.selected_by_percent / 100)

        # Base threshold
        if direction == "rise":
            threshold_pct = self.RISE_THRESHOLD_BASE
        else:
            threshold_pct = self.FALL_THRESHOLD_BASE

        # Adjust for price band (cheaper players change faster)
        if player.price < 5.0:
            threshold_pct *= 0.7  # 30% easier for cheap players
        elif player.price > 10.0:
            threshold_pct *= 1.3  # 30% harder for expensive players

        # Calculate threshold
        threshold = int(owners * threshold_pct)

        # Minimum threshold
        threshold = max(threshold, 1000)

        return threshold

    def calculate_threshold_progress(
        self,
        player: Player,
        net_transfers: int | None = None,
    ) -> PriceChangeThreshold:
        """
        Calculate how close a player is to a price change.

        Args:
            player: Player object
            net_transfers: Override net transfers (if known from tracking)

        Returns:
            PriceChangeThreshold with progress info
        """
        # Get net transfers (use event transfers if not provided)
        if net_transfers is None:
            transfers_in = getattr(player, "transfers_in_event", 0)
            transfers_out = getattr(player, "transfers_out_event", 0)
            net_transfers = transfers_in - transfers_out

        # Determine direction
        if net_transfers > 0:
            direction = "rise"
            threshold = self.estimate_threshold(player, "rise")
        elif net_transfers < 0:
            direction = "fall"
            threshold = self.estimate_threshold(player, "fall")
            net_transfers = abs(net_transfers)  # Use absolute value for comparison
        else:
            direction = "stable"
            threshold = self.estimate_threshold(player, "rise")

        # Calculate progress
        progress = min(1.0, net_transfers / threshold if threshold > 0 else 0)

        # Confidence based on data quality
        confidence = 0.7 if hasattr(player, "transfers_in_event") else 0.5

        return PriceChangeThreshold(
            player_id=player.id,
            current_price=player.price,
            net_transfers=net_transfers if direction != "fall" else -net_transfers,
            threshold_estimate=threshold,
            threshold_progress=progress,
            direction=direction,
            confidence=confidence,
        )

    def predict_price_change(
        self,
        player: Player,
        net_transfers: int | None = None,
    ) -> PricePrediction:
        """
        Predict if a player's price will change.

        Args:
            player: Player object
            net_transfers: Override net transfers (if known)

        Returns:
            PricePrediction with full analysis
        """
        threshold = self.calculate_threshold_progress(player, net_transfers)

        # Determine predicted change and probability
        if threshold.direction == "rise" and threshold.threshold_progress >= 0.95:
            predicted_change = 0.1
            probability = min(0.95, threshold.threshold_progress)
            expected_time = "tonight"
            recommendation = "Transfer IN now (before price rise)"
        elif threshold.direction == "rise" and threshold.threshold_progress >= 0.7:
            predicted_change = 0.1
            probability = threshold.threshold_progress * 0.8
            expected_time = "tomorrow"
            recommendation = "Watch closely (price rise likely soon)"
        elif threshold.direction == "fall" and threshold.threshold_progress >= 0.95:
            predicted_change = -0.1
            probability = min(0.95, threshold.threshold_progress)
            expected_time = "tonight"
            recommendation = "Transfer OUT now (before price drop)"
        elif threshold.direction == "fall" and threshold.threshold_progress >= 0.7:
            predicted_change = -0.1
            probability = threshold.threshold_progress * 0.8
            expected_time = "tomorrow"
            recommendation = "Consider selling (price drop likely soon)"
        elif threshold.direction == "rise" and threshold.threshold_progress >= 0.4:
            predicted_change = 0.1
            probability = threshold.threshold_progress * 0.6
            expected_time = "this week"
            recommendation = "Watch (trending towards rise)"
        elif threshold.direction == "fall" and threshold.threshold_progress >= 0.4:
            predicted_change = -0.1
            probability = threshold.threshold_progress * 0.6
            expected_time = "this week"
            recommendation = "Watch (trending towards fall)"
        else:
            predicted_change = 0
            probability = 0.2
            expected_time = "unlikely"
            recommendation = "Hold (price stable)"

        return PricePrediction(
            player_id=player.id,
            player_name=player.web_name,
            team_name="",  # Caller can fill in
            current_price=player.price,
            predicted_change=predicted_change,
            probability=probability,
            expected_time=expected_time,
            net_transfers_week=threshold.net_transfers,
            ownership_percent=player.selected_by_percent,
            recommendation=recommendation,
        )

    def get_rising_players(
        self,
        players: list[Player],
        min_probability: float = 0.5,
        limit: int = 10,
    ) -> list[PricePrediction]:
        """
        Get players most likely to rise in price.

        Args:
            players: List of players to analyze
            min_probability: Minimum probability threshold
            limit: Number of results to return

        Returns:
            List of PricePrediction sorted by probability
        """
        predictions = []

        for player in players:
            pred = self.predict_price_change(player)
            if pred.predicted_change > 0 and pred.probability >= min_probability:
                predictions.append(pred)

        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x.probability, reverse=True)

        return predictions[:limit]

    def get_falling_players(
        self,
        players: list[Player],
        min_probability: float = 0.5,
        limit: int = 10,
    ) -> list[PricePrediction]:
        """
        Get players most likely to fall in price.

        Args:
            players: List of players to analyze
            min_probability: Minimum probability threshold
            limit: Number of results to return

        Returns:
            List of PricePrediction sorted by probability
        """
        predictions = []

        for player in players:
            pred = self.predict_price_change(player)
            if pred.predicted_change < 0 and pred.probability >= min_probability:
                predictions.append(pred)

        # Sort by probability (highest first)
        predictions.sort(key=lambda x: x.probability, reverse=True)

        return predictions[:limit]

    def get_price_alerts(
        self,
        my_players: list[Player],
        all_players: list[Player],
        watchlist: list[int] | None = None,
    ) -> dict[str, list[PricePrediction]]:
        """
        Get price change alerts for owned and watched players.

        Args:
            my_players: Players in user's squad
            all_players: All available players
            watchlist: Optional list of player IDs to watch

        Returns:
            Dict with "urgent_sell", "consider_sell", "buy_soon", "rising_targets"
        """
        alerts = {
            "urgent_sell": [],  # Owned players about to fall
            "consider_sell": [],  # Owned players trending down
            "buy_soon": [],  # Watchlist players about to rise
            "rising_targets": [],  # High value players rising
        }

        my_player_ids = {p.id for p in my_players}

        # Check owned players for falls
        for player in my_players:
            pred = self.predict_price_change(player)
            if pred.predicted_change < 0:
                if pred.probability >= 0.8:
                    alerts["urgent_sell"].append(pred)
                elif pred.probability >= 0.5:
                    alerts["consider_sell"].append(pred)

        # Check all players for rises (potential targets)
        for player in all_players:
            if player.id in my_player_ids:
                continue

            pred = self.predict_price_change(player)
            if pred.predicted_change > 0 and pred.probability >= 0.5:
                if watchlist and player.id in watchlist:
                    alerts["buy_soon"].append(pred)
                elif pred.probability >= 0.7:
                    alerts["rising_targets"].append(pred)

        # Sort each category
        for key in alerts:
            alerts[key].sort(key=lambda x: x.probability, reverse=True)
            alerts[key] = alerts[key][:5]  # Limit to top 5

        return alerts

    def save_price_snapshot(
        self,
        players: list[Player],
    ) -> int:
        """
        Save current prices for historical tracking.

        Args:
            players: List of players with current prices

        Returns:
            Number of records saved
        """
        now = datetime.now().isoformat()
        count = 0

        with self.db.connection() as conn:
            for player in players:
                try:
                    transfers_in = getattr(player, "transfers_in_event", 0)
                    transfers_out = getattr(player, "transfers_out_event", 0)

                    conn.execute(
                        """
                        INSERT INTO price_history
                        (player_id, recorded_at, price, transfers_in_week, transfers_out_week)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            player.id,
                            now,
                            player.price,
                            transfers_in,
                            transfers_out,
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to save price for player {player.id}: {e}")

            conn.commit()

        logger.info(f"Saved price snapshot for {count} players")
        return count

    def save_prediction(
        self,
        prediction: PricePrediction,
    ) -> None:
        """Save a price prediction to the database."""
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_predictions
                (player_id, prediction_date, predicted_change, confidence, threshold_distance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    prediction.player_id,
                    datetime.now().date().isoformat(),
                    prediction.predicted_change,
                    prediction.probability,
                    1.0 - prediction.probability,  # Distance from threshold
                ),
            )
            conn.commit()


# Convenience functions
def get_price_predictor(db: Database) -> PricePredictor:
    """Get a price predictor instance."""
    return PricePredictor(db)


def predict_price_changes(
    players: list[Player],
    db: Database,
) -> list[PricePrediction]:
    """
    Predict price changes for all players.

    Args:
        players: List of players
        db: Database instance

    Returns:
        List of predictions for players likely to change
    """
    predictor = PricePredictor(db)

    results = []
    for player in players:
        pred = predictor.predict_price_change(player)
        if pred.predicted_change != 0 and pred.probability >= 0.4:
            results.append(pred)

    # Sort by absolute probability
    results.sort(key=lambda x: x.probability, reverse=True)

    return results
