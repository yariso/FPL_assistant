"""
Minutes Probability Model for Elite FPL.

The #1 predictor of FPL points is minutes played.
Elite prediction systems model:
- P(start): Probability of being in starting XI
- P(60+): Probability of playing 60+ minutes (2 appearance points)
- E[minutes]: Expected minutes (0-90)
- Rotation risk: Classification of rotation likelihood

This module estimates these based on available data:
- Recent playing time patterns
- Player status (injury/doubt flags)
- Fixture congestion (games in next 14 days)
- Team depth at position
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..data.models import Player, PlayerStatus, Position

logger = logging.getLogger(__name__)


class RotationRisk(StrEnum):
    """Rotation risk classification."""

    LOW = "low"        # Nailed starter, plays almost every game
    MEDIUM = "medium"  # Regular starter but occasional rotation
    HIGH = "high"      # Significant rotation risk
    UNKNOWN = "unknown"  # Insufficient data


@dataclass
class MinutesPrediction:
    """
    Comprehensive minutes prediction for a player.

    This is the core output of the minutes model.
    """

    player_id: int
    player_name: str

    # Core probabilities
    p_start: float       # Probability of starting (0-1)
    p_60_plus: float     # Probability of playing 60+ minutes (0-1)
    p_90_plus: float     # Probability of playing full 90 (0-1)
    e_minutes: float     # Expected minutes (0-90)

    # Risk classification
    rotation_risk: RotationRisk
    risk_factors: list[str]  # Specific reasons for rotation risk

    # Model confidence
    confidence: float    # 0-1, based on data quality
    data_points: int     # Number of games used for estimation

    @property
    def is_nailed(self) -> bool:
        """Check if player is considered 'nailed' (very likely to start)."""
        return self.p_start >= 0.85 and self.rotation_risk == RotationRisk.LOW

    @property
    def is_rotation_risk(self) -> bool:
        """Check if player has significant rotation risk."""
        return self.rotation_risk in [RotationRisk.MEDIUM, RotationRisk.HIGH]


class MinutesPredictor:
    """
    Predicts player minutes based on available signals.

    Key features used:
    1. Historical minutes per game
    2. Player status (injury/doubt)
    3. Recent starts vs benched
    4. Position competition (squad depth)
    5. Fixture congestion
    """

    # Thresholds for classification
    NAILED_MINUTES_THRESHOLD = 80  # Avg mins for "nailed" player
    ROTATION_MINUTES_THRESHOLD = 60  # Below this = rotation risk

    # Status-based probability adjustments
    STATUS_START_PROBABILITY = {
        PlayerStatus.AVAILABLE: 0.95,
        PlayerStatus.DOUBTFUL: 0.50,
        PlayerStatus.INJURED: 0.05,
        PlayerStatus.SUSPENDED: 0.00,
        PlayerStatus.UNAVAILABLE: 0.10,
        PlayerStatus.NOT_AVAILABLE: 0.00,
    }

    def __init__(
        self,
        players: list[Player],
        fixture_congestion: dict[int, int] | None = None,
    ):
        """
        Initialize the minutes predictor.

        Args:
            players: List of all players
            fixture_congestion: Optional dict of team_id -> games in next 14 days
        """
        self.players = {p.id: p for p in players}
        self.fixture_congestion = fixture_congestion or {}

        # Group players by team and position for depth analysis
        self._team_position_players: dict[tuple[int, Position], list[Player]] = {}
        for p in players:
            key = (p.team_id, p.position)
            if key not in self._team_position_players:
                self._team_position_players[key] = []
            self._team_position_players[key].append(p)

    def predict_minutes(self, player: Player) -> MinutesPrediction:
        """
        Generate minutes prediction for a single player.

        Args:
            player: Player to predict

        Returns:
            MinutesPrediction with all probability estimates
        """
        risk_factors = []

        # Base probability from status
        status_prob = self.STATUS_START_PROBABILITY.get(player.status, 0.5)
        if player.status != PlayerStatus.AVAILABLE:
            risk_factors.append(f"Status: {player.status.value}")

        # Override with chance_of_playing if available (more precise)
        if player.chance_of_playing is not None:
            cop_prob = player.chance_of_playing / 100
            status_prob = min(status_prob, cop_prob)
            if cop_prob < 0.75:
                risk_factors.append(f"Chance of playing: {player.chance_of_playing}%")

        # Analyze historical minutes pattern
        mins_analysis = self._analyze_minutes_history(player)

        # Calculate P(start) - probability of being in starting XI
        p_start = status_prob * mins_analysis["start_rate"]

        # Apply squad depth penalty
        depth_factor = self._calculate_depth_factor(player)
        if depth_factor < 0.9:
            risk_factors.append("Competition for place")
        p_start *= depth_factor

        # Apply fixture congestion penalty
        congestion = self.fixture_congestion.get(player.team_id, 0)
        if congestion >= 5:  # 5+ games in 14 days = heavy congestion
            congestion_factor = 0.85
            risk_factors.append("Fixture congestion")
        elif congestion >= 4:
            congestion_factor = 0.92
        else:
            congestion_factor = 1.0
        p_start *= congestion_factor

        # Clamp to valid range
        p_start = max(0.0, min(1.0, p_start))

        # Calculate P(60+) - conditional on starting
        # Players who start usually play 60+, but some get subbed early
        if p_start > 0:
            p_60_given_start = mins_analysis["p_60_if_start"]
            p_60_plus = p_start * p_60_given_start
        else:
            p_60_plus = 0.0

        # Calculate P(90+) - plays full game
        if p_start > 0:
            p_90_given_start = mins_analysis["p_90_if_start"]
            p_90_plus = p_start * p_90_given_start
        else:
            p_90_plus = 0.0

        # Calculate E[minutes]
        # E[mins] = P(start) * E[mins|start] + P(sub) * E[mins|sub]
        e_mins_if_start = mins_analysis["avg_mins_if_start"]
        e_mins_if_sub = 15  # Typical sub appearance
        p_sub = (1 - p_start) * 0.3  # 30% chance of coming off bench if not starting

        e_minutes = (p_start * e_mins_if_start) + (p_sub * e_mins_if_sub)
        e_minutes = max(0.0, min(90.0, e_minutes))

        # Determine rotation risk classification
        rotation_risk = self._classify_rotation_risk(
            p_start, mins_analysis["avg_mins"], risk_factors
        )

        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(player, mins_analysis)

        return MinutesPrediction(
            player_id=player.id,
            player_name=player.web_name,
            p_start=round(p_start, 3),
            p_60_plus=round(p_60_plus, 3),
            p_90_plus=round(p_90_plus, 3),
            e_minutes=round(e_minutes, 1),
            rotation_risk=rotation_risk,
            risk_factors=risk_factors,
            confidence=round(confidence, 2),
            data_points=mins_analysis["games_played"],
        )

    def _analyze_minutes_history(self, player: Player) -> dict:
        """
        Analyze player's historical minutes patterns.

        Returns dict with:
        - avg_mins: Average minutes per game
        - start_rate: Fraction of games started
        - p_60_if_start: P(60+|started)
        - p_90_if_start: P(90|started)
        - avg_mins_if_start: Average mins when starting
        - games_played: Number of games analyzed
        """
        # Use total minutes and points_per_game to estimate games played
        if player.minutes == 0:
            return {
                "avg_mins": 0,
                "start_rate": 0.5,  # Unknown, assume 50%
                "p_60_if_start": 0.85,
                "p_90_if_start": 0.6,
                "avg_mins_if_start": 75,
                "games_played": 0,
            }

        # Estimate games from minutes (assuming ~90 min games)
        # This is approximate - ideally we'd have game-by-game data
        if player.points_per_game > 0:
            estimated_games = player.total_points / player.points_per_game
        else:
            estimated_games = player.minutes / 90

        estimated_games = max(1, estimated_games)
        avg_mins = player.minutes / estimated_games

        # Estimate start rate from average minutes
        # Players with high avg mins usually start
        if avg_mins >= 85:
            start_rate = 0.98
            p_60_if_start = 0.98
            p_90_if_start = 0.85
            avg_mins_if_start = 88
        elif avg_mins >= 70:
            start_rate = 0.90
            p_60_if_start = 0.95
            p_90_if_start = 0.65
            avg_mins_if_start = 82
        elif avg_mins >= 55:
            start_rate = 0.75
            p_60_if_start = 0.90
            p_90_if_start = 0.45
            avg_mins_if_start = 75
        elif avg_mins >= 40:
            start_rate = 0.55
            p_60_if_start = 0.80
            p_90_if_start = 0.30
            avg_mins_if_start = 68
        elif avg_mins >= 20:
            start_rate = 0.35
            p_60_if_start = 0.65
            p_90_if_start = 0.15
            avg_mins_if_start = 55
        else:
            start_rate = 0.15
            p_60_if_start = 0.40
            p_90_if_start = 0.05
            avg_mins_if_start = 40

        return {
            "avg_mins": avg_mins,
            "start_rate": start_rate,
            "p_60_if_start": p_60_if_start,
            "p_90_if_start": p_90_if_start,
            "avg_mins_if_start": avg_mins_if_start,
            "games_played": int(estimated_games),
        }

    def _calculate_depth_factor(self, player: Player) -> float:
        """
        Calculate squad depth factor (competition for place).

        Returns 0.0-1.0 where 1.0 = no competition.
        """
        key = (player.team_id, player.position)
        competitors = self._team_position_players.get(key, [])

        if len(competitors) <= 1:
            return 1.0

        # Sort by minutes (more minutes = more established)
        competitors = sorted(competitors, key=lambda p: p.minutes, reverse=True)

        # Find player's rank
        player_rank = next(
            (i for i, p in enumerate(competitors) if p.id == player.id),
            len(competitors)
        )

        # Adjust based on position requirements
        if player.position == Position.GK:
            # Only 1 GK plays - being #1 is crucial
            return 1.0 if player_rank == 0 else 0.3
        elif player.position == Position.DEF:
            # Typically 3-5 DEF play
            if player_rank < 4:
                return 1.0 - (player_rank * 0.05)
            return 0.7
        elif player.position == Position.MID:
            # Typically 3-5 MID play
            if player_rank < 5:
                return 1.0 - (player_rank * 0.04)
            return 0.65
        else:  # FWD
            # Typically 1-2 FWD play
            if player_rank < 2:
                return 1.0 - (player_rank * 0.1)
            return 0.5

    def _classify_rotation_risk(
        self,
        p_start: float,
        avg_mins: float,
        risk_factors: list[str],
    ) -> RotationRisk:
        """Classify rotation risk based on indicators."""
        if avg_mins == 0 and not risk_factors:
            return RotationRisk.UNKNOWN

        if p_start >= 0.85 and avg_mins >= self.NAILED_MINUTES_THRESHOLD:
            return RotationRisk.LOW
        elif p_start >= 0.65 or avg_mins >= self.ROTATION_MINUTES_THRESHOLD:
            return RotationRisk.MEDIUM
        else:
            return RotationRisk.HIGH

    def _calculate_confidence(
        self,
        player: Player,
        mins_analysis: dict,
    ) -> float:
        """
        Calculate confidence in the prediction.

        Based on:
        - Amount of historical data
        - Player status clarity
        - Consistency of minutes
        """
        base_confidence = 0.5

        # More games = higher confidence
        games = mins_analysis["games_played"]
        if games >= 15:
            base_confidence += 0.3
        elif games >= 8:
            base_confidence += 0.2
        elif games >= 3:
            base_confidence += 0.1

        # Clear status = higher confidence
        if player.status == PlayerStatus.AVAILABLE:
            base_confidence += 0.15
        elif player.status in [PlayerStatus.INJURED, PlayerStatus.SUSPENDED]:
            base_confidence += 0.1  # At least we know something

        # Chance of playing gives more signal
        if player.chance_of_playing is not None:
            base_confidence += 0.05

        return min(1.0, base_confidence)

    def get_all_predictions(self) -> list[MinutesPrediction]:
        """Generate predictions for all players."""
        predictions = []
        for player in self.players.values():
            try:
                pred = self.predict_minutes(player)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Failed to predict minutes for {player.web_name}: {e}")
        return predictions

    def get_rotation_risks(
        self,
        players: list[Player] | None = None,
        min_price: float = 0,
    ) -> list[MinutesPrediction]:
        """
        Get players with significant rotation risk.

        Useful for identifying who might not play despite being selected.
        """
        if players is None:
            players = list(self.players.values())

        risks = []
        for player in players:
            if player.price < min_price:
                continue
            pred = self.predict_minutes(player)
            if pred.is_rotation_risk:
                risks.append(pred)

        # Sort by price (expensive rotation risks are most costly)
        risks.sort(key=lambda p: self.players[p.player_id].price, reverse=True)
        return risks

    def get_nailed_players(
        self,
        position: Position | None = None,
        min_price: float = 0,
    ) -> list[MinutesPrediction]:
        """
        Get players who are considered 'nailed' starters.

        These are the safest picks for consistent minutes.
        """
        nailed = []
        for player in self.players.values():
            if position and player.position != position:
                continue
            if player.price < min_price:
                continue
            pred = self.predict_minutes(player)
            if pred.is_nailed:
                nailed.append(pred)

        # Sort by expected minutes (highest first)
        nailed.sort(key=lambda p: p.e_minutes, reverse=True)
        return nailed


# Convenience functions
def get_minutes_predictor(players: list[Player]) -> MinutesPredictor:
    """Get a minutes predictor instance."""
    return MinutesPredictor(players)


def predict_player_minutes(player: Player, all_players: list[Player]) -> MinutesPrediction:
    """Predict minutes for a single player."""
    predictor = MinutesPredictor(all_players)
    return predictor.predict_minutes(player)
