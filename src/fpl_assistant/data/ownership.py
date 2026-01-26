"""
Ownership Tracking for Elite FPL Analysis.

Tracks player ownership changes and captain effective ownership (EO)
for differential captaincy decisions.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

from .models import Player
from .storage import Database

logger = logging.getLogger(__name__)


@dataclass
class OwnershipSnapshot:
    """A snapshot of player ownership at a point in time."""

    player_id: int
    gameweek: int
    recorded_date: date
    selected_by_percent: float
    transfers_in: int = 0
    transfers_out: int = 0
    transfers_in_event: int = 0
    transfers_out_event: int = 0


@dataclass
class CaptainEO:
    """Captain Effective Ownership data."""

    player_id: int
    gameweek: int
    overall_eo: float  # % of managers captaining this player
    top_10k_eo: float | None = None  # EO among top 10k
    top_1k_eo: float | None = None  # EO among top 1k
    regular_ownership: float = 0.0  # Regular ownership %
    captain_delta: float = 0.0  # Difference between captain EO and regular


@dataclass
class OwnershipTrend:
    """Ownership trend analysis for a player."""

    player_id: int
    player_name: str
    current_ownership: float
    ownership_change_1d: float  # Change in last day
    ownership_change_7d: float  # Change in last 7 days
    net_transfers_event: int  # Net transfers this gameweek
    trend: str  # "rising", "falling", "stable"
    momentum: float  # Rate of change


class OwnershipTracker:
    """
    Tracks player ownership for elite FPL analysis.

    Used for:
    1. Differential player identification
    2. Captain EO estimation
    3. Price change prediction (via transfer velocity)
    4. Ownership trajectory analysis
    """

    # Thresholds for trend classification
    RISING_THRESHOLD = 0.5  # >0.5% gain = rising
    FALLING_THRESHOLD = -0.5  # <-0.5% loss = falling

    def __init__(self, db: Database):
        """Initialize the ownership tracker."""
        self.db = db

    def record_ownership_snapshot(
        self,
        players: list[Player],
        gameweek: int,
        recorded_date: date | None = None,
    ) -> int:
        """
        Record current ownership for all players.

        Args:
            players: List of players with current ownership data
            gameweek: Current gameweek
            recorded_date: Date of snapshot (defaults to today)

        Returns:
            Number of records inserted
        """
        if recorded_date is None:
            recorded_date = date.today()

        count = 0
        with self.db.connection() as conn:
            for player in players:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO ownership_history
                        (player_id, gameweek, recorded_date, selected_by_percent,
                         transfers_in, transfers_out, transfers_in_event, transfers_out_event)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            player.id,
                            gameweek,
                            recorded_date.isoformat(),
                            player.selected_by_percent,
                            getattr(player, "transfers_in", 0),
                            getattr(player, "transfers_out", 0),
                            getattr(player, "transfers_in_event", 0),
                            getattr(player, "transfers_out_event", 0),
                        ),
                    )
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to record ownership for player {player.id}: {e}")

            conn.commit()

        logger.info(f"Recorded ownership for {count} players for GW{gameweek}")
        return count

    def get_ownership_history(
        self,
        player_id: int,
        days: int = 7,
    ) -> list[OwnershipSnapshot]:
        """
        Get ownership history for a player.

        Args:
            player_id: Player ID
            days: Number of days of history

        Returns:
            List of ownership snapshots
        """
        cutoff = (date.today() - timedelta(days=days)).isoformat()

        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT player_id, gameweek, recorded_date, selected_by_percent,
                       transfers_in, transfers_out, transfers_in_event, transfers_out_event
                FROM ownership_history
                WHERE player_id = ? AND recorded_date >= ?
                ORDER BY recorded_date DESC
                """,
                (player_id, cutoff),
            ).fetchall()

        return [
            OwnershipSnapshot(
                player_id=row[0],
                gameweek=row[1],
                recorded_date=date.fromisoformat(row[2]) if isinstance(row[2], str) else row[2],
                selected_by_percent=row[3],
                transfers_in=row[4] or 0,
                transfers_out=row[5] or 0,
                transfers_in_event=row[6] or 0,
                transfers_out_event=row[7] or 0,
            )
            for row in rows
        ]

    def calculate_ownership_trend(
        self,
        player: Player,
        history: list[OwnershipSnapshot] | None = None,
    ) -> OwnershipTrend:
        """
        Calculate ownership trend for a player.

        Args:
            player: Player object
            history: Optional pre-fetched history

        Returns:
            OwnershipTrend with analysis
        """
        if history is None:
            history = self.get_ownership_history(player.id, days=7)

        current = player.selected_by_percent

        # Calculate changes
        change_1d = 0.0
        change_7d = 0.0

        if history:
            # 1-day change (most recent vs current)
            if len(history) >= 1:
                change_1d = current - history[0].selected_by_percent

            # 7-day change (oldest in history vs current)
            if len(history) >= 2:
                change_7d = current - history[-1].selected_by_percent

        # Net transfers this event
        net_transfers = getattr(player, "transfers_in_event", 0) - getattr(player, "transfers_out_event", 0)

        # Determine trend
        if change_7d > self.RISING_THRESHOLD:
            trend = "rising"
        elif change_7d < self.FALLING_THRESHOLD:
            trend = "falling"
        else:
            trend = "stable"

        # Momentum: rate of change per day
        momentum = change_7d / 7 if len(history) >= 2 else change_1d

        return OwnershipTrend(
            player_id=player.id,
            player_name=player.web_name,
            current_ownership=current,
            ownership_change_1d=change_1d,
            ownership_change_7d=change_7d,
            net_transfers_event=net_transfers,
            trend=trend,
            momentum=momentum,
        )

    def estimate_captain_eo(
        self,
        player: Player,
        gameweek: int,
        captain_picks: list[tuple[Player, float]] | None = None,
    ) -> CaptainEO:
        """
        Estimate captain effective ownership for a player.

        Captain EO is typically higher than regular ownership for popular picks.
        We estimate it based on:
        - Regular ownership
        - Form ranking
        - Fixture difficulty
        - Historical captain pick patterns

        Args:
            player: Player to estimate EO for
            gameweek: Gameweek number
            captain_picks: Optional list of (player, xP) from captain analysis

        Returns:
            CaptainEO estimate
        """
        regular_eo = player.selected_by_percent

        # Base captain EO estimation
        # Top captain picks typically have 2-5x their regular ownership as captain EO
        # e.g., Salah at 40% ownership might be captained by 60-80% of managers

        # Factors that increase captain EO:
        # 1. High form (more likely to be captained)
        form_boost = min(2.0, 1.0 + player.form / 10)  # Up to 2x boost for form 10+

        # 2. Position (MID/FWD captained more than DEF/GK)
        position_mult = {1: 0.1, 2: 0.5, 3: 1.2, 4: 1.5}.get(player.position.value, 1.0)

        # 3. High ownership = high captain probability
        ownership_factor = regular_eo / 100  # 0-1 scale

        # Estimate captain EO as percentage of managers
        # Formula: regular_eo * form_boost * position_mult, capped at 90%
        estimated_captain_eo = min(90.0, regular_eo * form_boost * position_mult)

        # For truly elite picks (high ownership + high form), boost further
        if regular_eo > 50 and player.form > 7:
            estimated_captain_eo = min(90.0, estimated_captain_eo * 1.3)

        # Captain delta shows how much more they're captained vs owned
        captain_delta = estimated_captain_eo - regular_eo

        return CaptainEO(
            player_id=player.id,
            gameweek=gameweek,
            overall_eo=estimated_captain_eo,
            top_10k_eo=None,  # Would need livefpl.net data
            top_1k_eo=None,
            regular_ownership=regular_eo,
            captain_delta=captain_delta,
        )

    def save_captain_eo(self, captain_eo: CaptainEO) -> None:
        """Save captain EO to database."""
        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO captain_eo
                (player_id, gameweek, overall_eo, top_10k_eo, top_1k_eo,
                 regular_ownership, captain_delta, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    captain_eo.player_id,
                    captain_eo.gameweek,
                    captain_eo.overall_eo,
                    captain_eo.top_10k_eo,
                    captain_eo.top_1k_eo,
                    captain_eo.regular_ownership,
                    captain_eo.captain_delta,
                    datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_rising_players(
        self,
        players: list[Player],
        min_ownership: float = 1.0,
        limit: int = 10,
    ) -> list[OwnershipTrend]:
        """
        Get players with rising ownership (potential price rises).

        Args:
            players: List of players
            min_ownership: Minimum ownership to consider
            limit: Number of players to return

        Returns:
            List of rising players sorted by momentum
        """
        trends = []
        for player in players:
            if player.selected_by_percent < min_ownership:
                continue

            trend = self.calculate_ownership_trend(player)
            if trend.trend == "rising":
                trends.append(trend)

        # Sort by momentum (fastest rising first)
        trends.sort(key=lambda t: t.momentum, reverse=True)
        return trends[:limit]

    def get_falling_players(
        self,
        players: list[Player],
        min_ownership: float = 1.0,
        limit: int = 10,
    ) -> list[OwnershipTrend]:
        """
        Get players with falling ownership (potential price falls).

        Args:
            players: List of players
            min_ownership: Minimum ownership to consider
            limit: Number of players to return

        Returns:
            List of falling players sorted by momentum (most negative first)
        """
        trends = []
        for player in players:
            if player.selected_by_percent < min_ownership:
                continue

            trend = self.calculate_ownership_trend(player)
            if trend.trend == "falling":
                trends.append(trend)

        # Sort by momentum (fastest falling first)
        trends.sort(key=lambda t: t.momentum)
        return trends[:limit]

    def get_differential_candidates(
        self,
        players: list[Player],
        max_ownership: float = 10.0,
        min_form: float = 3.0,
        limit: int = 10,
    ) -> list[tuple[Player, OwnershipTrend]]:
        """
        Find differential candidates (low ownership, good potential).

        Args:
            players: List of players
            max_ownership: Maximum ownership percentage
            min_form: Minimum form rating
            limit: Number of candidates to return

        Returns:
            List of (player, trend) tuples
        """
        candidates = []

        for player in players:
            if player.selected_by_percent > max_ownership:
                continue
            if player.form < min_form:
                continue
            if not player.is_available:
                continue

            trend = self.calculate_ownership_trend(player)

            # Prefer rising differentials
            score = player.form + (trend.momentum * 10 if trend.trend == "rising" else 0)
            candidates.append((player, trend, score))

        # Sort by score
        candidates.sort(key=lambda x: x[2], reverse=True)

        return [(p, t) for p, t, _ in candidates[:limit]]


# Convenience functions
def get_ownership_tracker(db: Database) -> OwnershipTracker:
    """Get an ownership tracker instance."""
    return OwnershipTracker(db)
