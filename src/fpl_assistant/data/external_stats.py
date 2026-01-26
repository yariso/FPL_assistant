"""
External Statistics Integration for FPL.

Integrates data from multiple external sources:
1. FPL-Elo-Insights - Elo ratings, match stats, defensive actions
2. Understat - xG/xA data

These provide more accurate stats than the FPL API alone.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# FPL-Elo-Insights data URLs (CSV format, updated twice daily)
FPL_ELO_BASE_URL = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data"
FPL_ELO_PLAYERS_URL = f"{FPL_ELO_BASE_URL}/players.csv"
FPL_ELO_MATCHES_URL = f"{FPL_ELO_BASE_URL}/matches.csv"
FPL_ELO_TEAMS_URL = f"{FPL_ELO_BASE_URL}/teams.csv"


@dataclass
class MatchStats:
    """Match statistics from external sources."""

    match_id: int
    date: datetime
    home_team: str
    away_team: str

    # xG data
    home_xg: float
    away_xg: float

    # Defensive actions (for defensive contribution calculation)
    home_clearances: int
    home_blocks: int
    home_interceptions: int
    home_tackles: int
    away_clearances: int
    away_blocks: int
    away_interceptions: int
    away_tackles: int

    # Elo ratings
    home_elo: float
    away_elo: float


@dataclass
class PlayerMatchStats:
    """Player statistics for a specific match."""

    player_id: int
    player_name: str
    match_id: int

    # Playing time
    minutes: int
    started: bool

    # Offensive
    goals: int
    assists: int
    shots: int
    shots_on_target: int
    xg: float
    xa: float

    # Defensive (for defensive contribution points)
    clearances: int
    blocks: int
    interceptions: int
    tackles: int
    ball_recoveries: int

    @property
    def defensive_contributions_cbit(self) -> int:
        """CBIT: Clearances + Blocks + Interceptions + Tackles (for DEF)."""
        return self.clearances + self.blocks + self.interceptions + self.tackles

    @property
    def defensive_contributions_cbirt(self) -> int:
        """CBIRT: CBIT + Ball Recoveries (for MID/FWD)."""
        return self.defensive_contributions_cbit + self.ball_recoveries

    @property
    def qualifies_def_bonus(self) -> bool:
        """Would qualify for DEF defensive contribution bonus (10+)."""
        return self.defensive_contributions_cbit >= 10

    @property
    def qualifies_mid_fwd_bonus(self) -> bool:
        """Would qualify for MID/FWD defensive contribution bonus (12+)."""
        return self.defensive_contributions_cbirt >= 12


class ExternalStatsLoader:
    """
    Loads and parses external statistics from various sources.

    Supports:
    - FPL-Elo-Insights CSVs
    - Local CSV files
    - Understat API
    """

    def __init__(self, cache_dir: str | Path | None = None):
        """
        Initialize the stats loader.

        Args:
            cache_dir: Directory to cache downloaded CSVs
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/external")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._players_df: pd.DataFrame | None = None
        self._matches_df: pd.DataFrame | None = None
        self._teams_df: pd.DataFrame | None = None

    def load_fpl_elo_data(self, force_refresh: bool = False) -> bool:
        """
        Load data from FPL-Elo-Insights GitHub.

        Args:
            force_refresh: Force download even if cached

        Returns:
            True if successful
        """
        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed, cannot fetch external data")
            return False

        try:
            # Download players data
            players_path = self.cache_dir / "fpl_elo_players.csv"
            if force_refresh or not players_path.exists():
                response = httpx.get(FPL_ELO_PLAYERS_URL, timeout=30)
                response.raise_for_status()
                players_path.write_bytes(response.content)
                logger.info("Downloaded FPL-Elo players data")

            # Download matches data
            matches_path = self.cache_dir / "fpl_elo_matches.csv"
            if force_refresh or not matches_path.exists():
                response = httpx.get(FPL_ELO_MATCHES_URL, timeout=30)
                response.raise_for_status()
                matches_path.write_bytes(response.content)
                logger.info("Downloaded FPL-Elo matches data")

            # Download teams data
            teams_path = self.cache_dir / "fpl_elo_teams.csv"
            if force_refresh or not teams_path.exists():
                response = httpx.get(FPL_ELO_TEAMS_URL, timeout=30)
                response.raise_for_status()
                teams_path.write_bytes(response.content)
                logger.info("Downloaded FPL-Elo teams data")

            # Load into DataFrames
            self._players_df = pd.read_csv(players_path)
            self._matches_df = pd.read_csv(matches_path)
            self._teams_df = pd.read_csv(teams_path)

            logger.info(
                f"Loaded FPL-Elo data: {len(self._players_df)} players, "
                f"{len(self._matches_df)} matches, {len(self._teams_df)} teams"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load FPL-Elo data: {e}")
            return False

    def get_player_defensive_stats(
        self,
        player_id: int,
        last_n_matches: int = 5,
    ) -> dict[str, float]:
        """
        Get defensive stats for a player (for defensive contribution prediction).

        Args:
            player_id: FPL player ID
            last_n_matches: Number of recent matches to average

        Returns:
            Dict with avg defensive actions per 90
        """
        if self._players_df is None:
            return {}

        try:
            # Filter for player
            player_data = self._players_df[
                self._players_df["element"] == player_id
            ].sort_values("kickoff_time", ascending=False)

            if player_data.empty:
                return {}

            # Get last N matches
            recent = player_data.head(last_n_matches)

            # Calculate averages
            total_minutes = recent["minutes"].sum()
            if total_minutes == 0:
                return {}

            per_90_factor = total_minutes / 90

            return {
                "clearances_per_90": recent["clearances"].sum() / per_90_factor,
                "blocks_per_90": recent["blocks"].sum() / per_90_factor,
                "interceptions_per_90": recent["interceptions"].sum() / per_90_factor,
                "tackles_per_90": recent["tackles"].sum() / per_90_factor,
                "ball_recoveries_per_90": recent.get("ball_recoveries", pd.Series([0])).sum() / per_90_factor,
                "cbit_per_90": (
                    recent["clearances"].sum() +
                    recent["blocks"].sum() +
                    recent["interceptions"].sum() +
                    recent["tackles"].sum()
                ) / per_90_factor,
                "matches_analyzed": len(recent),
            }

        except Exception as e:
            logger.warning(f"Failed to get defensive stats for player {player_id}: {e}")
            return {}

    def get_team_elo(self, team_name: str) -> float | None:
        """
        Get current Elo rating for a team.

        Args:
            team_name: Team name

        Returns:
            Elo rating or None
        """
        if self._teams_df is None:
            return None

        try:
            team_data = self._teams_df[
                self._teams_df["team_name"].str.contains(team_name, case=False)
            ]
            if not team_data.empty:
                return team_data.iloc[-1]["elo"]
            return None
        except Exception as e:
            logger.warning(f"Failed to get Elo for {team_name}: {e}")
            return None

    def get_fixture_difficulty_by_elo(
        self,
        team_name: str,
        opponent_name: str,
        is_home: bool,
    ) -> float:
        """
        Calculate fixture difficulty based on Elo ratings.

        Args:
            team_name: Your team
            opponent_name: Opponent team
            is_home: Is playing at home

        Returns:
            Difficulty score (1-5 like FDR)
        """
        team_elo = self.get_team_elo(team_name) or 1500
        opp_elo = self.get_team_elo(opponent_name) or 1500

        # Home advantage adjustment
        if is_home:
            team_elo += 50

        # Calculate win probability
        elo_diff = team_elo - opp_elo
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # Convert to FDR scale (1-5)
        # Higher win prob = easier fixture = lower FDR
        if win_prob >= 0.7:
            return 1  # Very easy
        elif win_prob >= 0.55:
            return 2  # Easy
        elif win_prob >= 0.45:
            return 3  # Medium
        elif win_prob >= 0.35:
            return 4  # Hard
        else:
            return 5  # Very hard

    def estimate_defensive_contribution_probability(
        self,
        player_id: int,
        position: str,
        opponent_name: str,
    ) -> float:
        """
        Estimate probability of hitting defensive contribution threshold.

        Uses historical data to calculate likelihood of 10+ CBIT (DEF)
        or 12+ CBIRT (MID/FWD).

        Args:
            player_id: FPL player ID
            position: Position (DEF, MID, FWD)
            opponent_name: Opponent team name

        Returns:
            Probability (0-1)
        """
        stats = self.get_player_defensive_stats(player_id)
        if not stats:
            # Fallback to base rates
            base_rates = {"DEF": 0.35, "MID": 0.10, "FWD": 0.02}
            return base_rates.get(position, 0.0)

        # Threshold based on position
        if position == "DEF":
            avg_per_90 = stats.get("cbit_per_90", 0)
            threshold = 10
        else:
            avg_per_90 = stats.get("cbit_per_90", 0) + stats.get("ball_recoveries_per_90", 0)
            threshold = 12

        # Simple probability estimation
        # If averaging 8 per 90 with threshold 10, probability ~30%
        # If averaging 12 per 90 with threshold 10, probability ~80%
        if avg_per_90 >= threshold * 1.2:
            return 0.75
        elif avg_per_90 >= threshold:
            return 0.55
        elif avg_per_90 >= threshold * 0.8:
            return 0.35
        elif avg_per_90 >= threshold * 0.6:
            return 0.15
        else:
            return 0.05


# Singleton instance
_loader: ExternalStatsLoader | None = None


def get_external_stats_loader() -> ExternalStatsLoader:
    """Get singleton ExternalStatsLoader."""
    global _loader
    if _loader is None:
        _loader = ExternalStatsLoader()
    return _loader


def get_defensive_contribution_probability(
    player_id: int,
    position: str,
    opponent_name: str = "",
) -> float:
    """
    Quick function to get defensive contribution probability.

    Args:
        player_id: FPL player ID
        position: Position (DEF, MID, FWD)
        opponent_name: Optional opponent for context

    Returns:
        Probability (0-1)
    """
    loader = get_external_stats_loader()

    # Try to load data if not already loaded
    if loader._players_df is None:
        loader.load_fpl_elo_data()

    return loader.estimate_defensive_contribution_probability(
        player_id, position, opponent_name
    )
