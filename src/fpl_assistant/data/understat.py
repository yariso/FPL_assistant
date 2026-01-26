"""
Understat Data Integration for Enhanced FPL Stats.

Uses understatAPI to fetch real xG/xA data and player statistics
that aren't available in the official FPL API.

Install: pip install understatapi

Key data points from Understat:
- Accurate xG/xA per shot
- Shot locations and types
- Key passes
- Match-by-match breakdown
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UnderstatPlayer:
    """Player data from Understat."""

    understat_id: int
    name: str
    team: str

    # Season totals
    games: int
    minutes: int
    goals: int
    assists: int
    xg: float
    xa: float
    npxg: float  # Non-penalty xG
    xg_chain: float  # xG from build-up involvement
    xg_buildup: float  # xG from passes leading to shots

    # Per 90 stats (calculated)
    xg_per_90: float
    xa_per_90: float
    npxg_per_90: float

    # Shots
    shots: int
    key_passes: int

    @property
    def xgi(self) -> float:
        """Expected Goal Involvement."""
        return self.xg + self.xa

    @property
    def xgi_per_90(self) -> float:
        """xGI per 90 minutes."""
        return self.xg_per_90 + self.xa_per_90

    @property
    def goals_minus_xg(self) -> float:
        """Goals above/below expected (luck indicator)."""
        return self.goals - self.xg


@dataclass
class UnderstatMatch:
    """Match-level data from Understat."""

    match_id: int
    date: datetime
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    home_goals: int
    away_goals: int


class UnderstatFetcher:
    """
    Fetches data from Understat via understatAPI.

    Usage:
    ```
    fetcher = UnderstatFetcher()
    players = fetcher.get_all_players(season="2025")
    salah = fetcher.get_player_by_name("Mohamed Salah")
    ```
    """

    # Map FPL team names to Understat team names
    TEAM_NAME_MAP = {
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth",
        "Brentford": "Brentford",
        "Brighton": "Brighton",
        "Burnley": "Burnley",
        "Chelsea": "Chelsea",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Ipswich": "Ipswich",
        "Leeds": "Leeds",
        "Leicester": "Leicester",
        "Liverpool": "Liverpool",
        "Man City": "Manchester City",
        "Man Utd": "Manchester United",
        "Newcastle": "Newcastle United",
        "Nott'm Forest": "Nottingham Forest",
        "Southampton": "Southampton",
        "Spurs": "Tottenham",
        "Sunderland": "Sunderland",
        "West Ham": "West Ham",
        "Wolves": "Wolverhampton Wanderers",
    }

    def __init__(self):
        """Initialize the Understat fetcher."""
        self._client = None
        self._player_cache: dict[str, UnderstatPlayer] = {}
        self._initialized = False

    def _ensure_client(self):
        """Ensure understatAPI client is available."""
        if self._client is not None:
            return True

        try:
            from understatapi import UnderstatClient
            self._client = UnderstatClient()
            self._initialized = True
            return True
        except ImportError:
            logger.warning(
                "understatapi not installed. Install with: pip install understatapi"
            )
            return False

    def get_all_players(self, season: str = "2025") -> list[UnderstatPlayer]:
        """
        Get all Premier League players for a season.

        Args:
            season: Season year (e.g., "2025" for 2025/26)

        Returns:
            List of UnderstatPlayer objects
        """
        if not self._ensure_client():
            return []

        try:
            with self._client as understat:
                data = understat.league(league="EPL").get_player_data(season=season)

            players = []
            for p in data:
                minutes = int(p.get("time", 0))
                per_90_factor = minutes / 90 if minutes > 0 else 1

                player = UnderstatPlayer(
                    understat_id=int(p.get("id", 0)),
                    name=p.get("player_name", ""),
                    team=p.get("team_title", ""),
                    games=int(p.get("games", 0)),
                    minutes=minutes,
                    goals=int(p.get("goals", 0)),
                    assists=int(p.get("assists", 0)),
                    xg=float(p.get("xG", 0)),
                    xa=float(p.get("xA", 0)),
                    npxg=float(p.get("npxG", 0)),
                    xg_chain=float(p.get("xGChain", 0)),
                    xg_buildup=float(p.get("xGBuildup", 0)),
                    xg_per_90=float(p.get("xG", 0)) / per_90_factor if per_90_factor > 0 else 0,
                    xa_per_90=float(p.get("xA", 0)) / per_90_factor if per_90_factor > 0 else 0,
                    npxg_per_90=float(p.get("npxG", 0)) / per_90_factor if per_90_factor > 0 else 0,
                    shots=int(p.get("shots", 0)),
                    key_passes=int(p.get("key_passes", 0)),
                )
                players.append(player)
                self._player_cache[player.name.lower()] = player

            logger.info(f"Fetched {len(players)} players from Understat")
            return players

        except Exception as e:
            logger.error(f"Failed to fetch Understat players: {e}")
            return []

    def get_player_by_name(self, name: str) -> UnderstatPlayer | None:
        """
        Get a specific player by name.

        Args:
            name: Player name (fuzzy matching supported)

        Returns:
            UnderstatPlayer or None if not found
        """
        # Check cache first
        name_lower = name.lower()
        if name_lower in self._player_cache:
            return self._player_cache[name_lower]

        # Try partial match
        for cached_name, player in self._player_cache.items():
            if name_lower in cached_name or cached_name in name_lower:
                return player

        return None

    def get_team_xg(self, team_name: str, season: str = "2025") -> dict[str, float]:
        """
        Get team-level xG statistics.

        Args:
            team_name: Team name (FPL format)
            season: Season year

        Returns:
            Dict with xG stats: {xg_for, xg_against, xg_diff}
        """
        if not self._ensure_client():
            return {}

        # Map FPL team name to Understat name
        understat_name = self.TEAM_NAME_MAP.get(team_name, team_name)

        try:
            with self._client as understat:
                data = understat.team(team=understat_name, season=season).get_stats_data()

            # Parse team stats
            xg_for = sum(float(m.get("xG", 0)) for m in data.get("situation", {}).values())
            xg_against = sum(float(m.get("xGA", 0)) for m in data.get("situation", {}).values())

            return {
                "xg_for": xg_for,
                "xg_against": xg_against,
                "xg_diff": xg_for - xg_against,
            }

        except Exception as e:
            logger.error(f"Failed to fetch team xG for {team_name}: {e}")
            return {}

    def get_player_shot_data(self, player_id: int) -> list[dict]:
        """
        Get detailed shot data for a player.

        Useful for understanding:
        - Shot quality (xG per shot)
        - Shot locations
        - Big chances

        Args:
            player_id: Understat player ID

        Returns:
            List of shot dicts with xG, result, etc.
        """
        if not self._ensure_client():
            return []

        try:
            with self._client as understat:
                data = understat.player(player=player_id).get_shot_data()

            return data

        except Exception as e:
            logger.error(f"Failed to fetch shot data for player {player_id}: {e}")
            return []


class UnderstatEnhancer:
    """
    Enhances FPL player data with Understat stats.

    Merges accurate xG/xA from Understat with FPL player objects.
    """

    def __init__(self):
        """Initialize the enhancer."""
        self.fetcher = UnderstatFetcher()
        self._understat_players: dict[str, UnderstatPlayer] = {}

    def load_understat_data(self, season: str = "2025") -> bool:
        """
        Load all Understat data for the season.

        Args:
            season: Season year

        Returns:
            True if successful
        """
        players = self.fetcher.get_all_players(season)
        if not players:
            return False

        for p in players:
            self._understat_players[p.name.lower()] = p

        logger.info(f"Loaded {len(players)} Understat players for enhancement")
        return True

    def enhance_player(self, fpl_player: Any) -> dict[str, float]:
        """
        Get enhanced stats for an FPL player.

        Args:
            fpl_player: FPL Player object

        Returns:
            Dict with enhanced stats
        """
        # Try to find matching Understat player
        name = fpl_player.name.lower() if hasattr(fpl_player, 'name') else ""
        web_name = fpl_player.web_name.lower() if hasattr(fpl_player, 'web_name') else ""

        understat = None

        # Try full name first
        if name in self._understat_players:
            understat = self._understat_players[name]
        else:
            # Try partial matches
            for us_name, us_player in self._understat_players.items():
                if web_name in us_name or us_name in name:
                    understat = us_player
                    break

        if not understat:
            return {}

        return {
            "understat_xg": understat.xg,
            "understat_xa": understat.xa,
            "understat_npxg": understat.npxg,
            "understat_xg_per_90": understat.xg_per_90,
            "understat_xa_per_90": understat.xa_per_90,
            "understat_xgi_per_90": understat.xgi_per_90,
            "understat_shots": understat.shots,
            "understat_key_passes": understat.key_passes,
            "goals_minus_xg": understat.goals_minus_xg,
        }

    def get_underperformers(self, min_minutes: int = 450) -> list[tuple[str, float, float]]:
        """
        Find players underperforming their xG (due to score more).

        Args:
            min_minutes: Minimum minutes filter

        Returns:
            List of (name, xG, goals) tuples sorted by biggest gap
        """
        results = []
        for player in self._understat_players.values():
            if player.minutes < min_minutes:
                continue
            gap = player.xg - player.goals
            if gap > 1:  # At least 1 goal under
                results.append((player.name, player.xg, player.goals))

        results.sort(key=lambda x: x[1] - x[2], reverse=True)
        return results[:10]

    def get_overperformers(self, min_minutes: int = 450) -> list[tuple[str, float, float]]:
        """
        Find players overperforming their xG (due to regress).

        Args:
            min_minutes: Minimum minutes filter

        Returns:
            List of (name, xG, goals) tuples sorted by biggest gap
        """
        results = []
        for player in self._understat_players.values():
            if player.minutes < min_minutes:
                continue
            gap = player.goals - player.xg
            if gap > 1:  # At least 1 goal over
                results.append((player.name, player.xg, player.goals))

        results.sort(key=lambda x: x[2] - x[1], reverse=True)
        return results[:10]


# Convenience functions
_fetcher: UnderstatFetcher | None = None
_enhancer: UnderstatEnhancer | None = None


def get_understat_fetcher() -> UnderstatFetcher:
    """Get singleton Understat fetcher."""
    global _fetcher
    if _fetcher is None:
        _fetcher = UnderstatFetcher()
    return _fetcher


def get_understat_enhancer() -> UnderstatEnhancer:
    """Get singleton Understat enhancer."""
    global _enhancer
    if _enhancer is None:
        _enhancer = UnderstatEnhancer()
    return _enhancer


def fetch_understat_xg(player_name: str, season: str = "2025") -> dict[str, float]:
    """
    Quick function to get Understat xG for a player.

    Args:
        player_name: Player name
        season: Season year

    Returns:
        Dict with xG stats
    """
    enhancer = get_understat_enhancer()
    if not enhancer._understat_players:
        enhancer.load_understat_data(season)

    player = enhancer.fetcher.get_player_by_name(player_name)
    if player:
        return {
            "xg": player.xg,
            "xa": player.xa,
            "xgi": player.xgi,
            "xg_per_90": player.xg_per_90,
            "xa_per_90": player.xa_per_90,
        }
    return {}
