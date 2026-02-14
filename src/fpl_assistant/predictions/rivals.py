"""
Mini-League Rival Tracking Module.

Analyzes rival teams in your mini-league to identify:
- Ownership overlap (common players)
- Differential opportunities (players they don't have)
- Strategy recommendations (match vs differentiate)
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from fpl_assistant.data.models import Player, PlayerProjection


class RivalStrategy(StrEnum):
    """Recommended strategy against rivals."""

    MATCH = "MATCH"  # You're behind - match their picks
    DIFFERENTIATE = "DIFFERENTIATE"  # You're ahead - pick different players
    NEUTRAL = "NEUTRAL"  # Close - balance both strategies


@dataclass
class RivalEntry:
    """A rival manager's entry from league standings."""

    manager_id: int
    manager_name: str
    team_name: str
    rank: int
    total_points: int
    gameweek_points: int


@dataclass
class RivalTeam:
    """A rival's team for a specific gameweek."""

    manager_id: int
    player_ids: list[int]  # All 15 players
    starting_ids: list[int]  # 11 starters
    captain_id: int
    vice_captain_id: int
    chip_used: str | None = None


@dataclass
class RivalAnalysis:
    """Analysis of a single rival."""

    rival: RivalEntry
    points_gap: int  # Positive = you're ahead
    common_players: list[int]  # Player IDs you both own
    their_differentials: list[int]  # They own, you don't
    your_differentials: list[int]  # You own, they don't
    recommended_strategy: RivalStrategy

    @property
    def overlap_percentage(self) -> float:
        """Percentage of players you share with this rival."""
        total_players = len(self.common_players) + len(self.their_differentials)
        if total_players == 0:
            return 0.0
        return (len(self.common_players) / total_players) * 100

    @property
    def is_threat(self) -> bool:
        """Is this rival a threat to overtake you?"""
        return self.points_gap < 50  # Within 50 points


@dataclass
class DifferentialPick:
    """A player that could be a differential vs rivals."""

    player: Player
    projected_points: float
    rivals_owning: int  # How many rivals own this player
    ownership_in_league: float  # % of rivals who own
    is_template: bool  # Is this a common pick across rivals?

    @property
    def differential_score(self) -> float:
        """Higher = better differential (high xP, low rival ownership)."""
        ownership_factor = 1.0 - (self.ownership_in_league / 100)
        return self.projected_points * ownership_factor


@dataclass
class CaptainRiskMatrix:
    """Risk analysis for captain pick vs rivals."""

    captain: Player
    rivals_also_captaining: int
    rivals_owning_not_captaining: int
    rivals_not_owning: int

    # Outcome analysis
    haul_gain_if_unique: float  # Points gained if captain hauls and rivals don't have
    blank_loss_if_shared: float  # Points lost if captain blanks and rivals have diff captain

    @property
    def risk_level(self) -> str:
        """Risk level of this captain pick."""
        if self.rivals_also_captaining > 3:
            return "LOW RISK"  # Safe pick - many rivals same captain
        elif self.rivals_not_owning > 3:
            return "HIGH RISK/REWARD"  # Big swing potential
        return "MEDIUM RISK"


@dataclass
class LeagueAnalysis:
    """Complete analysis of a mini-league."""

    league_id: int
    league_name: str
    your_rank: int
    your_points: int
    rivals: list[RivalAnalysis] = field(default_factory=list)
    differential_targets: list[DifferentialPick] = field(default_factory=list)
    overall_strategy: RivalStrategy = RivalStrategy.NEUTRAL


class RivalTracker:
    """
    Tracks and analyzes mini-league rivals.

    Provides differential analysis and strategy recommendations
    for competing in mini-leagues.
    """

    # Strategy thresholds
    MATCH_THRESHOLD = -30  # Behind by 30+ points = MATCH strategy
    DIFFERENTIATE_THRESHOLD = 30  # Ahead by 30+ points = DIFFERENTIATE

    def __init__(
        self,
        players: dict[int, Player],
        projections: dict[int, float] | None = None,
    ):
        """
        Initialize tracker.

        Args:
            players: Dict of player_id -> Player
            projections: Dict of player_id -> projected points (optional)
        """
        self.players = players
        self.projections = projections or {}

    def analyze_rival(
        self,
        rival: RivalEntry,
        rival_team: RivalTeam,
        your_team: list[int],
        your_points: int,
    ) -> RivalAnalysis:
        """
        Analyze a single rival.

        Args:
            rival: Rival entry from league standings
            rival_team: Rival's team composition
            your_team: Your team player IDs
            your_points: Your total points
        """
        your_set = set(your_team)
        rival_set = set(rival_team.player_ids)

        common = list(your_set & rival_set)
        their_diff = list(rival_set - your_set)
        your_diff = list(your_set - rival_set)

        points_gap = your_points - rival.total_points

        # Determine strategy
        if points_gap <= self.MATCH_THRESHOLD:
            strategy = RivalStrategy.MATCH
        elif points_gap >= self.DIFFERENTIATE_THRESHOLD:
            strategy = RivalStrategy.DIFFERENTIATE
        else:
            strategy = RivalStrategy.NEUTRAL

        return RivalAnalysis(
            rival=rival,
            points_gap=points_gap,
            common_players=common,
            their_differentials=their_diff,
            your_differentials=your_diff,
            recommended_strategy=strategy,
        )

    def analyze_league(
        self,
        league_id: int,
        league_name: str,
        standings: list[RivalEntry],
        rival_teams: dict[int, RivalTeam],
        your_team: list[int],
        your_manager_id: int,
    ) -> LeagueAnalysis:
        """
        Analyze entire mini-league.

        Args:
            league_id: League ID
            league_name: League name
            standings: League standings
            rival_teams: Dict of manager_id -> RivalTeam
            your_team: Your team player IDs
            your_manager_id: Your manager ID
        """
        # Find your position
        your_entry = next(
            (e for e in standings if e.manager_id == your_manager_id),
            None
        )
        your_rank = your_entry.rank if your_entry else 0
        your_points = your_entry.total_points if your_entry else 0

        # Analyze rivals (top 10 or those within 100 points)
        rival_analyses = []
        for rival in standings:
            if rival.manager_id == your_manager_id:
                continue
            if rival.rank > 10 and abs(your_points - rival.total_points) > 100:
                continue

            rival_team = rival_teams.get(rival.manager_id)
            if rival_team:
                analysis = self.analyze_rival(
                    rival, rival_team, your_team, your_points
                )
                rival_analyses.append(analysis)

        # Find differential targets
        differential_targets = self._find_differential_targets(
            your_team, rival_teams, your_manager_id
        )

        # Determine overall strategy
        if rival_analyses:
            avg_gap = sum(r.points_gap for r in rival_analyses) / len(rival_analyses)
            if avg_gap <= self.MATCH_THRESHOLD:
                overall_strategy = RivalStrategy.MATCH
            elif avg_gap >= self.DIFFERENTIATE_THRESHOLD:
                overall_strategy = RivalStrategy.DIFFERENTIATE
            else:
                overall_strategy = RivalStrategy.NEUTRAL
        else:
            overall_strategy = RivalStrategy.NEUTRAL

        return LeagueAnalysis(
            league_id=league_id,
            league_name=league_name,
            your_rank=your_rank,
            your_points=your_points,
            rivals=rival_analyses,
            differential_targets=differential_targets,
            overall_strategy=overall_strategy,
        )

    def _find_differential_targets(
        self,
        your_team: list[int],
        rival_teams: dict[int, RivalTeam],
        your_manager_id: int,
    ) -> list[DifferentialPick]:
        """Find players that could be differentials vs rivals."""
        your_set = set(your_team)

        # Count rival ownership
        rival_ownership: dict[int, int] = {}
        num_rivals = 0

        for manager_id, team in rival_teams.items():
            if manager_id == your_manager_id:
                continue
            num_rivals += 1
            for player_id in team.player_ids:
                rival_ownership[player_id] = rival_ownership.get(player_id, 0) + 1

        if num_rivals == 0:
            return []

        # Find potential differentials
        targets = []
        for player_id, player in self.players.items():
            # Skip your current players
            if player_id in your_set:
                continue

            # Skip unavailable players
            if not player.is_available:
                continue

            rivals_owning = rival_ownership.get(player_id, 0)
            ownership_pct = (rivals_owning / num_rivals) * 100 if num_rivals > 0 else 0

            # Get projected points
            proj_points = self.projections.get(player_id, player.points_per_game)

            # Only include players with decent projections
            if proj_points < 4.0:
                continue

            targets.append(DifferentialPick(
                player=player,
                projected_points=proj_points,
                rivals_owning=rivals_owning,
                ownership_in_league=ownership_pct,
                is_template=ownership_pct > 50,
            ))

        # Sort by differential score
        return sorted(targets, key=lambda t: -t.differential_score)[:20]

    def get_captain_risk_matrix(
        self,
        captain: Player,
        rival_teams: dict[int, RivalTeam],
        your_manager_id: int,
    ) -> CaptainRiskMatrix:
        """
        Analyze captain pick risk vs rivals.

        Shows how your captain pick compares to rival choices.
        """
        captain_id = captain.id
        also_captaining = 0
        owning_not_captain = 0
        not_owning = 0

        for manager_id, team in rival_teams.items():
            if manager_id == your_manager_id:
                continue

            if team.captain_id == captain_id:
                also_captaining += 1
            elif captain_id in team.player_ids:
                owning_not_captain += 1
            else:
                not_owning += 1

        # Estimate point swings
        avg_captain_points = 8.0  # Approximate average captain return
        haul_points = 15.0  # Points in a haul

        haul_gain = not_owning * haul_points  # Gain if you haul, rivals don't have
        blank_loss = also_captaining * 0 + owning_not_captain * avg_captain_points

        return CaptainRiskMatrix(
            captain=captain,
            rivals_also_captaining=also_captaining,
            rivals_owning_not_captaining=owning_not_captain,
            rivals_not_owning=not_owning,
            haul_gain_if_unique=haul_gain,
            blank_loss_if_shared=blank_loss,
        )


def parse_league_standings(data: dict[str, Any]) -> tuple[str, list[RivalEntry]]:
    """
    Parse league standings from FPL API response.

    Returns:
        Tuple of (league_name, list of RivalEntry)
    """
    league_name = data.get("league", {}).get("name", "Unknown League")
    standings_data = data.get("standings", {}).get("results", [])

    entries = []
    for entry in standings_data:
        entries.append(RivalEntry(
            manager_id=entry.get("entry", 0),
            manager_name=entry.get("player_name", "Unknown"),
            team_name=entry.get("entry_name", "Unknown Team"),
            rank=entry.get("rank", 0),
            total_points=entry.get("total", 0),
            gameweek_points=entry.get("event_total", 0),
        ))

    return league_name, entries


def parse_rival_team(picks_data: dict[str, Any]) -> RivalTeam:
    """
    Parse rival team from FPL API picks response.

    Returns:
        RivalTeam with player composition
    """
    picks = picks_data.get("picks", [])

    all_player_ids = [p["element"] for p in picks]
    starting_ids = [p["element"] for p in picks if p["position"] <= 11]

    captain_id = next(
        (p["element"] for p in picks if p["is_captain"]),
        all_player_ids[0] if all_player_ids else 0
    )
    vice_captain_id = next(
        (p["element"] for p in picks if p["is_vice_captain"]),
        all_player_ids[1] if len(all_player_ids) > 1 else 0
    )

    chip_used = None
    active_chip = picks_data.get("active_chip")
    if active_chip:
        chip_used = active_chip

    return RivalTeam(
        manager_id=0,  # Set by caller
        player_ids=all_player_ids,
        starting_ids=starting_ids,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        chip_used=chip_used,
    )


def parse_h2h_standings(data: dict[str, Any]) -> tuple[str, list[RivalEntry]]:
    """
    Parse H2H league standings from FPL API response.

    H2H leagues use matches_won/drawn/lost and points_for instead of total.

    Returns:
        Tuple of (league_name, list of RivalEntry)
    """
    league_name = data.get("league", {}).get("name", "Unknown H2H League")
    standings_data = data.get("standings", {}).get("results", [])

    entries = []
    for entry in standings_data:
        # H2H points = 3*wins + 1*draws
        h2h_points = entry.get("total", 0)
        # points_for = actual FPL points scored (use for gap analysis)
        fpl_points = entry.get("points_for", 0)

        entries.append(RivalEntry(
            manager_id=entry.get("entry", 0),
            manager_name=entry.get("player_name", "Unknown"),
            team_name=entry.get("entry_name", "Unknown Team"),
            rank=entry.get("rank", 0),
            total_points=fpl_points,  # Use FPL points for analysis
            gameweek_points=entry.get("event_total", 0),
        ))

    return league_name, entries
