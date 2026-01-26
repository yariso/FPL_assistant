"""
Pydantic data models for FPL Assistant.

These models represent the core data structures for players, teams, fixtures,
squads, and optimization plans.
"""

from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Any

from pydantic import BaseModel, Field, computed_field


class Position(IntEnum):
    """Player positions in FPL (matches FPL element_type)."""

    GK = 1
    DEF = 2
    MID = 3
    FWD = 4


class PlayerStatus(StrEnum):
    """Player availability status."""

    AVAILABLE = "a"
    DOUBTFUL = "d"
    INJURED = "i"
    UNAVAILABLE = "u"
    NOT_AVAILABLE = "n"
    SUSPENDED = "s"


class ChipType(StrEnum):
    """Available FPL chips."""

    WILDCARD = "wildcard"
    FREE_HIT = "freehit"
    BENCH_BOOST = "bboost"
    TRIPLE_CAPTAIN = "3xc"


class ChipSet(IntEnum):
    """
    2025/26 FPL has TWO sets of chips.

    First set must be used BEFORE GW19 deadline.
    Second set becomes available after GW19.
    """

    FIRST_HALF = 1   # Must use before GW19
    SECOND_HALF = 2  # Available after GW19


# 2025/26 Season Rule: Chip deadline for first set
CHIP_FIRST_HALF_DEADLINE = 19  # Must use first set of chips before GW19


# =============================================================================
# Core Data Models
# =============================================================================


class Player(BaseModel):
    """Represents a Premier League player in FPL."""

    id: int = Field(description="FPL element ID")
    name: str = Field(description="Player's display name")
    web_name: str = Field(description="Short name shown in FPL")
    team_id: int = Field(description="Premier League team ID")
    position: Position = Field(description="Playing position")
    price: float = Field(description="Current price in millions (e.g., 10.5)")
    status: PlayerStatus = Field(
        default=PlayerStatus.AVAILABLE, description="Availability status"
    )
    news: str = Field(default="", description="Injury/suspension news")
    chance_of_playing: int | None = Field(
        default=None, description="Chance of playing next round (0-100)"
    )

    # Stats
    total_points: int = Field(default=0, description="Total FPL points this season")
    points_per_game: float = Field(default=0.0, description="Average points per game")
    form: float = Field(default=0.0, description="Recent form rating")
    selected_by_percent: float = Field(default=0.0, description="Ownership percentage")
    ict_index: float = Field(default=0.0, description="ICT index score")

    # Additional stats (optional)
    goals_scored: int = Field(default=0)
    assists: int = Field(default=0)
    clean_sheets: int = Field(default=0)
    minutes: int = Field(default=0)

    # xG stats (key for accurate projections!)
    expected_goals: float = Field(default=0.0, description="Cumulative xG this season")
    expected_assists: float = Field(default=0.0, description="Cumulative xA this season")
    expected_goal_involvements: float = Field(default=0.0, description="xG + xA combined")
    expected_goals_per_90: float = Field(default=0.0, description="xG per 90 minutes")
    expected_assists_per_90: float = Field(default=0.0, description="xA per 90 minutes")

    # Additional stats needed for accurate projections
    bonus: int = Field(default=0, description="Total bonus points this season")
    yellow_cards: int = Field(default=0, description="Yellow cards this season")
    red_cards: int = Field(default=0, description="Red cards this season")
    saves: int = Field(default=0, description="Saves this season (GK)")
    goals_conceded: int = Field(default=0, description="Goals conceded (GK/DEF)")
    own_goals: int = Field(default=0, description="Own goals this season")
    penalties_missed: int = Field(default=0, description="Penalties missed this season")
    penalties_saved: int = Field(default=0, description="Penalties saved (GK)")

    @computed_field
    @property
    def is_available(self) -> bool:
        """Check if player is likely available to play."""
        return self.status in (PlayerStatus.AVAILABLE, PlayerStatus.DOUBTFUL)

    @computed_field
    @property
    def position_name(self) -> str:
        """Human-readable position name."""
        return self.position.name

    @computed_field
    @property
    def xgi_per_90(self) -> float:
        """Expected Goal Involvement per 90 minutes (key predictive stat)."""
        return self.expected_goals_per_90 + self.expected_assists_per_90

    @computed_field
    @property
    def goals_vs_xg(self) -> float:
        """Actual goals minus xG - positive = overperforming, negative = underperforming."""
        return self.goals_scored - self.expected_goals

    @computed_field
    @property
    def assists_vs_xa(self) -> float:
        """Actual assists minus xA - positive = overperforming, negative = underperforming."""
        return self.assists - self.expected_assists

    @computed_field
    @property
    def games_played(self) -> float:
        """Estimated games played based on minutes."""
        if self.minutes == 0:
            return 0.0
        # Estimate games from minutes (assuming ~90 min games)
        return self.minutes / 90.0

    @computed_field
    @property
    def bonus_per_90(self) -> float:
        """Bonus points per 90 minutes with regression-to-mean for low samples."""
        if self.minutes < 90:
            return 0.3  # League average when no data
        games = self.games_played
        raw_rate = self.bonus / games if games > 0 else 0.3
        # Regression to mean: blend with league average based on sample size
        # More games = trust individual rate more
        league_avg = 0.3  # ~0.3 bonus per 90 is league average
        weight = min(1.0, games / 15.0)  # Full weight at 15+ games
        return raw_rate * weight + league_avg * (1 - weight)

    @computed_field
    @property
    def yellow_card_prob(self) -> float:
        """Probability of yellow card per game with regression-to-mean."""
        if self.minutes < 90:
            return 0.08  # League average ~8% per game
        games = self.games_played
        raw_rate = self.yellow_cards / games if games > 0 else 0.08
        league_avg = 0.08
        weight = min(1.0, games / 10.0)
        return raw_rate * weight + league_avg * (1 - weight)

    @computed_field
    @property
    def red_card_prob(self) -> float:
        """Probability of red card per game with regression-to-mean."""
        if self.minutes < 90:
            return 0.005  # League average ~0.5% per game
        games = self.games_played
        raw_rate = self.red_cards / games if games > 0 else 0.005
        league_avg = 0.005
        weight = min(1.0, games / 15.0)
        return raw_rate * weight + league_avg * (1 - weight)

    @computed_field
    @property
    def saves_per_90(self) -> float:
        """Saves per 90 minutes (GK only)."""
        if self.minutes < 90:
            return 3.0  # League average for GKs
        games = self.games_played
        return self.saves / games if games > 0 else 3.0

    @computed_field
    @property
    def goals_conceded_per_90(self) -> float:
        """Goals conceded per 90 minutes (GK/DEF)."""
        if self.minutes < 90:
            return 1.3  # League average
        games = self.games_played
        return self.goals_conceded / games if games > 0 else 1.3


class Team(BaseModel):
    """Represents a Premier League team."""

    id: int = Field(description="FPL team ID")
    name: str = Field(description="Full team name")
    short_name: str = Field(description="3-letter abbreviation")
    strength_home: int = Field(default=3, description="Home strength rating")
    strength_away: int = Field(default=3, description="Away strength rating")
    strength_attack_home: int = Field(default=0)
    strength_attack_away: int = Field(default=0)
    strength_defence_home: int = Field(default=0)
    strength_defence_away: int = Field(default=0)


class Fixture(BaseModel):
    """Represents a Premier League fixture."""

    id: int = Field(description="Fixture ID")
    gameweek: int = Field(description="Gameweek number")
    home_team_id: int = Field(description="Home team ID")
    away_team_id: int = Field(description="Away team ID")
    home_difficulty: int = Field(default=3, ge=1, le=5, description="FDR for home team")
    away_difficulty: int = Field(default=3, ge=1, le=5, description="FDR for away team")
    kickoff_time: datetime | None = Field(default=None, description="Match kickoff time")
    finished: bool = Field(default=False, description="Has the match finished")

    # Results (if finished)
    home_score: int | None = Field(default=None)
    away_score: int | None = Field(default=None)


class PlayerProjection(BaseModel):
    """Expected points projection for a player in a specific gameweek."""

    player_id: int = Field(description="Player's FPL element ID")
    gameweek: int = Field(description="Gameweek number")
    expected_points: float = Field(description="Projected points")
    minutes_probability: float = Field(
        default=1.0, ge=0, le=1, description="Probability of playing"
    )
    source: str = Field(default="internal", description="Projection source")
    updated_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# User Squad Models
# =============================================================================


class SquadPlayer(BaseModel):
    """A player in the user's squad with position info."""

    player_id: int = Field(description="Player's FPL element ID")
    position: int = Field(ge=1, le=15, description="Position in squad (1-15)")
    is_captain: bool = Field(default=False)
    is_vice_captain: bool = Field(default=False)
    purchase_price: float = Field(description="Price when purchased")
    selling_price: float = Field(description="Current selling price")

    @computed_field
    @property
    def is_starter(self) -> bool:
        """Check if player is in starting XI (positions 1-11)."""
        return self.position <= 11

    @computed_field
    @property
    def bench_order(self) -> int | None:
        """Return bench order (1-4) if on bench, else None."""
        if self.position > 11:
            return self.position - 11
        return None


class ChipStatus(BaseModel):
    """
    Status of a chip (used/available).

    2025/26 Rule: Two sets of chips exist.
    First set must be used before GW19 deadline.
    """

    chip_type: ChipType
    chip_set: ChipSet = Field(
        default=ChipSet.FIRST_HALF,
        description="Which half-season this chip belongs to"
    )
    used_gameweek: int | None = Field(
        default=None, description="Gameweek when used, None if available"
    )

    @computed_field
    @property
    def is_available(self) -> bool:
        """Check if chip is still available."""
        return self.used_gameweek is None

    def is_expiring_soon(self, current_gw: int) -> bool:
        """
        Check if this chip must be used soon (2025/26 rule).

        First-half chips expire at GW19.
        """
        if not self.is_available:
            return False
        if self.chip_set == ChipSet.FIRST_HALF:
            return current_gw >= CHIP_FIRST_HALF_DEADLINE - 3  # Warn 3 GWs early
        return False

    def gameweeks_until_expiry(self, current_gw: int) -> int | None:
        """
        Calculate gameweeks until this chip expires.

        Returns None if chip doesn't expire (second half or already used).
        """
        if not self.is_available:
            return None
        if self.chip_set == ChipSet.FIRST_HALF:
            return max(0, CHIP_FIRST_HALF_DEADLINE - current_gw)
        return None  # Second half chips don't expire mid-season


class Squad(BaseModel):
    """User's current FPL squad state."""

    players: list[SquadPlayer] = Field(description="15 players in squad")
    bank: float = Field(default=0.0, description="Money in bank (millions)")
    free_transfers: int = Field(default=1, ge=0, le=5, description="Available FTs")
    total_value: float = Field(default=100.0, description="Total squad value")

    # Chip availability (2025/26 has two sets)
    chips: list[ChipStatus] = Field(default_factory=list)

    @computed_field
    @property
    def starters(self) -> list[SquadPlayer]:
        """Get starting XI."""
        return [p for p in self.players if p.is_starter]

    @computed_field
    @property
    def bench(self) -> list[SquadPlayer]:
        """Get bench players in order."""
        return sorted([p for p in self.players if not p.is_starter], key=lambda p: p.position)

    def get_available_chips(self) -> list[ChipType]:
        """Get list of available chips."""
        return [c.chip_type for c in self.chips if c.is_available]

    def validate_squad(self) -> list[str]:
        """Validate squad against FPL rules. Returns list of violations."""
        violations = []

        if len(self.players) != 15:
            violations.append(f"Squad must have 15 players, has {len(self.players)}")

        # Would need player data to validate positions and team limits
        # This is handled in the optimizer constraints

        return violations


# =============================================================================
# Optimization Plan Models
# =============================================================================


class Transfer(BaseModel):
    """Represents a transfer (in or out)."""

    player_out_id: int = Field(description="Player being sold")
    player_in_id: int = Field(description="Player being bought")
    gameweek: int = Field(description="Gameweek of transfer")
    cost: float = Field(default=0.0, description="Net cost of transfer")


class WeekPlan(BaseModel):
    """Optimization plan for a single gameweek."""

    gameweek: int = Field(description="Gameweek number")
    transfers: list[Transfer] = Field(default_factory=list)
    chip_used: ChipType | None = Field(default=None)
    captain_id: int = Field(description="Captain player ID")
    vice_captain_id: int = Field(description="Vice captain player ID")
    starting_xi: list[int] = Field(description="List of 11 starting player IDs")
    bench_order: list[int] = Field(
        description="List of 4 bench player IDs in auto-sub order"
    )
    expected_points: float = Field(default=0.0)
    hit_cost: int = Field(default=0, description="Point hit for extra transfers")

    @computed_field
    @property
    def net_expected_points(self) -> float:
        """Expected points minus hit cost."""
        return self.expected_points - self.hit_cost


class MultiWeekPlan(BaseModel):
    """Complete optimization plan across multiple gameweeks."""

    week_plans: list[WeekPlan] = Field(description="Plan for each gameweek")
    horizon: int = Field(description="Number of gameweeks planned")
    total_expected_points: float = Field(default=0.0)
    total_hits: int = Field(default=0)
    created_at: datetime = Field(default_factory=datetime.now)

    # Metadata
    starting_squad: list[int] = Field(
        default_factory=list, description="Initial squad player IDs"
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Optimization parameters used"
    )

    @computed_field
    @property
    def net_expected_points(self) -> float:
        """Total expected points minus all hit costs."""
        return self.total_expected_points - (self.total_hits * 4)


# =============================================================================
# Gameweek Info
# =============================================================================


class GameweekInfo(BaseModel):
    """Information about a specific gameweek."""

    id: int = Field(description="Gameweek number")
    name: str = Field(description="Display name (e.g., 'Gameweek 10')")
    deadline: datetime = Field(description="Transfer deadline")
    is_current: bool = Field(default=False)
    is_next: bool = Field(default=False)
    finished: bool = Field(default=False)

    # Blank/Double info
    is_blank: bool = Field(default=False, description="Has teams without fixtures")
    is_double: bool = Field(default=False, description="Has teams with 2+ fixtures")
    blank_teams: list[int] = Field(default_factory=list)
    double_teams: list[int] = Field(default_factory=list)
