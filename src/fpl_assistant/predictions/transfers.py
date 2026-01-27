"""
Transfer Hit Calculator Module.

Analyzes whether taking a transfer hit (-4 points) is worth it
by computing break-even timelines and expected value.
"""

from dataclasses import dataclass
from enum import StrEnum

from fpl_assistant.data.models import Player, PlayerProjection


class HitRecommendation(StrEnum):
    """Hit recommendation levels."""

    TAKE_HIT = "TAKE HIT"
    AVOID_HIT = "AVOID HIT"
    MARGINAL = "MARGINAL"


@dataclass
class HitAnalysis:
    """Analysis of whether a transfer hit is worth taking."""

    player_out_id: int
    player_out_name: str
    player_in_id: int
    player_in_name: str

    # xP difference per week
    xp_gain_per_week: float
    weeks_horizon: int

    # Total analysis
    total_xp_out: float  # Total xP for player_out over horizon
    total_xp_in: float  # Total xP for player_in over horizon
    gross_gain: float  # xP_in - xP_out (before hit)
    net_gain: float  # gross_gain - 4 (after hit)

    # Break-even analysis
    break_even_weeks: float | None  # Weeks until hit pays off (None if never)

    # Recommendation
    recommendation: HitRecommendation
    confidence: str  # "High", "Medium", "Low"

    @property
    def hit_cost(self) -> int:
        """Hit cost in points."""
        return 4

    @property
    def pays_off(self) -> bool:
        """Whether the hit is expected to pay off."""
        return self.net_gain > 0

    @property
    def explanation(self) -> str:
        """Human-readable explanation of the analysis."""
        if self.recommendation == HitRecommendation.TAKE_HIT:
            return (
                f"Taking the hit gains {self.net_gain:.1f} points over {self.weeks_horizon} weeks. "
                f"Break-even in {self.break_even_weeks:.1f} weeks."
            )
        elif self.recommendation == HitRecommendation.AVOID_HIT:
            if self.net_gain < 0:
                return (
                    f"Hit loses {abs(self.net_gain):.1f} points over {self.weeks_horizon} weeks. "
                    "Not worth it."
                )
            else:
                return (
                    f"Only gains {self.net_gain:.1f} points - save the free transfer instead."
                )
        else:  # MARGINAL
            return (
                f"Close call: {self.net_gain:.1f} point gain over {self.weeks_horizon} weeks. "
                "Consider other factors like price changes or injury risk."
            )


class HitCalculator:
    """
    Calculator for transfer hit decisions.

    Determines whether taking a -4 hit is worth it based on
    expected points over a planning horizon.
    """

    # Thresholds for recommendations
    TAKE_HIT_THRESHOLD = 6.0  # Net gain > 6 = definitely take hit (150% of hit)
    AVOID_HIT_THRESHOLD = 2.0  # Net gain < 2 = definitely avoid
    # Between 2-6 = marginal

    def __init__(
        self,
        players: dict[int, Player],
        projections: dict[int, list[PlayerProjection]],
    ):
        """
        Initialize calculator.

        Args:
            players: Dict of player_id -> Player
            projections: Dict of player_id -> list of PlayerProjection (per GW)
        """
        self.players = players
        self.projections = projections

    def get_player_xp(
        self,
        player_id: int,
        start_gw: int,
        num_weeks: int,
    ) -> list[float]:
        """Get expected points for a player over multiple weeks."""
        player_projs = self.projections.get(player_id, [])
        xp_list = []

        for gw in range(start_gw, start_gw + num_weeks):
            proj = next(
                (p for p in player_projs if p.gameweek == gw),
                None
            )
            if proj:
                xp_list.append(proj.expected_points)
            else:
                # Fallback to player's ppg if no projection
                player = self.players.get(player_id)
                xp_list.append(player.points_per_game if player else 4.0)

        return xp_list

    def calculate_hit(
        self,
        player_out_id: int,
        player_in_id: int,
        start_gw: int,
        weeks_horizon: int = 5,
    ) -> HitAnalysis:
        """
        Calculate whether a transfer hit is worth taking.

        Args:
            player_out_id: Player being transferred out
            player_in_id: Player being transferred in
            start_gw: Starting gameweek
            weeks_horizon: Number of weeks to analyze (default 5)

        Returns:
            HitAnalysis with recommendation
        """
        player_out = self.players.get(player_out_id)
        player_in = self.players.get(player_in_id)

        if not player_out or not player_in:
            raise ValueError("Invalid player IDs")

        # Get xP for both players
        xp_out = self.get_player_xp(player_out_id, start_gw, weeks_horizon)
        xp_in = self.get_player_xp(player_in_id, start_gw, weeks_horizon)

        total_xp_out = sum(xp_out)
        total_xp_in = sum(xp_in)

        gross_gain = total_xp_in - total_xp_out
        net_gain = gross_gain - 4  # Subtract hit cost

        # Calculate break-even
        xp_gain_per_week = gross_gain / weeks_horizon if weeks_horizon > 0 else 0
        if xp_gain_per_week > 0:
            break_even_weeks = 4 / xp_gain_per_week
        else:
            break_even_weeks = None  # Never breaks even

        # Determine recommendation
        if net_gain >= self.TAKE_HIT_THRESHOLD:
            recommendation = HitRecommendation.TAKE_HIT
            confidence = "High"
        elif net_gain < self.AVOID_HIT_THRESHOLD:
            recommendation = HitRecommendation.AVOID_HIT
            confidence = "High" if net_gain < 0 else "Medium"
        else:
            recommendation = HitRecommendation.MARGINAL
            confidence = "Low"

        return HitAnalysis(
            player_out_id=player_out_id,
            player_out_name=player_out.web_name,
            player_in_id=player_in_id,
            player_in_name=player_in.web_name,
            xp_gain_per_week=xp_gain_per_week,
            weeks_horizon=weeks_horizon,
            total_xp_out=total_xp_out,
            total_xp_in=total_xp_in,
            gross_gain=gross_gain,
            net_gain=net_gain,
            break_even_weeks=break_even_weeks,
            recommendation=recommendation,
            confidence=confidence,
        )

    def analyze_multiple_hits(
        self,
        transfers: list[tuple[int, int]],  # List of (out_id, in_id)
        start_gw: int,
        weeks_horizon: int = 5,
    ) -> list[HitAnalysis]:
        """
        Analyze multiple potential transfer hits.

        Args:
            transfers: List of (player_out_id, player_in_id) tuples
            start_gw: Starting gameweek
            weeks_horizon: Planning horizon

        Returns:
            List of HitAnalysis sorted by net_gain (best first)
        """
        analyses = []
        for out_id, in_id in transfers:
            try:
                analysis = self.calculate_hit(out_id, in_id, start_gw, weeks_horizon)
                analyses.append(analysis)
            except ValueError:
                continue

        return sorted(analyses, key=lambda a: -a.net_gain)

    def should_take_hit(
        self,
        player_out_id: int,
        player_in_id: int,
        start_gw: int,
        weeks_horizon: int = 5,
    ) -> bool:
        """
        Simple boolean check for whether to take a hit.

        For quick decisions - use calculate_hit() for full analysis.
        """
        analysis = self.calculate_hit(
            player_out_id, player_in_id, start_gw, weeks_horizon
        )
        return analysis.recommendation == HitRecommendation.TAKE_HIT


def calculate_hit_value(
    player_out: Player,
    player_in: Player,
    out_xp: float,
    in_xp: float,
    weeks_horizon: int = 5,
) -> HitAnalysis:
    """
    Convenience function for quick hit analysis.

    Args:
        player_out: Player being sold
        player_in: Player being bought
        out_xp: Total expected points for player_out over horizon
        in_xp: Total expected points for player_in over horizon
        weeks_horizon: Number of weeks

    Returns:
        HitAnalysis with recommendation
    """
    gross_gain = in_xp - out_xp
    net_gain = gross_gain - 4
    xp_gain_per_week = gross_gain / weeks_horizon if weeks_horizon > 0 else 0

    if xp_gain_per_week > 0:
        break_even_weeks = 4 / xp_gain_per_week
    else:
        break_even_weeks = None

    # Determine recommendation
    if net_gain >= 6.0:
        recommendation = HitRecommendation.TAKE_HIT
        confidence = "High"
    elif net_gain < 2.0:
        recommendation = HitRecommendation.AVOID_HIT
        confidence = "High" if net_gain < 0 else "Medium"
    else:
        recommendation = HitRecommendation.MARGINAL
        confidence = "Low"

    return HitAnalysis(
        player_out_id=player_out.id,
        player_out_name=player_out.web_name,
        player_in_id=player_in.id,
        player_in_name=player_in.web_name,
        xp_gain_per_week=xp_gain_per_week,
        weeks_horizon=weeks_horizon,
        total_xp_out=out_xp,
        total_xp_in=in_xp,
        gross_gain=gross_gain,
        net_gain=net_gain,
        break_even_weeks=break_even_weeks,
        recommendation=recommendation,
        confidence=confidence,
    )
