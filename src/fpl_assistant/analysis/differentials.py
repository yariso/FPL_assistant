"""
Captain Differential Analysis for Elite FPL.

Analyzes the value of captaining differentially vs going with the template.
Key insight: Your rank only moves relative to other managers.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..data.models import Player
from ..data.ownership import CaptainEO, OwnershipTracker

logger = logging.getLogger(__name__)


class CaptainStrategy(Enum):
    """Captain strategy recommendation."""

    TEMPLATE = "template"  # Go with the crowd
    DIFFERENTIAL = "differential"  # Go against the crowd
    SLIGHT_DIFF = "slight_differential"  # Moderate risk


@dataclass
class DifferentialRecommendation:
    """Recommendation for captain differential decision."""

    recommended_captain: Player
    recommended_strategy: CaptainStrategy
    template_captain: Player
    template_captain_eo: float
    template_captain_xp: float
    differential_captain: Player | None
    differential_captain_eo: float
    differential_captain_xp: float
    expected_rank_gain_template: float
    expected_rank_gain_differential: float
    reasoning: str
    confidence: float  # 0-1


@dataclass
class CaptainOption:
    """A captain option with EO-adjusted value."""

    player: Player
    expected_points: float
    captain_eo: float  # % of managers captaining
    regular_eo: float  # % of managers owning
    differential_value: float  # Expected rank gain vs template
    is_template: bool  # Is this the template pick?


class CaptainDifferentialAnalyzer:
    """
    Analyzes captain choices through the lens of effective ownership.

    Key concepts:
    - Template captain: The most captained player (highest EO)
    - Differential captain: A less-captained player with good upside
    - Rank impact depends on EO, not just expected points

    The math:
    - If you captain the template and they score: no rank change (everyone has them)
    - If you captain a differential and they outscore: big rank gains
    - If you captain a differential and they blank: big rank losses
    """

    # EO thresholds for classification
    TEMPLATE_EO_THRESHOLD = 30.0  # >30% EO = template
    DIFFERENTIAL_EO_THRESHOLD = 15.0  # <15% EO = differential

    def __init__(
        self,
        ownership_tracker: OwnershipTracker,
    ):
        """Initialize the analyzer."""
        self.ownership_tracker = ownership_tracker

    def analyze_captain_options(
        self,
        candidates: list[tuple[Player, float]],  # (player, xP)
        gameweek: int,
    ) -> list[CaptainOption]:
        """
        Analyze all captain candidates with EO-adjusted values.

        Args:
            candidates: List of (player, expected_points) tuples
            gameweek: Current gameweek

        Returns:
            List of CaptainOption sorted by differential value
        """
        if not candidates:
            return []

        # Get EO estimates for all candidates
        options = []
        template_xp = candidates[0][1]  # Highest xP is template assumption

        for player, xp in candidates:
            # Estimate captain EO
            captain_eo = self.ownership_tracker.estimate_captain_eo(player, gameweek)

            # Calculate differential value
            # This is the expected rank gain compared to captaining the template
            diff_value = self._calculate_differential_value(
                player_xp=xp,
                player_eo=captain_eo.overall_eo,
                template_xp=template_xp,
                template_eo=candidates[0][1] if candidates else 0,  # Estimate template EO
            )

            options.append(CaptainOption(
                player=player,
                expected_points=xp,
                captain_eo=captain_eo.overall_eo,
                regular_eo=captain_eo.regular_ownership,
                differential_value=diff_value,
                is_template=captain_eo.overall_eo >= self.TEMPLATE_EO_THRESHOLD,
            ))

        # Sort by expected points (template ranking)
        options.sort(key=lambda x: x.expected_points, reverse=True)

        # Mark the actual template (highest EO among top 3 xP)
        top_3_by_xp = options[:3]
        template_pick = max(top_3_by_xp, key=lambda x: x.captain_eo)
        for opt in options:
            opt.is_template = (opt.player.id == template_pick.player.id)

        return options

    def _calculate_differential_value(
        self,
        player_xp: float,
        player_eo: float,
        template_xp: float,
        template_eo: float,
    ) -> float:
        """
        Calculate the differential value of a captain pick.

        Differential value = expected rank gain vs captaining template

        If player outscores template:
        - Gain vs managers who captained template: (1 - template_eo) * point_diff
        - Gain vs managers who didn't captain player: (1 - player_eo) * player_points

        This is simplified - real calculation would need more data.
        """
        if player_eo >= template_eo:
            # This IS the template, no differential value
            return 0.0

        # Expected points if player outperforms
        upside = (player_xp * 2) - (template_xp * 2)  # Captain points

        # Weight by probability and EO difference
        # Lower EO = higher potential gain when correct
        eo_advantage = (template_eo - player_eo) / 100  # 0-1 scale

        return upside * eo_advantage

    def get_recommendation(
        self,
        candidates: list[tuple[Player, float]],
        gameweek: int,
        league_position: str = "mid",  # "leading", "chasing", "mid"
        risk_tolerance: str = "medium",  # "low", "medium", "high"
    ) -> DifferentialRecommendation:
        """
        Get captain recommendation based on EO analysis.

        Args:
            candidates: List of (player, expected_points) tuples
            gameweek: Current gameweek
            league_position: Your mini-league position
            risk_tolerance: How much risk you want to take

        Returns:
            DifferentialRecommendation with analysis
        """
        options = self.analyze_captain_options(candidates, gameweek)

        if not options:
            raise ValueError("No captain candidates provided")

        # Identify template and best differential
        template = next((o for o in options if o.is_template), options[0])
        differentials = [o for o in options if not o.is_template and o.captain_eo < self.DIFFERENTIAL_EO_THRESHOLD]

        # Best differential by xP (but with low EO)
        best_diff = None
        if differentials:
            best_diff = max(differentials, key=lambda x: x.expected_points)

        # Decision logic based on position and risk
        strategy = self._determine_strategy(
            template, best_diff, league_position, risk_tolerance
        )

        # Build recommendation
        if strategy == CaptainStrategy.TEMPLATE:
            recommended = template.player
            reasoning = self._build_template_reasoning(template, league_position)
        elif strategy == CaptainStrategy.DIFFERENTIAL and best_diff:
            recommended = best_diff.player
            reasoning = self._build_differential_reasoning(template, best_diff, league_position)
        else:
            # Slight differential - second highest xP with moderate EO
            slight_diffs = [o for o in options if 15 <= o.captain_eo < 30]
            if slight_diffs:
                recommended = max(slight_diffs, key=lambda x: x.expected_points).player
                reasoning = f"Moderate differential with good upside"
            else:
                recommended = template.player
                reasoning = self._build_template_reasoning(template, league_position)

        return DifferentialRecommendation(
            recommended_captain=recommended,
            recommended_strategy=strategy,
            template_captain=template.player,
            template_captain_eo=template.captain_eo,
            template_captain_xp=template.expected_points,
            differential_captain=best_diff.player if best_diff else None,
            differential_captain_eo=best_diff.captain_eo if best_diff else 0,
            differential_captain_xp=best_diff.expected_points if best_diff else 0,
            expected_rank_gain_template=0,  # Baseline
            expected_rank_gain_differential=best_diff.differential_value if best_diff else 0,
            reasoning=reasoning,
            confidence=self._calculate_confidence(template, best_diff, strategy),
        )

    def _determine_strategy(
        self,
        template: CaptainOption,
        best_diff: CaptainOption | None,
        league_position: str,
        risk_tolerance: str,
    ) -> CaptainStrategy:
        """Determine the recommended strategy."""

        # If leading in mini-league, protect with template
        if league_position == "leading":
            return CaptainStrategy.TEMPLATE

        # If chasing, need to take risks
        if league_position == "chasing":
            if best_diff and best_diff.expected_points > template.expected_points * 0.85:
                return CaptainStrategy.DIFFERENTIAL

        # Risk tolerance
        if risk_tolerance == "low":
            return CaptainStrategy.TEMPLATE
        elif risk_tolerance == "high" and best_diff:
            return CaptainStrategy.DIFFERENTIAL

        # Default: template unless differential has similar xP
        if best_diff and best_diff.expected_points > template.expected_points * 0.95:
            return CaptainStrategy.SLIGHT_DIFF

        return CaptainStrategy.TEMPLATE

    def _build_template_reasoning(
        self,
        template: CaptainOption,
        league_position: str,
    ) -> str:
        """Build reasoning for template recommendation."""
        reasons = [f"{template.player.web_name} is the template captain ({template.captain_eo:.0f}% EO)"]

        if league_position == "leading":
            reasons.append("You're leading - protect your position")

        reasons.append(f"Highest projected points: {template.expected_points:.1f} xP")

        return ". ".join(reasons)

    def _build_differential_reasoning(
        self,
        template: CaptainOption,
        diff: CaptainOption,
        league_position: str,
    ) -> str:
        """Build reasoning for differential recommendation."""
        xp_gap = template.expected_points - diff.expected_points
        eo_gap = template.captain_eo - diff.captain_eo

        reasons = [f"{diff.player.web_name} is a differential ({diff.captain_eo:.0f}% EO)"]

        if league_position == "chasing":
            reasons.append("You're chasing - need to take calculated risks")

        reasons.append(f"Only {xp_gap:.1f} xP behind template")
        reasons.append(f"EO advantage of {eo_gap:.0f}% means big gains if correct")

        return ". ".join(reasons)

    def _calculate_confidence(
        self,
        template: CaptainOption,
        best_diff: CaptainOption | None,
        strategy: CaptainStrategy,
    ) -> float:
        """Calculate confidence in the recommendation (0-1)."""
        if strategy == CaptainStrategy.TEMPLATE:
            # High confidence if template is clear
            if template.captain_eo > 50:
                return 0.9
            elif template.captain_eo > 30:
                return 0.7
            else:
                return 0.6

        elif strategy == CaptainStrategy.DIFFERENTIAL and best_diff:
            # Lower confidence for differential picks
            xp_ratio = best_diff.expected_points / template.expected_points
            if xp_ratio > 0.95:
                return 0.6
            elif xp_ratio > 0.85:
                return 0.5
            else:
                return 0.4

        return 0.5


# Convenience function
def analyze_captain_differential(
    candidates: list[tuple[Player, float]],
    gameweek: int,
    db,
    league_position: str = "mid",
    risk_tolerance: str = "medium",
) -> DifferentialRecommendation:
    """
    Convenience function to analyze captain differential.

    Args:
        candidates: List of (player, expected_points) tuples
        gameweek: Current gameweek
        db: Database instance
        league_position: "leading", "chasing", or "mid"
        risk_tolerance: "low", "medium", or "high"

    Returns:
        DifferentialRecommendation
    """
    tracker = OwnershipTracker(db)
    analyzer = CaptainDifferentialAnalyzer(tracker)
    return analyzer.get_recommendation(
        candidates, gameweek, league_position, risk_tolerance
    )
