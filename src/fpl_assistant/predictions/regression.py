"""
xG/xA Regression Analysis Module.

Identifies players who are over/under-performing their expected stats
and are likely to regress to the mean.
"""

from dataclasses import dataclass
from enum import StrEnum

from fpl_assistant.data.models import Player, Position


class RegressionType(StrEnum):
    """Type of regression expected."""

    DUE_HAUL = "DUE HAUL"  # Underperforming xG/xA - likely to score soon
    DUE_BLANK = "DUE BLANK"  # Overperforming xG/xA - likely to blank soon
    NEUTRAL = "NEUTRAL"  # Performing close to expectations


@dataclass
class RegressionCandidate:
    """Player identified for potential regression."""

    player: Player
    regression_type: RegressionType

    # Goal regression
    goals: int
    expected_goals: float
    goals_diff: float  # goals - xG (negative = underperforming)

    # Assist regression
    assists: int
    expected_assists: float
    assists_diff: float  # assists - xA (negative = underperforming)

    # Combined
    total_diff: float  # Combined G+A vs xG+xA
    regression_score: float  # Higher = more likely to regress

    @property
    def is_buy_target(self) -> bool:
        """Player is underperforming - good buy target."""
        return self.regression_type == RegressionType.DUE_HAUL

    @property
    def is_sell_target(self) -> bool:
        """Player is overperforming - consider selling."""
        return self.regression_type == RegressionType.DUE_BLANK

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        if self.regression_type == RegressionType.DUE_HAUL:
            return (
                f"Underperforming by {abs(self.total_diff):.1f} G+A. "
                f"({self.goals}G vs {self.expected_goals:.1f}xG, "
                f"{self.assists}A vs {self.expected_assists:.1f}xA)"
            )
        elif self.regression_type == RegressionType.DUE_BLANK:
            return (
                f"Overperforming by {self.total_diff:.1f} G+A. "
                f"({self.goals}G vs {self.expected_goals:.1f}xG, "
                f"{self.assists}A vs {self.expected_assists:.1f}xA)"
            )
        return "Performing as expected"


class RegressionAnalyzer:
    """
    Analyzes xG/xA regression for FPL players.

    Identifies players who are significantly over/under-performing
    their expected statistics and are likely to regress.
    """

    # Thresholds for regression classification
    UNDERPERFORM_THRESHOLD = -1.5  # G+A below xG+xA
    OVERPERFORM_THRESHOLD = 1.5  # G+A above xG+xA

    # Minimum minutes to be considered (avoid small sample issues)
    MIN_MINUTES = 450  # ~5 full games

    def __init__(self, players: list[Player]):
        self.players = players

    def analyze_player(self, player: Player) -> RegressionCandidate | None:
        """
        Analyze a single player for regression potential.

        Returns None if player doesn't meet minimum requirements.
        """
        # Skip players with insufficient minutes
        if player.minutes < self.MIN_MINUTES:
            return None

        # Skip goalkeepers (xG not relevant)
        if player.position == Position.GK:
            return None

        goals_diff = player.goals_vs_xg
        assists_diff = player.assists_vs_xa
        total_diff = goals_diff + assists_diff

        # Determine regression type
        if total_diff <= self.UNDERPERFORM_THRESHOLD:
            regression_type = RegressionType.DUE_HAUL
            regression_score = abs(total_diff)
        elif total_diff >= self.OVERPERFORM_THRESHOLD:
            regression_type = RegressionType.DUE_BLANK
            regression_score = total_diff
        else:
            regression_type = RegressionType.NEUTRAL
            regression_score = 0

        return RegressionCandidate(
            player=player,
            regression_type=regression_type,
            goals=player.goals_scored,
            expected_goals=player.expected_goals,
            goals_diff=goals_diff,
            assists=player.assists,
            expected_assists=player.expected_assists,
            assists_diff=assists_diff,
            total_diff=total_diff,
            regression_score=regression_score,
        )

    def get_buy_targets(self, top_n: int = 10) -> list[RegressionCandidate]:
        """
        Get players who are underperforming xG/xA (due a haul).

        These are good buy targets as they're likely to score soon.
        """
        candidates = []
        for player in self.players:
            analysis = self.analyze_player(player)
            if analysis and analysis.is_buy_target:
                candidates.append(analysis)

        # Sort by regression score (most underperforming first)
        return sorted(candidates, key=lambda c: -c.regression_score)[:top_n]

    def get_sell_targets(self, top_n: int = 10) -> list[RegressionCandidate]:
        """
        Get players who are overperforming xG/xA (due to blank).

        These might be good sell targets as their returns may decline.
        """
        candidates = []
        for player in self.players:
            analysis = self.analyze_player(player)
            if analysis and analysis.is_sell_target:
                candidates.append(analysis)

        # Sort by regression score (most overperforming first)
        return sorted(candidates, key=lambda c: -c.regression_score)[:top_n]

    def get_all_regression_candidates(
        self,
        include_neutral: bool = False,
    ) -> dict[str, list[RegressionCandidate]]:
        """
        Get all regression candidates grouped by type.

        Returns:
            Dict with keys "buy_targets", "sell_targets", and optionally "neutral"
        """
        buy_targets = []
        sell_targets = []
        neutral = []

        for player in self.players:
            analysis = self.analyze_player(player)
            if not analysis:
                continue

            if analysis.is_buy_target:
                buy_targets.append(analysis)
            elif analysis.is_sell_target:
                sell_targets.append(analysis)
            elif include_neutral:
                neutral.append(analysis)

        result = {
            "buy_targets": sorted(buy_targets, key=lambda c: -c.regression_score),
            "sell_targets": sorted(sell_targets, key=lambda c: -c.regression_score),
        }

        if include_neutral:
            result["neutral"] = neutral

        return result


def get_regression_candidates(
    players: list[Player],
    min_minutes: int = 450,
) -> dict[str, list[RegressionCandidate]]:
    """
    Convenience function to get regression candidates.

    Args:
        players: List of all players
        min_minutes: Minimum minutes played to be considered

    Returns:
        Dict with "buy_targets" and "sell_targets" lists
    """
    analyzer = RegressionAnalyzer(players)
    analyzer.MIN_MINUTES = min_minutes
    return analyzer.get_all_regression_candidates()


def identify_regression_buys(
    players: list[Player],
    top_n: int = 10,
) -> list[RegressionCandidate]:
    """
    Quick function to get top buy targets based on xG regression.
    """
    analyzer = RegressionAnalyzer(players)
    return analyzer.get_buy_targets(top_n)


def identify_regression_sells(
    players: list[Player],
    top_n: int = 10,
) -> list[RegressionCandidate]:
    """
    Quick function to get top sell targets based on xG regression.
    """
    analyzer = RegressionAnalyzer(players)
    return analyzer.get_sell_targets(top_n)
