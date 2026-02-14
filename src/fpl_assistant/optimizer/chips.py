"""
Chip Timing Optimization for Elite FPL.

Analyzes the optimal timing for using FPL chips:
- Wildcard (WC): Unlimited free transfers for one week
- Free Hit (FH): Temporary squad for one week
- Bench Boost (BB): All 15 players score
- Triple Captain (TC): Captain scores 3x points

2025/26 RULE: Two sets of chips exist.
- First set (WC, FH, BB, TC) must be used BEFORE GW19
- Second set becomes available after GW19

Key insight: Proper chip timing can add 30-50 points per chip.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from ..data.models import (
    CHIP_FIRST_HALF_DEADLINE,
    ChipSet,
    ChipStatus,
    ChipType,
    GameweekInfo,
    Player,
    Squad,
)

logger = logging.getLogger(__name__)


@dataclass
class ChipValue:
    """Value estimate for using a chip in a specific gameweek."""

    chip: ChipType
    gameweek: int
    estimated_value: float  # Extra points gained vs not using chip
    confidence: float  # 0-1, how confident in the estimate
    reasoning: str
    is_recommended: bool = False

    # Supporting data
    is_double_gw: bool = False
    is_blank_gw: bool = False
    best_captain_xp: float = 0.0
    bench_total_xp: float = 0.0
    optimal_team_xp: float = 0.0
    current_team_xp: float = 0.0


@dataclass
class ChipRecommendation:
    """Full chip timing recommendation."""

    gameweek: int
    recommended_chip: ChipType | None
    chip_values: list[ChipValue]
    reasoning: str
    save_for_later: list[str]  # Reasons to save chips
    upcoming_opportunities: list[tuple[int, ChipType, str]]  # (gw, chip, reason)


@dataclass
class ChipDeadlineWarning:
    """Warning about chip expiring due to 2025/26 two-set rule."""

    chip: ChipType
    chip_set: ChipSet
    gameweeks_remaining: int
    deadline_gw: int
    urgency: str  # "critical", "warning", "info"
    message: str


@dataclass
class ChipTimingPlan:
    """Multi-week chip timing plan."""

    horizon: int
    recommendations_by_gw: dict[int, ChipRecommendation]
    best_bb_gw: int | None = None
    best_tc_gw: int | None = None
    best_fh_gw: int | None = None
    best_wc_gw: int | None = None
    total_chip_value: float = 0.0

    # 2025/26 Rule: Deadline warnings for first-half chips
    deadline_warnings: list[ChipDeadlineWarning] = field(default_factory=list)
    first_half_chips_remaining: int = 0  # How many first-set chips left unused


class ChipOptimizer:
    """
    Optimizes chip timing for maximum FPL points.

    Key strategies:
    1. Bench Boost: Use in Double GW when bench is strong
    2. Triple Captain: Use in Double GW on premium attacker
    3. Free Hit: Use in Blank GW or extreme fixture swing
    4. Wildcard: Use for major team restructure before good run
    """

    # Minimum value thresholds for chip usage
    BB_MIN_VALUE = 8.0  # At least 8 extra points expected
    TC_MIN_VALUE = 6.0  # At least 6 extra points expected
    FH_MIN_VALUE = 10.0  # At least 10 extra points expected
    WC_MIN_VALUE = 15.0  # At least 15 extra points expected over horizon

    def __init__(
        self,
        players: list[Player],
        gameweeks: list[GameweekInfo],
        projections_by_gw: dict[int, dict[int, float]],
    ):
        """
        Initialize the chip optimizer.

        Args:
            players: All available players
            gameweeks: Gameweek info including blank/double status
            projections_by_gw: Player projections by gameweek
        """
        self.players = {p.id: p for p in players}
        self.gameweeks = {gw.id: gw for gw in gameweeks}
        self.projections = projections_by_gw

    def calculate_bench_boost_value(
        self,
        squad: Squad,
        gameweek: int,
    ) -> ChipValue:
        """
        Calculate the value of using Bench Boost in a gameweek.

        BB value = sum of bench player projections (since they wouldn't normally score)
        Best in Double GW when bench players also have doubles.
        """
        gw_projections = self.projections.get(gameweek, {})
        gw_info = self.gameweeks.get(gameweek)

        # Get bench players (positions 12-15)
        bench_players = [
            sp for sp in squad.players
            if sp.position > 11
        ]

        # Calculate bench xP weighted by each player's probability of playing
        bench_xp = 0.0
        for sp in bench_players:
            player_xp = gw_projections.get(sp.player_id, 0)
            player = self.players.get(sp.player_id)
            if player:
                play_prob = self._get_playing_probability(player)
                bench_xp += player_xp * play_prob
            else:
                bench_xp += player_xp

        is_double = gw_info.is_double if gw_info else False

        # Double GW bonus - bench might have double fixtures too
        if is_double:
            bench_xp *= 1.5  # Approximate boost for double
            confidence = 0.8
            reasoning = f"Double GW {gameweek}: bench expected to score {bench_xp:.1f} pts"
        else:
            confidence = 0.6
            reasoning = f"GW {gameweek}: bench expected to score {bench_xp:.1f} pts"

        is_recommended = bench_xp >= self.BB_MIN_VALUE and is_double

        return ChipValue(
            chip=ChipType.BENCH_BOOST,
            gameweek=gameweek,
            estimated_value=bench_xp,
            confidence=confidence,
            reasoning=reasoning,
            is_recommended=is_recommended,
            is_double_gw=is_double,
            bench_total_xp=bench_xp,
        )

    def calculate_triple_captain_value(
        self,
        squad: Squad,
        gameweek: int,
    ) -> ChipValue:
        """
        Calculate the value of using Triple Captain in a gameweek.

        TC value = best captain's xP (the extra 1x beyond normal 2x)
        Best in Double GW with premium attacker having two good fixtures.
        """
        gw_projections = self.projections.get(gameweek, {})
        gw_info = self.gameweeks.get(gameweek)

        # Find best captain option in squad
        squad_pids = {sp.player_id for sp in squad.players}
        captain_options = []

        for pid in squad_pids:
            player = self.players.get(pid)
            if player and player.position.value in [3, 4]:  # MID or FWD
                xp = gw_projections.get(pid, 0)
                captain_options.append((player, xp))

        captain_options.sort(key=lambda x: x[1], reverse=True)

        if not captain_options:
            return ChipValue(
                chip=ChipType.TRIPLE_CAPTAIN,
                gameweek=gameweek,
                estimated_value=0,
                confidence=0.3,
                reasoning="No suitable captain options",
                is_recommended=False,
            )

        best_captain, best_xp = captain_options[0]

        # TC adds 1x captain points (normal captain = 2x, TC = 3x)
        tc_value = best_xp

        is_double = gw_info.is_double if gw_info else False

        # Significant boost for double GW
        if is_double:
            confidence = 0.8
            reasoning = f"Double GW {gameweek}: {best_captain.web_name} projected {best_xp:.1f} xP (TC adds {tc_value:.1f})"
        else:
            confidence = 0.5
            reasoning = f"GW {gameweek}: {best_captain.web_name} projected {best_xp:.1f} xP (TC adds {tc_value:.1f})"

        is_recommended = tc_value >= self.TC_MIN_VALUE and is_double

        return ChipValue(
            chip=ChipType.TRIPLE_CAPTAIN,
            gameweek=gameweek,
            estimated_value=tc_value,
            confidence=confidence,
            reasoning=reasoning,
            is_recommended=is_recommended,
            is_double_gw=is_double,
            best_captain_xp=best_xp,
        )

    def calculate_free_hit_value(
        self,
        squad: Squad,
        gameweek: int,
    ) -> ChipValue:
        """
        Calculate the value of using Free Hit in a gameweek.

        FH value = optimal_team_xP - current_team_xP
        Best in Blank GW when many of your players don't play.
        """
        gw_projections = self.projections.get(gameweek, {})
        gw_info = self.gameweeks.get(gameweek)

        # Calculate current team xP for this gameweek
        squad_pids = {sp.player_id for sp in squad.players}
        current_team_xp = sum(
            gw_projections.get(pid, 0)
            for pid in squad_pids
        )

        # Calculate optimal team xP (best possible team this week)
        # Get top 15 by xP respecting FPL constraints
        all_player_xp = [
            (pid, gw_projections.get(pid, 0))
            for pid in self.players
            if self.players[pid].is_available
        ]
        all_player_xp.sort(key=lambda x: x[1], reverse=True)

        # Simple estimate: sum of top 2 GK, 5 DEF, 5 MID, 3 FWD by xP
        optimal_by_pos = {1: [], 2: [], 3: [], 4: []}
        for pid, xp in all_player_xp:
            player = self.players.get(pid)
            if player:
                pos = player.position.value
                if len(optimal_by_pos[pos]) < [2, 5, 5, 3][pos - 1]:
                    optimal_by_pos[pos].append(xp)

        optimal_team_xp = (
            sum(optimal_by_pos[1][:1]) +  # 1 GK starts
            sum(sorted(optimal_by_pos[2], reverse=True)[:4]) +  # 4 DEF start
            sum(sorted(optimal_by_pos[3], reverse=True)[:4]) +  # 4 MID start
            sum(sorted(optimal_by_pos[4], reverse=True)[:2])  # 2 FWD start
        )

        fh_value = optimal_team_xp - current_team_xp

        is_blank = gw_info.is_blank if gw_info else False

        if is_blank:
            confidence = 0.85
            reasoning = f"Blank GW {gameweek}: FH gains {fh_value:.1f} pts ({optimal_team_xp:.1f} vs {current_team_xp:.1f})"
        else:
            confidence = 0.6
            reasoning = f"GW {gameweek}: FH gains {fh_value:.1f} pts ({optimal_team_xp:.1f} vs {current_team_xp:.1f})"

        is_recommended = fh_value >= self.FH_MIN_VALUE and (is_blank or fh_value >= 20)

        return ChipValue(
            chip=ChipType.FREE_HIT,
            gameweek=gameweek,
            estimated_value=fh_value,
            confidence=confidence,
            reasoning=reasoning,
            is_recommended=is_recommended,
            is_blank_gw=is_blank,
            optimal_team_xp=optimal_team_xp,
            current_team_xp=current_team_xp,
        )

    def calculate_wildcard_value(
        self,
        squad: Squad,
        gameweek: int,
        horizon: int = 5,
    ) -> ChipValue:
        """
        Calculate the value of using Wildcard in a gameweek.

        WC value = (optimal_team_total_xP - current_team_total_xP) over horizon
        Best before a fixture swing or to fix a broken squad.
        """
        # Calculate current squad total xP over horizon
        squad_pids = {sp.player_id for sp in squad.players}
        current_total_xp = 0

        for gw in range(gameweek, gameweek + horizon):
            gw_projections = self.projections.get(gw, {})
            current_total_xp += sum(
                gw_projections.get(pid, 0)
                for pid in squad_pids
            )

        # Estimate optimal squad total xP over horizon
        # This is complex - use rough estimate based on best players
        optimal_total_xp = 0

        for gw in range(gameweek, gameweek + horizon):
            gw_projections = self.projections.get(gw, {})
            all_xp = sorted(gw_projections.values(), reverse=True)
            # Top 11 average (approximate optimal starting XI)
            if len(all_xp) >= 11:
                optimal_total_xp += sum(all_xp[:11])

        wc_value = optimal_total_xp - current_total_xp

        # WC is worth more if team needs major restructure
        squad_strength = current_total_xp / max(1, optimal_total_xp)

        if squad_strength < 0.7:  # Team is weak
            confidence = 0.8
            reasoning = f"GW {gameweek}: Team needs rebuild - WC gains {wc_value:.1f} pts over {horizon} weeks"
            is_recommended = True
        elif squad_strength < 0.85:
            confidence = 0.6
            reasoning = f"GW {gameweek}: Team moderate - WC gains {wc_value:.1f} pts over {horizon} weeks"
            is_recommended = wc_value >= self.WC_MIN_VALUE
        else:
            confidence = 0.4
            reasoning = f"GW {gameweek}: Team strong - WC only gains {wc_value:.1f} pts over {horizon} weeks"
            is_recommended = False

        return ChipValue(
            chip=ChipType.WILDCARD,
            gameweek=gameweek,
            estimated_value=wc_value,
            confidence=confidence,
            reasoning=reasoning,
            is_recommended=is_recommended,
            optimal_team_xp=optimal_total_xp,
            current_team_xp=current_total_xp,
        )

    def get_chip_recommendation(
        self,
        squad: Squad,
        gameweek: int,
        available_chips: list[ChipType],
    ) -> ChipRecommendation:
        """
        Get chip recommendation for a specific gameweek.

        Args:
            squad: Current squad
            gameweek: Gameweek to analyze
            available_chips: Chips still available to use

        Returns:
            ChipRecommendation with analysis
        """
        chip_values = []
        save_for_later = []
        recommended_chip = None

        # Calculate value of each available chip
        if ChipType.BENCH_BOOST in available_chips:
            bb_value = self.calculate_bench_boost_value(squad, gameweek)
            chip_values.append(bb_value)
            if not bb_value.is_recommended and bb_value.estimated_value < self.BB_MIN_VALUE:
                save_for_later.append("Save Bench Boost for a Double GW with strong bench")

        if ChipType.TRIPLE_CAPTAIN in available_chips:
            tc_value = self.calculate_triple_captain_value(squad, gameweek)
            chip_values.append(tc_value)
            if not tc_value.is_recommended and tc_value.estimated_value < self.TC_MIN_VALUE:
                save_for_later.append("Save Triple Captain for a Double GW with premium captain")

        if ChipType.FREE_HIT in available_chips:
            fh_value = self.calculate_free_hit_value(squad, gameweek)
            chip_values.append(fh_value)
            if not fh_value.is_recommended and fh_value.estimated_value < self.FH_MIN_VALUE:
                save_for_later.append("Save Free Hit for a Blank GW")

        if ChipType.WILDCARD in available_chips:
            wc_value = self.calculate_wildcard_value(squad, gameweek)
            chip_values.append(wc_value)
            if not wc_value.is_recommended:
                save_for_later.append("Save Wildcard for major fixture swing or team crisis")

        # Sort by value
        chip_values.sort(key=lambda x: x.estimated_value, reverse=True)

        # Get the best recommended chip
        recommended = [cv for cv in chip_values if cv.is_recommended]
        if recommended:
            recommended_chip = recommended[0].chip
            reasoning = recommended[0].reasoning
        else:
            reasoning = "No chip recommended this week - save for better opportunities"

        # Look ahead for opportunities
        upcoming = self._find_upcoming_opportunities(squad, gameweek, available_chips)

        return ChipRecommendation(
            gameweek=gameweek,
            recommended_chip=recommended_chip,
            chip_values=chip_values,
            reasoning=reasoning,
            save_for_later=save_for_later,
            upcoming_opportunities=upcoming,
        )

    def create_chip_timing_plan(
        self,
        squad: Squad,
        start_gameweek: int,
        horizon: int = 10,
        available_chips: list[ChipType] | None = None,
    ) -> ChipTimingPlan:
        """
        Create a full chip timing plan over multiple gameweeks.

        Args:
            squad: Current squad
            start_gameweek: Starting gameweek
            horizon: Number of weeks to plan
            available_chips: Chips available (default: all)

        Returns:
            ChipTimingPlan with optimal chip timing
        """
        if available_chips is None:
            available_chips = [
                ChipType.BENCH_BOOST,
                ChipType.TRIPLE_CAPTAIN,
                ChipType.FREE_HIT,
                ChipType.WILDCARD,
            ]

        recommendations = {}
        best_values = {
            ChipType.BENCH_BOOST: (None, 0),
            ChipType.TRIPLE_CAPTAIN: (None, 0),
            ChipType.FREE_HIT: (None, 0),
            ChipType.WILDCARD: (None, 0),
        }

        for gw in range(start_gameweek, start_gameweek + horizon):
            rec = self.get_chip_recommendation(squad, gw, available_chips)
            recommendations[gw] = rec

            # Track best GW for each chip
            for cv in rec.chip_values:
                current_best_gw, current_best_value = best_values[cv.chip]
                if cv.estimated_value > current_best_value:
                    best_values[cv.chip] = (gw, cv.estimated_value)

        total_value = sum(v for gw, v in best_values.values() if gw is not None)

        # 2025/26 Rule: Check for first-half chip deadline warnings
        deadline_warnings = self._get_chip_deadline_warnings(
            current_gw=start_gameweek,
            available_chips=available_chips,
        )
        first_half_remaining = len([
            c for c in available_chips
            if start_gameweek < CHIP_FIRST_HALF_DEADLINE
        ])

        return ChipTimingPlan(
            horizon=horizon,
            recommendations_by_gw=recommendations,
            best_bb_gw=best_values[ChipType.BENCH_BOOST][0],
            best_tc_gw=best_values[ChipType.TRIPLE_CAPTAIN][0],
            best_fh_gw=best_values[ChipType.FREE_HIT][0],
            best_wc_gw=best_values[ChipType.WILDCARD][0],
            total_chip_value=total_value,
            deadline_warnings=deadline_warnings,
            first_half_chips_remaining=first_half_remaining,
        )

    def _get_chip_deadline_warnings(
        self,
        current_gw: int,
        available_chips: list[ChipType],
    ) -> list[ChipDeadlineWarning]:
        """
        Generate warnings for chips approaching 2025/26 first-half deadline.

        First set of chips must be used before GW19.
        """
        warnings = []

        # Only generate warnings if we're in the first half
        if current_gw >= CHIP_FIRST_HALF_DEADLINE:
            return warnings

        gws_remaining = CHIP_FIRST_HALF_DEADLINE - current_gw

        for chip in available_chips:
            # Determine urgency level
            if gws_remaining <= 1:
                urgency = "critical"
                message = f"⚠️ URGENT: {chip.value.upper()} expires after this GW! Use it or lose it!"
            elif gws_remaining <= 3:
                urgency = "warning"
                message = f"⏰ {chip.value.upper()} must be used within {gws_remaining} GWs (before GW{CHIP_FIRST_HALF_DEADLINE})"
            elif gws_remaining <= 5:
                urgency = "info"
                message = f"📅 {chip.value.upper()} expires in {gws_remaining} GWs - start planning"
            else:
                continue  # Don't warn too early

            warnings.append(ChipDeadlineWarning(
                chip=chip,
                chip_set=ChipSet.FIRST_HALF,
                gameweeks_remaining=gws_remaining,
                deadline_gw=CHIP_FIRST_HALF_DEADLINE,
                urgency=urgency,
                message=message,
            ))

        return warnings

    def _get_playing_probability(self, player: Player) -> float:
        """Get probability of player playing."""
        if player.chance_of_playing is not None:
            return player.chance_of_playing / 100
        if player.status.value == "a":  # Available
            return 0.95
        if player.status.value == "d":  # Doubtful
            return 0.5
        return 0.1

    def _find_upcoming_opportunities(
        self,
        squad: Squad,
        current_gw: int,
        available_chips: list[ChipType],
        look_ahead: int = 6,
    ) -> list[tuple[int, ChipType, str]]:
        """Find upcoming good opportunities for chip usage."""
        opportunities = []

        for gw in range(current_gw, current_gw + look_ahead):
            gw_info = self.gameweeks.get(gw)
            if not gw_info:
                continue

            if gw_info.is_double and ChipType.BENCH_BOOST in available_chips:
                opportunities.append((
                    gw,
                    ChipType.BENCH_BOOST,
                    f"GW{gw} is a Double - consider Bench Boost"
                ))

            if gw_info.is_double and ChipType.TRIPLE_CAPTAIN in available_chips:
                opportunities.append((
                    gw,
                    ChipType.TRIPLE_CAPTAIN,
                    f"GW{gw} is a Double - consider Triple Captain"
                ))

            if gw_info.is_blank and ChipType.FREE_HIT in available_chips:
                opportunities.append((
                    gw,
                    ChipType.FREE_HIT,
                    f"GW{gw} is a Blank - consider Free Hit"
                ))

        return opportunities


# Convenience functions
def get_chip_optimizer(
    players: list[Player],
    gameweeks: list[GameweekInfo],
    projections_by_gw: dict[int, dict[int, float]],
) -> ChipOptimizer:
    """Get a chip optimizer instance."""
    return ChipOptimizer(players, gameweeks, projections_by_gw)


def recommend_chip(
    squad: Squad,
    gameweek: int,
    players: list[Player],
    gameweeks: list[GameweekInfo],
    projections_by_gw: dict[int, dict[int, float]],
    available_chips: list[ChipType],
) -> ChipRecommendation:
    """
    Get chip recommendation for current gameweek.

    Args:
        squad: Current squad
        gameweek: Current gameweek
        players: All players
        gameweeks: All gameweek info
        projections_by_gw: Projections by gameweek
        available_chips: Available chips

    Returns:
        ChipRecommendation
    """
    optimizer = ChipOptimizer(players, gameweeks, projections_by_gw)
    return optimizer.get_chip_recommendation(squad, gameweek, available_chips)
