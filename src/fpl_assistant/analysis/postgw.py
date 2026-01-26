"""
Post-Gameweek Analysis for FPL.

Elite managers distinguish between bad luck and bad decisions.
This module analyzes what went right/wrong and why.

Key insight: A good process can have bad outcomes (luck) and
a bad process can have good outcomes (also luck). We want to
identify and reinforce good processes.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from ..data.models import Player, Position

logger = logging.getLogger(__name__)


class OutcomeType(StrEnum):
    """Classification of decision outcomes."""

    GOOD_PROCESS_GOOD_OUTCOME = "good_good"    # Expected and happened
    GOOD_PROCESS_BAD_OUTCOME = "good_bad"      # Unlucky - trust the process
    BAD_PROCESS_GOOD_OUTCOME = "bad_good"      # Got lucky - don't repeat
    BAD_PROCESS_BAD_OUTCOME = "bad_bad"        # Learn from this


@dataclass
class CaptainAnalysis:
    """Analysis of captain decision."""

    captain_name: str
    captain_actual_points: int
    captain_predicted_xp: float

    # Was it a good process?
    was_top_3_xp: bool          # Was captain in top 3 by xP?
    was_top_3_actual: bool      # Was captain in actual top 3?

    # Alternatives
    best_captain_name: str
    best_captain_actual_points: int
    points_lost_to_best: int

    # Verdict
    outcome_type: OutcomeType
    reasoning: str


@dataclass
class BenchAnalysis:
    """Analysis of bench decisions."""

    total_bench_points: int
    total_starter_points: int

    # Auto-subs that happened
    auto_subs_made: int
    auto_sub_points_gained: int

    # Bench players who outscored starters
    bench_beats_starter: list[tuple[str, int, str, int]]  # (bench_name, bench_pts, starter_name, starter_pts)

    # Verdict
    bench_order_correct: bool
    reasoning: str


@dataclass
class TransferAnalysis:
    """Analysis of transfer decisions."""

    transfers_made: int
    hits_taken: int
    hit_cost: int

    # Transfer outcomes
    transfer_outcomes: list[dict]  # {player_out, out_pts, player_in, in_pts, net_gain}

    # Net impact
    total_transfer_gain: int
    net_after_hits: int

    # Verdict
    transfers_worth_it: bool
    reasoning: str


@dataclass
class PostGWAnalysis:
    """
    Complete post-gameweek analysis.

    Helps answer: "Why did I get this score?"
    """

    gameweek: int

    # Points summary
    actual_points: int
    predicted_points: float
    variance: float              # actual - predicted

    # Component analysis
    captain_analysis: CaptainAnalysis
    bench_analysis: BenchAnalysis
    transfer_analysis: TransferAnalysis

    # Overall classification
    variance_contribution: float  # Points lost/gained to luck
    decision_contribution: float  # Points lost/gained to decisions
    overall_outcome: OutcomeType

    # Key learnings
    lessons: list[str]

    @property
    def was_lucky(self) -> bool:
        """Did luck help this week?"""
        return self.variance > 2

    @property
    def was_unlucky(self) -> bool:
        """Did luck hurt this week?"""
        return self.variance < -2


class PostGWAnalyzer:
    """
    Analyzes gameweek results to distinguish luck from decisions.

    Usage:
    ```
    analyzer = PostGWAnalyzer(players, my_picks, results)
    analysis = analyzer.analyze()
    print(analysis.lessons)
    ```
    """

    def __init__(
        self,
        players: dict[int, Player],
        my_picks: list[dict],           # From FPL API my-team
        actual_results: dict[int, int],  # player_id -> actual points
        predictions: dict[int, float],   # player_id -> predicted xP
        transfers_made: list[dict] | None = None,  # Transfers this GW
    ):
        """
        Initialize analyzer.

        Args:
            players: Dict of player_id -> Player
            my_picks: User's picks for this GW
            actual_results: Actual points scored
            predictions: Predicted xP before GW
            transfers_made: Optional list of transfers made
        """
        self.players = players
        self.my_picks = my_picks
        self.actual_results = actual_results
        self.predictions = predictions
        self.transfers_made = transfers_made or []

    def analyze(self, gameweek: int) -> PostGWAnalysis:
        """Run complete post-GW analysis."""

        # Calculate basic totals
        actual_total = self._calculate_actual_total()
        predicted_total = self._calculate_predicted_total()

        # Analyze each component
        captain_analysis = self._analyze_captain()
        bench_analysis = self._analyze_bench()
        transfer_analysis = self._analyze_transfers()

        # Calculate variance breakdown
        variance = actual_total - predicted_total
        variance_contribution = self._estimate_luck_contribution(variance)
        decision_contribution = variance - variance_contribution

        # Determine overall outcome
        good_decisions = (
            captain_analysis.was_top_3_xp
            and (transfer_analysis.transfers_worth_it or transfer_analysis.transfers_made == 0)
        )
        good_outcome = actual_total >= predicted_total - 2

        if good_decisions and good_outcome:
            overall_outcome = OutcomeType.GOOD_PROCESS_GOOD_OUTCOME
        elif good_decisions and not good_outcome:
            overall_outcome = OutcomeType.GOOD_PROCESS_BAD_OUTCOME
        elif not good_decisions and good_outcome:
            overall_outcome = OutcomeType.BAD_PROCESS_GOOD_OUTCOME
        else:
            overall_outcome = OutcomeType.BAD_PROCESS_BAD_OUTCOME

        # Generate lessons
        lessons = self._generate_lessons(
            captain_analysis,
            bench_analysis,
            transfer_analysis,
            overall_outcome,
        )

        return PostGWAnalysis(
            gameweek=gameweek,
            actual_points=actual_total,
            predicted_points=round(predicted_total, 1),
            variance=round(variance, 1),
            captain_analysis=captain_analysis,
            bench_analysis=bench_analysis,
            transfer_analysis=transfer_analysis,
            variance_contribution=round(variance_contribution, 1),
            decision_contribution=round(decision_contribution, 1),
            overall_outcome=overall_outcome,
            lessons=lessons,
        )

    def _calculate_actual_total(self) -> int:
        """Calculate actual points from picks."""
        total = 0
        for pick in self.my_picks:
            player_id = pick["element"]
            pts = self.actual_results.get(player_id, 0)

            # Captain gets 2x
            if pick.get("is_captain"):
                pts *= 2
            elif pick.get("multiplier", 1) > 1:
                pts *= pick["multiplier"]

            # Only count starters (position <= 11)
            if pick.get("position", 1) <= 11:
                total += pts

        return total

    def _calculate_predicted_total(self) -> float:
        """Calculate predicted points from picks."""
        total = 0.0
        for pick in self.my_picks:
            player_id = pick["element"]
            xp = self.predictions.get(player_id, 0)

            # Captain gets 2x
            if pick.get("is_captain"):
                xp *= 2

            # Only count starters
            if pick.get("position", 1) <= 11:
                total += xp

        return total

    def _analyze_captain(self) -> CaptainAnalysis:
        """Analyze captain decision."""

        # Find captain
        captain_pick = next((p for p in self.my_picks if p.get("is_captain")), None)
        if not captain_pick:
            return CaptainAnalysis(
                captain_name="Unknown",
                captain_actual_points=0,
                captain_predicted_xp=0,
                was_top_3_xp=False,
                was_top_3_actual=False,
                best_captain_name="Unknown",
                best_captain_actual_points=0,
                points_lost_to_best=0,
                outcome_type=OutcomeType.BAD_PROCESS_BAD_OUTCOME,
                reasoning="No captain found",
            )

        captain_id = captain_pick["element"]
        captain = self.players.get(captain_id)
        captain_name = captain.web_name if captain else "Unknown"
        captain_actual = self.actual_results.get(captain_id, 0)
        captain_xp = self.predictions.get(captain_id, 0)

        # Get all squad players sorted by xP and actual
        squad_by_xp = sorted(
            [(p["element"], self.predictions.get(p["element"], 0)) for p in self.my_picks],
            key=lambda x: -x[1]
        )
        squad_by_actual = sorted(
            [(p["element"], self.actual_results.get(p["element"], 0)) for p in self.my_picks],
            key=lambda x: -x[1]
        )

        # Was captain in top 3 by xP?
        top_3_xp_ids = [x[0] for x in squad_by_xp[:3]]
        was_top_3_xp = captain_id in top_3_xp_ids

        # Was captain in top 3 by actual?
        top_3_actual_ids = [x[0] for x in squad_by_actual[:3]]
        was_top_3_actual = captain_id in top_3_actual_ids

        # Best captain
        best_captain_id = squad_by_actual[0][0]
        best_captain = self.players.get(best_captain_id)
        best_captain_name = best_captain.web_name if best_captain else "Unknown"
        best_captain_actual = squad_by_actual[0][1]
        points_lost = (best_captain_actual - captain_actual)  # Difference (captain gets 2x)

        # Determine outcome
        if was_top_3_xp and was_top_3_actual:
            outcome = OutcomeType.GOOD_PROCESS_GOOD_OUTCOME
            reasoning = f"Good pick! {captain_name} was in your top 3 by xP and delivered."
        elif was_top_3_xp and not was_top_3_actual:
            outcome = OutcomeType.GOOD_PROCESS_BAD_OUTCOME
            reasoning = f"Unlucky! {captain_name} was a good pick by xP but underperformed. Keep trusting the process."
        elif not was_top_3_xp and was_top_3_actual:
            outcome = OutcomeType.BAD_PROCESS_GOOD_OUTCOME
            reasoning = f"Got lucky! {captain_name} wasn't in your top 3 by xP but still scored well. Don't rely on this."
        else:
            outcome = OutcomeType.BAD_PROCESS_BAD_OUTCOME
            reasoning = f"Bad pick. {captain_name} wasn't high xP and didn't deliver. Consider using xP-based captaincy."

        return CaptainAnalysis(
            captain_name=captain_name,
            captain_actual_points=captain_actual,
            captain_predicted_xp=captain_xp,
            was_top_3_xp=was_top_3_xp,
            was_top_3_actual=was_top_3_actual,
            best_captain_name=best_captain_name,
            best_captain_actual_points=best_captain_actual,
            points_lost_to_best=points_lost,
            outcome_type=outcome,
            reasoning=reasoning,
        )

    def _analyze_bench(self) -> BenchAnalysis:
        """Analyze bench decisions."""

        starters = [p for p in self.my_picks if p.get("position", 1) <= 11]
        bench = [p for p in self.my_picks if p.get("position", 1) > 11]

        starter_points = sum(
            self.actual_results.get(p["element"], 0)
            for p in starters
            if not p.get("is_captain")  # Don't double count captain
        )
        # Add captain separately (without 2x for this comparison)
        captain = next((p for p in starters if p.get("is_captain")), None)
        if captain:
            starter_points += self.actual_results.get(captain["element"], 0)

        bench_points = sum(
            self.actual_results.get(p["element"], 0)
            for p in bench
        )

        # Find bench players who outscored starters
        bench_beats = []
        bench_sorted = sorted(bench, key=lambda p: -self.actual_results.get(p["element"], 0))
        starters_sorted = sorted(
            [s for s in starters if not s.get("is_captain")],
            key=lambda p: self.actual_results.get(p["element"], 0)
        )

        for bp in bench_sorted:
            bp_pts = self.actual_results.get(bp["element"], 0)
            bp_player = self.players.get(bp["element"])
            for sp in starters_sorted:
                sp_pts = self.actual_results.get(sp["element"], 0)
                sp_player = self.players.get(sp["element"])
                if bp_pts > sp_pts:
                    bench_beats.append((
                        bp_player.web_name if bp_player else "?",
                        bp_pts,
                        sp_player.web_name if sp_player else "?",
                        sp_pts,
                    ))
                    break

        # Check bench order (was highest scorer first?)
        bench_order_correct = len(bench) <= 1 or (
            self.actual_results.get(bench[0]["element"], 0) >=
            max(self.actual_results.get(b["element"], 0) for b in bench[1:])
        )

        reasoning = ""
        if bench_beats:
            reasoning = f"Bench outscored {len(bench_beats)} starter(s). Consider rotation risk."
        elif bench_order_correct:
            reasoning = "Good bench order - highest scorer was first."
        else:
            reasoning = "Bench order could be optimized."

        return BenchAnalysis(
            total_bench_points=bench_points,
            total_starter_points=starter_points,
            auto_subs_made=0,  # Would need live data
            auto_sub_points_gained=0,
            bench_beats_starter=bench_beats,
            bench_order_correct=bench_order_correct,
            reasoning=reasoning,
        )

    def _analyze_transfers(self) -> TransferAnalysis:
        """Analyze transfer decisions."""

        if not self.transfers_made:
            return TransferAnalysis(
                transfers_made=0,
                hits_taken=0,
                hit_cost=0,
                transfer_outcomes=[],
                total_transfer_gain=0,
                net_after_hits=0,
                transfers_worth_it=True,
                reasoning="No transfers made this week.",
            )

        outcomes = []
        total_gain = 0

        for t in self.transfers_made:
            out_id = t.get("element_out")
            in_id = t.get("element_in")

            out_player = self.players.get(out_id)
            in_player = self.players.get(in_id)

            out_pts = self.actual_results.get(out_id, 0)
            in_pts = self.actual_results.get(in_id, 0)
            gain = in_pts - out_pts

            outcomes.append({
                "player_out": out_player.web_name if out_player else "?",
                "out_pts": out_pts,
                "player_in": in_player.web_name if in_player else "?",
                "in_pts": in_pts,
                "net_gain": gain,
            })
            total_gain += gain

        # Estimate hits (would need API data for accuracy)
        hits_taken = max(0, len(self.transfers_made) - 1)  # Assuming 1 FT
        hit_cost = hits_taken * 4
        net_after_hits = total_gain - hit_cost

        worth_it = net_after_hits >= 0

        if worth_it and total_gain > 0:
            reasoning = f"Good transfers! Gained {total_gain} pts."
        elif worth_it:
            reasoning = "Transfers broke even."
        else:
            reasoning = f"Transfers lost {-net_after_hits} pts after hits. Consider being more patient."

        return TransferAnalysis(
            transfers_made=len(self.transfers_made),
            hits_taken=hits_taken,
            hit_cost=hit_cost,
            transfer_outcomes=outcomes,
            total_transfer_gain=total_gain,
            net_after_hits=net_after_hits,
            transfers_worth_it=worth_it,
            reasoning=reasoning,
        )

    def _estimate_luck_contribution(self, variance: float) -> float:
        """
        Estimate how much of the variance was due to luck.

        Luck factors:
        - Bonus points (hard to predict)
        - Goals on low xG
        - Penalties
        - Red cards
        """
        # Simple heuristic: assume ~60% of variance is luck
        return variance * 0.6

    def _generate_lessons(
        self,
        captain: CaptainAnalysis,
        bench: BenchAnalysis,
        transfers: TransferAnalysis,
        overall: OutcomeType,
    ) -> list[str]:
        """Generate actionable lessons from analysis."""

        lessons = []

        # Captain lessons
        if captain.outcome_type == OutcomeType.GOOD_PROCESS_BAD_OUTCOME:
            lessons.append(f"Captain {captain.captain_name} was unlucky (-{captain.points_lost_to_best} vs best). Keep trusting xP-based picks.")
        elif captain.outcome_type == OutcomeType.BAD_PROCESS_GOOD_OUTCOME:
            lessons.append(f"Captain {captain.captain_name} got lucky. Next time, pick from top 3 by xP.")
        elif captain.outcome_type == OutcomeType.BAD_PROCESS_BAD_OUTCOME:
            lessons.append(f"Lost {captain.points_lost_to_best} captain pts. Use xP to guide captaincy.")

        # Bench lessons
        if bench.bench_beats_starter:
            for bench_name, bench_pts, starter_name, starter_pts in bench.bench_beats_starter[:2]:
                lessons.append(f"{bench_name} (bench: {bench_pts}pts) outscored {starter_name} ({starter_pts}pts). Check rotation risk.")

        # Transfer lessons
        if not transfers.transfers_worth_it and transfers.transfers_made > 0:
            lessons.append(f"Transfers cost {-transfers.net_after_hits} pts net. Be more patient with hits.")

        # Overall
        if overall == OutcomeType.GOOD_PROCESS_BAD_OUTCOME:
            lessons.append("Overall: Good decisions, bad luck. Variance will even out - trust the process!")
        elif overall == OutcomeType.BAD_PROCESS_GOOD_OUTCOME:
            lessons.append("Overall: Got lucky this week. Review decision process for sustainability.")

        return lessons


def analyze_gameweek(
    players: dict[int, Player],
    my_picks: list[dict],
    actual_results: dict[int, int],
    predictions: dict[int, float],
    gameweek: int,
    transfers: list[dict] | None = None,
) -> PostGWAnalysis:
    """Convenience function for post-GW analysis."""
    analyzer = PostGWAnalyzer(players, my_picks, actual_results, predictions, transfers)
    return analyzer.analyze(gameweek)
