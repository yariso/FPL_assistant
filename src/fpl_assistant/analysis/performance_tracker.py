"""
Performance Tracker for FPL Tool Evaluation.

Tracks how well the tool's advice performs over time:
- Actual vs Predicted points
- Captain accuracy
- Transfer effectiveness
- Rank improvements

Use this to evaluate if the tool is improving your scores.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GameweekPerformance:
    """Performance record for a single gameweek."""

    gameweek: int
    date: str

    # Points
    actual_points: int
    predicted_points: float
    variance: float  # actual - predicted

    # Captain
    captain_name: str
    captain_actual_points: int
    captain_was_recommended: bool
    best_captain_points: int  # What the best pick would have scored

    # Rank
    overall_rank: int
    rank_change: int  # vs previous week (negative = improvement)

    # Transfers
    transfers_made: int
    hits_taken: int
    transfer_net_gain: int  # Actual points gained from transfers

    @property
    def captain_accuracy(self) -> float:
        """How close was captain to best option (0-1)."""
        if self.best_captain_points == 0:
            return 0
        return min(1.0, self.captain_actual_points / self.best_captain_points)

    @property
    def prediction_accuracy(self) -> float:
        """How accurate was prediction (0-1, 1=perfect)."""
        if self.actual_points == 0:
            return 0
        error = abs(self.variance) / self.actual_points
        return max(0, 1 - error)


@dataclass
class PerformanceSummary:
    """Summary statistics across multiple gameweeks."""

    gameweeks_tracked: int
    first_gameweek: int
    last_gameweek: int

    # Points
    total_actual_points: int
    total_predicted_points: float
    avg_points_per_week: float
    avg_prediction_error: float

    # Captain
    captain_accuracy_rate: float  # % of times captain was top 3
    captain_points_captured: float  # % of best captain points captured
    avg_captain_points: float

    # Rank
    starting_rank: int
    current_rank: int
    rank_improvement: int  # Positive = improved
    best_rank: int

    # Transfers
    total_transfers: int
    total_hits: int
    total_hit_cost: int
    total_transfer_value: int  # Net points from transfers

    # Tool effectiveness
    weeks_beat_prediction: int  # Weeks where actual > predicted
    weeks_captain_was_best: int  # Weeks where our captain was highest


class PerformanceTracker:
    """
    Tracks and evaluates tool performance over time.

    Usage:
    ```
    tracker = PerformanceTracker()

    # After each gameweek
    tracker.record_gameweek(
        gameweek=23,
        actual_points=58,
        predicted_points=62,
        captain_name="Salah",
        captain_points=12,
        ...
    )

    # Get summary
    summary = tracker.get_summary()
    print(f"Captain accuracy: {summary.captain_accuracy_rate:.0%}")
    ```
    """

    DEFAULT_FILE = "data/performance_history.json"

    def __init__(self, history_file: str | Path | None = None):
        """Initialize tracker."""
        self.history_file = Path(history_file or self.DEFAULT_FILE)
        self._history: list[GameweekPerformance] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load history from file."""
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self._history = [
                    GameweekPerformance(**gw) for gw in data.get("gameweeks", [])
                ]
                logger.info(f"Loaded {len(self._history)} gameweeks of history")
            except Exception as e:
                logger.warning(f"Failed to load history: {e}")
                self._history = []

    def _save_history(self) -> None:
        """Save history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "gameweeks": [asdict(gw) for gw in self._history],
            "updated_at": datetime.now().isoformat(),
        }
        with open(self.history_file, "w") as f:
            json.dump(data, f, indent=2)

    def record_gameweek(
        self,
        gameweek: int,
        actual_points: int,
        predicted_points: float,
        captain_name: str,
        captain_actual_points: int,
        captain_was_recommended: bool,
        best_captain_points: int,
        overall_rank: int,
        transfers_made: int = 0,
        hits_taken: int = 0,
        transfer_net_gain: int = 0,
    ) -> GameweekPerformance:
        """
        Record performance for a gameweek.

        Args:
            gameweek: Gameweek number
            actual_points: Your actual GW points
            predicted_points: What the tool predicted
            captain_name: Who you captained
            captain_actual_points: Points captain got (before 2x)
            captain_was_recommended: Was this the tool's recommendation?
            best_captain_points: Points the best captain option got
            overall_rank: Your OR after this GW
            transfers_made: Number of transfers
            hits_taken: Number of hits (-4s)
            transfer_net_gain: Net points from transfers

        Returns:
            GameweekPerformance record
        """
        # Calculate rank change
        prev_rank = self._history[-1].overall_rank if self._history else overall_rank
        rank_change = overall_rank - prev_rank  # Negative = improvement

        perf = GameweekPerformance(
            gameweek=gameweek,
            date=datetime.now().strftime("%Y-%m-%d"),
            actual_points=actual_points,
            predicted_points=predicted_points,
            variance=actual_points - predicted_points,
            captain_name=captain_name,
            captain_actual_points=captain_actual_points,
            captain_was_recommended=captain_was_recommended,
            best_captain_points=best_captain_points,
            overall_rank=overall_rank,
            rank_change=rank_change,
            transfers_made=transfers_made,
            hits_taken=hits_taken,
            transfer_net_gain=transfer_net_gain,
        )

        # Update or add
        existing_idx = next(
            (i for i, gw in enumerate(self._history) if gw.gameweek == gameweek),
            None
        )
        if existing_idx is not None:
            self._history[existing_idx] = perf
        else:
            self._history.append(perf)
            self._history.sort(key=lambda x: x.gameweek)

        self._save_history()
        return perf

    def get_summary(self) -> PerformanceSummary:
        """Get performance summary across all tracked weeks."""
        if not self._history:
            return PerformanceSummary(
                gameweeks_tracked=0,
                first_gameweek=0,
                last_gameweek=0,
                total_actual_points=0,
                total_predicted_points=0,
                avg_points_per_week=0,
                avg_prediction_error=0,
                captain_accuracy_rate=0,
                captain_points_captured=0,
                avg_captain_points=0,
                starting_rank=0,
                current_rank=0,
                rank_improvement=0,
                best_rank=0,
                total_transfers=0,
                total_hits=0,
                total_hit_cost=0,
                total_transfer_value=0,
                weeks_beat_prediction=0,
                weeks_captain_was_best=0,
            )

        # Calculate metrics
        total_actual = sum(gw.actual_points for gw in self._history)
        total_predicted = sum(gw.predicted_points for gw in self._history)
        total_variance = sum(abs(gw.variance) for gw in self._history)

        captain_points = sum(gw.captain_actual_points for gw in self._history)
        best_captain_points = sum(gw.best_captain_points for gw in self._history)
        weeks_captain_best = sum(
            1 for gw in self._history
            if gw.captain_actual_points >= gw.best_captain_points * 0.9
        )

        return PerformanceSummary(
            gameweeks_tracked=len(self._history),
            first_gameweek=self._history[0].gameweek,
            last_gameweek=self._history[-1].gameweek,
            total_actual_points=total_actual,
            total_predicted_points=total_predicted,
            avg_points_per_week=total_actual / len(self._history),
            avg_prediction_error=total_variance / len(self._history),
            captain_accuracy_rate=weeks_captain_best / len(self._history),
            captain_points_captured=(
                captain_points / best_captain_points if best_captain_points > 0 else 0
            ),
            avg_captain_points=captain_points / len(self._history),
            starting_rank=self._history[0].overall_rank,
            current_rank=self._history[-1].overall_rank,
            rank_improvement=self._history[0].overall_rank - self._history[-1].overall_rank,
            best_rank=min(gw.overall_rank for gw in self._history),
            total_transfers=sum(gw.transfers_made for gw in self._history),
            total_hits=sum(gw.hits_taken for gw in self._history),
            total_hit_cost=sum(gw.hits_taken * 4 for gw in self._history),
            total_transfer_value=sum(gw.transfer_net_gain for gw in self._history),
            weeks_beat_prediction=sum(1 for gw in self._history if gw.variance > 0),
            weeks_captain_was_best=weeks_captain_best,
        )

    def get_weekly_trend(self) -> list[dict]:
        """Get week-by-week trend data."""
        return [
            {
                "gameweek": gw.gameweek,
                "actual": gw.actual_points,
                "predicted": gw.predicted_points,
                "rank": gw.overall_rank,
                "captain_captured": gw.captain_accuracy,
            }
            for gw in self._history
        ]

    def compare_to_average(self, average_points_per_week: float = 50) -> dict:
        """
        Compare your performance to league average.

        Args:
            average_points_per_week: Typical average (default 50)

        Returns:
            Dict with comparison metrics
        """
        if not self._history:
            return {}

        your_avg = sum(gw.actual_points for gw in self._history) / len(self._history)
        weeks_above_avg = sum(1 for gw in self._history if gw.actual_points > average_points_per_week)

        return {
            "your_average": your_avg,
            "league_average": average_points_per_week,
            "difference": your_avg - average_points_per_week,
            "weeks_above_average": weeks_above_avg,
            "percentage_above_average": weeks_above_avg / len(self._history),
        }

    def print_report(self) -> str:
        """Generate a printable performance report."""
        summary = self.get_summary()

        if summary.gameweeks_tracked == 0:
            return "No data recorded yet. Record your first gameweek!"

        report = []
        report.append("=" * 60)
        report.append("FPL TOOL PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append(f"\nTracking: GW{summary.first_gameweek} to GW{summary.last_gameweek}")
        report.append(f"Weeks tracked: {summary.gameweeks_tracked}")

        report.append("\n--- POINTS ---")
        report.append(f"Total actual points: {summary.total_actual_points}")
        report.append(f"Total predicted: {summary.total_predicted_points:.0f}")
        report.append(f"Average per week: {summary.avg_points_per_week:.1f}")
        report.append(f"Prediction error: ±{summary.avg_prediction_error:.1f} pts/week")
        report.append(f"Beat prediction: {summary.weeks_beat_prediction}/{summary.gameweeks_tracked} weeks")

        report.append("\n--- CAPTAIN ---")
        report.append(f"Captain accuracy: {summary.captain_accuracy_rate:.0%} (was top pick)")
        report.append(f"Points captured: {summary.captain_points_captured:.0%} of maximum")
        report.append(f"Avg captain points: {summary.avg_captain_points:.1f} (x2)")

        report.append("\n--- RANK ---")
        report.append(f"Starting rank: {summary.starting_rank:,}")
        report.append(f"Current rank: {summary.current_rank:,}")
        improvement = summary.rank_improvement
        if improvement > 0:
            report.append(f"Improvement: +{improvement:,} places 📈")
        else:
            report.append(f"Change: {improvement:,} places")
        report.append(f"Best rank: {summary.best_rank:,}")

        report.append("\n--- TRANSFERS ---")
        report.append(f"Total transfers: {summary.total_transfers}")
        report.append(f"Hits taken: {summary.total_hits} (-{summary.total_hit_cost} pts)")
        report.append(f"Net transfer value: {summary.total_transfer_value:+d} pts")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


# Singleton
_tracker: PerformanceTracker | None = None


def get_performance_tracker() -> PerformanceTracker:
    """Get singleton performance tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker


def record_gameweek_performance(**kwargs) -> GameweekPerformance:
    """Convenience function to record a gameweek."""
    return get_performance_tracker().record_gameweek(**kwargs)


def print_performance_report() -> str:
    """Get printable performance report."""
    return get_performance_tracker().print_report()


def auto_record_gameweek_from_api(
    manager_id: int,
    gameweek: int,
    predicted_points: float,
    captain_was_recommended: bool,
    api_client=None,
) -> GameweekPerformance | None:
    """
    Auto-record gameweek performance from FPL API data.

    This fetches actual points, captain choice, rank, transfers etc.
    automatically from the FPL API instead of manual entry.

    Args:
        manager_id: FPL manager ID
        gameweek: Gameweek to record
        predicted_points: What the tool predicted (from projections)
        captain_was_recommended: Whether the captain choice matched recommendation
        api_client: Optional FPL API client

    Returns:
        GameweekPerformance record or None if failed
    """
    try:
        # Import here to avoid circular imports
        from fpl_assistant.api import SyncFPLClient

        client = api_client or SyncFPLClient()

        # Get manager entry history
        entry_history = client.get_entry_history(manager_id)

        # Find the specific gameweek in history
        gw_history = None
        for gw in entry_history.get("current", []):
            if gw.get("event") == gameweek:
                gw_history = gw
                break

        if not gw_history:
            logger.warning(f"GW{gameweek} not found in history - may not be finished yet")
            return None

        # Get picks for this gameweek to find captain
        picks_data = client.get_entry_picks(manager_id, gameweek)
        picks = picks_data.get("picks", [])

        captain_id = None
        captain_player = None
        for pick in picks:
            if pick.get("is_captain"):
                captain_id = pick.get("element")
                break

        # Get bootstrap data for player details
        bootstrap = client.get_bootstrap_static()
        elements = {p["id"]: p for p in bootstrap.get("elements", [])}

        if captain_id and captain_id in elements:
            captain_player = elements[captain_id]

        # Calculate captain points from live data
        captain_actual_points = 0
        best_captain_points = 0

        if picks:
            # Get live event data for actual points
            try:
                live_data = client.get_live_event(gameweek)
                live_elements = {e["id"]: e for e in live_data.get("elements", [])}

                for pick in picks:
                    pid = pick.get("element")
                    if pid in live_elements:
                        pts = live_elements[pid].get("stats", {}).get("total_points", 0)
                        if pick.get("is_captain"):
                            captain_actual_points = pts
                        best_captain_points = max(best_captain_points, pts)
            except Exception:
                # Live data might not be available, use form estimate
                captain_actual_points = captain_player.get("event_points", 0) if captain_player else 0
                for pick in picks:
                    pid = pick.get("element")
                    if pid in elements:
                        best_captain_points = max(best_captain_points, elements[pid].get("event_points", 0))

        # Get transfer info
        transfers_made = gw_history.get("event_transfers", 0)
        hits_taken = gw_history.get("event_transfers_cost", 0) // 4

        # Record the performance
        tracker = get_performance_tracker()
        return tracker.record_gameweek(
            gameweek=gameweek,
            actual_points=gw_history.get("points", 0),
            predicted_points=predicted_points,
            captain_name=captain_player.get("web_name", "Unknown") if captain_player else "Unknown",
            captain_actual_points=captain_actual_points,
            captain_was_recommended=captain_was_recommended,
            best_captain_points=best_captain_points,
            overall_rank=gw_history.get("overall_rank", 0),
            transfers_made=transfers_made,
            hits_taken=hits_taken,
            transfer_net_gain=0,  # Hard to calculate, leaving as 0
        )

    except Exception as e:
        logger.error(f"Failed to auto-record GW{gameweek}: {e}")
        return None


def get_auto_performance_summary(manager_id: int, api_client=None) -> dict:
    """
    Get performance summary by fetching all gameweek data from API.

    This auto-calculates everything without manual entry.

    Args:
        manager_id: FPL manager ID
        api_client: Optional FPL API client

    Returns:
        Dict with performance metrics
    """
    try:
        from fpl_assistant.api import SyncFPLClient

        client = api_client or SyncFPLClient()

        # Get full history
        entry_history = client.get_entry_history(manager_id)
        current_history = entry_history.get("current", [])

        if not current_history:
            return {"error": "No gameweek history found"}

        # Calculate metrics
        total_points = sum(gw.get("points", 0) for gw in current_history)
        total_gws = len(current_history)
        avg_points = total_points / total_gws if total_gws > 0 else 0

        starting_rank = current_history[0].get("overall_rank", 0)
        current_rank = current_history[-1].get("overall_rank", 0)
        best_rank = min(gw.get("overall_rank", 9999999) for gw in current_history)

        total_transfers = sum(gw.get("event_transfers", 0) for gw in current_history)
        total_hits = sum(gw.get("event_transfers_cost", 0) // 4 for gw in current_history)
        total_hit_cost = sum(gw.get("event_transfers_cost", 0) for gw in current_history)

        # Team value tracking (value is in tenths, e.g. 1000 = £100.0m)
        weekly_values = [gw.get("value", 0) / 10 for gw in current_history]
        weekly_bank = [gw.get("bank", 0) / 10 for gw in current_history]
        starting_value = weekly_values[0] if weekly_values else 100.0
        current_value = weekly_values[-1] if weekly_values else 100.0
        value_gained = current_value - starting_value

        return {
            "gameweeks_tracked": total_gws,
            "total_points": total_points,
            "avg_points_per_week": round(avg_points, 1),
            "starting_rank": starting_rank,
            "current_rank": current_rank,
            "best_rank": best_rank,
            "rank_improvement": starting_rank - current_rank,
            "total_transfers": total_transfers,
            "total_hits": total_hits,
            "total_hit_cost": total_hit_cost,
            "weekly_points": [gw.get("points", 0) for gw in current_history],
            "weekly_ranks": [gw.get("overall_rank", 0) for gw in current_history],
            "weekly_values": weekly_values,
            "weekly_bank": weekly_bank,
            "starting_value": starting_value,
            "current_value": current_value,
            "value_gained": round(value_gained, 1),
        }

    except Exception as e:
        return {"error": str(e)}
