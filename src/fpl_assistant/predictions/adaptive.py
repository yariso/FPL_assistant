"""
Adaptive Weight Adjustment for Predictions.

Learns optimal weights from backtest results to improve prediction accuracy.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WeightConfig:
    """Configuration of projection weights."""

    form_weight: float = 0.20
    ict_weight: float = 0.15
    fdr_weight: float = 0.15
    consistency_weight: float = 0.10
    team_strength_weight: float = 0.05
    xg_weight: float = 0.35  # xG is the PRIMARY predictor

    # Performance metrics from when these weights were tested
    mae: float | None = None
    correlation: float | None = None
    top_10_hit_rate: float | None = None
    captain_accuracy: float | None = None
    tested_at: str | None = None
    gameweeks_tested: list[int] | None = None

    def to_array(self) -> list[float]:
        """Convert weights to array for optimization."""
        return [
            self.form_weight,
            self.ict_weight,
            self.fdr_weight,
            self.consistency_weight,
            self.team_strength_weight,
            self.xg_weight,
        ]

    @classmethod
    def from_array(cls, arr: list[float]) -> "WeightConfig":
        """Create WeightConfig from array."""
        return cls(
            form_weight=arr[0],
            ict_weight=arr[1],
            fdr_weight=arr[2],
            consistency_weight=arr[3],
            team_strength_weight=arr[4],
            xg_weight=arr[5],
        )


@dataclass
class AdaptiveState:
    """State of the adaptive learning system."""

    current_weights: WeightConfig
    historical_results: list[dict[str, Any]]
    last_updated: str
    version: int = 1


class AdaptiveWeightManager:
    """
    Manages adaptive weight adjustment based on backtest results.

    Stores historical performance and can suggest weight adjustments
    to improve prediction accuracy.
    """

    DEFAULT_STATE_FILE = "data/adaptive_weights.json"

    def __init__(self, state_file: str | Path | None = None):
        """Initialize the adaptive weight manager."""
        self.state_file = Path(state_file or self.DEFAULT_STATE_FILE)
        self._state: AdaptiveState | None = None

    def _load_state(self) -> AdaptiveState:
        """Load state from file or create default."""
        if self._state is not None:
            return self._state

        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)

                self._state = AdaptiveState(
                    current_weights=WeightConfig(**data.get("current_weights", {})),
                    historical_results=data.get("historical_results", []),
                    last_updated=data.get("last_updated", datetime.now().isoformat()),
                    version=data.get("version", 1),
                )
                return self._state
            except Exception as e:
                logger.warning(f"Failed to load adaptive state: {e}")

        # Create default state
        self._state = AdaptiveState(
            current_weights=WeightConfig(),
            historical_results=[],
            last_updated=datetime.now().isoformat(),
        )
        return self._state

    def _save_state(self) -> None:
        """Save state to file."""
        if self._state is None:
            return

        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "current_weights": asdict(self._state.current_weights),
            "historical_results": self._state.historical_results,
            "last_updated": self._state.last_updated,
            "version": self._state.version,
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def get_current_weights(self) -> WeightConfig:
        """Get the current optimized weights."""
        state = self._load_state()
        return state.current_weights

    def record_backtest_result(
        self,
        weights: WeightConfig,
        mae: float,
        correlation: float,
        top_10_hit_rate: float,
        captain_accuracy: float,
        gameweeks_tested: list[int],
    ) -> None:
        """
        Record a backtest result for learning.

        Args:
            weights: The weights used for this backtest
            mae: Mean absolute error
            correlation: Prediction-actual correlation
            top_10_hit_rate: Top 10 hit rate
            captain_accuracy: Captain pick accuracy
            gameweeks_tested: List of gameweeks tested
        """
        state = self._load_state()

        # Create result record
        result = {
            "weights": asdict(weights),
            "mae": mae,
            "correlation": correlation,
            "top_10_hit_rate": top_10_hit_rate,
            "captain_accuracy": captain_accuracy,
            "gameweeks_tested": gameweeks_tested,
            "tested_at": datetime.now().isoformat(),
        }

        # Add to history (keep last 50 results)
        state.historical_results.append(result)
        if len(state.historical_results) > 50:
            state.historical_results = state.historical_results[-50:]

        # Update the weights if this result is better
        if self._is_better_result(result, state.current_weights):
            logger.info("New weights perform better - updating!")
            state.current_weights = WeightConfig(
                form_weight=weights.form_weight,
                ict_weight=weights.ict_weight,
                fdr_weight=weights.fdr_weight,
                consistency_weight=weights.consistency_weight,
                team_strength_weight=weights.team_strength_weight,
                mae=mae,
                correlation=correlation,
                top_10_hit_rate=top_10_hit_rate,
                captain_accuracy=captain_accuracy,
                tested_at=datetime.now().isoformat(),
                gameweeks_tested=gameweeks_tested,
            )

        state.last_updated = datetime.now().isoformat()
        self._save_state()

    def _is_better_result(self, result: dict, current: WeightConfig) -> bool:
        """Check if a new result is better than current weights."""
        if current.correlation is None:
            return True  # No current benchmark

        # Scoring: higher correlation is better, lower MAE is better
        # Weight correlation more heavily as it indicates predictive power
        current_score = (
            (current.correlation or 0) * 0.4
            + (1 - (current.mae or 3) / 5) * 0.3  # Normalize MAE
            + (current.top_10_hit_rate or 0) * 0.2
            + (current.captain_accuracy or 0) * 0.1
        )

        new_score = (
            result["correlation"] * 0.4
            + (1 - result["mae"] / 5) * 0.3
            + result["top_10_hit_rate"] * 0.2
            + result["captain_accuracy"] * 0.1
        )

        return new_score > current_score

    def suggest_weight_adjustments(self) -> dict[str, tuple[float, str]]:
        """
        Analyze historical results and suggest weight adjustments.

        Returns:
            Dict of weight_name -> (suggested_value, reason)
        """
        state = self._load_state()

        if len(state.historical_results) < 3:
            return {}  # Not enough data

        suggestions = {}

        # Analyze trends in historical results
        recent = state.historical_results[-10:]

        # Find which configurations had best captain accuracy
        best_captain = max(recent, key=lambda r: r.get("captain_accuracy", 0))
        if best_captain["captain_accuracy"] > (state.current_weights.captain_accuracy or 0):
            best_weights = best_captain["weights"]
            if best_weights.get("form_weight", 0.35) != state.current_weights.form_weight:
                suggestions["form_weight"] = (
                    best_weights["form_weight"],
                    f"Best captain accuracy ({best_captain['captain_accuracy']*100:.0f}%) used this weight",
                )

        # Find which had best correlation
        best_corr = max(recent, key=lambda r: r.get("correlation", 0))
        if best_corr["correlation"] > (state.current_weights.correlation or 0):
            best_weights = best_corr["weights"]
            if best_weights.get("fdr_weight", 0.25) != state.current_weights.fdr_weight:
                suggestions["fdr_weight"] = (
                    best_weights["fdr_weight"],
                    f"Best correlation ({best_corr['correlation']:.3f}) used this weight",
                )

        return suggestions

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of current performance and history."""
        state = self._load_state()

        return {
            "current_weights": asdict(state.current_weights),
            "last_updated": state.last_updated,
            "num_historical_results": len(state.historical_results),
            "best_correlation": max(
                (r.get("correlation", 0) for r in state.historical_results),
                default=None,
            ),
            "best_captain_accuracy": max(
                (r.get("captain_accuracy", 0) for r in state.historical_results),
                default=None,
            ),
        }

    def run_weight_optimization(
        self,
        backtest_func=None,
        gameweeks: list[int] | None = None,
        iterations: int = 10,
    ) -> WeightConfig:
        """
        Run an optimization loop to find better weights.

        Args:
            backtest_func: Function that takes weights dict and returns backtest result
            gameweeks: List of gameweeks to test (used if backtest_func not provided)
            iterations: Number of random variations to try

        Returns:
            Best WeightConfig found
        """
        import random

        # Create internal backtest function if not provided
        if backtest_func is None:
            if gameweeks is None:
                raise ValueError("Must provide either backtest_func or gameweeks")

            from .backtest import Backtester

            def internal_backtest(weights: WeightConfig):
                """Run backtest with specific weights."""
                backtester = Backtester()
                try:
                    custom_weights = {
                        "form_weight": weights.form_weight,
                        "ict_weight": weights.ict_weight,
                        "fdr_weight": weights.fdr_weight,
                        "consistency_weight": weights.consistency_weight,
                        "team_strength_weight": weights.team_strength_weight,
                        "xg_weight": weights.xg_weight,
                    }
                    return backtester.run_backtest(gameweeks, custom_weights=custom_weights)
                finally:
                    backtester.close()

            backtest_func = internal_backtest

        state = self._load_state()
        best = state.current_weights
        best_score = self._calculate_score(best)

        logger.info(f"Starting weight optimization with {iterations} iterations")

        for i in range(iterations):
            # Generate random variation of current weights
            new_weights = WeightConfig(
                form_weight=max(0.05, min(0.5, best.form_weight + random.uniform(-0.1, 0.1))),
                ict_weight=max(0.05, min(0.4, best.ict_weight + random.uniform(-0.08, 0.08))),
                fdr_weight=max(0.05, min(0.4, best.fdr_weight + random.uniform(-0.1, 0.1))),
                consistency_weight=max(0.05, min(0.3, best.consistency_weight + random.uniform(-0.05, 0.05))),
                team_strength_weight=max(0.05, min(0.3, best.team_strength_weight + random.uniform(-0.05, 0.05))),
                xg_weight=max(0.1, min(0.6, best.xg_weight + random.uniform(-0.1, 0.1))),  # xG weight
            )

            # Normalize to sum to ~1.0
            total = (
                new_weights.form_weight
                + new_weights.ict_weight
                + new_weights.fdr_weight
                + new_weights.consistency_weight
                + new_weights.team_strength_weight
                + new_weights.xg_weight
            )
            if total > 0:
                new_weights.form_weight /= total
                new_weights.ict_weight /= total
                new_weights.fdr_weight /= total
                new_weights.consistency_weight /= total
                new_weights.team_strength_weight /= total
                new_weights.xg_weight /= total

            try:
                # Run backtest with new weights
                result = backtest_func(new_weights)

                # Record and check if better
                self.record_backtest_result(
                    new_weights,
                    result.mean_absolute_error,
                    result.correlation,
                    result.top_10_hit_rate,
                    result.captain_accuracy,
                    result.gameweeks_tested,
                )

                new_score = self._calculate_score_from_result(result)
                if new_score > best_score:
                    logger.info(f"Iteration {i+1}: Found better weights (score: {new_score:.3f} vs {best_score:.3f})")
                    best = new_weights
                    best_score = new_score

            except Exception as e:
                logger.warning(f"Iteration {i+1} failed: {e}")

        return best

    def _calculate_score(self, weights: WeightConfig) -> float:
        """Calculate overall performance score for weights."""
        if weights.correlation is None:
            return 0.0

        return (
            (weights.correlation or 0) * 0.4
            + (1 - (weights.mae or 3) / 5) * 0.3
            + (weights.top_10_hit_rate or 0) * 0.2
            + (weights.captain_accuracy or 0) * 0.1
        )

    def _calculate_score_from_result(self, result) -> float:
        """Calculate score from a backtest result."""
        return (
            result.correlation * 0.4
            + (1 - result.mean_absolute_error / 5) * 0.3
            + result.top_10_hit_rate * 0.2
            + result.captain_accuracy * 0.1
        )

    def run_gradient_optimization(
        self,
        backtest_func,
        max_iterations: int = 50,
    ) -> WeightConfig:
        """
        Run gradient-based optimization using scipy Nelder-Mead.

        This is more efficient than random search - it follows the
        gradient of improvement to find better weights faster.

        Args:
            backtest_func: Function that takes weights and returns backtest result
            max_iterations: Maximum optimization iterations

        Returns:
            Best WeightConfig found
        """
        try:
            from scipy.optimize import minimize
        except ImportError:
            logger.warning("scipy not installed, falling back to random search")
            return self.run_weight_optimization(backtest_func, iterations=max_iterations)

        state = self._load_state()
        initial_weights = state.current_weights.to_array()
        best_result = None
        best_score = 0.0

        def objective(weights_array):
            """Objective function to minimize (negative score)."""
            nonlocal best_result, best_score

            # Normalize weights to sum to 1
            weights_array = [max(0.05, w) for w in weights_array]
            total = sum(weights_array)
            weights_array = [w / total for w in weights_array]

            weights = WeightConfig.from_array(weights_array)

            try:
                result = backtest_func(weights)

                # Record result
                self.record_backtest_result(
                    weights,
                    result.mean_absolute_error,
                    result.correlation,
                    result.top_10_hit_rate,
                    result.captain_accuracy,
                    result.gameweeks_tested,
                )

                score = self._calculate_score_from_result(result)

                if score > best_score:
                    best_score = score
                    best_result = weights
                    logger.info(f"Gradient opt: New best score {score:.3f}")

                # Return negative because we minimize
                return -score

            except Exception as e:
                logger.warning(f"Gradient opt iteration failed: {e}")
                return 0.0  # Penalize failed iterations

        # Run Nelder-Mead optimization
        logger.info(f"Starting gradient optimization with max {max_iterations} iterations")

        # Bounds for weights: each between 0.05 and 0.6
        bounds = [(0.05, 0.6)] * 6

        result = minimize(
            objective,
            initial_weights,
            method='Nelder-Mead',
            options={'maxiter': max_iterations, 'disp': False},
        )

        if best_result is not None:
            # Update state with best weights
            state.current_weights = best_result
            state.last_updated = datetime.now().isoformat()
            self._save_state()
            return best_result

        return state.current_weights

    def should_auto_optimize(self) -> bool:
        """
        Check if auto-optimization should run.

        Triggers when:
        - New gameweek data is available
        - No optimization in last 7 days
        - Performance has degraded
        """
        state = self._load_state()

        # Check last update time
        if state.last_updated:
            try:
                last_update = datetime.fromisoformat(state.last_updated)
                days_since = (datetime.now() - last_update).days
                if days_since >= 7:
                    logger.info(f"Auto-optimize triggered: {days_since} days since last update")
                    return True
            except:
                pass

        # Check if recent results show degraded performance
        if len(state.historical_results) >= 2:
            recent = state.historical_results[-1]
            previous = state.historical_results[-2]
            if recent.get("correlation", 0) < previous.get("correlation", 0) - 0.05:
                logger.info("Auto-optimize triggered: performance degradation detected")
                return True

        return False


# Convenience functions
_manager: AdaptiveWeightManager | None = None


def get_adaptive_manager() -> AdaptiveWeightManager:
    """Get the singleton adaptive weight manager."""
    global _manager
    if _manager is None:
        _manager = AdaptiveWeightManager()
    return _manager


def get_optimized_weights() -> WeightConfig:
    """Get the current optimized weights."""
    return get_adaptive_manager().get_current_weights()
