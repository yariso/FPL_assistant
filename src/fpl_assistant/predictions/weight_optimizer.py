"""
Weight Optimization for FPL Predictions.

Uses backtesting and optimization algorithms to find the best
weight configuration for maximum prediction accuracy.
"""

import json
import logging
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .enhanced_weights import EnhancedWeightConfig, EnhancedSignalCalculator, convert_signals_to_projection

logger = logging.getLogger(__name__)


class WeightOptimizer:
    """
    Finds optimal weights through backtesting.

    Uses a combination of:
    - Grid search for initial exploration
    - Random search for fine-tuning
    - Hill climbing for local optimization
    """

    CACHE_FILE = "data/optimized_weights.json"

    def __init__(self):
        self.best_weights: EnhancedWeightConfig | None = None
        self.best_score: float = 0.0
        self.optimization_history: list[dict] = []

    def optimize(
        self,
        backtest_fn: Callable[[EnhancedWeightConfig], dict[str, float]],
        iterations: int = 50,
        method: str = "hybrid",
    ) -> EnhancedWeightConfig:
        """
        Run optimization to find best weights.

        Args:
            backtest_fn: Function that takes weights and returns metrics dict with:
                - mae: Mean Absolute Error (lower is better)
                - correlation: Correlation with actual points (higher is better)
                - captain_accuracy: % of correct captain picks (higher is better)
                - top_10_hit_rate: % of top 10 projections in actual top 10
            iterations: Number of optimization iterations
            method: "grid", "random", "hill_climb", or "hybrid"

        Returns:
            Optimized EnhancedWeightConfig
        """
        logger.info(f"Starting weight optimization with {iterations} iterations using {method} method")

        # Start with default weights
        self.best_weights = EnhancedWeightConfig()
        initial_result = backtest_fn(self.best_weights)
        self.best_score = self._calculate_score(initial_result)

        logger.info(f"Initial score: {self.best_score:.4f}")

        if method == "grid":
            self._grid_search(backtest_fn, iterations)
        elif method == "random":
            self._random_search(backtest_fn, iterations)
        elif method == "hill_climb":
            self._hill_climb(backtest_fn, iterations)
        else:  # hybrid
            # Phase 1: Random exploration (40% of iterations)
            self._random_search(backtest_fn, int(iterations * 0.4))
            # Phase 2: Hill climbing from best found (60% of iterations)
            self._hill_climb(backtest_fn, int(iterations * 0.6))

        logger.info(f"Optimization complete. Best score: {self.best_score:.4f}")
        self._save_results()

        return self.best_weights

    def _calculate_score(self, metrics: dict[str, float]) -> float:
        """
        Calculate composite optimization score from metrics.

        Optimizes for:
        - Low MAE (30% weight)
        - High correlation (30% weight)
        - High captain accuracy (25% weight) - critical for FPL success
        - High top 10 hit rate (15% weight)
        """
        mae = metrics.get("mae", 3.0)
        correlation = metrics.get("correlation", 0.0)
        captain_acc = metrics.get("captain_accuracy", 0.0)
        top_10_rate = metrics.get("top_10_hit_rate", 0.0)

        # Normalize MAE (0-5 scale, lower is better -> invert)
        mae_score = max(0, 1 - mae / 5)

        score = (
            mae_score * 0.30 +
            correlation * 0.30 +
            captain_acc * 0.25 +
            top_10_rate * 0.15
        )

        return score

    def _random_search(
        self,
        backtest_fn: Callable[[EnhancedWeightConfig], dict],
        iterations: int,
    ) -> None:
        """Random search for weight configurations."""
        for i in range(iterations):
            # Generate random weights
            weights = self._generate_random_weights()

            try:
                result = backtest_fn(weights)
                score = self._calculate_score(result)

                self.optimization_history.append({
                    "iteration": i,
                    "method": "random",
                    "weights": asdict(weights),
                    "metrics": result,
                    "score": score,
                })

                if score > self.best_score:
                    self.best_score = score
                    self.best_weights = weights
                    logger.info(f"New best score: {score:.4f} at iteration {i}")

            except Exception as e:
                logger.warning(f"Backtest failed at iteration {i}: {e}")

    def _hill_climb(
        self,
        backtest_fn: Callable[[EnhancedWeightConfig], dict],
        iterations: int,
    ) -> None:
        """Hill climbing from current best."""
        current = self.best_weights
        current_score = self.best_score

        for i in range(iterations):
            # Generate neighbor by perturbing one weight
            neighbor = self._perturb_weights(current)

            try:
                result = backtest_fn(neighbor)
                score = self._calculate_score(result)

                self.optimization_history.append({
                    "iteration": i,
                    "method": "hill_climb",
                    "weights": asdict(neighbor),
                    "metrics": result,
                    "score": score,
                })

                if score > current_score:
                    current = neighbor
                    current_score = score

                    if score > self.best_score:
                        self.best_score = score
                        self.best_weights = neighbor
                        logger.info(f"New best score: {score:.4f} at iteration {i}")

            except Exception as e:
                logger.warning(f"Backtest failed at iteration {i}: {e}")

    def _grid_search(
        self,
        backtest_fn: Callable[[EnhancedWeightConfig], dict],
        iterations: int,
    ) -> None:
        """Coarse grid search over weight space."""
        # Define grid points for key weights
        grid_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

        # Focus on most impactful weights
        count = 0
        for recent_form in grid_values:
            for rolling_xg in grid_values:
                for fixture_diff in grid_values[:4]:  # Fewer options
                    if count >= iterations:
                        return

                    weights = EnhancedWeightConfig(
                        recent_form_weight=recent_form,
                        rolling_xg_weight=rolling_xg,
                        fixture_difficulty_weight=fixture_diff,
                    ).normalize()

                    try:
                        result = backtest_fn(weights)
                        score = self._calculate_score(result)

                        self.optimization_history.append({
                            "iteration": count,
                            "method": "grid",
                            "weights": asdict(weights),
                            "metrics": result,
                            "score": score,
                        })

                        if score > self.best_score:
                            self.best_score = score
                            self.best_weights = weights
                            logger.info(f"New best score: {score:.4f}")

                        count += 1

                    except Exception as e:
                        logger.warning(f"Backtest failed: {e}")
                        count += 1

    def _generate_random_weights(self) -> EnhancedWeightConfig:
        """Generate random weight configuration."""
        weights = EnhancedWeightConfig(
            recent_form_weight=random.uniform(0.10, 0.35),
            rolling_xg_weight=random.uniform(0.15, 0.40),
            fixture_difficulty_weight=random.uniform(0.08, 0.25),
            season_form_weight=random.uniform(0.02, 0.15),
            ict_index_weight=random.uniform(0.02, 0.12),
            team_momentum_weight=random.uniform(0.03, 0.15),
            home_away_weight=random.uniform(0.02, 0.10),
            opposition_weakness_weight=random.uniform(0.02, 0.12),
            minutes_certainty_weight=random.uniform(0.02, 0.10),
            ownership_trend_weight=random.uniform(0.01, 0.08),
        )
        return weights.normalize()

    def _perturb_weights(self, weights: EnhancedWeightConfig) -> EnhancedWeightConfig:
        """Perturb weights slightly for hill climbing."""
        weight_names = [
            "recent_form_weight",
            "rolling_xg_weight",
            "fixture_difficulty_weight",
            "season_form_weight",
            "ict_index_weight",
            "team_momentum_weight",
            "home_away_weight",
            "opposition_weakness_weight",
            "minutes_certainty_weight",
            "ownership_trend_weight",
        ]

        # Pick random weight to perturb
        name = random.choice(weight_names)
        current_val = getattr(weights, name)

        # Perturb by ±10%
        new_val = current_val + random.uniform(-0.05, 0.05)
        new_val = max(0.01, min(0.50, new_val))

        # Create new config
        new_weights = EnhancedWeightConfig(**{
            n: (new_val if n == name else getattr(weights, n))
            for n in weight_names
        })

        return new_weights.normalize()

    def _save_results(self) -> None:
        """Save optimization results to file."""
        try:
            cache_path = Path(self.CACHE_FILE)
            cache_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "best_weights": asdict(self.best_weights) if self.best_weights else {},
                "best_score": self.best_score,
                "optimization_history": self.optimization_history[-100:],  # Keep last 100
                "optimized_at": datetime.now().isoformat(),
            }

            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved optimization results to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save optimization results: {e}")

    @classmethod
    def load_optimized_weights(cls) -> EnhancedWeightConfig | None:
        """Load previously optimized weights."""
        try:
            cache_path = Path(cls.CACHE_FILE)
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)

                weights_dict = data.get("best_weights", {})
                if weights_dict:
                    return EnhancedWeightConfig(**weights_dict)

        except Exception as e:
            logger.warning(f"Failed to load optimized weights: {e}")

        return None


def create_backtest_function(
    players: list,
    teams: list,
    fixtures: list,
    historical_data: dict[int, list[tuple[int, float]]],  # player_id -> [(gw, points)]
    test_gameweeks: list[int],
):
    """
    Create a backtest function for weight optimization.

    Args:
        players: List of players
        teams: List of teams
        fixtures: List of fixtures
        historical_data: Dict mapping player_id to list of (gameweek, actual_points)
        test_gameweeks: Gameweeks to test on

    Returns:
        Function that takes weights and returns metrics
    """
    def backtest(weights: EnhancedWeightConfig) -> dict[str, float]:
        """Run backtest with specific weights."""
        calculator = EnhancedSignalCalculator(players, teams, fixtures)

        all_predictions = []
        all_actuals = []
        captain_correct = 0
        captain_total = 0
        top_10_hits = 0
        top_10_total = 0

        for gw in test_gameweeks:
            gw_predictions = []
            gw_actuals = []

            for player in players:
                # Get actual points for this GW
                player_history = historical_data.get(player.id, [])
                actual = next((pts for g, pts in player_history if g == gw), None)

                if actual is None:
                    continue

                # Calculate prediction
                signals = calculator.calculate_signals(player, gw)
                predicted = convert_signals_to_projection(signals, weights, horizon=1)

                gw_predictions.append((player.id, predicted))
                gw_actuals.append((player.id, actual))
                all_predictions.append(predicted)
                all_actuals.append(actual)

            # Check captain pick (top predicted vs top actual)
            if gw_predictions and gw_actuals:
                pred_sorted = sorted(gw_predictions, key=lambda x: -x[1])
                actual_sorted = sorted(gw_actuals, key=lambda x: -x[1])

                captain_pick = pred_sorted[0][0]
                best_actual = actual_sorted[0][0]
                captain_correct += 1 if captain_pick == best_actual else 0
                captain_total += 1

                # Check top 10 overlap
                pred_top_10 = {p[0] for p in pred_sorted[:10]}
                actual_top_10 = {p[0] for p in actual_sorted[:10]}
                top_10_hits += len(pred_top_10 & actual_top_10)
                top_10_total += 10

        # Calculate metrics
        if len(all_predictions) < 10:
            return {"mae": 5.0, "correlation": 0.0, "captain_accuracy": 0.0, "top_10_hit_rate": 0.0}

        predictions = np.array(all_predictions)
        actuals = np.array(all_actuals)

        mae = float(np.mean(np.abs(predictions - actuals)))
        correlation = float(np.corrcoef(predictions, actuals)[0, 1]) if len(predictions) > 1 else 0.0
        captain_accuracy = captain_correct / captain_total if captain_total > 0 else 0.0
        top_10_rate = top_10_hits / top_10_total if top_10_total > 0 else 0.0

        return {
            "mae": mae,
            "correlation": max(0, correlation),  # Clamp negative correlations
            "captain_accuracy": captain_accuracy,
            "top_10_hit_rate": top_10_rate,
        }

    return backtest
