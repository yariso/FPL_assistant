"""
Uncertainty and Distribution Forecasting for FPL.

Elite managers don't use single point estimates - they understand:
- P10 (floor): What's the worst likely outcome?
- P50 (median): What's the typical outcome?
- P90 (ceiling): What's the best likely outcome?

This enables better decisions:
- Captaincy: Who has the highest ceiling?
- Hits: What's the probability of breaking even?
- Chips: How much variance am I taking on?
"""

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from ..data.models import Player, Position

logger = logging.getLogger(__name__)


class VarianceLevel(StrEnum):
    """Classification of outcome variance."""

    LOW = "low"        # Consistent player, predictable
    MEDIUM = "medium"  # Some variance but reasonable
    HIGH = "high"      # High-ceiling, high-floor spread


@dataclass
class PointsDistribution:
    """
    Probability distribution of expected points.

    Instead of just xP = 5.0, this gives:
    - mean = 5.0 (expected value)
    - p10 = 2.1 (10th percentile - floor)
    - p50 = 4.5 (50th percentile - median)
    - p90 = 8.5 (90th percentile - ceiling)
    """

    player_id: int
    player_name: str

    # Distribution parameters
    mean: float           # Expected points (current projection)
    std_dev: float        # Standard deviation
    p10: float            # 10th percentile (floor)
    p50: float            # 50th percentile (median)
    p90: float            # 90th percentile (ceiling)

    # Variance classification
    variance_level: VarianceLevel

    # Context
    confidence: float     # Model confidence (0-1)
    factors: list[str]    # Factors affecting variance

    def probability_above(self, threshold: float) -> float:
        """
        Calculate P(points > threshold).

        Useful for hit decisions:
        - If threshold = 4 (hit cost), what's probability of breaking even?

        Uses normal distribution approximation.
        """
        if self.std_dev <= 0:
            return 1.0 if self.mean > threshold else 0.0

        # Z-score
        z = (threshold - self.mean) / self.std_dev
        # Use error function for normal CDF approximation
        return 0.5 * (1 + math.erf(-z / math.sqrt(2)))

    def probability_between(self, low: float, high: float) -> float:
        """Calculate P(low < points < high)."""
        return self.probability_above(low) - self.probability_above(high)

    @property
    def upside(self) -> float:
        """Potential upside: P90 - mean."""
        return self.p90 - self.mean

    @property
    def downside(self) -> float:
        """Potential downside: mean - P10."""
        return self.mean - self.p10

    @property
    def risk_reward_ratio(self) -> float:
        """Upside/Downside ratio. >1 = more upside potential."""
        if self.downside <= 0:
            return float('inf')
        return self.upside / self.downside


# Position-specific variance (based on historical FPL data)
# Forwards have highest variance, defenders lowest
POSITION_STD_DEV_BASE = {
    Position.GK: 2.0,    # Low variance (clean sheet binary)
    Position.DEF: 2.3,   # Moderate variance
    Position.MID: 3.0,   # High variance (goal involvement)
    Position.FWD: 3.5,   # Highest variance
}

# How variance scales with expected points
# Higher xP = higher absolute variance
VARIANCE_SCALE_FACTOR = 0.35


class UncertaintyModel:
    """
    Models uncertainty in FPL point predictions.

    Key insight: Elite managers pick high-ceiling players for captaincy
    but low-variance players for bench fodder.
    """

    def __init__(self, players: list[Player]):
        """
        Initialize uncertainty model.

        Args:
            players: List of all players for context
        """
        self.players = {p.id: p for p in players}

        # Compute league averages for comparison
        self._position_avg_points = self._compute_position_averages()

    def _compute_position_averages(self) -> dict[Position, float]:
        """Compute average points per game by position."""
        position_totals: dict[Position, list[float]] = {
            Position.GK: [],
            Position.DEF: [],
            Position.MID: [],
            Position.FWD: [],
        }

        for player in self.players.values():
            if player.points_per_game > 0:
                position_totals[player.position].append(player.points_per_game)

        return {
            pos: sum(pts) / len(pts) if pts else 3.0
            for pos, pts in position_totals.items()
        }

    def estimate_distribution(
        self,
        player: Player,
        base_xp: float,
        fixture_difficulty: int = 3,
    ) -> PointsDistribution:
        """
        Estimate the full points distribution for a player.

        Args:
            player: Player to estimate
            base_xp: Base expected points (from ProjectionEngine)
            fixture_difficulty: FDR (1=easy, 5=hard)

        Returns:
            PointsDistribution with P10/P50/P90
        """
        factors = []

        # Base standard deviation from position
        base_std = POSITION_STD_DEV_BASE.get(player.position, 2.5)

        # Scale variance with expected points
        # Higher xP players have higher absolute variance
        std_dev = base_std + (base_xp * VARIANCE_SCALE_FACTOR)

        # Fixture difficulty affects variance
        # Easy fixtures = more predictable, hard = more variance
        if fixture_difficulty <= 2:
            std_dev *= 0.85
            factors.append("Easy fixture (lower variance)")
        elif fixture_difficulty >= 4:
            std_dev *= 1.2
            factors.append("Tough fixture (higher variance)")

        # Player form consistency
        if player.form > 0 and player.points_per_game > 0:
            # Higher form relative to average = more consistent recently
            form_ratio = player.form / max(player.points_per_game, 1)
            if form_ratio > 1.2:
                std_dev *= 0.9  # Hot form, more predictable
                factors.append("Hot form (lower variance)")
            elif form_ratio < 0.8:
                std_dev *= 1.15  # Cold form, less predictable
                factors.append("Poor form (higher variance)")

        # Premium players tend to be more consistent
        if player.price >= 10.0:
            std_dev *= 0.9
            factors.append("Premium player (more consistent)")
        elif player.price <= 5.0:
            std_dev *= 1.1
            factors.append("Budget player (less consistent)")

        # Minutes consistency affects variance
        if player.minutes > 0:
            games_estimate = max(1, player.minutes / 90)
            avg_minutes = player.minutes / games_estimate
            if avg_minutes >= 85:
                std_dev *= 0.85
                factors.append("Nailed starter (lower variance)")
            elif avg_minutes < 60:
                std_dev *= 1.3
                factors.append("Rotation risk (higher variance)")

        # Calculate percentiles using normal distribution
        # P10, P50, P90 z-scores: -1.28, 0, 1.28
        p10 = max(0, base_xp - 1.28 * std_dev)
        p50 = max(0, base_xp)  # Median ≈ mean for symmetric distribution
        p90 = base_xp + 1.28 * std_dev

        # Adjust for floor effects (can't score negative)
        # This creates slight positive skew
        if p10 < 1:
            p10 = max(0, p10)
            # Redistribute to maintain mean
            p90 += (1 - p10) * 0.3

        # Classify variance level
        if std_dev < 2.5:
            variance_level = VarianceLevel.LOW
        elif std_dev < 3.5:
            variance_level = VarianceLevel.MEDIUM
        else:
            variance_level = VarianceLevel.HIGH

        # Calculate confidence
        confidence = self._calculate_confidence(player, base_xp)

        return PointsDistribution(
            player_id=player.id,
            player_name=player.web_name,
            mean=round(base_xp, 2),
            std_dev=round(std_dev, 2),
            p10=round(p10, 2),
            p50=round(p50, 2),
            p90=round(p90, 2),
            variance_level=variance_level,
            confidence=round(confidence, 2),
            factors=factors,
        )

    def _calculate_confidence(self, player: Player, base_xp: float) -> float:
        """Calculate confidence in the distribution estimate."""
        confidence = 0.5

        # More minutes = more data = higher confidence
        if player.minutes >= 1500:
            confidence += 0.3
        elif player.minutes >= 900:
            confidence += 0.2
        elif player.minutes >= 450:
            confidence += 0.1

        # Consistent performers are more predictable
        if player.points_per_game > 0 and player.form > 0:
            consistency = 1 - abs(player.form - player.points_per_game) / max(player.points_per_game, 1)
            confidence += consistency * 0.15

        return min(0.95, max(0.3, confidence))

    def get_captain_upside(
        self,
        player: Player,
        base_xp: float,
        fixture_difficulty: int = 3,
    ) -> float:
        """
        Get captain upside score for optimal captaincy decisions.

        Captaincy should prioritize ceiling (P90) not just mean.
        2x points means 2x the upside.

        Returns:
            Upside-weighted captain score
        """
        dist = self.estimate_distribution(player, base_xp, fixture_difficulty)

        # Captain score = mean + upside bonus
        # Weight ceiling more for captaincy decisions
        upside_weight = 0.3
        return dist.mean + (dist.upside * upside_weight)

    def get_hit_probability(
        self,
        player_out_xp: float,
        player_in: Player,
        player_in_xp: float,
        fixture_difficulty: int = 3,
    ) -> float:
        """
        Calculate probability that a transfer hit will pay off.

        A hit costs 4 points, so we need P(gain > 4).

        Args:
            player_out_xp: xP of player being sold
            player_in: Player being bought
            player_in_xp: xP of player being bought
            fixture_difficulty: FDR for incoming player

        Returns:
            Probability that the hit will be worth it (0-1)
        """
        # Get distribution for incoming player
        dist_in = self.estimate_distribution(player_in, player_in_xp, fixture_difficulty)

        # Expected gain from the transfer
        expected_gain = player_in_xp - player_out_xp

        # We need gain > 4 (hit cost)
        threshold = 4 - expected_gain

        # What's P(actual_in > expected_out + 4)?
        # This is P(actual_in > player_out_xp + 4)
        return dist_in.probability_above(player_out_xp + 4)


def get_uncertainty_model(players: list[Player]) -> UncertaintyModel:
    """Get an uncertainty model instance."""
    return UncertaintyModel(players)


def estimate_player_distribution(
    player: Player,
    base_xp: float,
    all_players: list[Player],
    fixture_difficulty: int = 3,
) -> PointsDistribution:
    """Convenience function to estimate distribution for a single player."""
    model = UncertaintyModel(all_players)
    return model.estimate_distribution(player, base_xp, fixture_difficulty)


# =============================================================================
# Monte Carlo Simulation for Scenario Testing
# =============================================================================

import random
from dataclasses import field as dataclass_field


@dataclass
class MonteCarloResult:
    """Result from Monte Carlo simulation."""

    player_id: int
    player_name: str
    simulations: int

    # Statistics across simulations
    mean_points: float
    median_points: float
    std_dev: float
    p10: float
    p50: float
    p90: float

    # Captain win rate (% of simulations this player was top scorer)
    captain_win_rate: float
    top_3_rate: float  # % of times in top 3

    # Sample outcomes for visualization
    sample_outcomes: list[float]


@dataclass
class ScenarioTestResult:
    """Result from testing multiple team scenarios."""

    scenario_name: str
    team_players: list[int]  # Player IDs
    captain_id: int

    # Aggregated results
    mean_total_points: float
    median_total_points: float
    p10_total: float
    p90_total: float

    # Individual player results
    player_results: list[MonteCarloResult]

    # Win rate vs other scenarios
    win_rate_vs_others: float


class MonteCarloSimulator:
    """
    Monte Carlo simulation for FPL outcome analysis.

    Elite managers use simulation to:
    1. Compare captain options probabilistically
    2. Test team selection under uncertainty
    3. Evaluate hit decisions with realistic variance
    4. Validate optimal team is truly optimal

    Based on FPL Optimized and FPL Review methodologies.
    """

    def __init__(
        self,
        players: list[Player],
        projections: dict[int, float],  # player_id -> xP
        fixture_difficulties: dict[int, int] | None = None,  # player_id -> FDR
    ):
        """
        Initialize Monte Carlo simulator.

        Args:
            players: List of all players
            projections: Dictionary of player_id -> expected points
            fixture_difficulties: Optional FDR per player
        """
        self.players = {p.id: p for p in players}
        self.projections = projections
        self.fixture_difficulties = fixture_difficulties or {}

        # Create uncertainty model for distribution estimation
        self.uncertainty_model = UncertaintyModel(players)

    def simulate_player(
        self,
        player_id: int,
        n_simulations: int = 10000,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for a single player.

        Args:
            player_id: Player to simulate
            n_simulations: Number of simulations (default 10000)

        Returns:
            MonteCarloResult with statistics
        """
        player = self.players.get(player_id)
        if not player:
            raise ValueError(f"Player {player_id} not found")

        base_xp = self.projections.get(player_id, 0)
        fdr = self.fixture_difficulties.get(player_id, 3)

        # Get distribution
        dist = self.uncertainty_model.estimate_distribution(player, base_xp, fdr)

        # Run simulations using normal distribution sampling
        # FPL points CAN be negative (cards, own goals, goals conceded)
        # Position-dependent floors based on realistic worst-case scenarios
        POSITION_FLOOR = {
            Position.GK: -6,   # Red card (-3) + own goal (-2) + conceded (-1)
            Position.DEF: -6,  # Same as GK
            Position.MID: -4,  # Red card (-3) + own goal (-2) + missed pen (-2)
            Position.FWD: -4,  # Same as MID
        }
        min_floor = POSITION_FLOOR.get(player.position, -4)

        outcomes = []
        for _ in range(n_simulations):
            # Sample from normal distribution
            sample = random.gauss(dist.mean, dist.std_dev)

            # Floor at position-dependent minimum (allows negative scores)
            sample = max(min_floor, sample)

            # Add occasional haul probability for attackers
            if player.position in [Position.MID, Position.FWD]:
                # 5% chance of a haul (double+ expected points)
                if random.random() < 0.05:
                    sample = max(sample, dist.mean * 2 + random.gauss(2, 1))

            outcomes.append(sample)

        # Calculate statistics
        outcomes.sort()
        mean_pts = sum(outcomes) / len(outcomes)
        median_pts = outcomes[len(outcomes) // 2]
        std = (sum((x - mean_pts) ** 2 for x in outcomes) / len(outcomes)) ** 0.5

        p10_idx = int(len(outcomes) * 0.1)
        p50_idx = int(len(outcomes) * 0.5)
        p90_idx = int(len(outcomes) * 0.9)

        return MonteCarloResult(
            player_id=player_id,
            player_name=player.web_name,
            simulations=n_simulations,
            mean_points=round(mean_pts, 2),
            median_points=round(median_pts, 2),
            std_dev=round(std, 2),
            p10=round(outcomes[p10_idx], 2),
            p50=round(outcomes[p50_idx], 2),
            p90=round(outcomes[p90_idx], 2),
            captain_win_rate=0.0,  # Calculated in compare_captains
            top_3_rate=0.0,
            sample_outcomes=random.sample(outcomes, min(100, len(outcomes))),
        )

    def compare_captains(
        self,
        captain_candidates: list[int],  # Player IDs
        n_simulations: int = 10000,
    ) -> list[MonteCarloResult]:
        """
        Compare captain options using Monte Carlo simulation.

        This is THE key method for captain decisions:
        - Simulates all candidates simultaneously
        - Tracks who "wins" (scores most) in each simulation
        - Returns win rate for each option

        Args:
            captain_candidates: List of player IDs to compare
            n_simulations: Number of simulations

        Returns:
            List of MonteCarloResult sorted by win rate (best first)
        """
        if not captain_candidates:
            return []

        # Get distributions for all candidates
        distributions = {}
        for pid in captain_candidates:
            player = self.players.get(pid)
            if not player:
                continue
            base_xp = self.projections.get(pid, 0)
            fdr = self.fixture_difficulties.get(pid, 3)
            distributions[pid] = self.uncertainty_model.estimate_distribution(
                player, base_xp, fdr
            )

        # Track outcomes
        wins = {pid: 0 for pid in captain_candidates}
        top_3_counts = {pid: 0 for pid in captain_candidates}
        all_outcomes = {pid: [] for pid in captain_candidates}

        # Position-dependent floors (allows negative scores)
        POSITION_FLOOR = {
            Position.GK: -6,
            Position.DEF: -6,
            Position.MID: -4,
            Position.FWD: -4,
        }

        # Run simulations
        for _ in range(n_simulations):
            sim_results = {}

            for pid, dist in distributions.items():
                player = self.players[pid]

                # Sample from distribution
                sample = random.gauss(dist.mean, dist.std_dev)
                min_floor = POSITION_FLOOR.get(player.position, -4)
                sample = max(min_floor, sample)

                # Haul probability for attackers
                if player.position in [Position.MID, Position.FWD]:
                    if random.random() < 0.05:
                        sample = max(sample, dist.mean * 2 + random.gauss(2, 1))

                # Captain = double points
                sim_results[pid] = sample * 2
                all_outcomes[pid].append(sample)

            # Determine winner and top 3
            sorted_results = sorted(sim_results.items(), key=lambda x: -x[1])
            if sorted_results:
                wins[sorted_results[0][0]] += 1
                for pid, _ in sorted_results[:3]:
                    top_3_counts[pid] += 1

        # Build results
        results = []
        for pid in captain_candidates:
            player = self.players.get(pid)
            if not player:
                continue

            outcomes = all_outcomes[pid]
            outcomes.sort()
            mean_pts = sum(outcomes) / len(outcomes) if outcomes else 0
            median_pts = outcomes[len(outcomes) // 2] if outcomes else 0
            std = (sum((x - mean_pts) ** 2 for x in outcomes) / len(outcomes)) ** 0.5 if outcomes else 0

            p10_idx = int(len(outcomes) * 0.1)
            p50_idx = int(len(outcomes) * 0.5)
            p90_idx = int(len(outcomes) * 0.9)

            results.append(MonteCarloResult(
                player_id=pid,
                player_name=player.web_name,
                simulations=n_simulations,
                mean_points=round(mean_pts, 2),
                median_points=round(median_pts, 2),
                std_dev=round(std, 2),
                p10=round(outcomes[p10_idx], 2) if outcomes else 0,
                p50=round(outcomes[p50_idx], 2) if outcomes else 0,
                p90=round(outcomes[p90_idx], 2) if outcomes else 0,
                captain_win_rate=round(wins[pid] / n_simulations * 100, 1),
                top_3_rate=round(top_3_counts[pid] / n_simulations * 100, 1),
                sample_outcomes=random.sample(outcomes, min(100, len(outcomes))) if outcomes else [],
            ))

        # Sort by win rate
        results.sort(key=lambda x: -x.captain_win_rate)
        return results

    def test_team_scenario(
        self,
        team_player_ids: list[int],
        captain_id: int,
        scenario_name: str = "Current Team",
        n_simulations: int = 10000,
    ) -> ScenarioTestResult:
        """
        Test a team selection scenario with Monte Carlo simulation.

        Args:
            team_player_ids: List of 11 starting player IDs
            captain_id: Captain player ID
            scenario_name: Name for this scenario
            n_simulations: Number of simulations

        Returns:
            ScenarioTestResult with aggregated statistics
        """
        total_points = []
        player_outcomes = {pid: [] for pid in team_player_ids}

        # Get distributions
        distributions = {}
        for pid in team_player_ids:
            player = self.players.get(pid)
            if not player:
                continue
            base_xp = self.projections.get(pid, 0)
            fdr = self.fixture_difficulties.get(pid, 3)
            distributions[pid] = self.uncertainty_model.estimate_distribution(
                player, base_xp, fdr
            )

        # Position-dependent floors (allows negative scores)
        POSITION_FLOOR = {
            Position.GK: -6,
            Position.DEF: -6,
            Position.MID: -4,
            Position.FWD: -4,
        }

        # Run simulations
        for _ in range(n_simulations):
            sim_total = 0

            for pid, dist in distributions.items():
                player = self.players[pid]

                # Sample points
                sample = random.gauss(dist.mean, dist.std_dev)
                min_floor = POSITION_FLOOR.get(player.position, -4)
                sample = max(min_floor, sample)

                if player.position in [Position.MID, Position.FWD]:
                    if random.random() < 0.05:
                        sample = max(sample, dist.mean * 2 + random.gauss(2, 1))

                # Captain = double
                if pid == captain_id:
                    sample *= 2

                sim_total += sample
                player_outcomes[pid].append(sample / 2 if pid == captain_id else sample)

            total_points.append(sim_total)

        # Calculate statistics
        total_points.sort()
        mean_total = sum(total_points) / len(total_points)
        median_total = total_points[len(total_points) // 2]

        p10_idx = int(len(total_points) * 0.1)
        p90_idx = int(len(total_points) * 0.9)

        # Build player results
        player_results = []
        for pid in team_player_ids:
            player = self.players.get(pid)
            if not player:
                continue

            outcomes = player_outcomes[pid]
            outcomes.sort()
            mean_pts = sum(outcomes) / len(outcomes) if outcomes else 0
            std = (sum((x - mean_pts) ** 2 for x in outcomes) / len(outcomes)) ** 0.5 if outcomes else 0

            p10 = outcomes[int(len(outcomes) * 0.1)] if outcomes else 0
            p50 = outcomes[int(len(outcomes) * 0.5)] if outcomes else 0
            p90 = outcomes[int(len(outcomes) * 0.9)] if outcomes else 0

            player_results.append(MonteCarloResult(
                player_id=pid,
                player_name=player.web_name,
                simulations=n_simulations,
                mean_points=round(mean_pts, 2),
                median_points=round(p50, 2),
                std_dev=round(std, 2),
                p10=round(p10, 2),
                p50=round(p50, 2),
                p90=round(p90, 2),
                captain_win_rate=100.0 if pid == captain_id else 0.0,
                top_3_rate=0.0,
                sample_outcomes=[],
            ))

        return ScenarioTestResult(
            scenario_name=scenario_name,
            team_players=team_player_ids,
            captain_id=captain_id,
            mean_total_points=round(mean_total, 2),
            median_total_points=round(median_total, 2),
            p10_total=round(total_points[p10_idx], 2),
            p90_total=round(total_points[p90_idx], 2),
            player_results=player_results,
            win_rate_vs_others=0.0,  # Set when comparing multiple scenarios
        )

    def compare_team_scenarios(
        self,
        scenarios: list[tuple[list[int], int, str]],  # [(player_ids, captain_id, name), ...]
        n_simulations: int = 10000,
    ) -> list[ScenarioTestResult]:
        """
        Compare multiple team scenarios head-to-head.

        Args:
            scenarios: List of (player_ids, captain_id, name) tuples
            n_simulations: Number of simulations

        Returns:
            List of ScenarioTestResult sorted by win rate
        """
        if not scenarios:
            return []

        # Get distributions for all players
        all_player_ids = set()
        for player_ids, _, _ in scenarios:
            all_player_ids.update(player_ids)

        distributions = {}
        for pid in all_player_ids:
            player = self.players.get(pid)
            if not player:
                continue
            base_xp = self.projections.get(pid, 0)
            fdr = self.fixture_difficulties.get(pid, 3)
            distributions[pid] = self.uncertainty_model.estimate_distribution(
                player, base_xp, fdr
            )

        # Track wins and totals per scenario
        scenario_wins = {i: 0 for i in range(len(scenarios))}
        scenario_totals = {i: [] for i in range(len(scenarios))}

        # Position-dependent floors (allows negative scores)
        POSITION_FLOOR = {
            Position.GK: -6,
            Position.DEF: -6,
            Position.MID: -4,
            Position.FWD: -4,
        }

        # Run simulations
        for _ in range(n_simulations):
            # Sample points for all players once
            sampled_points = {}
            for pid, dist in distributions.items():
                player = self.players[pid]
                sample = random.gauss(dist.mean, dist.std_dev)
                min_floor = POSITION_FLOOR.get(player.position, -4)
                sample = max(min_floor, sample)
                if player.position in [Position.MID, Position.FWD]:
                    if random.random() < 0.05:
                        sample = max(sample, dist.mean * 2 + random.gauss(2, 1))
                sampled_points[pid] = sample

            # Calculate total for each scenario
            scenario_scores = []
            for i, (player_ids, captain_id, _) in enumerate(scenarios):
                total = 0
                for pid in player_ids:
                    pts = sampled_points.get(pid, 0)
                    if pid == captain_id:
                        pts *= 2
                    total += pts
                scenario_scores.append((i, total))
                scenario_totals[i].append(total)

            # Determine winner
            winner = max(scenario_scores, key=lambda x: x[1])[0]
            scenario_wins[winner] += 1

        # Build results
        results = []
        for i, (player_ids, captain_id, name) in enumerate(scenarios):
            totals = scenario_totals[i]
            totals.sort()

            results.append(ScenarioTestResult(
                scenario_name=name,
                team_players=player_ids,
                captain_id=captain_id,
                mean_total_points=round(sum(totals) / len(totals), 2),
                median_total_points=round(totals[len(totals) // 2], 2),
                p10_total=round(totals[int(len(totals) * 0.1)], 2),
                p90_total=round(totals[int(len(totals) * 0.9)], 2),
                player_results=[],  # Not populated for comparison
                win_rate_vs_others=round(scenario_wins[i] / n_simulations * 100, 1),
            ))

        # Sort by win rate
        results.sort(key=lambda x: -x.win_rate_vs_others)
        return results


def run_monte_carlo_captain_comparison(
    players: list[Player],
    projections: dict[int, float],
    captain_candidates: list[int],
    n_simulations: int = 10000,
) -> list[MonteCarloResult]:
    """
    Convenience function to compare captain options.

    Args:
        players: All players
        projections: Player ID -> xP mapping
        captain_candidates: Player IDs to compare
        n_simulations: Number of simulations

    Returns:
        List of results sorted by win rate
    """
    simulator = MonteCarloSimulator(players, projections)
    return simulator.compare_captains(captain_candidates, n_simulations)
