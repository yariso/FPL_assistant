"""
Event-Based Monte Carlo Simulation for FPL Points.

Uses Poisson/Bernoulli distributions to simulate individual scoring events
(goals, assists, clean sheets, cards, saves) and compute:
- P(haul) - probability of 10+ points
- P(blank) - probability of 2 or fewer points
- Ceiling percentiles (90th, 95th)
- Captain "win rate" comparisons

This provides more accurate captaincy and hit decisions than point estimates alone.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..data.models import Player, Position

logger = logging.getLogger(__name__)


# =============================================================================
# FPL Scoring Constants (2025/26)
# =============================================================================

GOAL_POINTS = {
    Position.GK: 6,
    Position.DEF: 6,
    Position.MID: 5,
    Position.FWD: 4,
}

CLEAN_SHEET_POINTS = {
    Position.GK: 4,
    Position.DEF: 4,
    Position.MID: 1,
    Position.FWD: 0,
}

ASSIST_POINTS = 3
YELLOW_CARD_POINTS = -1
RED_CARD_POINTS = -3
SAVE_POINTS_PER_3 = 1  # 1 pt per 3 saves
GOALS_CONCEDED_POINTS_PER_2 = -1  # -1 pt per 2 goals conceded (GK/DEF only)


# =============================================================================
# Simulation Data Classes
# =============================================================================

@dataclass
class SimulationParams:
    """Parameters for simulating a single player's gameweek."""

    player_id: int
    position: Position

    # Minutes model
    p_start: float          # P(starts the game)
    p_sub: float            # P(comes on as sub | didn't start)
    e_minutes_if_start: float   # Expected minutes if starting
    e_minutes_if_sub: float     # Expected minutes if subbed on

    # Attacking events (lambda for Poisson)
    lambda_goals: float     # Expected goals
    lambda_assists: float   # Expected assists

    # Defensive events
    p_clean_sheet: float    # P(clean sheet | plays 60+)
    lambda_goals_conceded: float  # Expected goals conceded (for GK/DEF)
    lambda_saves: float     # Expected saves (for GK)

    # Cards (probability for Bernoulli)
    p_yellow: float         # P(yellow card | plays)
    p_red: float            # P(red card | plays)

    # Bonus (simplified model)
    base_bonus: float       # Expected bonus points


@dataclass
class SimulatedEvents:
    """Events from a single simulation run."""

    minutes: int = 0
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False
    goals_conceded: int = 0
    saves: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    bonus: int = 0


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation of a player."""

    player_id: int
    n_simulations: int

    # Point distribution
    expected_points: float      # Mean of simulations
    std_dev: float              # Standard deviation
    median_points: float        # Median outcome

    # Key probabilities
    p_haul: float               # P(points >= 10)
    p_double_digit: float       # P(points >= 10) - alias for clarity
    p_returns: float            # P(points >= 5) - "returned"
    p_blank: float              # P(points <= 2)

    # Ceiling analysis
    percentile_75: float        # 75th percentile
    percentile_90: float        # 90th percentile
    percentile_95: float        # 95th percentile
    max_points: float           # Maximum simulated

    # Captain comparison metrics
    ceiling_score: float        # Weighted ceiling metric for captain picks

    # Optional: raw distribution for advanced analysis
    distribution: list[int] = field(default_factory=list)


# =============================================================================
# Event Simulator
# =============================================================================

class EventSimulator:
    """
    Poisson/Bernoulli event-based FPL points simulator.

    Uses Monte Carlo simulation to generate realistic FPL point distributions
    based on underlying event probabilities (goals, assists, CS, cards, etc.).
    """

    def __init__(self, seed: int | None = None):
        """
        Initialize the simulator.

        Args:
            seed: Random seed for reproducibility (None for random)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def simulate_player(
        self,
        params: SimulationParams,
        n_sims: int = 10000,
        store_distribution: bool = False,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a player.

        Args:
            params: Simulation parameters for the player
            n_sims: Number of simulations to run
            store_distribution: Whether to store raw point distribution

        Returns:
            SimulationResult with probability metrics
        """
        points = np.zeros(n_sims, dtype=np.int32)

        for i in range(n_sims):
            events = self._simulate_single(params)
            points[i] = self._calculate_fpl_points(events, params.position)

        # Calculate statistics
        expected = float(np.mean(points))
        std_dev = float(np.std(points))
        median = float(np.median(points))

        # Probability metrics
        p_haul = float(np.mean(points >= 10))
        p_returns = float(np.mean(points >= 5))
        p_blank = float(np.mean(points <= 2))

        # Percentiles
        p75 = float(np.percentile(points, 75))
        p90 = float(np.percentile(points, 90))
        p95 = float(np.percentile(points, 95))
        max_pts = float(np.max(points))

        # Captain ceiling score: weighted combination of expected + ceiling
        # This captures both consistency and upside for captain decisions
        ceiling_score = expected * 0.6 + p90 * 0.3 + p_haul * 10 * 0.1

        return SimulationResult(
            player_id=params.player_id,
            n_simulations=n_sims,
            expected_points=expected,
            std_dev=std_dev,
            median_points=median,
            p_haul=p_haul,
            p_double_digit=p_haul,
            p_returns=p_returns,
            p_blank=p_blank,
            percentile_75=p75,
            percentile_90=p90,
            percentile_95=p95,
            max_points=max_pts,
            ceiling_score=ceiling_score,
            distribution=points.tolist() if store_distribution else [],
        )

    def _simulate_single(self, params: SimulationParams) -> SimulatedEvents:
        """
        Simulate a single gameweek for a player.

        Returns simulated events (goals, assists, minutes, etc.)
        """
        events = SimulatedEvents()

        # 1. MINUTES - Custom model based on start/sub probabilities
        events.minutes = self._simulate_minutes(params)

        if events.minutes == 0:
            # Didn't play - no points possible except maybe auto-sub
            return events

        # Scale factor for partial minutes (for events that scale with time)
        mins_factor = events.minutes / 90.0

        # 2. GOALS - Poisson distribution
        # Scale lambda by minutes factor (more time = more goal opportunities)
        adjusted_lambda_goals = params.lambda_goals * mins_factor
        events.goals = np.random.poisson(adjusted_lambda_goals)

        # 3. ASSISTS - Poisson distribution
        adjusted_lambda_assists = params.lambda_assists * mins_factor
        events.assists = np.random.poisson(adjusted_lambda_assists)

        # 4. CLEAN SHEET - Bernoulli (only if 60+ minutes)
        if events.minutes >= 60 and params.position in [Position.GK, Position.DEF, Position.MID]:
            events.clean_sheet = np.random.random() < params.p_clean_sheet

        # 5. GOALS CONCEDED - Poisson (GK/DEF only, 60+ mins)
        if events.minutes >= 60 and params.position in [Position.GK, Position.DEF]:
            events.goals_conceded = np.random.poisson(params.lambda_goals_conceded)

        # 6. SAVES - Poisson (GK only)
        if params.position == Position.GK:
            adjusted_lambda_saves = params.lambda_saves * mins_factor
            events.saves = np.random.poisson(adjusted_lambda_saves)

        # 7. CARDS - Bernoulli (per-match events, not scaled by minutes)
        # A player either gets carded or doesn't in the match
        events.yellow_cards = 1 if np.random.random() < params.p_yellow else 0
        events.red_cards = 1 if np.random.random() < params.p_red else 0

        # If red card, no yellow (can't have both, red supersedes)
        if events.red_cards > 0:
            events.yellow_cards = 0

        # 8. BONUS - Simplified model based on attacking returns
        events.bonus = self._simulate_bonus(events, params)

        return events

    def _simulate_minutes(self, params: SimulationParams) -> int:
        """
        Simulate minutes played using start/sub probabilities.

        Returns expected minutes (0, ~15-30 for sub, ~75-90 for starter)
        """
        # Does player start?
        if np.random.random() < params.p_start:
            # Started - expected to play most of the game
            # Add some variance around expected minutes
            base = params.e_minutes_if_start
            variance = min(15, base * 0.1)
            minutes = int(np.random.normal(base, variance))
            return max(1, min(90, minutes))

        # Didn't start - does player come on as sub?
        if np.random.random() < params.p_sub:
            # Came on as sub
            base = params.e_minutes_if_sub
            variance = min(10, base * 0.2)
            minutes = int(np.random.normal(base, variance))
            return max(1, min(45, minutes))  # Subs max ~45 mins

        # Didn't play at all
        return 0

    def _simulate_bonus(self, events: SimulatedEvents, params: SimulationParams) -> int:
        """
        Simplified bonus point model.

        Bonus is correlated with goals, assists, clean sheets, and saves.
        This is a rough approximation - actual BPS is complex.
        """
        if events.minutes == 0:
            return 0

        # Base probability of getting any bonus
        bonus_probability = params.base_bonus / 3.0  # Normalize to ~probability

        # Boost for attacking returns
        if events.goals >= 2:
            bonus_probability += 0.7  # Very likely 3 bonus
        elif events.goals == 1:
            bonus_probability += 0.35  # Good chance of bonus

        if events.assists >= 2:
            bonus_probability += 0.5
        elif events.assists == 1:
            bonus_probability += 0.2

        # Clean sheet bonus (GK/DEF)
        if events.clean_sheet and params.position in [Position.GK, Position.DEF]:
            bonus_probability += 0.25

        # Saves bonus (GK)
        if events.saves >= 5:
            bonus_probability += 0.3
        elif events.saves >= 3:
            bonus_probability += 0.15

        # Clamp probability
        bonus_probability = min(0.95, bonus_probability)

        # Determine bonus points (0, 1, 2, or 3)
        if np.random.random() > bonus_probability:
            return 0

        # Got some bonus - how much?
        if events.goals >= 2 or (events.goals >= 1 and events.assists >= 1):
            # High chance of 3 bonus
            return np.random.choice([2, 3, 3], p=[0.2, 0.3, 0.5])
        elif events.goals == 1 or events.assists >= 2:
            # Decent chance of 2-3 bonus
            return np.random.choice([1, 2, 3], p=[0.3, 0.4, 0.3])
        else:
            # Lower bonus likely
            return np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])

    def _calculate_fpl_points(self, events: SimulatedEvents, position: Position) -> int:
        """
        Calculate FPL points from simulated events.

        Uses official 2025/26 FPL scoring rules.
        """
        pts = 0

        # 1. Appearance points
        if events.minutes > 0:
            pts += 1  # Played any minutes
        if events.minutes >= 60:
            pts += 1  # Played 60+ minutes

        # 2. Goals
        pts += events.goals * GOAL_POINTS[position]

        # 3. Assists
        pts += events.assists * ASSIST_POINTS

        # 4. Clean sheet (60+ mins required)
        if events.clean_sheet and events.minutes >= 60:
            pts += CLEAN_SHEET_POINTS[position]

        # 5. Goals conceded (GK/DEF only, 60+ mins)
        if position in [Position.GK, Position.DEF] and events.minutes >= 60:
            pts += (events.goals_conceded // 2) * GOALS_CONCEDED_POINTS_PER_2

        # 6. Saves (GK only)
        if position == Position.GK:
            pts += (events.saves // 3) * SAVE_POINTS_PER_3

        # 7. Cards
        pts += events.yellow_cards * YELLOW_CARD_POINTS
        pts += events.red_cards * RED_CARD_POINTS

        # 8. Bonus
        pts += events.bonus

        return pts

    def compare_captains(
        self,
        candidates: list[SimulationParams],
        n_sims: int = 10000,
    ) -> list[tuple[SimulationResult, float]]:
        """
        Compare captain candidates and compute win rates.

        Args:
            candidates: List of SimulationParams for captain candidates
            n_sims: Number of simulations per player

        Returns:
            List of (SimulationResult, win_rate) tuples, sorted by ceiling_score
        """
        results = []
        all_points = []

        # Run simulations for each candidate
        for params in candidates:
            result = self.simulate_player(params, n_sims, store_distribution=True)
            results.append(result)
            all_points.append(np.array(result.distribution))

        # Calculate head-to-head win rates
        win_rates = []
        for i, (result, points_i) in enumerate(zip(results, all_points)):
            wins = 0
            for j, points_j in enumerate(all_points):
                if i != j:
                    wins += np.sum(points_i > points_j)

            total_comparisons = (len(candidates) - 1) * n_sims
            win_rate = wins / total_comparisons if total_comparisons > 0 else 0
            win_rates.append(win_rate)

        # Combine results with win rates
        combined = list(zip(results, win_rates))

        # Sort by ceiling_score (descending)
        combined.sort(key=lambda x: x[0].ceiling_score, reverse=True)

        return combined


# =============================================================================
# Helper Functions
# =============================================================================

def create_simulation_params(
    player: Player,
    p_start: float,
    p_60_plus: float,
    e_minutes: float,
    p_clean_sheet: float,
    lambda_goals_conceded: float,
    opp_attack: float,
    fixture_mult: float = 1.0,
) -> SimulationParams:
    """
    Create SimulationParams from a Player and projection context.

    This bridges the projection engine's calculations with the simulator.
    """
    # Get per-90 rates from player
    xg_per_90 = getattr(player, 'xg_per_90', 0.0) or 0.0
    xa_per_90 = getattr(player, 'xa_per_90', 0.0) or 0.0
    yellow_prob = getattr(player, 'yellow_card_prob', 0.08) or 0.08
    red_prob = getattr(player, 'red_card_prob', 0.005) or 0.005
    saves_per_90 = getattr(player, 'saves_per_90', 3.0) or 3.0
    bonus_per_90 = getattr(player, 'bonus_per_90', 0.3) or 0.3

    # Calculate expected minutes breakdown
    if p_start > 0.8:
        e_mins_if_start = min(90, e_minutes / p_start) if p_start > 0 else 85
        e_mins_if_sub = 20
        p_sub = 0.1
    elif p_start > 0.3:
        e_mins_if_start = 80
        e_mins_if_sub = 25
        p_sub = (p_60_plus - p_start * 0.9) / 0.5 if p_start < 0.9 else 0.2
        p_sub = max(0.0, min(0.8, p_sub))
    else:
        # Likely a sub or rotational player
        e_mins_if_start = 75
        e_mins_if_sub = e_minutes / max(0.1, 1 - p_start) if p_start < 0.9 else 20
        p_sub = max(0.1, p_60_plus - p_start)

    # Apply fixture multiplier to attacking rates
    lambda_goals = xg_per_90 * fixture_mult
    lambda_assists = xa_per_90 * fixture_mult

    # Adjust saves by opponent attack strength
    lambda_saves = saves_per_90 * opp_attack if player.position == Position.GK else 0.0

    return SimulationParams(
        player_id=player.id,
        position=player.position,
        p_start=p_start,
        p_sub=p_sub,
        e_minutes_if_start=e_mins_if_start,
        e_minutes_if_sub=e_mins_if_sub,
        lambda_goals=lambda_goals,
        lambda_assists=lambda_assists,
        p_clean_sheet=p_clean_sheet,
        lambda_goals_conceded=lambda_goals_conceded,
        lambda_saves=lambda_saves,
        p_yellow=yellow_prob,
        p_red=red_prob,
        base_bonus=bonus_per_90,
    )
