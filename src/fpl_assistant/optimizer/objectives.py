"""
Objective Functions for FPL Optimization.

Defines the point maximization objectives for single-week and multi-week planning.
"""

from typing import TYPE_CHECKING

import pulp

if TYPE_CHECKING:
    from ..data.models import Player

from .constraints import HIT_COST


def create_single_week_objective(
    start_vars: dict[int, pulp.LpVariable],
    captain_vars: dict[int, pulp.LpVariable],
    projections: dict[int, float],
    triple_captain: bool = False,
) -> pulp.LpAffineExpression:
    """
    Create objective function for single gameweek maximization.

    Points = sum of starting XI points + captain bonus points

    Captain gets double points (or triple with Triple Captain chip).

    Args:
        start_vars: Starting XI selection variables
        captain_vars: Captain selection variables
        projections: Dict of player_id -> expected points
        triple_captain: Whether Triple Captain chip is active

    Returns:
        PuLP expression to maximize
    """
    captain_multiplier = 2 if triple_captain else 1

    # Base points for starters
    base_points = pulp.lpSum(
        start_vars[pid] * projections.get(pid, 0.0)
        for pid in start_vars
    )

    # Captain bonus (captain gets an extra 1x or 2x their points)
    captain_bonus = pulp.lpSum(
        captain_vars[pid] * projections.get(pid, 0.0) * captain_multiplier
        for pid in captain_vars
    )

    return base_points + captain_bonus


def create_bench_boost_objective(
    squad_vars: dict[int, pulp.LpVariable],
    captain_vars: dict[int, pulp.LpVariable],
    projections: dict[int, float],
) -> pulp.LpAffineExpression:
    """
    Create objective function when Bench Boost is active.

    All 15 squad players score points.

    Args:
        squad_vars: Squad selection variables
        captain_vars: Captain selection variables
        projections: Dict of player_id -> expected points

    Returns:
        PuLP expression to maximize
    """
    # All squad players score
    base_points = pulp.lpSum(
        squad_vars[pid] * projections.get(pid, 0.0)
        for pid in squad_vars
    )

    # Captain still gets bonus
    captain_bonus = pulp.lpSum(
        captain_vars[pid] * projections.get(pid, 0.0)
        for pid in captain_vars
    )

    return base_points + captain_bonus


def create_multi_week_objective(
    week_objectives: list[pulp.LpAffineExpression],
    week_hit_vars: list[pulp.LpVariable],
    discount_rate: float = 0.98,
) -> pulp.LpAffineExpression:
    """
    Create objective function for multi-week optimization.

    Total = sum of weekly points - hit costs, with optional time discounting.

    Args:
        week_objectives: List of weekly objective expressions
        week_hit_vars: List of hit variables per week
        discount_rate: Discount factor per week (1.0 = no discounting)

    Returns:
        PuLP expression to maximize
    """
    total = pulp.LpAffineExpression()

    for week_idx, (week_obj, hits) in enumerate(zip(week_objectives, week_hit_vars)):
        discount = discount_rate ** week_idx
        total += discount * week_obj
        total -= discount * HIT_COST * hits

    return total


def add_effective_ownership_penalty(
    objective: pulp.LpAffineExpression,
    squad_vars: dict[int, pulp.LpVariable],
    captain_vars: dict[int, pulp.LpVariable],
    players: dict[int, "Player"],
    penalty_weight: float = 0.01,
) -> pulp.LpAffineExpression:
    """
    Add penalty for selecting highly-owned players (for differential strategy).

    This encourages picking differentials over template picks.
    Negative weight = prefer differentials.
    Positive weight = follow template.

    Args:
        objective: Base objective expression
        squad_vars: Squad selection variables
        captain_vars: Captain selection variables
        players: Dict of player_id -> Player
        penalty_weight: Weight for ownership penalty (negative = avoid popular)

    Returns:
        Modified objective expression
    """
    ownership_penalty = pulp.lpSum(
        squad_vars[pid] * (players[pid].selected_by_percent / 100.0) * penalty_weight
        for pid in squad_vars
        if pid in players
    )

    # Higher penalty for captaining highly-owned players
    captain_ownership = pulp.lpSum(
        captain_vars[pid] * (players[pid].selected_by_percent / 100.0) * penalty_weight * 2
        for pid in captain_vars
        if pid in players
    )

    return objective - ownership_penalty - captain_ownership


def calculate_expected_points(
    squad_player_ids: list[int],
    starting_xi_ids: list[int],
    captain_id: int,
    projections: dict[int, float],
    bench_boost: bool = False,
    triple_captain: bool = False,
) -> float:
    """
    Calculate expected points for a given lineup.

    Args:
        squad_player_ids: List of all 15 squad player IDs
        starting_xi_ids: List of 11 starting player IDs
        captain_id: Captain player ID
        projections: Dict of player_id -> expected points
        bench_boost: Whether Bench Boost is active
        triple_captain: Whether Triple Captain is active

    Returns:
        Total expected points
    """
    if bench_boost:
        scoring_players = squad_player_ids
    else:
        scoring_players = starting_xi_ids

    total = sum(projections.get(pid, 0.0) for pid in scoring_players)

    # Captain bonus
    captain_multiplier = 2 if triple_captain else 1
    captain_points = projections.get(captain_id, 0.0)
    total += captain_points * captain_multiplier

    return total


def rank_captaincy_options(
    starting_xi_ids: list[int],
    projections: dict[int, float],
    players: dict[int, "Player"],
    top_n: int = 5,
) -> list[tuple[int, float, float]]:
    """
    Rank players by captaincy potential.

    Returns list of (player_id, expected_points, effective_ownership).

    Args:
        starting_xi_ids: List of 11 starting player IDs
        projections: Dict of player_id -> expected points
        players: Dict of player_id -> Player
        top_n: Number of options to return

    Returns:
        List of (player_id, projected_points, ownership) tuples, sorted by points
    """
    options = []

    for pid in starting_xi_ids:
        proj = projections.get(pid, 0.0)
        player = players.get(pid)
        ownership = player.selected_by_percent if player else 0.0
        options.append((pid, proj, ownership))

    # Sort by projected points descending
    options.sort(key=lambda x: x[1], reverse=True)

    return options[:top_n]
