"""
Solver Wrapper for FPL Optimization.

Provides convenience functions for running the optimizer
with common configurations and handling results.
"""

import logging
from dataclasses import dataclass
from typing import Any

from ..data.models import (
    ChipType,
    GameweekInfo,
    MultiWeekPlan,
    Player,
    PlayerProjection,
    Squad,
    WeekPlan,
)
from .model import FPLOptimizer

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from optimization run."""

    success: bool
    plan: MultiWeekPlan | WeekPlan | None
    message: str
    stats: dict[str, Any]


class FPLSolver:
    """
    High-level solver interface for FPL optimization.

    Provides convenient methods for common optimization scenarios.
    """

    def __init__(
        self,
        time_limit: int = 60,
        gap_tolerance: float = 0.01,
    ):
        """
        Initialize the solver.

        Args:
            time_limit: Max solve time in seconds
            gap_tolerance: Acceptable optimality gap
        """
        self.optimizer = FPLOptimizer(
            solver_time_limit=time_limit,
            solver_gap_tolerance=gap_tolerance,
        )

    def optimize_week(
        self,
        players: list[Player],
        projections: list[PlayerProjection],
        current_squad: Squad,
        gameweek: int,
        chip: ChipType | None = None,
        allow_hits: bool = True,
    ) -> OptimizationResult:
        """
        Optimize for a single gameweek.

        Args:
            players: List of all available players
            projections: List of projections for this gameweek
            current_squad: Current squad state
            gameweek: Gameweek number
            chip: Optional chip to activate
            allow_hits: Whether to allow point hits

        Returns:
            OptimizationResult with plan or error message
        """
        # Convert to dictionaries
        players_dict = {p.id: p for p in players}
        proj_dict = {
            p.player_id: p.expected_points
            for p in projections
            if p.gameweek == gameweek
        }

        # Use form as fallback if no projections
        for p in players:
            if p.id not in proj_dict:
                # Simple fallback: use recent form * 2 as proxy
                proj_dict[p.id] = p.form * 2

        try:
            plan = self.optimizer.optimize_single_week(
                players=players_dict,
                projections=proj_dict,
                current_squad=current_squad,
                chip=chip,
                allow_hits=allow_hits,
            )

            if plan is None:
                return OptimizationResult(
                    success=False,
                    plan=None,
                    message="Optimization infeasible - check constraints",
                    stats={},
                )

            plan.gameweek = gameweek

            return OptimizationResult(
                success=True,
                plan=plan,
                message="Optimization successful",
                stats={
                    "expected_points": plan.expected_points,
                    "hit_cost": plan.hit_cost,
                    "net_points": plan.net_expected_points,
                    "transfers": len(plan.transfers),
                    "chip": chip.value if chip else None,
                },
            )

        except Exception as e:
            logger.exception("Optimization failed")
            return OptimizationResult(
                success=False,
                plan=None,
                message=f"Optimization error: {str(e)}",
                stats={},
            )

    def optimize_horizon(
        self,
        players: list[Player],
        projections: list[PlayerProjection],
        current_squad: Squad,
        gameweeks: list[GameweekInfo],
        horizon: int = 5,
        available_chips: list[ChipType] | None = None,
        allow_hits: bool = True,
    ) -> OptimizationResult:
        """
        Optimize across multiple gameweeks.

        Args:
            players: List of all available players
            projections: List of projections for all gameweeks
            current_squad: Current squad state
            gameweeks: List of gameweek info
            horizon: Number of weeks to plan
            available_chips: Chips that can be used
            allow_hits: Whether to allow point hits

        Returns:
            OptimizationResult with multi-week plan or error message
        """
        players_dict = {p.id: p for p in players}

        # Group projections by gameweek
        proj_by_week: dict[int, dict[int, float]] = {}
        for proj in projections:
            if proj.gameweek not in proj_by_week:
                proj_by_week[proj.gameweek] = {}
            proj_by_week[proj.gameweek][proj.player_id] = proj.expected_points

        # Fill in missing projections with form-based estimates
        upcoming_gws = [
            gw.id for gw in gameweeks
            if not gw.finished
        ][:horizon]

        for gw in upcoming_gws:
            if gw not in proj_by_week:
                proj_by_week[gw] = {}
            for p in players:
                if p.id not in proj_by_week[gw]:
                    proj_by_week[gw][p.id] = p.form * 2

        try:
            plan = self.optimizer.optimize_multi_week(
                players=players_dict,
                projections_by_week=proj_by_week,
                current_squad=current_squad,
                horizon=horizon,
                available_chips=available_chips,
                allow_hits=allow_hits,
            )

            if plan is None:
                return OptimizationResult(
                    success=False,
                    plan=None,
                    message="Multi-week optimization infeasible",
                    stats={},
                )

            return OptimizationResult(
                success=True,
                plan=plan,
                message=f"Optimized {len(plan.week_plans)} weeks",
                stats={
                    "total_expected": plan.total_expected_points,
                    "total_hits": plan.total_hits,
                    "net_points": plan.net_expected_points,
                    "weeks_planned": len(plan.week_plans),
                },
            )

        except Exception as e:
            logger.exception("Multi-week optimization failed")
            return OptimizationResult(
                success=False,
                plan=None,
                message=f"Optimization error: {str(e)}",
                stats={},
            )

    def suggest_best_transfers(
        self,
        players: list[Player],
        projections: list[PlayerProjection],
        current_squad: Squad,
        gameweek: int,
        num_suggestions: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Suggest best transfer options without running full optimization.

        Args:
            players: List of all available players
            projections: List of projections for this gameweek
            current_squad: Current squad state
            gameweek: Gameweek number
            num_suggestions: Number of suggestions to return

        Returns:
            List of transfer suggestions with player info
        """
        players_dict = {p.id: p for p in players}
        proj_dict = {
            p.player_id: p.expected_points
            for p in projections
            if p.gameweek == gameweek
        }

        # Fill gaps with form
        for p in players:
            if p.id not in proj_dict:
                proj_dict[p.id] = p.form * 2

        raw_suggestions = self.optimizer.suggest_transfers(
            players=players_dict,
            projections=proj_dict,
            current_squad=current_squad,
            num_suggestions=num_suggestions,
        )

        # Enrich with player info
        suggestions = []
        for out_id, in_id, gain in raw_suggestions:
            out_player = players_dict.get(out_id)
            in_player = players_dict.get(in_id)

            if out_player and in_player:
                suggestions.append({
                    "out": {
                        "id": out_id,
                        "name": out_player.web_name,
                        "team_id": out_player.team_id,
                        "price": out_player.price,
                        "projected": proj_dict.get(out_id, 0.0),
                    },
                    "in": {
                        "id": in_id,
                        "name": in_player.web_name,
                        "team_id": in_player.team_id,
                        "price": in_player.price,
                        "projected": proj_dict.get(in_id, 0.0),
                    },
                    "gain": gain,
                })

        return suggestions


def create_solver(
    fast: bool = False,
    thorough: bool = False,
) -> FPLSolver:
    """
    Factory function to create a solver with preset configurations.

    Args:
        fast: Use fast settings (shorter time limit, larger gap)
        thorough: Use thorough settings (longer time limit, smaller gap)

    Returns:
        Configured FPLSolver instance
    """
    if fast:
        return FPLSolver(time_limit=15, gap_tolerance=0.05)
    elif thorough:
        return FPLSolver(time_limit=300, gap_tolerance=0.001)
    else:
        return FPLSolver()  # Default settings
