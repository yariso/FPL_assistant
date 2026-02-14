"""
FPL Optimization Model.

Main optimization model using PuLP for single and multi-week squad planning.
"""

import logging
from typing import Any

import pulp

from ..data.models import (
    ChipType,
    MultiWeekPlan,
    Player,
    Squad,
    Transfer,
    WeekPlan,
)
from .constraints import (
    INITIAL_BUDGET,
    SQUAD_SIZE,
    add_bench_boost_constraints,
    add_budget_constraint,
    add_captain_constraints,
    add_position_constraints,
    add_squad_size_constraint,
    add_starter_in_squad_constraint,
    add_starting_xi_constraints,
    add_team_constraint,
    add_transfer_constraints,
    add_wildcard_constraints,
)
from .objectives import (
    create_bench_boost_objective,
    create_multi_week_objective,
    create_single_week_objective,
)

logger = logging.getLogger(__name__)


class FPLOptimizer:
    """
    Main FPL squad optimization engine.

    Uses linear programming (PuLP with HiGHS solver) to find optimal:
    - Squad selection (15 players)
    - Starting XI and formation
    - Captain and vice captain
    - Transfers across multiple gameweeks
    - Chip timing (Wildcard, Bench Boost, Free Hit, Triple Captain)
    """

    def __init__(
        self,
        solver_time_limit: int = 60,
        solver_gap_tolerance: float = 0.01,
    ):
        """
        Initialize the optimizer.

        Args:
            solver_time_limit: Max solve time in seconds
            solver_gap_tolerance: Acceptable optimality gap (0.01 = 1%)
        """
        self.time_limit = solver_time_limit
        self.gap_tolerance = solver_gap_tolerance
        self._solver = self._get_solver()

    def _get_solver(self) -> pulp.LpSolver:
        """Get the configured solver (CBC bundled with PuLP)."""
        # Use CBC which is bundled with PuLP (always available)
        try:
            solver = pulp.PULP_CBC_CMD(
                msg=0,
                timeLimit=self.time_limit,
                gapRel=self.gap_tolerance,
            )
            return solver
        except Exception as e:
            logger.warning(f"CBC solver error: {e}")
            # Ultimate fallback
            return pulp.PULP_CBC_CMD(msg=0)

    def optimize_single_week(
        self,
        players: dict[int, Player],
        projections: dict[int, float],
        current_squad: Squad | None = None,
        budget: float | None = None,
        chip: ChipType | None = None,
        allow_hits: bool = True,
        max_hits: int = 4,
    ) -> WeekPlan | None:
        """
        Optimize squad for a single gameweek.

        Args:
            players: Dict of player_id -> Player
            projections: Dict of player_id -> expected points for this GW
            current_squad: Current squad (if None, builds from scratch)
            budget: Available budget (if None, uses current squad value or default)
            chip: Chip to use (Wildcard, Bench Boost, Free Hit, Triple Captain)
            allow_hits: Whether to allow point hits for transfers
            max_hits: Maximum hits to take

        Returns:
            WeekPlan with optimal lineup, or None if infeasible
        """
        logger.info(f"Optimizing single week, {len(players)} players available")

        # Create problem
        prob = pulp.LpProblem("FPL_SingleWeek", pulp.LpMaximize)

        # Filter available players
        available_players = {
            pid: p for pid, p in players.items()
            if p.is_available
        }

        # Determine budget
        if budget is not None:
            total_budget = budget
        elif current_squad is not None:
            total_budget = current_squad.total_value + current_squad.bank
        else:
            total_budget = INITIAL_BUDGET

        # Create decision variables
        squad_vars = {
            pid: pulp.LpVariable(f"squad_{pid}", cat="Binary")
            for pid in available_players
        }

        start_vars = {
            pid: pulp.LpVariable(f"start_{pid}", cat="Binary")
            for pid in available_players
        }

        captain_vars = {
            pid: pulp.LpVariable(f"captain_{pid}", cat="Binary")
            for pid in available_players
        }

        vice_captain_vars = {
            pid: pulp.LpVariable(f"vice_{pid}", cat="Binary")
            for pid in available_players
        }

        # Add constraints
        add_squad_size_constraint(prob, squad_vars)
        add_position_constraints(prob, squad_vars, available_players)
        add_team_constraint(prob, squad_vars, available_players)
        add_budget_constraint(prob, squad_vars, available_players, total_budget)

        add_starter_in_squad_constraint(prob, squad_vars, start_vars)
        add_starting_xi_constraints(prob, start_vars, available_players)

        add_captain_constraints(prob, captain_vars, vice_captain_vars, start_vars)

        # Handle transfers if we have a current squad
        hits_var = None
        if current_squad is not None and chip != ChipType.WILDCARD and chip != ChipType.FREE_HIT:
            current_pids = {sp.player_id for sp in current_squad.players}

            transfer_out_vars = {
                pid: pulp.LpVariable(f"out_{pid}", cat="Binary")
                for pid in current_pids
            }

            transfer_in_vars = {
                pid: pulp.LpVariable(f"in_{pid}", cat="Binary")
                for pid in available_players
                if pid not in current_pids
            }

            hits_var = add_transfer_constraints(
                prob,
                current_pids,
                squad_vars,
                transfer_out_vars,
                transfer_in_vars,
                current_squad.free_transfers,
                allow_hits,
                max_hits,
            )

        # Handle Wildcard (no transfer cost)
        if chip == ChipType.WILDCARD and hits_var is not None:
            add_wildcard_constraints(prob, True, hits_var=hits_var)

        # Create objective
        triple_captain = chip == ChipType.TRIPLE_CAPTAIN

        if chip == ChipType.BENCH_BOOST:
            objective = create_bench_boost_objective(
                squad_vars, captain_vars, projections
            )
        else:
            objective = create_single_week_objective(
                start_vars, captain_vars, projections, triple_captain
            )

        # Subtract hit costs from objective
        if hits_var is not None:
            objective -= 4 * hits_var

        prob += objective, "maximize_points"

        # Solve
        status = prob.solve(self._solver)

        if status != pulp.LpStatusOptimal:
            logger.warning(f"Optimization status: {pulp.LpStatus[status]}")
            if status == pulp.LpStatusInfeasible:
                return None

        # Extract solution
        return self._extract_single_week_solution(
            prob,
            squad_vars,
            start_vars,
            captain_vars,
            vice_captain_vars,
            projections,
            current_squad,
            chip,
            hits_var,
        )

    def _extract_single_week_solution(
        self,
        prob: pulp.LpProblem,
        squad_vars: dict[int, pulp.LpVariable],
        start_vars: dict[int, pulp.LpVariable],
        captain_vars: dict[int, pulp.LpVariable],
        vice_captain_vars: dict[int, pulp.LpVariable],
        projections: dict[int, float],
        current_squad: Squad | None,
        chip: ChipType | None,
        hits_var: pulp.LpVariable | None,
    ) -> WeekPlan:
        """Extract solution from solved problem."""
        # Get selected squad
        squad_ids = [
            pid for pid, var in squad_vars.items()
            if var.value() and var.value() > 0.5
        ]

        # Get starters
        starting_xi = [
            pid for pid, var in start_vars.items()
            if var.value() and var.value() > 0.5
        ]

        # Get bench (squad - starters)
        bench = [pid for pid in squad_ids if pid not in starting_xi]

        # Get captain
        captain_id = next(
            (pid for pid, var in captain_vars.items()
             if var.value() and var.value() > 0.5),
            starting_xi[0] if starting_xi else 0
        )

        # Get vice captain
        vice_captain_id = next(
            (pid for pid, var in vice_captain_vars.items()
             if var.value() and var.value() > 0.5),
            starting_xi[1] if len(starting_xi) > 1 else 0
        )

        # Calculate transfers
        transfers = []
        if current_squad is not None:
            current_pids = {sp.player_id for sp in current_squad.players}
            new_pids = set(squad_ids)

            players_out = current_pids - new_pids
            players_in = new_pids - current_pids

            # Match transfers (assumes same number in/out)
            for out_id, in_id in zip(sorted(players_out), sorted(players_in)):
                transfers.append(Transfer(
                    player_out_id=out_id,
                    player_in_id=in_id,
                    gameweek=0,  # Will be set by caller
                ))

        # Calculate expected points
        if chip == ChipType.BENCH_BOOST:
            scoring_players = squad_ids
        else:
            scoring_players = starting_xi

        expected_points = sum(projections.get(pid, 0.0) for pid in scoring_players)

        # Add captain bonus
        captain_multiplier = 2 if chip == ChipType.TRIPLE_CAPTAIN else 1
        expected_points += projections.get(captain_id, 0.0) * captain_multiplier

        # Get hit cost
        hit_cost = 0
        if hits_var is not None and hits_var.value():
            hit_cost = int(hits_var.value()) * 4

        return WeekPlan(
            gameweek=0,  # Will be set by caller
            transfers=transfers,
            chip_used=chip,
            captain_id=captain_id,
            vice_captain_id=vice_captain_id,
            starting_xi=starting_xi,
            bench_order=bench,
            expected_points=expected_points,
            hit_cost=hit_cost,
        )

    def optimize_multi_week(
        self,
        players: dict[int, Player],
        projections_by_week: dict[int, dict[int, float]],
        current_squad: Squad,
        horizon: int = 5,
        available_chips: list[ChipType] | None = None,
        allow_hits: bool = True,
        max_hits_per_week: int = 3,
        use_coupled_optimization: bool = True,
    ) -> MultiWeekPlan | None:
        """
        Optimize squad across multiple gameweeks.

        This is the main optimization method that considers:
        - Transfers across the planning horizon
        - Chip timing optimization
        - Rolling free transfer accumulation

        Args:
            players: Dict of player_id -> Player
            projections_by_week: Dict of gameweek -> (player_id -> expected points)
            current_squad: Current squad state
            horizon: Number of gameweeks to plan
            available_chips: Chips that can be used
            allow_hits: Whether to allow point hits
            max_hits_per_week: Max hits per gameweek
            use_coupled_optimization: Use full coupled MILP (True) or greedy week-by-week (False)

        Returns:
            MultiWeekPlan with optimal strategy, or None if infeasible
        """
        logger.info(f"Optimizing {horizon} weeks, {len(players)} players, coupled={use_coupled_optimization}")

        if horizon < 1:
            raise ValueError("Horizon must be at least 1")

        gameweeks = sorted(projections_by_week.keys())[:horizon]

        if len(gameweeks) < horizon:
            logger.warning(
                f"Only {len(gameweeks)} weeks of projections available, "
                f"requested {horizon}"
            )
            horizon = len(gameweeks)

        # Use coupled optimization for better results
        if use_coupled_optimization and horizon > 1:
            return self._optimize_multi_week_coupled(
                players=players,
                projections_by_week=projections_by_week,
                current_squad=current_squad,
                gameweeks=gameweeks,
                available_chips=available_chips,
                allow_hits=allow_hits,
                max_hits_per_week=max_hits_per_week,
            )

        # Fallback: Greedy week-by-week optimization
        return self._optimize_multi_week_greedy(
            players=players,
            projections_by_week=projections_by_week,
            current_squad=current_squad,
            gameweeks=gameweeks,
            available_chips=available_chips,
            allow_hits=allow_hits,
            max_hits_per_week=max_hits_per_week,
        )

    def _optimize_multi_week_coupled(
        self,
        players: dict[int, Player],
        projections_by_week: dict[int, dict[int, float]],
        current_squad: Squad,
        gameweeks: list[int],
        available_chips: list[ChipType] | None = None,
        allow_hits: bool = True,
        max_hits_per_week: int = 3,
    ) -> MultiWeekPlan | None:
        """
        True coupled multi-week optimization using MILP.

        This formulation jointly optimizes transfers across all gameweeks,
        properly modeling the tradeoff between immediate gains and future flexibility.

        Key insight: A transfer that looks suboptimal in week 1 might enable
        a much better team in week 3. The coupled formulation captures this.
        """
        logger.info(f"Running coupled {len(gameweeks)}-week optimization")

        # Create problem
        prob = pulp.LpProblem("FPL_MultiWeek_Coupled", pulp.LpMaximize)

        # Filter available players
        available_players = {
            pid: p for pid, p in players.items()
            if p.is_available
        }

        # Get current squad player IDs
        current_pids = {sp.player_id for sp in current_squad.players}
        initial_free_transfers = current_squad.free_transfers

        # Budget (total value available)
        total_budget = current_squad.total_value + current_squad.bank

        # ================================================================
        # DECISION VARIABLES
        # ================================================================

        # squad[player, week] = 1 if player in squad for that week
        squad_vars = {}
        for gw in gameweeks:
            for pid in available_players:
                squad_vars[(pid, gw)] = pulp.LpVariable(
                    f"squad_{pid}_{gw}", cat="Binary"
                )

        # start[player, week] = 1 if player starts that week
        start_vars = {}
        for gw in gameweeks:
            for pid in available_players:
                start_vars[(pid, gw)] = pulp.LpVariable(
                    f"start_{pid}_{gw}", cat="Binary"
                )

        # captain[player, week] = 1 if player is captain that week
        captain_vars = {}
        for gw in gameweeks:
            for pid in available_players:
                captain_vars[(pid, gw)] = pulp.LpVariable(
                    f"captain_{pid}_{gw}", cat="Binary"
                )

        # transfer_in[player, week] = 1 if player transferred IN that week
        transfer_in_vars = {}
        for gw in gameweeks:
            for pid in available_players:
                transfer_in_vars[(pid, gw)] = pulp.LpVariable(
                    f"in_{pid}_{gw}", cat="Binary"
                )

        # transfer_out[player, week] = 1 if player transferred OUT that week
        transfer_out_vars = {}
        for gw in gameweeks:
            for pid in available_players:
                transfer_out_vars[(pid, gw)] = pulp.LpVariable(
                    f"out_{pid}_{gw}", cat="Binary"
                )

        # num_transfers[week] = number of transfers made that week
        num_transfers_vars = {}
        for gw in gameweeks:
            num_transfers_vars[gw] = pulp.LpVariable(
                f"num_transfers_{gw}", lowBound=0, upBound=SQUAD_SIZE, cat="Integer"
            )

        # hits[week] = number of hits taken that week (transfers beyond free)
        hits_vars = {}
        for gw in gameweeks:
            hits_vars[gw] = pulp.LpVariable(
                f"hits_{gw}", lowBound=0, upBound=max_hits_per_week, cat="Integer"
            )

        # free_transfers[week] = free transfers available at START of week
        # (We need to track FT accumulation)
        ft_vars = {}
        for i, gw in enumerate(gameweeks):
            if i == 0:
                # First week: use current FTs
                ft_vars[gw] = initial_free_transfers
            else:
                # Rolling FTs (1 or 2)
                ft_vars[gw] = pulp.LpVariable(
                    f"ft_{gw}", lowBound=1, upBound=2, cat="Integer"
                )

        # ================================================================
        # CONSTRAINTS
        # ================================================================

        # 1. Squad size = 15 each week
        for gw in gameweeks:
            prob += (
                pulp.lpSum(squad_vars[(pid, gw)] for pid in available_players) == SQUAD_SIZE,
                f"squad_size_{gw}"
            )

        # 2. Position constraints each week
        for gw in gameweeks:
            # 2 GK
            prob += (
                pulp.lpSum(
                    squad_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 1
                ) == 2,
                f"gk_count_{gw}"
            )
            # 5 DEF
            prob += (
                pulp.lpSum(
                    squad_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 2
                ) == 5,
                f"def_count_{gw}"
            )
            # 5 MID
            prob += (
                pulp.lpSum(
                    squad_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 3
                ) == 5,
                f"mid_count_{gw}"
            )
            # 3 FWD
            prob += (
                pulp.lpSum(
                    squad_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 4
                ) == 3,
                f"fwd_count_{gw}"
            )

        # 3. Max 3 players per team each week
        team_ids = set(p.team_id for p in available_players.values())
        for gw in gameweeks:
            for team_id in team_ids:
                prob += (
                    pulp.lpSum(
                        squad_vars[(pid, gw)]
                        for pid, p in available_players.items()
                        if p.team_id == team_id
                    ) <= 3,
                    f"team_limit_{team_id}_{gw}"
                )

        # 4. Budget constraint each week
        for gw in gameweeks:
            prob += (
                pulp.lpSum(
                    squad_vars[(pid, gw)] * available_players[pid].price
                    for pid in available_players
                ) <= total_budget,
                f"budget_{gw}"
            )

        # 5. Starting XI = 11 each week
        for gw in gameweeks:
            prob += (
                pulp.lpSum(start_vars[(pid, gw)] for pid in available_players) == 11,
                f"starting_xi_{gw}"
            )

        # 6. Can only start if in squad
        for gw in gameweeks:
            for pid in available_players:
                prob += (
                    start_vars[(pid, gw)] <= squad_vars[(pid, gw)],
                    f"start_in_squad_{pid}_{gw}"
                )

        # 7. Formation constraints (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD in starting XI)
        for gw in gameweeks:
            # Exactly 1 GK starts
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 1
                ) == 1,
                f"start_gk_{gw}"
            )
            # 3-5 DEF start
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 2
                ) >= 3,
                f"start_def_min_{gw}"
            )
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 2
                ) <= 5,
                f"start_def_max_{gw}"
            )
            # 2-5 MID start
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 3
                ) >= 2,
                f"start_mid_min_{gw}"
            )
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 3
                ) <= 5,
                f"start_mid_max_{gw}"
            )
            # 1-3 FWD start
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 4
                ) >= 1,
                f"start_fwd_min_{gw}"
            )
            prob += (
                pulp.lpSum(
                    start_vars[(pid, gw)]
                    for pid, p in available_players.items()
                    if p.position.value == 4
                ) <= 3,
                f"start_fwd_max_{gw}"
            )

        # 8. Exactly one captain who must be in starting XI
        for gw in gameweeks:
            prob += (
                pulp.lpSum(captain_vars[(pid, gw)] for pid in available_players) == 1,
                f"one_captain_{gw}"
            )
            for pid in available_players:
                prob += (
                    captain_vars[(pid, gw)] <= start_vars[(pid, gw)],
                    f"captain_starts_{pid}_{gw}"
                )

        # 9. COUPLING CONSTRAINT: Squad evolution through transfers
        # squad[p,gw] = squad[p,gw-1] + transfer_in[p,gw] - transfer_out[p,gw]
        for i, gw in enumerate(gameweeks):
            for pid in available_players:
                if i == 0:
                    # First week: compare to initial squad
                    was_in_squad = 1 if pid in current_pids else 0
                    prob += (
                        squad_vars[(pid, gw)] == was_in_squad + transfer_in_vars[(pid, gw)] - transfer_out_vars[(pid, gw)],
                        f"squad_evolution_{pid}_{gw}"
                    )
                else:
                    prev_gw = gameweeks[i - 1]
                    prob += (
                        squad_vars[(pid, gw)] == squad_vars[(pid, prev_gw)] + transfer_in_vars[(pid, gw)] - transfer_out_vars[(pid, gw)],
                        f"squad_evolution_{pid}_{gw}"
                    )

        # 10. Can only transfer out if was in squad
        for i, gw in enumerate(gameweeks):
            for pid in available_players:
                if i == 0:
                    was_in_squad = 1 if pid in current_pids else 0
                    prob += (
                        transfer_out_vars[(pid, gw)] <= was_in_squad,
                        f"transfer_out_valid_{pid}_{gw}"
                    )
                else:
                    prev_gw = gameweeks[i - 1]
                    prob += (
                        transfer_out_vars[(pid, gw)] <= squad_vars[(pid, prev_gw)],
                        f"transfer_out_valid_{pid}_{gw}"
                    )

        # 11. Can only transfer in if not already in squad
        for i, gw in enumerate(gameweeks):
            for pid in available_players:
                if i == 0:
                    was_in_squad = 1 if pid in current_pids else 0
                    prob += (
                        transfer_in_vars[(pid, gw)] <= 1 - was_in_squad,
                        f"transfer_in_valid_{pid}_{gw}"
                    )
                else:
                    prev_gw = gameweeks[i - 1]
                    prob += (
                        transfer_in_vars[(pid, gw)] <= 1 - squad_vars[(pid, prev_gw)],
                        f"transfer_in_valid_{pid}_{gw}"
                    )

        # 12. Number of transfers = sum of transfers in (= sum of transfers out)
        for gw in gameweeks:
            prob += (
                num_transfers_vars[gw] == pulp.lpSum(transfer_in_vars[(pid, gw)] for pid in available_players),
                f"count_transfers_{gw}"
            )

        # 13. Hit calculation: hits = max(0, transfers - free_transfers)
        # Using linearization: hits >= transfers - ft, hits >= 0
        for i, gw in enumerate(gameweeks):
            if allow_hits:
                if i == 0:
                    prob += (
                        hits_vars[gw] >= num_transfers_vars[gw] - initial_free_transfers,
                        f"hits_calc_{gw}"
                    )
                else:
                    prob += (
                        hits_vars[gw] >= num_transfers_vars[gw] - ft_vars[gw],
                        f"hits_calc_{gw}"
                    )
            else:
                # No hits allowed
                prob += (hits_vars[gw] == 0, f"no_hits_{gw}")

        # 14. Free transfer accumulation (for weeks 2+)
        # ft[gw] = min(2, 1 + max(0, ft[gw-1] - transfers[gw-1]))
        # Simplified: ft[gw] = 1 if took transfer, 2 if banked (max 2)
        for i, gw in enumerate(gameweeks):
            if i > 0:
                prev_gw = gameweeks[i - 1]
                # If no transfers last week, gain a FT (max 2)
                # If transfers >= ft last week, reset to 1
                # This is hard to model exactly in MILP, so we use approximation:
                # ft[gw] = 1 + (1 if transfers[prev] == 0 else 0), capped at 2
                # Simplification: assume we use at least 1 transfer most weeks
                # More accurate: ft_next = min(2, ft_current - transfers_used + 1)
                # Using aux variable for unused FTs
                unused_ft = pulp.LpVariable(f"unused_ft_{prev_gw}", lowBound=0, upBound=1, cat="Integer")
                if i == 1:
                    prob += (unused_ft >= initial_free_transfers - num_transfers_vars[prev_gw], f"unused_ft_calc_{prev_gw}")
                else:
                    prob += (unused_ft >= ft_vars[prev_gw] - num_transfers_vars[prev_gw], f"unused_ft_calc_{prev_gw}")
                prob += (unused_ft <= 1, f"unused_ft_max_{prev_gw}")
                prob += (ft_vars[gw] == 1 + unused_ft, f"ft_accumulate_{gw}")

        # ================================================================
        # OBJECTIVE: Maximize total expected points minus hit costs
        # ================================================================
        objective = pulp.LpSum(
            # Points from starting XI
            projections_by_week.get(gw, {}).get(pid, 0) * start_vars[(pid, gw)]
            for gw in gameweeks
            for pid in available_players
        ) + pulp.LpSum(
            # Captain bonus (extra xP for captain)
            projections_by_week.get(gw, {}).get(pid, 0) * captain_vars[(pid, gw)]
            for gw in gameweeks
            for pid in available_players
        ) - pulp.LpSum(
            # Hit costs
            4 * hits_vars[gw]
            for gw in gameweeks
        )

        prob += objective, "maximize_total_points"

        # ================================================================
        # SOLVE
        # ================================================================
        logger.info("Solving coupled multi-week optimization...")
        status = prob.solve(self._solver)

        if status != pulp.LpStatusOptimal:
            logger.warning(f"Optimization status: {pulp.LpStatus[status]}")
            if status == pulp.LpStatusInfeasible:
                logger.error("Problem is infeasible - falling back to greedy")
                return None

        # ================================================================
        # EXTRACT SOLUTION
        # ================================================================
        logger.info("Extracting solution...")
        week_plans = []

        for i, gw in enumerate(gameweeks):
            # Get squad for this week
            squad_ids = [
                pid for pid in available_players
                if squad_vars[(pid, gw)].value() and squad_vars[(pid, gw)].value() > 0.5
            ]

            # Get starters
            starting_xi = [
                pid for pid in available_players
                if start_vars[(pid, gw)].value() and start_vars[(pid, gw)].value() > 0.5
            ]

            # Get bench
            bench = [pid for pid in squad_ids if pid not in starting_xi]

            # Get captain
            captain_id = next(
                (pid for pid in available_players
                 if captain_vars[(pid, gw)].value() and captain_vars[(pid, gw)].value() > 0.5),
                starting_xi[0] if starting_xi else 0
            )

            # Get vice captain (second highest projected in starting XI)
            starting_projections = [
                (pid, projections_by_week.get(gw, {}).get(pid, 0))
                for pid in starting_xi if pid != captain_id
            ]
            starting_projections.sort(key=lambda x: x[1], reverse=True)
            vice_captain_id = starting_projections[0][0] if starting_projections else captain_id

            # Get transfers for this week
            transfers = []
            transfers_out = [
                pid for pid in available_players
                if transfer_out_vars[(pid, gw)].value() and transfer_out_vars[(pid, gw)].value() > 0.5
            ]
            transfers_in = [
                pid for pid in available_players
                if transfer_in_vars[(pid, gw)].value() and transfer_in_vars[(pid, gw)].value() > 0.5
            ]

            for out_id, in_id in zip(sorted(transfers_out), sorted(transfers_in)):
                transfers.append(Transfer(
                    player_out_id=out_id,
                    player_in_id=in_id,
                    gameweek=gw,
                ))

            # Calculate expected points
            expected_points = sum(
                projections_by_week.get(gw, {}).get(pid, 0)
                for pid in starting_xi
            )
            # Add captain bonus
            expected_points += projections_by_week.get(gw, {}).get(captain_id, 0)

            # Get hit cost
            hit_cost = 0
            if hits_vars[gw].value():
                hit_cost = int(hits_vars[gw].value()) * 4

            week_plans.append(WeekPlan(
                gameweek=gw,
                transfers=transfers,
                chip_used=None,  # Set by post-solve chip evaluation below
                captain_id=captain_id,
                vice_captain_id=vice_captain_id,
                starting_xi=starting_xi,
                bench_order=bench,
                expected_points=expected_points,
                hit_cost=hit_cost,
            ))

        # Post-solve: evaluate chips for each week
        if available_chips:
            remaining_chips = list(available_chips)
            try:
                from .chips import ChipOptimizer
                chip_optimizer = ChipOptimizer(
                    list(players.values()), [], projections_by_week,
                )
                for wp in week_plans:
                    if not remaining_chips:
                        break
                    chip_rec = chip_optimizer.get_chip_recommendation(
                        current_squad, wp.gameweek, remaining_chips,
                    )
                    if chip_rec.recommended_chip:
                        wp.chip_used = chip_rec.recommended_chip
                        remaining_chips.remove(chip_rec.recommended_chip)
                        logger.info(f"GW{wp.gameweek}: Recommending chip {wp.chip_used.value}")
            except Exception as e:
                logger.debug(f"Post-solve chip evaluation failed: {e}")

        # Calculate totals
        total_expected = sum(wp.expected_points for wp in week_plans)
        total_hits = sum(wp.hit_cost // 4 for wp in week_plans)

        logger.info(f"Coupled optimization complete: {total_expected:.1f} xP, {total_hits} hits")

        return MultiWeekPlan(
            week_plans=week_plans,
            horizon=len(gameweeks),
            total_expected_points=total_expected,
            total_hits=total_hits,
            starting_squad=[sp.player_id for sp in current_squad.players],
            parameters={
                "allow_hits": allow_hits,
                "max_hits_per_week": max_hits_per_week,
                "coupled_optimization": True,
            },
        )

    def _optimize_multi_week_greedy(
        self,
        players: dict[int, Player],
        projections_by_week: dict[int, dict[int, float]],
        current_squad: Squad,
        gameweeks: list[int],
        available_chips: list[ChipType] | None = None,
        allow_hits: bool = True,
        max_hits_per_week: int = 3,
    ) -> MultiWeekPlan | None:
        """
        Fallback greedy week-by-week optimization.

        Simpler but may miss cross-week synergies.
        """
        logger.info(f"Running greedy {len(gameweeks)}-week optimization")

        week_plans = []
        running_squad = current_squad
        running_free_transfers = current_squad.free_transfers

        # Track which chips have been used
        remaining_chips = list(available_chips) if available_chips else []

        for gw in gameweeks:
            week_projections = projections_by_week.get(gw, {})

            # Evaluate chip usage for this GW using ChipOptimizer
            chip = None
            if remaining_chips:
                try:
                    from .chips import ChipOptimizer
                    chip_optimizer = ChipOptimizer(
                        list(players.values()),
                        [],  # gameweek info populated from projections
                        projections_by_week,
                    )
                    chip_rec = chip_optimizer.get_chip_recommendation(
                        running_squad, gw, remaining_chips
                    )
                    if chip_rec.recommended_chip:
                        chip = chip_rec.recommended_chip
                        remaining_chips.remove(chip)
                        logger.info(f"GW{gw}: Using chip {chip.value}")
                except Exception as e:
                    logger.debug(f"Chip evaluation failed for GW{gw}: {e}")

            week_plan = self.optimize_single_week(
                players=players,
                projections=week_projections,
                current_squad=running_squad,
                chip=chip,
                allow_hits=allow_hits,
                max_hits=max_hits_per_week,
            )

            if week_plan is None:
                logger.error(f"Failed to optimize week {gw}")
                return None

            week_plan.gameweek = gw
            for t in week_plan.transfers:
                t.gameweek = gw

            week_plans.append(week_plan)

            # Update running state
            # Build new squad from transfers
            current_pids = {sp.player_id for sp in running_squad.players}
            for transfer in week_plan.transfers:
                current_pids.discard(transfer.player_out_id)
                current_pids.add(transfer.player_in_id)

            # Update free transfers (simplified)
            transfers_made = len(week_plan.transfers)
            if week_plan.chip_used == ChipType.WILDCARD:
                running_free_transfers = 1
            else:
                remaining_ft = running_free_transfers - transfers_made
                if remaining_ft >= 0:
                    # Roll unused FTs (max 2)
                    running_free_transfers = min(2, 1 + max(0, remaining_ft))
                else:
                    running_free_transfers = 1

            # Update squad object (simplified - just update player IDs)
            from ..data.models import SquadPlayer
            new_players = []
            for i, pid in enumerate(current_pids):
                sp = next(
                    (sp for sp in running_squad.players if sp.player_id == pid),
                    None
                )
                if sp:
                    new_players.append(sp)
                else:
                    # New player from transfer
                    player = players.get(pid)
                    if player:
                        new_players.append(SquadPlayer(
                            player_id=pid,
                            position=i + 1,
                            purchase_price=player.price,
                            selling_price=player.price,
                        ))

            running_squad = Squad(
                players=new_players,
                bank=running_squad.bank,  # Simplified
                free_transfers=running_free_transfers,
                total_value=running_squad.total_value,
                chips=running_squad.chips,
            )

        # Calculate totals
        total_expected = sum(wp.expected_points for wp in week_plans)
        total_hits = sum(wp.hit_cost // 4 for wp in week_plans)

        return MultiWeekPlan(
            week_plans=week_plans,
            horizon=len(gameweeks),
            total_expected_points=total_expected,
            total_hits=total_hits,
            starting_squad=[sp.player_id for sp in current_squad.players],
            parameters={
                "allow_hits": allow_hits,
                "max_hits_per_week": max_hits_per_week,
                "coupled_optimization": False,
            },
        )

    def suggest_transfers(
        self,
        players: dict[int, Player],
        projections: dict[int, float],
        current_squad: Squad,
        num_suggestions: int = 5,
    ) -> list[tuple[int, int, float]]:
        """
        Suggest best transfer options.

        Returns list of (player_out_id, player_in_id, expected_point_gain).

        Args:
            players: Dict of player_id -> Player
            projections: Dict of player_id -> expected points
            current_squad: Current squad
            num_suggestions: Number of suggestions to return

        Returns:
            List of transfer suggestions sorted by expected gain
        """
        suggestions = []

        current_pids = {sp.player_id for sp in current_squad.players}

        for out_id in current_pids:
            out_player = players.get(out_id)
            if not out_player:
                continue

            out_proj = projections.get(out_id, 0.0)

            # Find potential replacements
            for in_id, in_player in players.items():
                if in_id in current_pids:
                    continue
                if not in_player.is_available:
                    continue
                if in_player.position != out_player.position:
                    continue

                # Check team limit
                team_count = sum(
                    1 for pid in current_pids
                    if players.get(pid, out_player).team_id == in_player.team_id
                )
                if out_player.team_id != in_player.team_id and team_count >= 3:
                    continue

                # Check price
                squad_player = next(
                    (sp for sp in current_squad.players if sp.player_id == out_id),
                    None
                )
                sell_price = squad_player.selling_price if squad_player else out_player.price
                available = current_squad.bank + sell_price

                if in_player.price > available:
                    continue

                in_proj = projections.get(in_id, 0.0)
                gain = in_proj - out_proj

                if gain > 0:
                    suggestions.append((out_id, in_id, gain))

        # Sort by gain and return top N
        suggestions.sort(key=lambda x: x[2], reverse=True)
        return suggestions[:num_suggestions]
