"""
FPL Rule Constraints for Optimization.

Encodes all Fantasy Premier League rules as linear programming constraints
that can be used with PuLP.
"""

from typing import TYPE_CHECKING

import pulp

if TYPE_CHECKING:
    from ..data.models import Player, Position


# =============================================================================
# Squad Composition Constants
# =============================================================================

SQUAD_SIZE = 15
STARTING_XI_SIZE = 11
BENCH_SIZE = 4

# Position limits
POSITION_LIMITS = {
    1: (2, 2),   # GK: exactly 2
    2: (5, 5),   # DEF: exactly 5
    3: (5, 5),   # MID: exactly 5
    4: (3, 3),   # FWD: exactly 3
}

# Starting XI formation constraints
STARTING_XI_LIMITS = {
    1: (1, 1),   # GK: exactly 1
    2: (3, 5),   # DEF: 3-5
    3: (2, 5),   # MID: 2-5
    4: (1, 3),   # FWD: 1-3
}

# Max players per Premier League team
MAX_PER_TEAM = 3

# Transfer costs
HIT_COST = 4  # Points deducted per extra transfer

# Initial budget
INITIAL_BUDGET = 100.0  # £100 million


def add_squad_size_constraint(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    name: str = "squad_size",
) -> None:
    """
    Add constraint: Squad must have exactly 15 players.

    Args:
        prob: PuLP problem to add constraint to
        squad_vars: Dict mapping player_id to binary selection variable
        name: Constraint name
    """
    prob += (
        pulp.lpSum(squad_vars.values()) == SQUAD_SIZE,
        name,
    )


def add_position_constraints(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    players: dict[int, "Player"],
    name_prefix: str = "position",
) -> None:
    """
    Add constraints: Exactly 2 GK, 5 DEF, 5 MID, 3 FWD in squad.

    Args:
        prob: PuLP problem
        squad_vars: Selection variables
        players: Dict of player_id -> Player
        name_prefix: Prefix for constraint names
    """
    for pos, (min_count, max_count) in POSITION_LIMITS.items():
        pos_players = [
            squad_vars[pid] for pid, p in players.items()
            if p.position.value == pos and pid in squad_vars
        ]

        if min_count == max_count:
            prob += (
                pulp.lpSum(pos_players) == min_count,
                f"{name_prefix}_{pos}_exact",
            )
        else:
            prob += (
                pulp.lpSum(pos_players) >= min_count,
                f"{name_prefix}_{pos}_min",
            )
            prob += (
                pulp.lpSum(pos_players) <= max_count,
                f"{name_prefix}_{pos}_max",
            )


def add_starting_xi_constraints(
    prob: pulp.LpProblem,
    start_vars: dict[int, pulp.LpVariable],
    players: dict[int, "Player"],
    name_prefix: str = "starting",
) -> None:
    """
    Add constraints for starting XI: 11 players with valid formation.

    Formation must be:
    - 1 GK
    - 3-5 DEF
    - 2-5 MID
    - 1-3 FWD

    Args:
        prob: PuLP problem
        start_vars: Starting XI selection variables
        players: Dict of player_id -> Player
        name_prefix: Prefix for constraint names
    """
    # Total starting XI size
    prob += (
        pulp.lpSum(start_vars.values()) == STARTING_XI_SIZE,
        f"{name_prefix}_size",
    )

    # Formation constraints
    for pos, (min_count, max_count) in STARTING_XI_LIMITS.items():
        pos_players = [
            start_vars[pid] for pid, p in players.items()
            if p.position.value == pos and pid in start_vars
        ]

        if min_count == max_count:
            prob += (
                pulp.lpSum(pos_players) == min_count,
                f"{name_prefix}_pos_{pos}_exact",
            )
        else:
            prob += (
                pulp.lpSum(pos_players) >= min_count,
                f"{name_prefix}_pos_{pos}_min",
            )
            prob += (
                pulp.lpSum(pos_players) <= max_count,
                f"{name_prefix}_pos_{pos}_max",
            )


def add_team_constraint(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    players: dict[int, "Player"],
    name_prefix: str = "team",
) -> None:
    """
    Add constraint: Max 3 players from any Premier League team.

    Args:
        prob: PuLP problem
        squad_vars: Selection variables
        players: Dict of player_id -> Player
        name_prefix: Prefix for constraint names
    """
    # Group players by team
    teams: dict[int, list[int]] = {}
    for pid, player in players.items():
        if pid in squad_vars:
            if player.team_id not in teams:
                teams[player.team_id] = []
            teams[player.team_id].append(pid)

    # Add constraint for each team
    for team_id, team_players in teams.items():
        prob += (
            pulp.lpSum(squad_vars[pid] for pid in team_players) <= MAX_PER_TEAM,
            f"{name_prefix}_{team_id}_max",
        )


def add_budget_constraint(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    players: dict[int, "Player"],
    budget: float,
    name: str = "budget",
) -> None:
    """
    Add constraint: Total squad cost must not exceed budget.

    Args:
        prob: PuLP problem
        squad_vars: Selection variables
        players: Dict of player_id -> Player
        budget: Available budget in millions
        name: Constraint name
    """
    prob += (
        pulp.lpSum(
            squad_vars[pid] * player.price
            for pid, player in players.items()
            if pid in squad_vars
        ) <= budget,
        name,
    )


def add_starter_in_squad_constraint(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    start_vars: dict[int, pulp.LpVariable],
    name_prefix: str = "starter_in_squad",
) -> None:
    """
    Add constraint: Starters must be in squad.

    A player can only start if they're in the squad.

    Args:
        prob: PuLP problem
        squad_vars: Squad selection variables
        start_vars: Starting XI selection variables
        name_prefix: Prefix for constraint names
    """
    for pid in start_vars:
        if pid in squad_vars:
            prob += (
                start_vars[pid] <= squad_vars[pid],
                f"{name_prefix}_{pid}",
            )


def add_captain_constraints(
    prob: pulp.LpProblem,
    captain_vars: dict[int, pulp.LpVariable],
    vice_captain_vars: dict[int, pulp.LpVariable],
    start_vars: dict[int, pulp.LpVariable],
    name_prefix: str = "captain",
) -> None:
    """
    Add constraints for captain and vice captain selection.

    - Exactly 1 captain
    - Exactly 1 vice captain
    - Captain must be in starting XI
    - Vice captain must be in starting XI
    - Captain and vice captain must be different

    Args:
        prob: PuLP problem
        captain_vars: Captain selection variables
        vice_captain_vars: Vice captain selection variables
        start_vars: Starting XI selection variables
        name_prefix: Prefix for constraint names
    """
    # Exactly one captain
    prob += (
        pulp.lpSum(captain_vars.values()) == 1,
        f"{name_prefix}_one",
    )

    # Exactly one vice captain
    prob += (
        pulp.lpSum(vice_captain_vars.values()) == 1,
        f"{name_prefix}_vice_one",
    )

    # Captain must start
    for pid in captain_vars:
        if pid in start_vars:
            prob += (
                captain_vars[pid] <= start_vars[pid],
                f"{name_prefix}_{pid}_starts",
            )

    # Vice captain must start
    for pid in vice_captain_vars:
        if pid in start_vars:
            prob += (
                vice_captain_vars[pid] <= start_vars[pid],
                f"{name_prefix}_vice_{pid}_starts",
            )

    # Cannot be both captain and vice captain
    for pid in captain_vars:
        if pid in vice_captain_vars:
            prob += (
                captain_vars[pid] + vice_captain_vars[pid] <= 1,
                f"{name_prefix}_{pid}_not_both",
            )


def add_transfer_constraints(
    prob: pulp.LpProblem,
    current_squad: set[int],
    new_squad_vars: dict[int, pulp.LpVariable],
    transfer_out_vars: dict[int, pulp.LpVariable],
    transfer_in_vars: dict[int, pulp.LpVariable],
    free_transfers: int,
    allow_hits: bool = True,
    max_hits: int = 4,
    name_prefix: str = "transfer",
) -> pulp.LpVariable:
    """
    Add constraints for transfer logic.

    Args:
        prob: PuLP problem
        current_squad: Set of current player IDs
        new_squad_vars: New squad selection variables
        transfer_out_vars: Transfer out indicator variables
        transfer_in_vars: Transfer in indicator variables
        free_transfers: Number of free transfers available
        allow_hits: Whether to allow point hits
        max_hits: Maximum number of hits allowed
        name_prefix: Prefix for constraint names

    Returns:
        Variable representing total hits taken
    """
    # Transfer out: player in current squad but not in new squad
    for pid in current_squad:
        if pid in new_squad_vars and pid in transfer_out_vars:
            prob += (
                transfer_out_vars[pid] >= 1 - new_squad_vars[pid],
                f"{name_prefix}_out_{pid}",
            )

    # Transfer in: player not in current squad but in new squad
    for pid in new_squad_vars:
        if pid not in current_squad and pid in transfer_in_vars:
            prob += (
                transfer_in_vars[pid] >= new_squad_vars[pid],
                f"{name_prefix}_in_{pid}",
            )

    # Number of transfers out must equal transfers in
    total_out = pulp.lpSum(transfer_out_vars.values())
    total_in = pulp.lpSum(transfer_in_vars.values())
    prob += (
        total_out == total_in,
        f"{name_prefix}_balance",
    )

    # Hits calculation
    hits_var = pulp.LpVariable(f"{name_prefix}_hits", lowBound=0, cat="Integer")

    # hits = max(0, transfers - free_transfers)
    prob += (
        hits_var >= total_out - free_transfers,
        f"{name_prefix}_hits_min",
    )

    if not allow_hits:
        prob += (
            hits_var == 0,
            f"{name_prefix}_no_hits",
        )
    else:
        prob += (
            hits_var <= max_hits,
            f"{name_prefix}_max_hits",
        )

    return hits_var


def add_wildcard_constraints(
    prob: pulp.LpProblem,
    is_wildcard_active: bool,
    transfer_out_vars: dict[int, pulp.LpVariable] | None = None,
    hits_var: pulp.LpVariable | None = None,
    name_prefix: str = "wildcard",
) -> None:
    """
    Add constraints when Wildcard is active.

    When Wildcard is active:
    - All transfers are free (no hits)

    Args:
        prob: PuLP problem
        is_wildcard_active: Whether Wildcard is being used
        transfer_out_vars: Transfer out variables (ignored when WC active)
        hits_var: Hits variable to set to 0
        name_prefix: Prefix for constraint names
    """
    if is_wildcard_active and hits_var is not None:
        prob += (
            hits_var == 0,
            f"{name_prefix}_no_hits",
        )


def add_bench_boost_constraints(
    prob: pulp.LpProblem,
    squad_vars: dict[int, pulp.LpVariable],
    scoring_vars: dict[int, pulp.LpVariable],
    name_prefix: str = "bench_boost",
) -> None:
    """
    Add constraints when Bench Boost is active.

    When Bench Boost is active:
    - All 15 players score points (not just starting 11)

    Args:
        prob: PuLP problem
        squad_vars: Squad selection variables
        scoring_vars: Variables for players who score
        name_prefix: Prefix for constraint names
    """
    # All squad players score
    for pid in squad_vars:
        if pid in scoring_vars:
            prob += (
                scoring_vars[pid] == squad_vars[pid],
                f"{name_prefix}_{pid}",
            )


def validate_formation(players_by_position: dict[int, int]) -> tuple[bool, list[str]]:
    """
    Validate that a formation is legal.

    Args:
        players_by_position: Dict of position -> count

    Returns:
        (is_valid, list of violation messages)
    """
    violations = []

    for pos, (min_count, max_count) in STARTING_XI_LIMITS.items():
        count = players_by_position.get(pos, 0)
        if count < min_count:
            violations.append(f"Position {pos}: need at least {min_count}, have {count}")
        if count > max_count:
            violations.append(f"Position {pos}: max {max_count}, have {count}")

    total = sum(players_by_position.values())
    if total != STARTING_XI_SIZE:
        violations.append(f"Starting XI must have {STARTING_XI_SIZE} players, have {total}")

    return len(violations) == 0, violations
