"""
FPL Optimization Engine.

Multi-week squad optimization using linear programming.
"""

from .constraints import (
    HIT_COST,
    INITIAL_BUDGET,
    MAX_PER_TEAM,
    SQUAD_SIZE,
    STARTING_XI_SIZE,
    validate_formation,
)
from .model import FPLOptimizer
from .objectives import (
    calculate_expected_points,
    rank_captaincy_options,
)
from .solver import (
    FPLSolver,
    OptimizationResult,
    create_solver,
)
from .chips import (
    ChipOptimizer,
    ChipRecommendation,
    ChipTimingPlan,
    ChipValue,
    get_chip_optimizer,
    recommend_chip,
)
from .transfers import (
    TransferValueIndex,
    TransferRecommendation,
    TransferValueCalculator,
    calculate_transfer_value,
)

__all__ = [
    # Main classes
    "FPLOptimizer",
    "FPLSolver",
    "OptimizationResult",
    # Factory
    "create_solver",
    # Constants
    "SQUAD_SIZE",
    "STARTING_XI_SIZE",
    "MAX_PER_TEAM",
    "INITIAL_BUDGET",
    "HIT_COST",
    # Utility functions
    "validate_formation",
    "calculate_expected_points",
    "rank_captaincy_options",
    # Chip optimization
    "ChipOptimizer",
    "ChipRecommendation",
    "ChipTimingPlan",
    "ChipValue",
    "get_chip_optimizer",
    "recommend_chip",
    # Transfer Value Index
    "TransferValueIndex",
    "TransferRecommendation",
    "TransferValueCalculator",
    "calculate_transfer_value",
]
