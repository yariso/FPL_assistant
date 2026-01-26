"""
Predictions Module for FPL Assistant.

Data-driven player projections using multiple factors:
- Form and historical performance
- Fixture difficulty
- Team strength analysis
- Scout picks and community data
- Adaptive weight learning from backtests
"""

from .adaptive import (
    AdaptiveWeightManager,
    WeightConfig,
    get_adaptive_manager,
    get_optimized_weights,
)
from .backtest import (
    BacktestResult,
    Backtester,
    PlayerResult,
    print_backtest_report,
    run_backtest,
)
from .external import (
    ProjectionImporter,
    import_projections_from_directory,
)
from .projections import (
    ProjectionEngine,
    generate_projections,
    get_best_xi,
)
from .scout import (
    ScoutFetcher,
    ScoutPick,
    ScoutReport,
    fetch_community_tips,
    fetch_differentials,
    fetch_scout_picks,
)
from .prices import (
    PricePredictor,
    PricePrediction,
    PriceChangeThreshold,
    get_price_predictor,
    predict_price_changes,
)
from .minutes import (
    MinutesPrediction,
    MinutesPredictor,
    RotationRisk,
    get_minutes_predictor,
    predict_player_minutes,
)
from .uncertainty import (
    PointsDistribution,
    UncertaintyModel,
    VarianceLevel,
    get_uncertainty_model,
    estimate_player_distribution,
)
from .simulation import (
    EventSimulator,
    SimulatedEvents,
    SimulationParams,
    SimulationResult,
    create_simulation_params,
)
from .intelligence import (
    IntelligenceFetcher,
    IntelligenceReport,
    NewsItem,
    PlayerIntelligence,
    TeamIntelligence,
    fetch_intelligence_report,
    get_flagged_players,
)
from .enhanced_weights import (
    EnhancedWeightConfig,
    EnhancedSignalCalculator,
    PlayerSignals,
    convert_signals_to_projection,
    get_enhanced_weights,
)
from .weight_optimizer import (
    WeightOptimizer,
    create_backtest_function,
)

__all__ = [
    # Projection Engine
    "ProjectionEngine",
    "generate_projections",
    "get_best_xi",
    # Adaptive weights
    "AdaptiveWeightManager",
    "WeightConfig",
    "get_adaptive_manager",
    "get_optimized_weights",
    # External imports
    "ProjectionImporter",
    "import_projections_from_directory",
    # Scout
    "ScoutFetcher",
    "ScoutPick",
    "ScoutReport",
    "fetch_scout_picks",
    "fetch_community_tips",
    "fetch_differentials",
    # Backtesting
    "Backtester",
    "BacktestResult",
    "PlayerResult",
    "run_backtest",
    "print_backtest_report",
    # Price predictions
    "PricePredictor",
    "PricePrediction",
    "PriceChangeThreshold",
    "get_price_predictor",
    "predict_price_changes",
    # Minutes probability
    "MinutesPrediction",
    "MinutesPredictor",
    "RotationRisk",
    "get_minutes_predictor",
    "predict_player_minutes",
    # Uncertainty/Distribution
    "PointsDistribution",
    "UncertaintyModel",
    "VarianceLevel",
    "get_uncertainty_model",
    "estimate_player_distribution",
    # Event-based Simulation
    "EventSimulator",
    "SimulatedEvents",
    "SimulationParams",
    "SimulationResult",
    "create_simulation_params",
    # Team Intelligence
    "IntelligenceFetcher",
    "IntelligenceReport",
    "NewsItem",
    "PlayerIntelligence",
    "TeamIntelligence",
    "fetch_intelligence_report",
    "get_flagged_players",
    # Enhanced Multi-Signal Weights
    "EnhancedWeightConfig",
    "EnhancedSignalCalculator",
    "PlayerSignals",
    "convert_signals_to_projection",
    "get_enhanced_weights",
    # Weight Optimizer
    "WeightOptimizer",
    "create_backtest_function",
]
