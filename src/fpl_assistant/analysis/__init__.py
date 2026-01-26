"""
Analysis Module for Elite FPL Features.

Contains advanced analytics for competitive FPL play:
- Differential captaincy analysis
- Rival tracking and comparison
- Strategic recommendations
"""

from .differentials import (
    CaptainDifferentialAnalyzer,
    DifferentialRecommendation,
    analyze_captain_differential,
)
from .postgw import (
    PostGWAnalysis,
    PostGWAnalyzer,
    CaptainAnalysis,
    BenchAnalysis,
    TransferAnalysis,
    OutcomeType,
    analyze_gameweek,
)
from .performance_tracker import (
    PerformanceTracker,
    PerformanceSummary,
    GameweekPerformance,
    get_performance_tracker,
    record_gameweek_performance,
    print_performance_report,
)

__all__ = [
    "CaptainDifferentialAnalyzer",
    "DifferentialRecommendation",
    "analyze_captain_differential",
    # Post-GW Analysis
    "PostGWAnalysis",
    "PostGWAnalyzer",
    "CaptainAnalysis",
    "BenchAnalysis",
    "TransferAnalysis",
    "OutcomeType",
    "analyze_gameweek",
    # Performance Tracking
    "PerformanceTracker",
    "PerformanceSummary",
    "GameweekPerformance",
    "get_performance_tracker",
    "record_gameweek_performance",
    "print_performance_report",
]
