"""
Data Models and Storage Module.

Contains Pydantic models for FPL entities and SQLite storage operations.
"""

from .models import (
    ChipStatus,
    ChipType,
    Fixture,
    GameweekInfo,
    MultiWeekPlan,
    Player,
    PlayerProjection,
    PlayerStatus,
    Position,
    Squad,
    SquadPlayer,
    Team,
    Transfer,
    WeekPlan,
)
from .processors import (
    calculate_fixture_difficulty_score,
    get_player_fixtures,
    identify_blank_double_gameweeks,
    process_bootstrap_static,
    process_chips,
    process_fixtures,
    process_gameweeks,
    process_my_team,
    process_players,
    process_teams,
    update_gameweeks_with_blank_double,
)
from .storage import Database, get_database
from .understat import (
    UnderstatPlayer,
    UnderstatFetcher,
    UnderstatEnhancer,
    get_understat_fetcher,
    get_understat_enhancer,
    fetch_understat_xg,
)

__all__ = [
    # Models
    "Player",
    "PlayerStatus",
    "Position",
    "Team",
    "Fixture",
    "PlayerProjection",
    "SquadPlayer",
    "ChipStatus",
    "ChipType",
    "Squad",
    "Transfer",
    "WeekPlan",
    "MultiWeekPlan",
    "GameweekInfo",
    # Storage
    "Database",
    "get_database",
    # Processors
    "process_bootstrap_static",
    "process_players",
    "process_teams",
    "process_gameweeks",
    "process_fixtures",
    "process_my_team",
    "process_chips",
    "identify_blank_double_gameweeks",
    "update_gameweeks_with_blank_double",
    "get_player_fixtures",
    "calculate_fixture_difficulty_score",
    # Understat integration
    "UnderstatPlayer",
    "UnderstatFetcher",
    "UnderstatEnhancer",
    "get_understat_fetcher",
    "get_understat_enhancer",
    "fetch_understat_xg",
]
