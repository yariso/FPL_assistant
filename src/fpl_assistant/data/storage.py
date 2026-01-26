"""
SQLite database storage operations for FPL Assistant.

Handles persistence of player data, fixtures, projections, and user squad state.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from .models import (
    ChipStatus,
    ChipType,
    Fixture,
    GameweekInfo,
    Player,
    PlayerProjection,
    PlayerStatus,
    Position,
    Squad,
    SquadPlayer,
    Team,
)


# =============================================================================
# Database Schema
# =============================================================================

SCHEMA = """
-- Players table
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    web_name TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    price REAL NOT NULL,
    status TEXT DEFAULT 'a',
    news TEXT DEFAULT '',
    chance_of_playing INTEGER,
    total_points INTEGER DEFAULT 0,
    points_per_game REAL DEFAULT 0.0,
    form REAL DEFAULT 0.0,
    selected_by_percent REAL DEFAULT 0.0,
    ict_index REAL DEFAULT 0.0,
    goals_scored INTEGER DEFAULT 0,
    assists INTEGER DEFAULT 0,
    clean_sheets INTEGER DEFAULT 0,
    minutes INTEGER DEFAULT 0,
    expected_goals REAL DEFAULT 0.0,
    expected_assists REAL DEFAULT 0.0,
    expected_goal_involvements REAL DEFAULT 0.0,
    expected_goals_per_90 REAL DEFAULT 0.0,
    expected_assists_per_90 REAL DEFAULT 0.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

-- Teams table
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    short_name TEXT NOT NULL,
    strength_home INTEGER DEFAULT 3,
    strength_away INTEGER DEFAULT 3,
    strength_attack_home INTEGER DEFAULT 0,
    strength_attack_away INTEGER DEFAULT 0,
    strength_defence_home INTEGER DEFAULT 0,
    strength_defence_away INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fixtures table
CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    gameweek INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_difficulty INTEGER DEFAULT 3,
    away_difficulty INTEGER DEFAULT 3,
    kickoff_time TIMESTAMP,
    finished INTEGER DEFAULT 0,
    home_score INTEGER,
    away_score INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (home_team_id) REFERENCES teams(id),
    FOREIGN KEY (away_team_id) REFERENCES teams(id)
);

-- Projections table (expected points per player per gameweek)
CREATE TABLE IF NOT EXISTS projections (
    player_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    expected_points REAL NOT NULL,
    minutes_probability REAL DEFAULT 1.0,
    source TEXT DEFAULT 'internal',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek, source),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- User squad state
CREATE TABLE IF NOT EXISTS user_squad (
    player_id INTEGER PRIMARY KEY,
    position INTEGER NOT NULL,
    is_captain INTEGER DEFAULT 0,
    is_vice_captain INTEGER DEFAULT 0,
    purchase_price REAL NOT NULL,
    selling_price REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- User state (bank, free transfers, etc.)
CREATE TABLE IF NOT EXISTS user_state (
    id INTEGER PRIMARY KEY DEFAULT 1,
    bank REAL DEFAULT 0.0,
    free_transfers INTEGER DEFAULT 1,
    total_value REAL DEFAULT 100.0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Chip availability
CREATE TABLE IF NOT EXISTS chips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chip_type TEXT NOT NULL,
    used_gameweek INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gameweek info
CREATE TABLE IF NOT EXISTS gameweeks (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    deadline TIMESTAMP NOT NULL,
    is_current INTEGER DEFAULT 0,
    is_next INTEGER DEFAULT 0,
    finished INTEGER DEFAULT 0,
    is_blank INTEGER DEFAULT 0,
    is_double INTEGER DEFAULT 0,
    blank_teams TEXT DEFAULT '[]',
    double_teams TEXT DEFAULT '[]',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Saved plans
CREATE TABLE IF NOT EXISTS plans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    horizon INTEGER NOT NULL,
    total_expected_points REAL DEFAULT 0.0,
    total_hits INTEGER DEFAULT 0,
    starting_squad TEXT NOT NULL,
    parameters TEXT DEFAULT '{}',
    week_plans TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- ELITE FEATURES: Ownership Tracking
-- =============================================================================

-- Ownership history (track daily changes for price predictions and EO analysis)
CREATE TABLE IF NOT EXISTS ownership_history (
    player_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    recorded_date DATE NOT NULL,
    selected_by_percent REAL NOT NULL,
    transfers_in INTEGER DEFAULT 0,
    transfers_out INTEGER DEFAULT 0,
    transfers_in_event INTEGER DEFAULT 0,
    transfers_out_event INTEGER DEFAULT 0,
    PRIMARY KEY (player_id, gameweek, recorded_date),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Captain effective ownership (for differential analysis)
CREATE TABLE IF NOT EXISTS captain_eo (
    player_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    overall_eo REAL NOT NULL,
    top_10k_eo REAL DEFAULT NULL,
    top_1k_eo REAL DEFAULT NULL,
    regular_ownership REAL NOT NULL,
    captain_delta REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- =============================================================================
-- ELITE FEATURES: Price Change Tracking
-- =============================================================================

-- Price history (track all price changes)
CREATE TABLE IF NOT EXISTS price_history (
    player_id INTEGER NOT NULL,
    recorded_at TIMESTAMP NOT NULL,
    price REAL NOT NULL,
    transfers_in_week INTEGER DEFAULT 0,
    transfers_out_week INTEGER DEFAULT 0,
    ownership_percent REAL DEFAULT 0,
    PRIMARY KEY (player_id, recorded_at),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Price predictions (forecast rises/falls)
CREATE TABLE IF NOT EXISTS price_predictions (
    player_id INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_change REAL NOT NULL,
    confidence REAL DEFAULT 0.5,
    net_transfers INTEGER NOT NULL,
    threshold_distance REAL NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, prediction_date),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- =============================================================================
-- ELITE FEATURES: Mini-League Rival Tracking
-- =============================================================================

-- Rivals to track
CREATE TABLE IF NOT EXISTS rivals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manager_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    team_name TEXT,
    league_name TEXT,
    priority INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rival squad snapshots
CREATE TABLE IF NOT EXISTS rival_squads (
    rival_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    player_ids TEXT NOT NULL,
    captain_id INTEGER NOT NULL,
    vice_captain_id INTEGER,
    chip_used TEXT,
    total_points INTEGER DEFAULT 0,
    gameweek_points INTEGER DEFAULT 0,
    overall_rank INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rival_id, gameweek),
    FOREIGN KEY (rival_id) REFERENCES rivals(id)
);

-- =============================================================================
-- ELITE FEATURES: xG/xA Stats
-- =============================================================================

-- Player expected stats from external sources
CREATE TABLE IF NOT EXISTS player_xg_stats (
    player_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    xg REAL DEFAULT 0.0,
    xa REAL DEFAULT 0.0,
    xgi REAL DEFAULT 0.0,
    npxg REAL DEFAULT 0.0,
    shots INTEGER DEFAULT 0,
    key_passes INTEGER DEFAULT 0,
    source TEXT DEFAULT 'understat',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek, source),
    FOREIGN KEY (player_id) REFERENCES players(id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_fixtures_gameweek ON fixtures(gameweek);
CREATE INDEX IF NOT EXISTS idx_projections_gameweek ON projections(gameweek);
CREATE INDEX IF NOT EXISTS idx_projections_player ON projections(player_id);
CREATE INDEX IF NOT EXISTS idx_ownership_history_player ON ownership_history(player_id);
CREATE INDEX IF NOT EXISTS idx_ownership_history_gw ON ownership_history(gameweek);
CREATE INDEX IF NOT EXISTS idx_price_history_player ON price_history(player_id);
CREATE INDEX IF NOT EXISTS idx_rival_squads_gw ON rival_squads(gameweek);
CREATE INDEX IF NOT EXISTS idx_player_xg_gw ON player_xg_stats(gameweek);
"""


# =============================================================================
# Database Connection Management
# =============================================================================


class Database:
    """SQLite database manager for FPL Assistant."""

    def __init__(self, db_path: str | Path = "data/fpl.db"):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self.connection() as conn:
            conn.executescript(SCHEMA)

            # Migration: Add xG columns to existing databases
            xg_migrations = [
                "ALTER TABLE players ADD COLUMN expected_goals REAL DEFAULT 0.0",
                "ALTER TABLE players ADD COLUMN expected_assists REAL DEFAULT 0.0",
                "ALTER TABLE players ADD COLUMN expected_goal_involvements REAL DEFAULT 0.0",
                "ALTER TABLE players ADD COLUMN expected_goals_per_90 REAL DEFAULT 0.0",
                "ALTER TABLE players ADD COLUMN expected_assists_per_90 REAL DEFAULT 0.0",
            ]
            for migration in xg_migrations:
                try:
                    conn.execute(migration)
                except sqlite3.OperationalError:
                    pass  # Column already exists

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # =========================================================================
    # Player Operations
    # =========================================================================

    def upsert_player(self, player: Player) -> None:
        """Insert or update a player."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO players
                (id, name, web_name, team_id, position, price, status, news,
                 chance_of_playing, total_points, points_per_game, form,
                 selected_by_percent, ict_index, goals_scored, assists,
                 clean_sheets, minutes, expected_goals, expected_assists,
                 expected_goal_involvements, expected_goals_per_90,
                 expected_assists_per_90, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    player.id,
                    player.name,
                    player.web_name,
                    player.team_id,
                    player.position.value,
                    player.price,
                    player.status.value,
                    player.news,
                    player.chance_of_playing,
                    player.total_points,
                    player.points_per_game,
                    player.form,
                    player.selected_by_percent,
                    player.ict_index,
                    player.goals_scored,
                    player.assists,
                    player.clean_sheets,
                    player.minutes,
                    player.expected_goals,
                    player.expected_assists,
                    player.expected_goal_involvements,
                    player.expected_goals_per_90,
                    player.expected_assists_per_90,
                    datetime.now(),
                ),
            )

    def upsert_players(self, players: list[Player]) -> None:
        """Bulk insert or update players."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO players
                (id, name, web_name, team_id, position, price, status, news,
                 chance_of_playing, total_points, points_per_game, form,
                 selected_by_percent, ict_index, goals_scored, assists,
                 clean_sheets, minutes, expected_goals, expected_assists,
                 expected_goal_involvements, expected_goals_per_90,
                 expected_assists_per_90, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.id,
                        p.name,
                        p.web_name,
                        p.team_id,
                        p.position.value,
                        p.price,
                        p.status.value,
                        p.news,
                        p.chance_of_playing,
                        p.total_points,
                        p.points_per_game,
                        p.form,
                        p.selected_by_percent,
                        p.ict_index,
                        p.goals_scored,
                        p.assists,
                        p.clean_sheets,
                        p.minutes,
                        p.expected_goals,
                        p.expected_assists,
                        p.expected_goal_involvements,
                        p.expected_goals_per_90,
                        p.expected_assists_per_90,
                        datetime.now(),
                    )
                    for p in players
                ],
            )

    def get_player(self, player_id: int) -> Player | None:
        """Get a player by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM players WHERE id = ?", (player_id,)
            ).fetchone()
            if row:
                return self._row_to_player(row)
            return None

    def get_all_players(self) -> list[Player]:
        """Get all players."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM players ORDER BY id").fetchall()
            return [self._row_to_player(row) for row in rows]

    def get_players_by_team(self, team_id: int) -> list[Player]:
        """Get all players for a team."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM players WHERE team_id = ? ORDER BY position, price DESC",
                (team_id,),
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def get_players_by_position(self, position: Position) -> list[Player]:
        """Get all players of a position."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM players WHERE position = ? ORDER BY price DESC",
                (position.value,),
            ).fetchall()
            return [self._row_to_player(row) for row in rows]

    def _row_to_player(self, row: sqlite3.Row) -> Player:
        """Convert database row to Player model."""
        return Player(
            id=row["id"],
            name=row["name"],
            web_name=row["web_name"],
            team_id=row["team_id"],
            position=Position(row["position"]),
            price=row["price"],
            status=PlayerStatus(row["status"]),
            news=row["news"] or "",
            chance_of_playing=row["chance_of_playing"],
            total_points=row["total_points"],
            points_per_game=row["points_per_game"],
            form=row["form"],
            selected_by_percent=row["selected_by_percent"],
            ict_index=row["ict_index"],
            goals_scored=row["goals_scored"],
            assists=row["assists"],
            clean_sheets=row["clean_sheets"],
            minutes=row["minutes"],
            expected_goals=row["expected_goals"] or 0.0,
            expected_assists=row["expected_assists"] or 0.0,
            expected_goal_involvements=row["expected_goal_involvements"] or 0.0,
            expected_goals_per_90=row["expected_goals_per_90"] or 0.0,
            expected_assists_per_90=row["expected_assists_per_90"] or 0.0,
        )

    # =========================================================================
    # Team Operations
    # =========================================================================

    def upsert_team(self, team: Team) -> None:
        """Insert or update a team."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO teams
                (id, name, short_name, strength_home, strength_away,
                 strength_attack_home, strength_attack_away,
                 strength_defence_home, strength_defence_away, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team.id,
                    team.name,
                    team.short_name,
                    team.strength_home,
                    team.strength_away,
                    team.strength_attack_home,
                    team.strength_attack_away,
                    team.strength_defence_home,
                    team.strength_defence_away,
                    datetime.now(),
                ),
            )

    def upsert_teams(self, teams: list[Team]) -> None:
        """Bulk insert or update teams."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO teams
                (id, name, short_name, strength_home, strength_away,
                 strength_attack_home, strength_attack_away,
                 strength_defence_home, strength_defence_away, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        t.id,
                        t.name,
                        t.short_name,
                        t.strength_home,
                        t.strength_away,
                        t.strength_attack_home,
                        t.strength_attack_away,
                        t.strength_defence_home,
                        t.strength_defence_away,
                        datetime.now(),
                    )
                    for t in teams
                ],
            )

    def get_team(self, team_id: int) -> Team | None:
        """Get a team by ID."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM teams WHERE id = ?", (team_id,)).fetchone()
            if row:
                return Team(
                    id=row["id"],
                    name=row["name"],
                    short_name=row["short_name"],
                    strength_home=row["strength_home"],
                    strength_away=row["strength_away"],
                    strength_attack_home=row["strength_attack_home"],
                    strength_attack_away=row["strength_attack_away"],
                    strength_defence_home=row["strength_defence_home"],
                    strength_defence_away=row["strength_defence_away"],
                )
            return None

    def get_all_teams(self) -> list[Team]:
        """Get all teams."""
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM teams ORDER BY id").fetchall()
            return [
                Team(
                    id=row["id"],
                    name=row["name"],
                    short_name=row["short_name"],
                    strength_home=row["strength_home"],
                    strength_away=row["strength_away"],
                    strength_attack_home=row["strength_attack_home"],
                    strength_attack_away=row["strength_attack_away"],
                    strength_defence_home=row["strength_defence_home"],
                    strength_defence_away=row["strength_defence_away"],
                )
                for row in rows
            ]

    # =========================================================================
    # Fixture Operations
    # =========================================================================

    def upsert_fixture(self, fixture: Fixture) -> None:
        """Insert or update a fixture."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO fixtures
                (id, gameweek, home_team_id, away_team_id, home_difficulty,
                 away_difficulty, kickoff_time, finished, home_score, away_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fixture.id,
                    fixture.gameweek,
                    fixture.home_team_id,
                    fixture.away_team_id,
                    fixture.home_difficulty,
                    fixture.away_difficulty,
                    fixture.kickoff_time,
                    int(fixture.finished),
                    fixture.home_score,
                    fixture.away_score,
                    datetime.now(),
                ),
            )

    def upsert_fixtures(self, fixtures: list[Fixture]) -> None:
        """Bulk insert or update fixtures."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO fixtures
                (id, gameweek, home_team_id, away_team_id, home_difficulty,
                 away_difficulty, kickoff_time, finished, home_score, away_score, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        f.id,
                        f.gameweek,
                        f.home_team_id,
                        f.away_team_id,
                        f.home_difficulty,
                        f.away_difficulty,
                        f.kickoff_time,
                        int(f.finished),
                        f.home_score,
                        f.away_score,
                        datetime.now(),
                    )
                    for f in fixtures
                ],
            )

    def get_fixtures_by_gameweek(self, gameweek: int) -> list[Fixture]:
        """Get all fixtures for a gameweek."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM fixtures WHERE gameweek = ? ORDER BY kickoff_time",
                (gameweek,),
            ).fetchall()
            return [self._row_to_fixture(row) for row in rows]

    def get_fixtures_for_team(
        self, team_id: int, start_gw: int, end_gw: int
    ) -> list[Fixture]:
        """Get fixtures for a team in a gameweek range."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM fixtures
                WHERE (home_team_id = ? OR away_team_id = ?)
                AND gameweek BETWEEN ? AND ?
                ORDER BY gameweek, kickoff_time
                """,
                (team_id, team_id, start_gw, end_gw),
            ).fetchall()
            return [self._row_to_fixture(row) for row in rows]

    def get_all_fixtures(self) -> list[Fixture]:
        """Get all fixtures."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM fixtures ORDER BY gameweek, kickoff_time"
            ).fetchall()
            return [self._row_to_fixture(row) for row in rows]

    def _row_to_fixture(self, row: sqlite3.Row) -> Fixture:
        """Convert database row to Fixture model."""
        kickoff = None
        if row["kickoff_time"]:
            kickoff = datetime.fromisoformat(row["kickoff_time"])
        return Fixture(
            id=row["id"],
            gameweek=row["gameweek"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            home_difficulty=row["home_difficulty"],
            away_difficulty=row["away_difficulty"],
            kickoff_time=kickoff,
            finished=bool(row["finished"]),
            home_score=row["home_score"],
            away_score=row["away_score"],
        )

    # =========================================================================
    # Projection Operations
    # =========================================================================

    def upsert_projection(self, projection: PlayerProjection) -> None:
        """Insert or update a projection."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO projections
                (player_id, gameweek, expected_points, minutes_probability, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    projection.player_id,
                    projection.gameweek,
                    projection.expected_points,
                    projection.minutes_probability,
                    projection.source,
                    datetime.now(),
                ),
            )

    def upsert_projections(self, projections: list[PlayerProjection]) -> None:
        """Bulk insert or update projections."""
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO projections
                (player_id, gameweek, expected_points, minutes_probability, source, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.player_id,
                        p.gameweek,
                        p.expected_points,
                        p.minutes_probability,
                        p.source,
                        datetime.now(),
                    )
                    for p in projections
                ],
            )

    def get_projections_for_gameweek(
        self, gameweek: int, source: str = "internal"
    ) -> list[PlayerProjection]:
        """Get all projections for a gameweek."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM projections
                WHERE gameweek = ? AND source = ?
                ORDER BY expected_points DESC
                """,
                (gameweek, source),
            ).fetchall()
            return [
                PlayerProjection(
                    player_id=row["player_id"],
                    gameweek=row["gameweek"],
                    expected_points=row["expected_points"],
                    minutes_probability=row["minutes_probability"],
                    source=row["source"],
                )
                for row in rows
            ]

    def get_player_projections(
        self, player_id: int, start_gw: int, end_gw: int, source: str = "internal"
    ) -> list[PlayerProjection]:
        """Get projections for a player across gameweeks."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM projections
                WHERE player_id = ? AND gameweek BETWEEN ? AND ? AND source = ?
                ORDER BY gameweek
                """,
                (player_id, start_gw, end_gw, source),
            ).fetchall()
            return [
                PlayerProjection(
                    player_id=row["player_id"],
                    gameweek=row["gameweek"],
                    expected_points=row["expected_points"],
                    minutes_probability=row["minutes_probability"],
                    source=row["source"],
                )
                for row in rows
            ]

    # =========================================================================
    # User Squad Operations
    # =========================================================================

    def save_user_squad(self, squad: Squad) -> None:
        """Save user's squad state."""
        with self.connection() as conn:
            # Clear existing squad
            conn.execute("DELETE FROM user_squad")
            conn.execute("DELETE FROM user_state")
            conn.execute("DELETE FROM chips")

            # Insert squad players
            conn.executemany(
                """
                INSERT INTO user_squad
                (player_id, position, is_captain, is_vice_captain,
                 purchase_price, selling_price, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.player_id,
                        p.position,
                        int(p.is_captain),
                        int(p.is_vice_captain),
                        p.purchase_price,
                        p.selling_price,
                        datetime.now(),
                    )
                    for p in squad.players
                ],
            )

            # Insert user state
            conn.execute(
                """
                INSERT INTO user_state (id, bank, free_transfers, total_value, updated_at)
                VALUES (1, ?, ?, ?, ?)
                """,
                (squad.bank, squad.free_transfers, squad.total_value, datetime.now()),
            )

            # Insert chips
            conn.executemany(
                """
                INSERT INTO chips (chip_type, used_gameweek, updated_at)
                VALUES (?, ?, ?)
                """,
                [
                    (c.chip_type.value, c.used_gameweek, datetime.now())
                    for c in squad.chips
                ],
            )

    def get_user_squad(self) -> Squad | None:
        """Get user's current squad state."""
        with self.connection() as conn:
            # Get squad players
            player_rows = conn.execute(
                "SELECT * FROM user_squad ORDER BY position"
            ).fetchall()
            if not player_rows:
                return None

            players = [
                SquadPlayer(
                    player_id=row["player_id"],
                    position=row["position"],
                    is_captain=bool(row["is_captain"]),
                    is_vice_captain=bool(row["is_vice_captain"]),
                    purchase_price=row["purchase_price"],
                    selling_price=row["selling_price"],
                )
                for row in player_rows
            ]

            # Get user state
            state_row = conn.execute(
                "SELECT * FROM user_state WHERE id = 1"
            ).fetchone()
            if not state_row:
                return None

            # Get chips
            chip_rows = conn.execute("SELECT * FROM chips").fetchall()
            chips = [
                ChipStatus(
                    chip_type=ChipType(row["chip_type"]),
                    used_gameweek=row["used_gameweek"],
                )
                for row in chip_rows
            ]

            return Squad(
                players=players,
                bank=state_row["bank"],
                free_transfers=state_row["free_transfers"],
                total_value=state_row["total_value"],
                chips=chips,
            )

    # =========================================================================
    # Gameweek Operations
    # =========================================================================

    def upsert_gameweek(self, gw: GameweekInfo) -> None:
        """Insert or update gameweek info."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO gameweeks
                (id, name, deadline, is_current, is_next, finished,
                 is_blank, is_double, blank_teams, double_teams, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gw.id,
                    gw.name,
                    gw.deadline,
                    int(gw.is_current),
                    int(gw.is_next),
                    int(gw.finished),
                    int(gw.is_blank),
                    int(gw.is_double),
                    json.dumps(gw.blank_teams),
                    json.dumps(gw.double_teams),
                    datetime.now(),
                ),
            )

    def get_gameweek(self, gameweek_id: int) -> GameweekInfo | None:
        """Get gameweek info by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM gameweeks WHERE id = ?", (gameweek_id,)
            ).fetchone()
            if row:
                return self._row_to_gameweek(row)
            return None

    def get_current_gameweek(self) -> GameweekInfo | None:
        """Get the current gameweek."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM gameweeks WHERE is_current = 1"
            ).fetchone()
            if row:
                return self._row_to_gameweek(row)
            return None

    def get_next_gameweek(self) -> GameweekInfo | None:
        """Get the next gameweek."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM gameweeks WHERE is_next = 1"
            ).fetchone()
            if row:
                return self._row_to_gameweek(row)
            return None

    def _row_to_gameweek(self, row: sqlite3.Row) -> GameweekInfo:
        """Convert database row to GameweekInfo model."""
        return GameweekInfo(
            id=row["id"],
            name=row["name"],
            deadline=datetime.fromisoformat(row["deadline"]),
            is_current=bool(row["is_current"]),
            is_next=bool(row["is_next"]),
            finished=bool(row["finished"]),
            is_blank=bool(row["is_blank"]),
            is_double=bool(row["is_double"]),
            blank_teams=json.loads(row["blank_teams"]),
            double_teams=json.loads(row["double_teams"]),
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def get_player_count(self) -> int:
        """Get total number of players in database."""
        with self.connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM players").fetchone()
            return result[0] if result else 0

    def get_last_updated(self, table: str) -> datetime | None:
        """Get the last update time for a table."""
        with self.connection() as conn:
            result = conn.execute(
                f"SELECT MAX(updated_at) FROM {table}"  # noqa: S608
            ).fetchone()
            if result and result[0]:
                return datetime.fromisoformat(result[0])
            return None

    def clear_all_data(self) -> None:
        """Clear all data from all tables (use with caution)."""
        with self.connection() as conn:
            tables = [
                "players",
                "teams",
                "fixtures",
                "projections",
                "user_squad",
                "user_state",
                "chips",
                "gameweeks",
                "plans",
            ]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")  # noqa: S608


# =============================================================================
# Module-level convenience functions
# =============================================================================

_db: Database | None = None


def get_database(db_path: str | Path = "data/fpl.db") -> Database:
    """Get or create the database instance."""
    global _db
    if _db is None:
        _db = Database(db_path)
    return _db
