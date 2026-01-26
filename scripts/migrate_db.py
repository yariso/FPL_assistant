#!/usr/bin/env python3
"""
Database migration script to add new elite features tables.
Run this once to add the new tables without losing existing data.
"""

import sqlite3
from pathlib import Path

# New tables to add
NEW_TABLES = """
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
    PRIMARY KEY (player_id, gameweek, recorded_date)
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
    PRIMARY KEY (player_id, gameweek)
);

-- Price history (track all price changes)
CREATE TABLE IF NOT EXISTS price_history (
    player_id INTEGER NOT NULL,
    recorded_at TIMESTAMP NOT NULL,
    price REAL NOT NULL,
    transfers_in_week INTEGER DEFAULT 0,
    transfers_out_week INTEGER DEFAULT 0,
    ownership_percent REAL DEFAULT 0,
    PRIMARY KEY (player_id, recorded_at)
);

-- Price change predictions
CREATE TABLE IF NOT EXISTS price_predictions (
    player_id INTEGER NOT NULL,
    prediction_date DATE NOT NULL,
    predicted_change REAL NOT NULL,
    confidence REAL DEFAULT 0.5,
    threshold_distance REAL DEFAULT 0,
    PRIMARY KEY (player_id, prediction_date)
);

-- Mini-league rivals
CREATE TABLE IF NOT EXISTS rivals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    manager_id INTEGER NOT NULL UNIQUE,
    name TEXT NOT NULL,
    league_name TEXT DEFAULT '',
    priority INTEGER DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Rival squad snapshots
CREATE TABLE IF NOT EXISTS rival_squads (
    rival_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    player_ids TEXT NOT NULL,
    captain_id INTEGER DEFAULT NULL,
    chip_used TEXT DEFAULT NULL,
    total_points INTEGER DEFAULT 0,
    gameweek_points INTEGER DEFAULT 0,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (rival_id, gameweek)
);

-- Player xG stats (for external data)
CREATE TABLE IF NOT EXISTS player_xg_stats (
    player_id INTEGER NOT NULL,
    gameweek INTEGER NOT NULL,
    xg REAL DEFAULT 0.0,
    xa REAL DEFAULT 0.0,
    npxg REAL DEFAULT 0.0,
    xg_per_90 REAL DEFAULT 0.0,
    xa_per_90 REAL DEFAULT 0.0,
    source TEXT DEFAULT 'understat',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (player_id, gameweek, source)
);

-- Adaptive weight history
CREATE TABLE IF NOT EXISTS adaptive_weights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    form_weight REAL NOT NULL,
    ict_weight REAL NOT NULL,
    fdr_weight REAL NOT NULL,
    consistency_weight REAL NOT NULL,
    team_strength_weight REAL NOT NULL,
    mae REAL,
    correlation REAL,
    top_10_hit_rate REAL,
    captain_accuracy REAL,
    gameweeks_tested TEXT,
    tested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

def migrate():
    """Run the migration."""
    db_path = Path(__file__).parent.parent / "data" / "fpl.db"

    if not db_path.exists():
        print(f"Database not found at {db_path}")
        print("Run 'Update' in the app to create the database first.")
        return

    print(f"Migrating database: {db_path}")

    # Use timeout to wait for any locks to clear
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()

    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL")

    # Execute each statement
    for statement in NEW_TABLES.split(";"):
        statement = statement.strip()
        if statement:
            try:
                cursor.execute(statement)
                # Extract table name for logging
                if "CREATE TABLE" in statement:
                    table_name = statement.split("EXISTS")[1].split("(")[0].strip()
                    print(f"  Created/verified table: {table_name}")
            except Exception as e:
                print(f"  Error: {e}")

    conn.commit()
    conn.close()

    print("\nMigration complete!")
    print("Restart the app to use the new features.")

if __name__ == "__main__":
    migrate()
