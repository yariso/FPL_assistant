#!/usr/bin/env python3
"""
Database setup script for FPL Assistant.

Run this script to initialize the SQLite database with the required schema.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fpl_assistant.data.storage import Database


def main() -> None:
    """Initialize the database."""
    db_path = Path(__file__).parent.parent / "data" / "fpl.db"
    print(f"Initializing database at: {db_path}")

    # Create data directory if it doesn't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database (creates schema)
    db = Database(db_path)

    # Verify tables were created
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Created {len(tables)} tables:")
    for table in tables:
        print(f"  - {table}")

    print("\nDatabase initialized successfully!")


if __name__ == "__main__":
    main()
