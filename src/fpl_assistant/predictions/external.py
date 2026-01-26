"""
External Projections Import.

Import player projections from external sources like FPL Review, FPL Form, etc.
Supports CSV files with player ID mapping.
"""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..data.models import Player, PlayerProjection

logger = logging.getLogger(__name__)


class ProjectionImporter:
    """
    Import projections from external CSV files.

    Supports various formats from popular FPL projection sources.
    """

    # Known column mappings for different sources
    COLUMN_MAPPINGS = {
        "fplreview": {
            "player_id": ["id", "element", "player_id", "fpl_id"],
            "name": ["name", "player_name", "web_name"],
            "expected_points": ["xp", "expected_points", "ep", "pts"],
            "gameweek": ["gw", "gameweek", "event"],
        },
        "fplform": {
            "player_id": ["id", "element"],
            "name": ["name", "player"],
            "expected_points": ["predicted_points", "xpts"],
            "gameweek": ["gw", "round"],
        },
        "generic": {
            "player_id": ["id", "player_id", "element", "fpl_id"],
            "name": ["name", "player_name", "web_name", "player"],
            "expected_points": ["xp", "expected_points", "ep", "pts", "xpts", "predicted_points"],
            "gameweek": ["gw", "gameweek", "event", "round"],
        },
    }

    def __init__(self, players: list[Player] | None = None):
        """
        Initialize the importer.

        Args:
            players: List of players for name-to-ID matching
        """
        self.players = players or []
        self._name_to_id: dict[str, int] = {}
        self._build_name_index()

    def _build_name_index(self) -> None:
        """Build index for name-to-ID matching."""
        for player in self.players:
            # Multiple name variations for matching
            self._name_to_id[player.web_name.lower()] = player.id
            self._name_to_id[player.name.lower()] = player.id
            # Handle common variations
            name_parts = player.name.lower().split()
            if len(name_parts) > 1:
                # Last name only
                self._name_to_id[name_parts[-1]] = player.id

    def _find_column(self, headers: list[str], field: str, source: str = "generic") -> int | None:
        """Find column index for a field using known mappings."""
        mapping = self.COLUMN_MAPPINGS.get(source, self.COLUMN_MAPPINGS["generic"])
        possible_names = mapping.get(field, [])

        headers_lower = [h.lower().strip() for h in headers]

        for name in possible_names:
            if name.lower() in headers_lower:
                return headers_lower.index(name.lower())

        return None

    def _match_player_id(self, row_data: dict[str, Any], headers: list[str]) -> int | None:
        """Try to match a row to a player ID."""
        # First try direct ID match
        id_col = self._find_column(headers, "player_id")
        if id_col is not None:
            try:
                player_id = int(list(row_data.values())[id_col])
                return player_id
            except (ValueError, IndexError):
                pass

        # Fall back to name matching
        name_col = self._find_column(headers, "name")
        if name_col is not None:
            try:
                name = str(list(row_data.values())[name_col]).lower().strip()
                if name in self._name_to_id:
                    return self._name_to_id[name]
            except (ValueError, IndexError):
                pass

        return None

    def import_csv(
        self,
        file_path: str | Path,
        source: str = "generic",
        default_gameweek: int | None = None,
    ) -> list[PlayerProjection]:
        """
        Import projections from a CSV file.

        Args:
            file_path: Path to CSV file
            source: Source identifier for column mapping
            default_gameweek: Default gameweek if not in data

        Returns:
            List of PlayerProjection objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Projection file not found: {file_path}")

        projections = []

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            xp_col = self._find_column(headers, "expected_points", source)
            gw_col = self._find_column(headers, "gameweek", source)

            if xp_col is None:
                logger.warning(f"Could not find expected_points column in {file_path}")
                return []

            for row in reader:
                row_values = list(row.values())

                # Get player ID
                player_id = self._match_player_id(row, headers)
                if player_id is None:
                    continue

                # Get expected points
                try:
                    xp = float(row_values[xp_col])
                except (ValueError, IndexError):
                    continue

                # Get gameweek
                gameweek = default_gameweek
                if gw_col is not None:
                    try:
                        gameweek = int(row_values[gw_col])
                    except (ValueError, IndexError):
                        pass

                if gameweek is None:
                    logger.warning(f"No gameweek for player {player_id}, skipping")
                    continue

                projections.append(PlayerProjection(
                    player_id=player_id,
                    gameweek=gameweek,
                    expected_points=xp,
                    source=source,
                    updated_at=datetime.now(),
                ))

        logger.info(f"Imported {len(projections)} projections from {file_path}")
        return projections

    def import_fplreview(
        self,
        file_path: str | Path,
        gameweek: int,
    ) -> list[PlayerProjection]:
        """
        Import FPL Review projections.

        FPL Review CSV format typically has columns:
        id, name, team, pos, price, GW1, GW2, GW3, ... (points per GW)

        Args:
            file_path: Path to FPL Review CSV
            gameweek: Starting gameweek for projections

        Returns:
            List of projections for multiple gameweeks
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"FPL Review file not found: {file_path}")

        projections = []

        with open(file_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Find GW columns (GW1, GW2, etc. or just numbers)
            gw_columns: dict[int, str] = {}
            for h in headers:
                h_lower = h.lower()
                if h_lower.startswith("gw"):
                    try:
                        gw_num = int(h_lower.replace("gw", ""))
                        gw_columns[gw_num] = h
                    except ValueError:
                        pass
                elif h.isdigit():
                    gw_columns[int(h)] = h

            if not gw_columns:
                logger.warning("No gameweek columns found in FPL Review file")
                return self.import_csv(file_path, "fplreview", gameweek)

            for row in reader:
                # Get player ID
                player_id = None
                if "id" in row:
                    try:
                        player_id = int(row["id"])
                    except ValueError:
                        pass

                if player_id is None:
                    # Try name matching
                    name = row.get("name", row.get("Name", "")).lower().strip()
                    player_id = self._name_to_id.get(name)

                if player_id is None:
                    continue

                # Extract projections for each gameweek
                for gw_num, col_name in gw_columns.items():
                    try:
                        xp = float(row[col_name])
                        projections.append(PlayerProjection(
                            player_id=player_id,
                            gameweek=gw_num,
                            expected_points=xp,
                            source="fplreview",
                            updated_at=datetime.now(),
                        ))
                    except (ValueError, KeyError):
                        continue

        logger.info(f"Imported {len(projections)} FPL Review projections")
        return projections


def import_projections_from_directory(
    directory: str | Path,
    players: list[Player],
    gameweek: int,
) -> list[PlayerProjection]:
    """
    Import all projection files from a directory.

    Args:
        directory: Directory containing projection CSV files
        players: List of players for ID matching
        gameweek: Default gameweek for projections

    Returns:
        Combined list of projections from all files
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Projections directory not found: {directory}")
        return []

    importer = ProjectionImporter(players)
    all_projections = []

    for csv_file in directory.glob("*.csv"):
        try:
            # Detect source from filename
            filename = csv_file.stem.lower()
            if "fplreview" in filename or "review" in filename:
                projections = importer.import_fplreview(csv_file, gameweek)
            elif "fplform" in filename or "form" in filename:
                projections = importer.import_csv(csv_file, "fplform", gameweek)
            else:
                projections = importer.import_csv(csv_file, "generic", gameweek)

            all_projections.extend(projections)
            logger.info(f"Imported {len(projections)} projections from {csv_file.name}")

        except Exception as e:
            logger.error(f"Error importing {csv_file}: {e}")

    return all_projections
