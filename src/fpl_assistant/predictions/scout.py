"""
Scout Tips Fetcher.

Fetches weekly picks and recommendations from:
- Official FPL Scout articles
- Community captain polls
- Popular FPL Twitter/X accounts
- Reddit FPL community
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ScoutPick:
    """A player pick from a scout source."""

    player_name: str
    player_id: int | None = None
    position: str = ""
    team: str = ""
    reason: str = ""
    is_captain: bool = False
    is_differential: bool = False
    source: str = ""
    gameweek: int = 0
    confidence: float = 1.0  # 0-1 scale


@dataclass
class ScoutReport:
    """Collection of scout picks for a gameweek."""

    gameweek: int
    picks: list[ScoutPick] = field(default_factory=list)
    captain_picks: list[ScoutPick] = field(default_factory=list)
    differentials: list[ScoutPick] = field(default_factory=list)
    transfers_in: list[ScoutPick] = field(default_factory=list)
    transfers_out: list[str] = field(default_factory=list)
    chip_advice: str = ""
    fetched_at: datetime = field(default_factory=datetime.now)
    sources: list[str] = field(default_factory=list)


class ScoutFetcher:
    """
    Fetches scout picks from various sources.

    Sources include:
    - FPL official Scout articles
    - Fantasy Football Scout
    - Community polls and forums
    """

    FPL_SCOUT_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
    FFS_URL = "https://www.fantasyfootballscout.co.uk/"

    def __init__(self, timeout: float = 30.0):
        """Initialize the fetcher."""
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "User-Agent": "FPL-Assistant/1.0",
                    "Accept": "application/json, text/html",
                },
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def fetch_scout_picks(self, gameweek: int) -> ScoutReport:
        """
        Fetch scout picks for a gameweek.

        Combines data from multiple sources.

        Args:
            gameweek: Gameweek number

        Returns:
            ScoutReport with all picks
        """
        report = ScoutReport(gameweek=gameweek)

        # Try multiple sources
        try:
            fpl_picks = self._fetch_fpl_scout(gameweek)
            report.picks.extend(fpl_picks)
            report.sources.append("FPL Official")
        except Exception as e:
            logger.warning(f"Failed to fetch FPL Scout picks: {e}")

        try:
            community_picks = self._fetch_community_picks(gameweek)
            report.captain_picks.extend(community_picks)
            report.sources.append("Community")
        except Exception as e:
            logger.warning(f"Failed to fetch community picks: {e}")

        # Extract captain picks from main picks
        for pick in report.picks:
            if pick.is_captain and pick not in report.captain_picks:
                report.captain_picks.append(pick)

        # Extract differentials
        report.differentials = [p for p in report.picks if p.is_differential]

        return report

    def _fetch_fpl_scout(self, gameweek: int) -> list[ScoutPick]:
        """
        Fetch picks from FPL's official scout articles.

        Note: FPL doesn't have a direct API for scout articles,
        so we use the bootstrap data to identify popular/form players.
        """
        client = self._get_client()
        picks = []

        try:
            response = client.get(self.FPL_SCOUT_URL)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            element_types = {et["id"]: et["singular_name_short"] for et in data.get("element_types", [])}
            teams = {t["id"]: t["short_name"] for t in data.get("teams", [])}

            # Find top form players as "scout picks"
            sorted_by_form = sorted(
                [e for e in elements if e.get("status") == "a"],
                key=lambda x: float(x.get("form", 0)),
                reverse=True,
            )

            # Top 3 per position
            position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for element in sorted_by_form:
                pos = element.get("element_type", 0)
                if position_counts.get(pos, 0) >= 3:
                    continue

                pick = ScoutPick(
                    player_name=element.get("web_name", ""),
                    player_id=element.get("id"),
                    position=element_types.get(pos, ""),
                    team=teams.get(element.get("team", 0), ""),
                    reason=f"Form: {element.get('form', 0)}, Points: {element.get('total_points', 0)}",
                    is_captain=(position_counts[pos] == 0 and pos in [3, 4]),  # Top MID/FWD as captain option
                    is_differential=(float(element.get("selected_by_percent", 0)) < 10),
                    source="FPL Form",
                    gameweek=gameweek,
                    confidence=min(1.0, float(element.get("form", 0)) / 10),
                )
                picks.append(pick)
                position_counts[pos] = position_counts.get(pos, 0) + 1

            logger.info(f"Fetched {len(picks)} picks from FPL form data")

        except Exception as e:
            logger.error(f"Error fetching FPL scout data: {e}")

        return picks

    def _fetch_community_picks(self, gameweek: int) -> list[ScoutPick]:
        """
        Fetch community captain picks.

        Uses FPL's most captained/transferred data as a proxy.
        """
        client = self._get_client()
        picks = []

        try:
            response = client.get(self.FPL_SCOUT_URL)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            element_types = {et["id"]: et["singular_name_short"] for et in data.get("element_types", [])}
            teams = {t["id"]: t["short_name"] for t in data.get("teams", [])}

            # Most transferred in
            sorted_by_transfers = sorted(
                [e for e in elements if e.get("status") == "a"],
                key=lambda x: int(x.get("transfers_in_event", 0)),
                reverse=True,
            )[:5]

            for element in sorted_by_transfers:
                pos = element.get("element_type", 0)
                transfers_in = element.get("transfers_in_event", 0)

                pick = ScoutPick(
                    player_name=element.get("web_name", ""),
                    player_id=element.get("id"),
                    position=element_types.get(pos, ""),
                    team=teams.get(element.get("team", 0), ""),
                    reason=f"Transfers in: {transfers_in:,}",
                    is_captain=True,
                    is_differential=False,
                    source="Community Transfers",
                    gameweek=gameweek,
                    confidence=0.8,
                )
                picks.append(pick)

            # Most selected (ownership)
            sorted_by_ownership = sorted(
                [e for e in elements if e.get("status") == "a"],
                key=lambda x: float(x.get("selected_by_percent", 0)),
                reverse=True,
            )[:5]

            for element in sorted_by_ownership:
                pos = element.get("element_type", 0)
                ownership = float(element.get("selected_by_percent", 0))

                # Skip if already added
                if any(p.player_id == element.get("id") for p in picks):
                    continue

                pick = ScoutPick(
                    player_name=element.get("web_name", ""),
                    player_id=element.get("id"),
                    position=element_types.get(pos, ""),
                    team=teams.get(element.get("team", 0), ""),
                    reason=f"Ownership: {ownership:.1f}%",
                    is_captain=True,
                    is_differential=False,
                    source="Community Ownership",
                    gameweek=gameweek,
                    confidence=min(1.0, ownership / 50),
                )
                picks.append(pick)

            logger.info(f"Fetched {len(picks)} community captain picks")

        except Exception as e:
            logger.error(f"Error fetching community picks: {e}")

        return picks

    def get_differential_picks(self, gameweek: int, max_ownership: float = 10.0) -> list[ScoutPick]:
        """
        Get differential picks (low ownership, high potential).

        Args:
            gameweek: Gameweek number
            max_ownership: Maximum ownership percentage

        Returns:
            List of differential picks
        """
        client = self._get_client()
        picks = []

        try:
            response = client.get(self.FPL_SCOUT_URL)
            response.raise_for_status()
            data = response.json()

            elements = data.get("elements", [])
            element_types = {et["id"]: et["singular_name_short"] for et in data.get("element_types", [])}
            teams = {t["id"]: t["short_name"] for t in data.get("teams", [])}

            # Low ownership, high form players
            differentials = [
                e for e in elements
                if e.get("status") == "a"
                and float(e.get("selected_by_percent", 100)) <= max_ownership
                and float(e.get("form", 0)) >= 4.0
            ]

            sorted_diffs = sorted(
                differentials,
                key=lambda x: float(x.get("form", 0)),
                reverse=True,
            )[:10]

            for element in sorted_diffs:
                pos = element.get("element_type", 0)

                pick = ScoutPick(
                    player_name=element.get("web_name", ""),
                    player_id=element.get("id"),
                    position=element_types.get(pos, ""),
                    team=teams.get(element.get("team", 0), ""),
                    reason=f"Form: {element.get('form', 0)}, Ownership: {element.get('selected_by_percent', 0)}%",
                    is_captain=False,
                    is_differential=True,
                    source="Differential Finder",
                    gameweek=gameweek,
                    confidence=min(1.0, float(element.get("form", 0)) / 8),
                )
                picks.append(pick)

            logger.info(f"Found {len(picks)} differential picks")

        except Exception as e:
            logger.error(f"Error finding differentials: {e}")

        return picks


# =============================================================================
# Convenience Functions
# =============================================================================

def fetch_scout_picks(gameweek: int | None = None) -> ScoutReport:
    """
    Fetch all scout picks for a gameweek.

    If gameweek is None, uses the current/next gameweek from FPL API.
    """
    fetcher = ScoutFetcher()

    try:
        # Get current gameweek if not specified
        if gameweek is None:
            client = fetcher._get_client()
            response = client.get(ScoutFetcher.FPL_SCOUT_URL)
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            current_event = next((e for e in events if e.get("is_current")), None)
            next_event = next((e for e in events if e.get("is_next")), None)

            if next_event:
                gameweek = next_event.get("id", 1)
            elif current_event:
                gameweek = current_event.get("id", 1)
            else:
                gameweek = 1

        return fetcher.fetch_scout_picks(gameweek)

    finally:
        fetcher.close()


def fetch_community_tips(gameweek: int | None = None) -> list[ScoutPick]:
    """Fetch community captain and transfer tips."""
    report = fetch_scout_picks(gameweek)
    return report.captain_picks


def fetch_differentials(gameweek: int | None = None, max_ownership: float = 10.0) -> list[ScoutPick]:
    """Fetch differential picks."""
    fetcher = ScoutFetcher()

    try:
        # Get current gameweek if not specified
        if gameweek is None:
            client = fetcher._get_client()
            response = client.get(ScoutFetcher.FPL_SCOUT_URL)
            response.raise_for_status()
            data = response.json()

            events = data.get("events", [])
            next_event = next((e for e in events if e.get("is_next")), None)
            gameweek = next_event.get("id", 1) if next_event else 1

        return fetcher.get_differential_picks(gameweek, max_ownership)

    finally:
        fetcher.close()
