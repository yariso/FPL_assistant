"""
Team Intelligence Module for FPL Assistant.

Aggregates deadline intelligence from multiple sources:
- FPL API injury/news data
- Press conference summaries
- Reliable journalist sources
- Team news aggregators

Provides an "information edge" by surfacing relevant news
for transfer/captain decisions before deadline.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import httpx

from ..data.models import Player, PlayerStatus

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class NewsItem:
    """A single news item about a player or team."""

    player_id: int | None = None
    player_name: str = ""
    team_name: str = ""
    headline: str = ""
    summary: str = ""
    source: str = ""
    url: str = ""
    published_at: datetime | None = None
    sentiment: str = "neutral"  # positive, negative, neutral
    impact: str = "low"  # low, medium, high, critical
    category: str = "general"  # injury, lineup, transfer, form, tactical


@dataclass
class PlayerIntelligence:
    """Aggregated intelligence for a player."""

    player_id: int
    player_name: str
    team_name: str

    # FPL API data
    fpl_news: str = ""
    chance_of_playing: int | None = None
    status: PlayerStatus = PlayerStatus.AVAILABLE

    # Aggregated news
    news_items: list[NewsItem] = field(default_factory=list)

    # Risk assessment
    rotation_risk: str = "low"  # low, medium, high
    injury_risk: str = "low"
    minutes_confidence: str = "high"  # high, medium, low

    # Flags
    has_press_conference_news: bool = False
    has_lineup_leak: bool = False
    has_injury_update: bool = False
    is_flagged: bool = False
    flag_reason: str = ""

    @property
    def risk_level(self) -> str:
        """Overall risk level based on all factors."""
        if self.chance_of_playing is not None:
            if self.chance_of_playing == 0:
                return "critical"
            elif self.chance_of_playing <= 25:
                return "high"
            elif self.chance_of_playing <= 50:
                return "medium"
            elif self.chance_of_playing <= 75:
                return "low"

        if self.status in [PlayerStatus.INJURED, PlayerStatus.SUSPENDED]:
            return "critical"
        if self.status == PlayerStatus.DOUBTFUL:
            return "medium"

        if self.rotation_risk == "high" or self.injury_risk == "high":
            return "medium"

        return "low"


@dataclass
class TeamIntelligence:
    """Intelligence for a team."""

    team_id: int
    team_name: str

    # Press conference summary
    press_conference_summary: str = ""
    press_conference_date: datetime | None = None

    # Lineup intelligence
    expected_lineup_changes: list[str] = field(default_factory=list)
    rotation_expected: bool = False
    rotation_reason: str = ""

    # Fixture context
    has_midweek_game: bool = False
    competition_context: str = ""  # "League only", "Champions League", etc.

    # Player updates from presser
    player_updates: list[str] = field(default_factory=list)


@dataclass
class IntelligenceReport:
    """Full intelligence report for a gameweek."""

    gameweek: int
    generated_at: datetime = field(default_factory=datetime.now)

    # Flagged players requiring attention
    flagged_players: list[PlayerIntelligence] = field(default_factory=list)

    # Team-level intelligence
    team_intelligence: dict[int, TeamIntelligence] = field(default_factory=dict)

    # All player intelligence
    player_intelligence: dict[int, PlayerIntelligence] = field(default_factory=dict)

    # Summary
    key_headlines: list[str] = field(default_factory=list)
    sources_checked: list[str] = field(default_factory=list)


# =============================================================================
# Intelligence Keywords for Classification
# =============================================================================

INJURY_KEYWORDS = [
    "injury", "injured", "knock", "fitness", "doubt", "doubtful",
    "scan", "treatment", "recovering", "rehabilitation", "hamstring",
    "muscle", "ankle", "knee", "groin", "calf", "back", "shoulder",
    "illness", "sick", "unwell", "assessed", "setback",
]

POSITIVE_KEYWORDS = [
    "fit", "available", "training", "trained", "ready", "recovered",
    "back", "return", "full", "clearance", "green light", "passed",
]

ROTATION_KEYWORDS = [
    "rest", "rested", "rotate", "rotation", "manage", "managed",
    "minutes", "workload", "fresh", "squad", "changes", "bench",
]

LINEUP_KEYWORDS = [
    "start", "starting", "lineup", "xi", "team sheet", "selection",
    "pick", "picked", "dropped", "benched",
]


# =============================================================================
# Intelligence Fetcher
# =============================================================================

class IntelligenceFetcher:
    """
    Fetches and aggregates team/player intelligence.

    Sources:
    - FPL API (news, chance_of_playing)
    - BBC Sport injury tables
    - Sky Sports team news
    - Press conference quotes (from aggregators)
    """

    # News source URLs
    BBC_INJURIES_URL = "https://www.bbc.co.uk/sport/football/premier-league/injuries"
    SKY_TEAM_NEWS_URL = "https://www.skysports.com/football/news"

    def __init__(self, timeout: float = 30.0):
        """Initialize the intelligence fetcher."""
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                },
                follow_redirects=True,
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def generate_report(
        self,
        players: list[Player],
        gameweek: int,
        include_web_sources: bool = True,
    ) -> IntelligenceReport:
        """
        Generate a full intelligence report for a gameweek.

        Args:
            players: List of all players
            gameweek: Current gameweek number
            include_web_sources: Whether to fetch from web sources

        Returns:
            IntelligenceReport with all aggregated intelligence
        """
        report = IntelligenceReport(gameweek=gameweek)

        # 1. Process FPL API data
        logger.info("Processing FPL API news data...")
        self._process_fpl_news(players, report)

        # 2. Fetch external sources if enabled
        if include_web_sources:
            logger.info("Fetching external intelligence sources...")
            self._fetch_bbc_injuries(report, players)

        # 3. Flag players requiring attention
        self._flag_players(report)

        # 4. Generate key headlines
        self._generate_headlines(report)

        return report

    def _process_fpl_news(
        self,
        players: list[Player],
        report: IntelligenceReport,
    ) -> None:
        """Process news from FPL API player data."""
        report.sources_checked.append("FPL Official API")

        for player in players:
            intel = PlayerIntelligence(
                player_id=player.id,
                player_name=player.web_name,
                team_name=str(player.team_id),  # Will be replaced with team name
                fpl_news=player.news,
                chance_of_playing=player.chance_of_playing,
                status=player.status,
            )

            # Analyze FPL news text
            if player.news:
                news_lower = player.news.lower()

                # Check for injury keywords
                if any(kw in news_lower for kw in INJURY_KEYWORDS):
                    intel.has_injury_update = True
                    intel.injury_risk = "high" if player.chance_of_playing and player.chance_of_playing < 75 else "medium"

                # Check for rotation keywords
                if any(kw in news_lower for kw in ROTATION_KEYWORDS):
                    intel.rotation_risk = "medium"

                # Create news item from FPL news
                sentiment = self._classify_sentiment(player.news)
                impact = "high" if player.chance_of_playing and player.chance_of_playing < 75 else "medium"

                intel.news_items.append(NewsItem(
                    player_id=player.id,
                    player_name=player.web_name,
                    headline=player.news[:100] if len(player.news) > 100 else player.news,
                    summary=player.news,
                    source="FPL Official",
                    sentiment=sentiment,
                    impact=impact,
                    category="injury" if intel.has_injury_update else "general",
                ))

            report.player_intelligence[player.id] = intel

    def _fetch_bbc_injuries(
        self,
        report: IntelligenceReport,
        players: list[Player],
    ) -> None:
        """Fetch injury data from BBC Sport."""
        try:
            client = self._get_client()
            response = client.get(self.BBC_INJURIES_URL)

            if response.status_code != 200:
                logger.warning(f"BBC injuries fetch failed: {response.status_code}")
                return

            report.sources_checked.append("BBC Sport Injuries")

            # Parse HTML for injury mentions
            html = response.text

            # Create player name -> id mapping for matching
            player_names = {p.web_name.lower(): p.id for p in players}
            full_names = {}
            for p in players:
                if hasattr(p, 'first_name') and hasattr(p, 'second_name'):
                    full_name = f"{p.first_name} {p.second_name}".lower()
                    full_names[full_name] = p.id
                    # Also try just surname
                    full_names[p.second_name.lower()] = p.id

            # Look for player mentions in injury context
            # BBC typically lists injuries in table format
            for player in players:
                if player.id not in report.player_intelligence:
                    continue

                intel = report.player_intelligence[player.id]

                # Simple check: is player name mentioned near injury words?
                player_pattern = re.escape(player.web_name)
                matches = re.findall(
                    rf'.{{0,100}}{player_pattern}.{{0,100}}',
                    html,
                    re.IGNORECASE
                )

                for match in matches:
                    match_lower = match.lower()
                    if any(kw in match_lower for kw in INJURY_KEYWORDS):
                        # Found injury mention
                        if not intel.has_injury_update:
                            intel.has_injury_update = True

                        intel.news_items.append(NewsItem(
                            player_id=player.id,
                            player_name=player.web_name,
                            headline=f"BBC: {player.web_name} injury mention",
                            summary=match.strip()[:200],
                            source="BBC Sport",
                            url=self.BBC_INJURIES_URL,
                            sentiment="negative",
                            impact="medium",
                            category="injury",
                        ))
                        break  # Only one BBC mention per player

        except Exception as e:
            logger.warning(f"Error fetching BBC injuries: {e}")

    def _classify_sentiment(self, text: str) -> str:
        """Classify sentiment of news text."""
        text_lower = text.lower()

        # Check for positive keywords
        positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)

        # Check for negative keywords
        negative_count = sum(1 for kw in INJURY_KEYWORDS if kw in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _flag_players(self, report: IntelligenceReport) -> None:
        """Flag players requiring attention before deadline."""
        for player_id, intel in report.player_intelligence.items():
            should_flag = False
            reasons = []

            # Flag if chance of playing is low
            if intel.chance_of_playing is not None:
                if intel.chance_of_playing <= 50:
                    should_flag = True
                    reasons.append(f"{intel.chance_of_playing}% chance of playing")
                elif intel.chance_of_playing <= 75:
                    should_flag = True
                    reasons.append(f"Doubtful ({intel.chance_of_playing}%)")

            # Flag if status is not available
            if intel.status in [PlayerStatus.INJURED, PlayerStatus.SUSPENDED]:
                should_flag = True
                reasons.append(f"Status: {intel.status.value}")
            elif intel.status == PlayerStatus.DOUBTFUL:
                should_flag = True
                reasons.append("Flagged as doubtful")

            # Flag if has significant news
            if intel.has_injury_update and not should_flag:
                should_flag = True
                reasons.append("Recent injury news")

            if intel.has_lineup_leak:
                should_flag = True
                reasons.append("Lineup leak suggests benching")

            if should_flag:
                intel.is_flagged = True
                intel.flag_reason = "; ".join(reasons)
                report.flagged_players.append(intel)

        # Sort flagged players by risk level
        risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        report.flagged_players.sort(key=lambda x: risk_order.get(x.risk_level, 4))

    def _generate_headlines(self, report: IntelligenceReport) -> None:
        """Generate key headlines summary."""
        headlines = []

        # Count by risk level
        critical = sum(1 for p in report.flagged_players if p.risk_level == "critical")
        high = sum(1 for p in report.flagged_players if p.risk_level == "high")
        medium = sum(1 for p in report.flagged_players if p.risk_level == "medium")

        if critical > 0:
            headlines.append(f"⛔ {critical} player(s) ruled OUT or highly doubtful")

        if high > 0:
            headlines.append(f"⚠️ {high} player(s) with significant injury concerns")

        if medium > 0:
            headlines.append(f"⚡ {medium} player(s) flagged for monitoring")

        # Add specific player mentions for critical/high
        for intel in report.flagged_players[:5]:
            if intel.risk_level in ["critical", "high"]:
                news_summary = intel.fpl_news[:50] if intel.fpl_news else intel.flag_reason
                headlines.append(f"• {intel.player_name}: {news_summary}")

        report.key_headlines = headlines


# =============================================================================
# Convenience Functions
# =============================================================================

def fetch_intelligence_report(
    players: list[Player],
    gameweek: int,
    include_web: bool = True,
) -> IntelligenceReport:
    """
    Fetch intelligence report for a gameweek.

    Args:
        players: List of all players
        gameweek: Current gameweek
        include_web: Whether to fetch from web sources

    Returns:
        IntelligenceReport
    """
    fetcher = IntelligenceFetcher()
    try:
        return fetcher.generate_report(players, gameweek, include_web)
    finally:
        fetcher.close()


def get_flagged_players(
    players: list[Player],
    gameweek: int,
) -> list[PlayerIntelligence]:
    """
    Get list of flagged players requiring attention.

    Quick access to players with injury/rotation concerns.
    """
    report = fetch_intelligence_report(players, gameweek, include_web=False)
    return report.flagged_players


def get_player_intelligence(
    player: Player,
    players: list[Player],
    gameweek: int,
) -> PlayerIntelligence | None:
    """Get intelligence for a specific player."""
    report = fetch_intelligence_report(players, gameweek, include_web=False)
    return report.player_intelligence.get(player.id)
