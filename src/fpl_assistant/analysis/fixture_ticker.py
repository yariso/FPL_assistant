"""
Fixture Ticker Module.

Provides fixture run analysis for all Premier League teams,
helping identify teams with favorable upcoming schedules.
"""

from dataclasses import dataclass, field
from typing import Literal

from fpl_assistant.data.models import Fixture, GameweekInfo, Team


@dataclass
class FixtureDetail:
    """Single fixture detail for a team."""

    gameweek: int
    opponent_id: int
    opponent_name: str
    is_home: bool
    fdr: int  # Fixture Difficulty Rating 1-5
    is_blank: bool = False
    is_double: bool = False  # True if team has multiple fixtures this GW


@dataclass
class FixtureRun:
    """Fixture difficulty run for a team over multiple gameweeks."""

    team_id: int
    team_name: str
    team_short_name: str
    fixtures: list[FixtureDetail] = field(default_factory=list)

    @property
    def fdr_list(self) -> list[int | None]:
        """Get FDR values (None for blanks)."""
        return [f.fdr if not f.is_blank else None for f in self.fixtures]

    @property
    def avg_fdr(self) -> float:
        """Average fixture difficulty (excluding blanks)."""
        fdrs = [f.fdr for f in self.fixtures if not f.is_blank]
        return sum(fdrs) / len(fdrs) if fdrs else 3.0

    @property
    def green_count(self) -> int:
        """Count of easy fixtures (FDR 1-2)."""
        return sum(1 for f in self.fixtures if not f.is_blank and f.fdr <= 2)

    @property
    def red_count(self) -> int:
        """Count of hard fixtures (FDR 4-5)."""
        return sum(1 for f in self.fixtures if not f.is_blank and f.fdr >= 4)

    @property
    def blank_count(self) -> int:
        """Count of blank gameweeks."""
        return sum(1 for f in self.fixtures if f.is_blank)

    @property
    def double_count(self) -> int:
        """Count of double gameweeks."""
        return sum(1 for f in self.fixtures if f.is_double)

    @property
    def attack_score(self) -> float:
        """
        Score for attacking potential (lower FDR = better for scoring).
        Scale: 0-100 where higher is better.
        """
        if not self.fixtures:
            return 50.0
        # Invert FDR so lower difficulty = higher score
        # FDR 1 = 100, FDR 5 = 0
        scores = [(5 - f.fdr) * 25 for f in self.fixtures if not f.is_blank]
        return sum(scores) / len(scores) if scores else 50.0

    @property
    def defense_score(self) -> float:
        """
        Score for clean sheet potential (lower FDR = better for CS).
        Scale: 0-100 where higher is better.
        Same as attack_score for now - could be refined with team-specific data.
        """
        return self.attack_score


class FixtureTicker:
    """
    Fixture analysis tool for planning transfers.

    Generates color-coded fixture difficulty runs for all teams
    to help identify favorable schedules.
    """

    def __init__(
        self,
        teams: list[Team],
        fixtures: list[Fixture],
        gameweeks: list[GameweekInfo],
    ):
        self.teams = {t.id: t for t in teams}
        self.fixtures = fixtures
        self.gameweeks = {gw.id: gw for gw in gameweeks}

    def get_team_run(
        self,
        team_id: int,
        start_gw: int,
        num_weeks: int = 6,
    ) -> FixtureRun:
        """
        Get fixture run for a single team.

        Args:
            team_id: Team to analyze
            start_gw: Starting gameweek
            num_weeks: Number of weeks to look ahead
        """
        team = self.teams.get(team_id)
        if not team:
            raise ValueError(f"Team {team_id} not found")

        run = FixtureRun(
            team_id=team_id,
            team_name=team.name,
            team_short_name=team.short_name,
        )

        for gw in range(start_gw, start_gw + num_weeks):
            gw_fixtures = [
                f for f in self.fixtures
                if f.gameweek == gw and (f.home_team_id == team_id or f.away_team_id == team_id)
            ]

            if not gw_fixtures:
                # Blank gameweek
                run.fixtures.append(FixtureDetail(
                    gameweek=gw,
                    opponent_id=0,
                    opponent_name="BLANK",
                    is_home=False,
                    fdr=0,
                    is_blank=True,
                ))
            else:
                for i, fixture in enumerate(gw_fixtures):
                    is_home = fixture.home_team_id == team_id
                    opponent_id = fixture.away_team_id if is_home else fixture.home_team_id
                    opponent = self.teams.get(opponent_id)
                    fdr = fixture.home_difficulty if is_home else fixture.away_difficulty

                    run.fixtures.append(FixtureDetail(
                        gameweek=gw,
                        opponent_id=opponent_id,
                        opponent_name=opponent.short_name if opponent else "???",
                        is_home=is_home,
                        fdr=fdr,
                        is_double=len(gw_fixtures) > 1,
                    ))

        return run

    def get_all_team_runs(
        self,
        start_gw: int,
        num_weeks: int = 6,
    ) -> list[FixtureRun]:
        """
        Get fixture runs for all teams, sorted by avg FDR (best first).
        """
        runs = [
            self.get_team_run(team_id, start_gw, num_weeks)
            for team_id in self.teams
        ]
        return sorted(runs, key=lambda r: r.avg_fdr)

    def get_best_attack_fixtures(
        self,
        start_gw: int,
        num_weeks: int = 6,
        top_n: int = 5,
    ) -> list[FixtureRun]:
        """
        Get teams with best fixtures for attacking returns.
        """
        runs = self.get_all_team_runs(start_gw, num_weeks)
        return sorted(runs, key=lambda r: -r.attack_score)[:top_n]

    def get_best_defense_fixtures(
        self,
        start_gw: int,
        num_weeks: int = 6,
        top_n: int = 5,
    ) -> list[FixtureRun]:
        """
        Get teams with best fixtures for clean sheets.
        """
        runs = self.get_all_team_runs(start_gw, num_weeks)
        return sorted(runs, key=lambda r: -r.defense_score)[:top_n]

    def get_teams_with_doubles(
        self,
        start_gw: int,
        num_weeks: int = 6,
    ) -> list[FixtureRun]:
        """Get teams that have double gameweeks in the range."""
        runs = self.get_all_team_runs(start_gw, num_weeks)
        return [r for r in runs if r.double_count > 0]

    def get_teams_with_blanks(
        self,
        start_gw: int,
        num_weeks: int = 6,
    ) -> list[FixtureRun]:
        """Get teams that have blank gameweeks in the range."""
        runs = self.get_all_team_runs(start_gw, num_weeks)
        return [r for r in runs if r.blank_count > 0]


def get_fdr_color(fdr: int | None) -> str:
    """
    Get color class for FDR value (for UI display).

    Returns CSS-compatible color name.
    """
    if fdr is None or fdr == 0:
        return "#808080"  # Gray for blank

    colors = {
        1: "#00FF00",  # Bright green - very easy
        2: "#90EE90",  # Light green - easy
        3: "#FFFF00",  # Yellow - medium
        4: "#FF6B6B",  # Light red - hard
        5: "#FF0000",  # Bright red - very hard
    }
    return colors.get(fdr, "#FFFF00")


def get_fdr_emoji(fdr: int | None) -> str:
    """Get emoji for FDR value."""
    if fdr is None or fdr == 0:
        return "⬜"  # Blank

    emojis = {
        1: "🟢",  # Very easy
        2: "🟩",  # Easy
        3: "🟨",  # Medium
        4: "🟧",  # Hard
        5: "🔴",  # Very hard
    }
    return emojis.get(fdr, "🟨")
