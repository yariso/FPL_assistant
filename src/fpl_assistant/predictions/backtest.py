"""
Prediction Backtesting.

Test predictions against actual historical results to validate the model.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..data.models import (
    Fixture,
    GameweekInfo,
    Player,
    PlayerStatus,
    Position,
    Team,
)
from .projections import ProjectionEngine

logger = logging.getLogger(__name__)


@dataclass
class PlayerResult:
    """Actual result for a player in a gameweek."""

    player_id: int
    player_name: str
    gameweek: int
    actual_points: int
    predicted_points: float
    minutes: int  # Minutes played THIS gameweek
    season_minutes: int = 0  # Total season minutes (for filtering captain candidates)
    goals: int = 0
    assists: int = 0
    clean_sheet: bool = False
    bonus: int = 0


@dataclass
class BacktestResult:
    """Results from backtesting predictions."""

    gameweeks_tested: list[int] = field(default_factory=list)
    total_predictions: int = 0
    player_results: list[PlayerResult] = field(default_factory=list)

    # Aggregate metrics
    mean_absolute_error: float = 0.0
    root_mean_square_error: float = 0.0
    correlation: float = 0.0

    # By position breakdown
    mae_by_position: dict[str, float] = field(default_factory=dict)

    # Top pick accuracy
    top_10_hit_rate: float = 0.0  # How often top 10 predicted are in actual top 10
    captain_accuracy: float = 0.0  # How often #1 captain pick was in actual top 5
    captain_top_3_rate: float = 0.0  # How often #1 captain pick was in actual top 3
    captain_exact_hit: float = 0.0  # How often #1 captain pick was THE top scorer (very hard!)


class Backtester:
    """
    Backtest predictions against actual FPL results.

    Fetches historical data and compares predictions to reality.
    """

    FPL_API = "https://fantasy.premierleague.com/api"

    def __init__(self, timeout: float = 30.0):
        """Initialize the backtester."""
        self.timeout = timeout
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Get HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": "FPL-Assistant/1.0"},
            )
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def fetch_live_gameweek(self, gameweek: int) -> dict[int, dict[str, Any]]:
        """
        Fetch actual points for a completed gameweek.

        Returns dict of player_id -> {points, minutes, goals, assists, etc.}
        """
        client = self._get_client()
        url = f"{self.FPL_API}/event/{gameweek}/live/"

        try:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()

            results = {}
            for element in data.get("elements", []):
                player_id = element.get("id")
                stats = element.get("stats", {})

                results[player_id] = {
                    "points": stats.get("total_points", 0),
                    "minutes": stats.get("minutes", 0),
                    "goals": stats.get("goals_scored", 0),
                    "assists": stats.get("assists", 0),
                    "clean_sheets": stats.get("clean_sheets", 0),
                    "bonus": stats.get("bonus", 0),
                    "saves": stats.get("saves", 0),
                    "yellow_cards": stats.get("yellow_cards", 0),
                    "red_cards": stats.get("red_cards", 0),
                }

            return results

        except Exception as e:
            logger.error(f"Failed to fetch GW{gameweek} live data: {e}")
            return {}

    def fetch_bootstrap_data(self) -> dict[str, Any]:
        """Fetch current bootstrap-static data."""
        client = self._get_client()
        url = f"{self.FPL_API}/bootstrap-static/"

        response = client.get(url)
        response.raise_for_status()
        return response.json()

    def run_backtest(
        self,
        gameweeks: list[int],
        use_current_data: bool = True,
        custom_weights: dict[str, float] | None = None,
    ) -> BacktestResult:
        """
        Run backtest for specified gameweeks.

        Args:
            gameweeks: List of gameweeks to test (must be completed)
            use_current_data: Whether to use current player stats for projection
            custom_weights: Optional dict of custom weight overrides for projections

        Returns:
            BacktestResult with all metrics
        """
        result = BacktestResult(gameweeks_tested=gameweeks)

        # Fetch current data
        logger.info("Fetching bootstrap data...")
        bootstrap = self.fetch_bootstrap_data()

        # Parse players
        players = self._parse_players(bootstrap)
        teams = self._parse_teams(bootstrap)
        fixtures = self._parse_fixtures()

        # Create projection engine with optional custom weights
        engine = ProjectionEngine(
            players, teams, fixtures,
            use_adaptive_weights=custom_weights is None,
            custom_weights=custom_weights,
        )

        all_errors = []
        position_errors: dict[str, list[float]] = {
            "GK": [], "DEF": [], "MID": [], "FWD": [],
        }
        top_10_hits = 0
        top_10_total = 0
        captain_in_top_5 = 0
        captain_in_top_3 = 0
        captain_exact_hits = 0
        captain_total = 0

        for gw in gameweeks:
            logger.info(f"Testing gameweek {gw}...")

            # Fetch actual results
            actual_results = self.fetch_live_gameweek(gw)
            if not actual_results:
                logger.warning(f"No results for GW{gw}, skipping")
                continue

            # Generate predictions
            predictions = {}
            for player in players:
                try:
                    xp = engine.project_single_player(player, gw)
                    predictions[player.id] = xp
                except Exception as e:
                    logger.debug(f"Could not project {player.web_name}: {e}")

            # Compare predictions to actual
            gw_results = []
            for player_id, actual in actual_results.items():
                if player_id not in predictions:
                    continue

                player = next((p for p in players if p.id == player_id), None)
                if not player:
                    continue

                predicted = predictions[player_id]
                actual_points = actual["points"]

                player_result = PlayerResult(
                    player_id=player_id,
                    player_name=player.web_name,
                    gameweek=gw,
                    actual_points=actual_points,
                    predicted_points=predicted,
                    minutes=actual["minutes"],
                    season_minutes=player.minutes,  # Total season minutes from Player object
                    goals=actual["goals"],
                    assists=actual["assists"],
                    clean_sheet=actual["clean_sheets"] > 0,
                    bonus=actual["bonus"],
                )
                result.player_results.append(player_result)
                gw_results.append(player_result)

                # Track errors
                error = abs(actual_points - predicted)
                all_errors.append(error)
                position_errors[player.position.name].append(error)

            # Check top 10 accuracy
            if gw_results:
                # Filter to players with reasonable SEASON minutes (180+ = 2 full games)
                # to avoid inflated per-90 stats from low-minutes players
                # Note: r.season_minutes is SEASON total, not gameweek minutes
                captainable_results = [r for r in gw_results if r.season_minutes >= 180]

                # Use all results for top 10 comparison
                by_predicted = sorted(gw_results, key=lambda x: x.predicted_points, reverse=True)[:10]
                by_actual = sorted(gw_results, key=lambda x: x.actual_points, reverse=True)[:10]

                predicted_ids = {r.player_id for r in by_predicted}
                actual_ids = {r.player_id for r in by_actual}

                hits = len(predicted_ids & actual_ids)
                top_10_hits += hits
                top_10_total += 10

                # Captain accuracy - use captainable players only
                # Sort by predicted (captain picks)
                cap_by_predicted = sorted(captainable_results, key=lambda x: x.predicted_points, reverse=True)
                # Sort by actual
                cap_by_actual = sorted(captainable_results, key=lambda x: x.actual_points, reverse=True)

                if cap_by_predicted and len(cap_by_actual) >= 5:
                    top_pick_id = cap_by_predicted[0].player_id
                    actual_top_5_ids = {r.player_id for r in cap_by_actual[:5]}
                    actual_top_3_ids = {r.player_id for r in cap_by_actual[:3]}
                    actual_top_1_id = cap_by_actual[0].player_id

                    # Was our #1 pick in top 5 actual? (realistic captain success)
                    if top_pick_id in actual_top_5_ids:
                        captain_in_top_5 += 1
                    # Was our #1 pick in top 3 actual? (good captain pick)
                    if top_pick_id in actual_top_3_ids:
                        captain_in_top_3 += 1
                    # Exact hit (very hard!)
                    if top_pick_id == actual_top_1_id:
                        captain_exact_hits += 1
                    captain_total += 1

        result.total_predictions = len(result.player_results)

        # Calculate aggregate metrics
        if all_errors:
            result.mean_absolute_error = sum(all_errors) / len(all_errors)
            result.root_mean_square_error = (sum(e**2 for e in all_errors) / len(all_errors)) ** 0.5

        for pos, errors in position_errors.items():
            if errors:
                result.mae_by_position[pos] = sum(errors) / len(errors)

        if top_10_total > 0:
            result.top_10_hit_rate = top_10_hits / top_10_total

        if captain_total > 0:
            result.captain_accuracy = captain_in_top_5 / captain_total  # In top 5 = success
            result.captain_top_3_rate = captain_in_top_3 / captain_total
            result.captain_exact_hit = captain_exact_hits / captain_total

        # Calculate correlation
        if result.player_results:
            result.correlation = self._calculate_correlation(result.player_results)

        return result

    def _parse_players(self, bootstrap: dict) -> list[Player]:
        """Parse players from bootstrap data."""
        players = []
        for elem in bootstrap.get("elements", []):
            try:
                status_map = {
                    "a": PlayerStatus.AVAILABLE,
                    "d": PlayerStatus.DOUBTFUL,
                    "i": PlayerStatus.INJURED,
                    "u": PlayerStatus.UNAVAILABLE,
                    "n": PlayerStatus.NOT_AVAILABLE,
                    "s": PlayerStatus.SUSPENDED,
                }

                players.append(Player(
                    id=elem["id"],
                    name=elem.get("first_name", "") + " " + elem.get("second_name", ""),
                    web_name=elem["web_name"],
                    team_id=elem["team"],
                    position=Position(elem["element_type"]),
                    price=elem["now_cost"] / 10,
                    status=status_map.get(elem.get("status", "a"), PlayerStatus.AVAILABLE),
                    news=elem.get("news", ""),
                    chance_of_playing=elem.get("chance_of_playing_next_round"),
                    total_points=elem.get("total_points", 0),
                    points_per_game=float(elem.get("points_per_game", 0)),
                    form=float(elem.get("form", 0)),
                    selected_by_percent=float(elem.get("selected_by_percent", 0)),
                    ict_index=float(elem.get("ict_index", 0)),
                    goals_scored=elem.get("goals_scored", 0),
                    assists=elem.get("assists", 0),
                    clean_sheets=elem.get("clean_sheets", 0),
                    minutes=elem.get("minutes", 0),
                    # xG stats - critical for accurate projections!
                    expected_goals=float(elem.get("expected_goals", 0) or 0),
                    expected_assists=float(elem.get("expected_assists", 0) or 0),
                    expected_goal_involvements=float(elem.get("expected_goal_involvements", 0) or 0),
                    expected_goals_per_90=float(elem.get("expected_goals_per_90", 0) or 0),
                    expected_assists_per_90=float(elem.get("expected_assists_per_90", 0) or 0),
                ))
            except Exception as e:
                logger.debug(f"Error parsing player: {e}")

        return players

    def _parse_teams(self, bootstrap: dict) -> list[Team]:
        """Parse teams from bootstrap data."""
        teams = []
        for t in bootstrap.get("teams", []):
            teams.append(Team(
                id=t["id"],
                name=t["name"],
                short_name=t["short_name"],
                strength_home=t.get("strength_overall_home", 3),
                strength_away=t.get("strength_overall_away", 3),
                strength_attack_home=t.get("strength_attack_home", 1200),
                strength_attack_away=t.get("strength_attack_away", 1200),
                strength_defence_home=t.get("strength_defence_home", 1200),
                strength_defence_away=t.get("strength_defence_away", 1200),
            ))
        return teams

    def _parse_fixtures(self) -> list[Fixture]:
        """Fetch and parse fixtures."""
        client = self._get_client()
        url = f"{self.FPL_API}/fixtures/"

        response = client.get(url)
        response.raise_for_status()
        data = response.json()

        fixtures = []
        for f in data:
            try:
                from datetime import datetime
                kickoff = None
                if f.get("kickoff_time"):
                    kickoff = datetime.fromisoformat(f["kickoff_time"].replace("Z", "+00:00"))

                fixtures.append(Fixture(
                    id=f["id"],
                    gameweek=f.get("event") or 0,
                    home_team_id=f["team_h"],
                    away_team_id=f["team_a"],
                    home_difficulty=f.get("team_h_difficulty", 3),
                    away_difficulty=f.get("team_a_difficulty", 3),
                    kickoff_time=kickoff,
                    finished=f.get("finished", False),
                    home_score=f.get("team_h_score"),
                    away_score=f.get("team_a_score"),
                ))
            except Exception as e:
                logger.debug(f"Error parsing fixture: {e}")

        return fixtures

    def _calculate_correlation(self, results: list[PlayerResult]) -> float:
        """Calculate Pearson correlation between predicted and actual."""
        if len(results) < 2:
            return 0.0

        predicted = [r.predicted_points for r in results]
        actual = [r.actual_points for r in results]

        n = len(predicted)
        mean_p = sum(predicted) / n
        mean_a = sum(actual) / n

        numerator = sum((p - mean_p) * (a - mean_a) for p, a in zip(predicted, actual))
        denom_p = sum((p - mean_p) ** 2 for p in predicted) ** 0.5
        denom_a = sum((a - mean_a) ** 2 for a in actual) ** 0.5

        if denom_p * denom_a == 0:
            return 0.0

        return numerator / (denom_p * denom_a)


def run_backtest(
    gameweeks: list[int] | None = None,
    num_gameweeks: int = 3,
    record_to_adaptive: bool = True,
) -> BacktestResult:
    """
    Run backtest on recent gameweeks.

    Args:
        gameweeks: Specific gameweeks to test (optional)
        num_gameweeks: Number of recent completed gameweeks to test (default 3)
        record_to_adaptive: Whether to record results to adaptive weight system
    """
    backtester = Backtester()

    try:
        if gameweeks is None:
            # Find completed gameweeks
            bootstrap = backtester.fetch_bootstrap_data()
            events = bootstrap.get("events", [])

            completed = [
                e["id"] for e in events
                if e.get("finished") and e.get("data_checked")
            ]

            # Take last N completed gameweeks
            gameweeks = completed[-num_gameweeks:] if len(completed) >= num_gameweeks else completed

        if not gameweeks:
            logger.warning("No completed gameweeks to test")
            return BacktestResult()

        logger.info(f"Running backtest on gameweeks: {gameweeks}")
        result = backtester.run_backtest(gameweeks)

        # Record to adaptive system for learning
        if record_to_adaptive and result.total_predictions > 0:
            try:
                from .adaptive import WeightConfig, get_adaptive_manager

                manager = get_adaptive_manager()
                current_weights = WeightConfig()  # Uses current defaults

                manager.record_backtest_result(
                    weights=current_weights,
                    mae=result.mean_absolute_error,
                    correlation=result.correlation,
                    top_10_hit_rate=result.top_10_hit_rate,
                    captain_accuracy=result.captain_accuracy,
                    gameweeks_tested=result.gameweeks_tested,
                )
                logger.info("Recorded backtest results to adaptive system")
            except Exception as e:
                logger.warning(f"Failed to record to adaptive system: {e}")

        return result

    finally:
        backtester.close()


def print_backtest_report(result: BacktestResult) -> str:
    """Generate a formatted backtest report."""
    lines = [
        "=" * 60,
        "PREDICTION BACKTEST REPORT",
        "=" * 60,
        f"Gameweeks tested: {result.gameweeks_tested}",
        f"Total predictions: {result.total_predictions}",
        "",
        "ACCURACY METRICS:",
        f"  Mean Absolute Error: {result.mean_absolute_error:.2f} pts",
        f"  Root Mean Square Error: {result.root_mean_square_error:.2f} pts",
        f"  Correlation (predicted vs actual): {result.correlation:.3f}",
        "",
        "BY POSITION:",
    ]

    for pos, mae in result.mae_by_position.items():
        lines.append(f"  {pos}: MAE = {mae:.2f} pts")

    lines.extend([
        "",
        "TOP PICK ACCURACY:",
        f"  Top 10 Hit Rate: {result.top_10_hit_rate * 100:.1f}%",
        f"  Captain in Top 5: {result.captain_accuracy * 100:.1f}% (realistic success rate)",
        f"  Captain in Top 3: {result.captain_top_3_rate * 100:.1f}% (good pick)",
        f"  Captain Exact #1: {result.captain_exact_hit * 100:.1f}% (very difficult!)",
        "",
    ])

    # Show some examples
    if result.player_results:
        lines.append("SAMPLE PREDICTIONS (best and worst):")

        # Sort by error
        sorted_results = sorted(
            result.player_results,
            key=lambda x: abs(x.actual_points - x.predicted_points),
        )

        # Best predictions
        lines.append("  Best predictions:")
        for r in sorted_results[:5]:
            error = r.actual_points - r.predicted_points
            lines.append(
                f"    {r.player_name}: Predicted {r.predicted_points:.1f}, "
                f"Actual {r.actual_points} (error: {error:+.1f})"
            )

        # Worst predictions
        lines.append("  Worst predictions:")
        for r in sorted_results[-5:]:
            error = r.actual_points - r.predicted_points
            lines.append(
                f"    {r.player_name}: Predicted {r.predicted_points:.1f}, "
                f"Actual {r.actual_points} (error: {error:+.1f})"
            )

    lines.append("=" * 60)

    return "\n".join(lines)
