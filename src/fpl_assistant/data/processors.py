"""
Data Processors for FPL API Responses.

Transforms raw API JSON responses into typed Pydantic models.
Handles data cleaning, validation, and derived calculations.
"""

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from .models import (
    ChipStatus,
    ChipType,
    Fixture,
    GameweekInfo,
    Player,
    PlayerStatus,
    Position,
    Squad,
    SquadPlayer,
    Team,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Bootstrap Static Processing
# =============================================================================


def process_players(elements: list[dict[str, Any]]) -> list[Player]:
    """
    Process player data from bootstrap-static 'elements' array.

    Args:
        elements: List of player dictionaries from API

    Returns:
        List of Player models
    """
    players = []

    for elem in elements:
        try:
            # Map status character to enum
            status_char = elem.get("status", "a")
            try:
                status = PlayerStatus(status_char)
            except ValueError:
                status = PlayerStatus.UNAVAILABLE

            # Map element_type to Position enum
            position = Position(elem.get("element_type", 1))

            # Price is in tenths (e.g., 100 = £10.0)
            price = elem.get("now_cost", 0) / 10.0

            player = Player(
                id=elem["id"],
                name=f"{elem.get('first_name', '')} {elem.get('second_name', '')}".strip(),
                web_name=elem.get("web_name", ""),
                team_id=elem.get("team", 0),
                position=position,
                price=price,
                status=status,
                news=elem.get("news", "") or "",
                chance_of_playing=elem.get("chance_of_playing_next_round"),
                total_points=elem.get("total_points", 0),
                points_per_game=float(elem.get("points_per_game", 0) or 0),
                form=float(elem.get("form", 0) or 0),
                selected_by_percent=float(elem.get("selected_by_percent", 0) or 0),
                ict_index=float(elem.get("ict_index", 0) or 0),
                goals_scored=elem.get("goals_scored", 0),
                assists=elem.get("assists", 0),
                clean_sheets=elem.get("clean_sheets", 0),
                minutes=elem.get("minutes", 0),
                # xG stats - the key to accurate predictions!
                expected_goals=float(elem.get("expected_goals", 0) or 0),
                expected_assists=float(elem.get("expected_assists", 0) or 0),
                expected_goal_involvements=float(elem.get("expected_goal_involvements", 0) or 0),
                expected_goals_per_90=float(elem.get("expected_goals_per_90", 0) or 0),
                expected_assists_per_90=float(elem.get("expected_assists_per_90", 0) or 0),
                # Additional stats for accurate projections
                bonus=elem.get("bonus", 0),
                yellow_cards=elem.get("yellow_cards", 0),
                red_cards=elem.get("red_cards", 0),
                saves=elem.get("saves", 0),
                goals_conceded=elem.get("goals_conceded", 0),
                own_goals=elem.get("own_goals", 0),
                penalties_missed=elem.get("penalties_missed", 0),
                penalties_saved=elem.get("penalties_saved", 0),
            )
            players.append(player)

        except Exception as e:
            logger.warning(f"Error processing player {elem.get('id')}: {e}")
            continue

    logger.info(f"Processed {len(players)} players")
    return players


def process_teams(teams_data: list[dict[str, Any]]) -> list[Team]:
    """
    Process team data from bootstrap-static 'teams' array.

    Args:
        teams_data: List of team dictionaries from API

    Returns:
        List of Team models
    """
    teams = []

    for team_data in teams_data:
        try:
            team = Team(
                id=team_data["id"],
                name=team_data.get("name", ""),
                short_name=team_data.get("short_name", ""),
                strength_home=team_data.get("strength_overall_home", 3),
                strength_away=team_data.get("strength_overall_away", 3),
                strength_attack_home=team_data.get("strength_attack_home", 0),
                strength_attack_away=team_data.get("strength_attack_away", 0),
                strength_defence_home=team_data.get("strength_defence_home", 0),
                strength_defence_away=team_data.get("strength_defence_away", 0),
            )
            teams.append(team)

        except Exception as e:
            logger.warning(f"Error processing team {team_data.get('id')}: {e}")
            continue

    logger.info(f"Processed {len(teams)} teams")
    return teams


def process_gameweeks(events: list[dict[str, Any]]) -> list[GameweekInfo]:
    """
    Process gameweek data from bootstrap-static 'events' array.

    Args:
        events: List of event/gameweek dictionaries from API

    Returns:
        List of GameweekInfo models
    """
    gameweeks = []

    for event in events:
        try:
            # Parse deadline time
            deadline_str = event.get("deadline_time", "")
            if deadline_str:
                # Handle ISO format with or without timezone
                deadline_str = deadline_str.replace("Z", "+00:00")
                deadline = datetime.fromisoformat(deadline_str)
            else:
                deadline = datetime.now()

            gw = GameweekInfo(
                id=event["id"],
                name=event.get("name", f"Gameweek {event['id']}"),
                deadline=deadline,
                is_current=event.get("is_current", False),
                is_next=event.get("is_next", False),
                finished=event.get("finished", False),
                # Blank/double will be calculated from fixtures
                is_blank=False,
                is_double=False,
                blank_teams=[],
                double_teams=[],
            )
            gameweeks.append(gw)

        except Exception as e:
            logger.warning(f"Error processing gameweek {event.get('id')}: {e}")
            continue

    logger.info(f"Processed {len(gameweeks)} gameweeks")
    return gameweeks


# =============================================================================
# Fixture Processing
# =============================================================================


def process_fixtures(fixtures_data: list[dict[str, Any]]) -> list[Fixture]:
    """
    Process fixture data from fixtures endpoint.

    Args:
        fixtures_data: List of fixture dictionaries from API

    Returns:
        List of Fixture models
    """
    fixtures = []

    for fix in fixtures_data:
        try:
            # Parse kickoff time
            kickoff_str = fix.get("kickoff_time")
            kickoff = None
            if kickoff_str:
                kickoff_str = kickoff_str.replace("Z", "+00:00")
                kickoff = datetime.fromisoformat(kickoff_str)

            fixture = Fixture(
                id=fix["id"],
                gameweek=fix.get("event") or 0,  # Can be None for unscheduled
                home_team_id=fix.get("team_h", 0),
                away_team_id=fix.get("team_a", 0),
                home_difficulty=fix.get("team_h_difficulty", 3),
                away_difficulty=fix.get("team_a_difficulty", 3),
                kickoff_time=kickoff,
                finished=fix.get("finished", False),
                home_score=fix.get("team_h_score"),
                away_score=fix.get("team_a_score"),
            )
            fixtures.append(fixture)

        except Exception as e:
            logger.warning(f"Error processing fixture {fix.get('id')}: {e}")
            continue

    logger.info(f"Processed {len(fixtures)} fixtures")
    return fixtures


def identify_blank_double_gameweeks(
    fixtures: list[Fixture],
    num_teams: int = 20,
) -> dict[int, dict[str, list[int]]]:
    """
    Identify blank and double gameweeks from fixture list.

    A blank gameweek is when a team has 0 fixtures.
    A double gameweek is when a team has 2+ fixtures.

    Args:
        fixtures: List of Fixture models
        num_teams: Number of teams in the league (default 20)

    Returns:
        Dict mapping gameweek to {"blank_teams": [...], "double_teams": [...]}
    """
    # Count fixtures per team per gameweek
    team_fixtures: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for fixture in fixtures:
        if fixture.gameweek > 0:  # Skip unscheduled fixtures
            team_fixtures[fixture.gameweek][fixture.home_team_id] += 1
            team_fixtures[fixture.gameweek][fixture.away_team_id] += 1

    result: dict[int, dict[str, list[int]]] = {}

    for gw, teams in sorted(team_fixtures.items()):
        blank_teams = []
        double_teams = []

        # Check all teams (1-20)
        for team_id in range(1, num_teams + 1):
            fixture_count = teams.get(team_id, 0)

            if fixture_count == 0:
                blank_teams.append(team_id)
            elif fixture_count >= 2:
                double_teams.append(team_id)

        if blank_teams or double_teams:
            result[gw] = {
                "blank_teams": blank_teams,
                "double_teams": double_teams,
            }

    return result


def update_gameweeks_with_blank_double(
    gameweeks: list[GameweekInfo],
    blank_double_info: dict[int, dict[str, list[int]]],
) -> list[GameweekInfo]:
    """
    Update gameweek info with blank/double team lists.

    Args:
        gameweeks: List of GameweekInfo models
        blank_double_info: Output from identify_blank_double_gameweeks

    Returns:
        Updated list of GameweekInfo models
    """
    for gw in gameweeks:
        info = blank_double_info.get(gw.id, {})
        gw.blank_teams = info.get("blank_teams", [])
        gw.double_teams = info.get("double_teams", [])
        gw.is_blank = len(gw.blank_teams) > 0
        gw.is_double = len(gw.double_teams) > 0

    return gameweeks


# =============================================================================
# User Squad Processing
# =============================================================================


def process_my_team(my_team_data: dict[str, Any]) -> Squad:
    """
    Process my-team endpoint response into Squad model.

    Args:
        my_team_data: Response from my-team API endpoint

    Returns:
        Squad model with players and state
    """
    # Process squad picks
    picks = my_team_data.get("picks", [])
    players = []

    for pick in picks:
        squad_player = SquadPlayer(
            player_id=pick["element"],
            position=pick["position"],
            is_captain=pick.get("is_captain", False),
            is_vice_captain=pick.get("is_vice_captain", False),
            purchase_price=pick.get("purchase_price", 0) / 10.0,
            selling_price=pick.get("selling_price", 0) / 10.0,
        )
        players.append(squad_player)

    # Process chips
    chips_data = my_team_data.get("chips", [])
    chips = process_chips(chips_data)

    # Get transfers info
    transfers = my_team_data.get("transfers", {})
    bank = transfers.get("bank", 0) / 10.0

    # Calculate total value
    total_value = sum(p.selling_price for p in players) + bank

    # Free transfers - this might be in different places depending on API version
    # Check multiple possible locations
    free_transfers = 1  # Default
    if "transfers" in my_team_data:
        free_transfers = my_team_data["transfers"].get("limit", 1)
        # Subtract any transfers already made this gameweek
        made = my_team_data["transfers"].get("made", 0)
        free_transfers = max(0, free_transfers - made)

    squad = Squad(
        players=players,
        bank=bank,
        free_transfers=free_transfers,
        total_value=total_value,
        chips=chips,
    )

    logger.info(
        f"Processed squad: {len(players)} players, "
        f"bank={bank:.1f}, FTs={free_transfers}"
    )
    return squad


def process_chips(chips_data: list[dict[str, Any]]) -> list[ChipStatus]:
    """
    Process chips data from my-team response.

    Args:
        chips_data: List of chip status dictionaries

    Returns:
        List of ChipStatus models
    """
    # Map FPL chip names to our enum
    chip_name_map = {
        "wildcard": ChipType.WILDCARD,
        "freehit": ChipType.FREE_HIT,
        "bboost": ChipType.BENCH_BOOST,
        "3xc": ChipType.TRIPLE_CAPTAIN,
    }

    chips = []
    for chip in chips_data:
        chip_name = chip.get("name", "").lower()
        chip_type = chip_name_map.get(chip_name)

        if chip_type:
            # time_used indicates when chip was used (None if available)
            used_event = chip.get("event")  # Gameweek when used

            chip_status = ChipStatus(
                chip_type=chip_type,
                used_gameweek=used_event,
            )
            chips.append(chip_status)

    return chips


# =============================================================================
# Full Bootstrap Processing
# =============================================================================


def process_bootstrap_static(
    data: dict[str, Any]
) -> tuple[list[Player], list[Team], list[GameweekInfo]]:
    """
    Process complete bootstrap-static response.

    Args:
        data: Full bootstrap-static API response

    Returns:
        Tuple of (players, teams, gameweeks)
    """
    players = process_players(data.get("elements", []))
    teams = process_teams(data.get("teams", []))
    gameweeks = process_gameweeks(data.get("events", []))

    return players, teams, gameweeks


# =============================================================================
# Fixture Difficulty Analysis
# =============================================================================


def get_player_fixtures(
    player: Player,
    fixtures: list[Fixture],
    gameweek_start: int,
    gameweek_end: int,
) -> list[dict[str, Any]]:
    """
    Get a player's fixtures for a gameweek range.

    Args:
        player: Player model
        fixtures: All fixtures
        gameweek_start: Starting gameweek (inclusive)
        gameweek_end: Ending gameweek (inclusive)

    Returns:
        List of fixture info dicts with difficulty ratings
    """
    player_fixtures = []

    for fixture in fixtures:
        if gameweek_start <= fixture.gameweek <= gameweek_end:
            if fixture.home_team_id == player.team_id:
                player_fixtures.append({
                    "gameweek": fixture.gameweek,
                    "opponent_id": fixture.away_team_id,
                    "is_home": True,
                    "difficulty": fixture.home_difficulty,
                    "kickoff_time": fixture.kickoff_time,
                })
            elif fixture.away_team_id == player.team_id:
                player_fixtures.append({
                    "gameweek": fixture.gameweek,
                    "opponent_id": fixture.home_team_id,
                    "is_home": False,
                    "difficulty": fixture.away_difficulty,
                    "kickoff_time": fixture.kickoff_time,
                })

    return sorted(player_fixtures, key=lambda x: (x["gameweek"], x["kickoff_time"] or datetime.max))


def calculate_fixture_difficulty_score(
    player: Player,
    fixtures: list[Fixture],
    gameweek_start: int,
    gameweek_end: int,
) -> float:
    """
    Calculate average fixture difficulty for a player over a range.

    Lower score = easier fixtures.

    Args:
        player: Player model
        fixtures: All fixtures
        gameweek_start: Starting gameweek
        gameweek_end: Ending gameweek

    Returns:
        Average difficulty score (1-5 scale)
    """
    player_fixtures = get_player_fixtures(
        player, fixtures, gameweek_start, gameweek_end
    )

    if not player_fixtures:
        return 3.0  # Default medium difficulty

    total_difficulty = sum(f["difficulty"] for f in player_fixtures)
    return total_difficulty / len(player_fixtures)
