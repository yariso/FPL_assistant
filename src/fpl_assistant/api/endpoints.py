"""
FPL API endpoint definitions.

All URLs for the Fantasy Premier League API endpoints.
Note: The FPL API is undocumented and unofficial - URLs may change.
"""

# Base URLs
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_LOGIN_URL = "https://users.premierleague.com/accounts/login/"

# =============================================================================
# Public Endpoints (No authentication required)
# =============================================================================

# Bootstrap Static - Main data endpoint
# Contains: all players, teams, game settings, gameweek info
# Recommended refresh: Once per hour
BOOTSTRAP_STATIC = f"{FPL_BASE_URL}/bootstrap-static/"

# Fixtures - All match fixtures
# Contains: fixture IDs, teams, difficulties, kickoff times, scores
# Can filter with ?event=N for specific gameweek
# Can filter with ?future=1 for upcoming only
FIXTURES = f"{FPL_BASE_URL}/fixtures/"

# Element Summary - Individual player details
# Contains: past gameweek history, upcoming fixtures
# Use sparingly - don't bulk fetch all players
ELEMENT_SUMMARY = f"{FPL_BASE_URL}/element-summary/{{element_id}}/"

# Entry (Manager) - Public manager info
# Contains: overall points, rank, team name
ENTRY = f"{FPL_BASE_URL}/entry/{{manager_id}}/"

# Entry History - Manager's season history
# Contains: points per gameweek, overall rank progression
ENTRY_HISTORY = f"{FPL_BASE_URL}/entry/{{manager_id}}/history/"

# Entry Picks - Manager's team for a specific gameweek
# Note: Only works if team is public, otherwise needs auth
ENTRY_PICKS = f"{FPL_BASE_URL}/entry/{{manager_id}}/event/{{event_id}}/picks/"

# Entry Transfers - Manager's transfer history
ENTRY_TRANSFERS = f"{FPL_BASE_URL}/entry/{{manager_id}}/transfers/"

# Event Status - Current gameweek status
# Contains: whether scoring is live, bonus processing, etc.
EVENT_STATUS = f"{FPL_BASE_URL}/event-status/"

# Live Event - Real-time points during gameweek
# Contains: live player points for active gameweek
LIVE_EVENT = f"{FPL_BASE_URL}/event/{{event_id}}/live/"

# Dream Team - Best XI for a gameweek
DREAM_TEAM = f"{FPL_BASE_URL}/dream-team/{{event_id}}/"

# Set Piece Notes - Penalty and free kick takers
SET_PIECE_NOTES = f"{FPL_BASE_URL}/set-piece-notes/"

# =============================================================================
# Authenticated Endpoints (Require FPL login)
# =============================================================================

# My Team - User's current squad (requires auth)
# Contains: squad picks, bank, free transfers, chips
MY_TEAM = f"{FPL_BASE_URL}/my-team/{{manager_id}}/"

# Transfers - Make transfers (requires auth)
# POST endpoint for executing transfers
TRANSFERS = f"{FPL_BASE_URL}/transfers/"

# =============================================================================
# Endpoint Helper Functions
# =============================================================================


def get_element_summary_url(element_id: int) -> str:
    """Get URL for player element summary."""
    return ELEMENT_SUMMARY.format(element_id=element_id)


def get_entry_url(manager_id: int) -> str:
    """Get URL for manager entry."""
    return ENTRY.format(manager_id=manager_id)


def get_entry_history_url(manager_id: int) -> str:
    """Get URL for manager history."""
    return ENTRY_HISTORY.format(manager_id=manager_id)


def get_entry_picks_url(manager_id: int, event_id: int) -> str:
    """Get URL for manager's picks in a gameweek."""
    return ENTRY_PICKS.format(manager_id=manager_id, event_id=event_id)


def get_entry_transfers_url(manager_id: int) -> str:
    """Get URL for manager's transfer history."""
    return ENTRY_TRANSFERS.format(manager_id=manager_id)


def get_live_event_url(event_id: int) -> str:
    """Get URL for live event data."""
    return LIVE_EVENT.format(event_id=event_id)


def get_dream_team_url(event_id: int) -> str:
    """Get URL for dream team."""
    return DREAM_TEAM.format(event_id=event_id)


def get_my_team_url(manager_id: int) -> str:
    """Get URL for user's team (authenticated)."""
    return MY_TEAM.format(manager_id=manager_id)


def get_fixtures_url(event_id: int | None = None, future_only: bool = False) -> str:
    """
    Get URL for fixtures with optional filters.

    Args:
        event_id: Specific gameweek to filter by
        future_only: Only return upcoming fixtures
    """
    url = FIXTURES
    params = []

    if event_id is not None:
        params.append(f"event={event_id}")
    if future_only:
        params.append("future=1")

    if params:
        url += "?" + "&".join(params)

    return url
