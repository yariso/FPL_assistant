"""
FPL API Client.

Async HTTP client for interacting with the Fantasy Premier League API.
Includes retry logic, error handling, and response validation.
"""

import asyncio
import logging
from typing import Any

import httpx

from .endpoints import (
    BOOTSTRAP_STATIC,
    EVENT_STATUS,
    FIXTURES,
    SET_PIECE_NOTES,
    get_classic_league_url,
    get_dream_team_url,
    get_element_summary_url,
    get_entry_history_url,
    get_entry_picks_url,
    get_entry_url,
    get_fixtures_url,
    get_h2h_league_url,
    get_live_event_url,
    get_my_team_url,
)

logger = logging.getLogger(__name__)


class FPLAPIError(Exception):
    """Base exception for FPL API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class FPLAuthenticationError(FPLAPIError):
    """Raised when authentication fails."""

    pass


class FPLRateLimitError(FPLAPIError):
    """Raised when rate limited by the API."""

    pass


class FPLNotFoundError(FPLAPIError):
    """Raised when resource not found."""

    pass


class FPLClient:
    """
    Async client for the FPL API.

    Handles HTTP requests with automatic retries, timeouts, and error handling.
    """

    DEFAULT_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    RETRY_BACKOFF = 2.0  # multiplier for exponential backoff

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = MAX_RETRIES,
    ):
        """
        Initialize the FPL client.

        Args:
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.AsyncClient | None = None
        self._cookies: dict[str, str] = {}

    async def __aenter__(self) -> "FPLClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True,
                headers={
                    "User-Agent": "FPL-Assistant/1.0",
                    "Accept": "application/json",
                },
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def set_cookies(self, cookies: dict[str, str]) -> None:
        """Set authentication cookies."""
        self._cookies = cookies

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional arguments for httpx

        Returns:
            JSON response data

        Raises:
            FPLAPIError: On request failure after retries
        """
        await self._ensure_client()
        assert self._client is not None

        # Add cookies if set
        if self._cookies:
            kwargs.setdefault("cookies", {}).update(self._cookies)

        last_error: Exception | None = None
        delay = self.RETRY_DELAY

        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Request attempt {attempt + 1}: {method} {url}")

                response = await self._client.request(method, url, **kwargs)

                # Handle different status codes
                if response.status_code == 200:
                    return response.json()

                elif response.status_code == 401:
                    raise FPLAuthenticationError(
                        "Authentication required or session expired",
                        status_code=401,
                    )

                elif response.status_code == 403:
                    raise FPLAuthenticationError(
                        "Access forbidden - check credentials",
                        status_code=403,
                    )

                elif response.status_code == 404:
                    raise FPLNotFoundError(
                        f"Resource not found: {url}",
                        status_code=404,
                    )

                elif response.status_code == 429:
                    # Rate limited - wait and retry
                    retry_after = int(response.headers.get("Retry-After", delay * 2))
                    logger.warning(f"Rate limited, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    continue

                elif response.status_code >= 500:
                    # Server error - retry with backoff
                    logger.warning(
                        f"Server error {response.status_code}, "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    delay *= self.RETRY_BACKOFF
                    continue

                else:
                    raise FPLAPIError(
                        f"Unexpected status code: {response.status_code}",
                        status_code=response.status_code,
                    )

            except httpx.TimeoutException as e:
                logger.warning(f"Request timeout, attempt {attempt + 1}")
                last_error = e
                await asyncio.sleep(delay)
                delay *= self.RETRY_BACKOFF

            except httpx.RequestError as e:
                logger.warning(f"Request error: {e}, attempt {attempt + 1}")
                last_error = e
                await asyncio.sleep(delay)
                delay *= self.RETRY_BACKOFF

        # All retries exhausted
        raise FPLAPIError(
            f"Request failed after {self.max_retries} attempts: {last_error}"
        )

    async def get(self, url: str, **kwargs: Any) -> dict[str, Any] | list[Any]:
        """Make a GET request."""
        return await self._request("GET", url, **kwargs)

    async def post(
        self, url: str, data: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any] | list[Any]:
        """Make a POST request."""
        return await self._request("POST", url, json=data, **kwargs)

    # =========================================================================
    # Public API Methods
    # =========================================================================

    async def get_bootstrap_static(self) -> dict[str, Any]:
        """
        Get bootstrap-static data.

        Returns all players, teams, game settings, and gameweek info.
        This is the main data endpoint - cache aggressively.
        """
        data = await self.get(BOOTSTRAP_STATIC)
        self._validate_bootstrap(data)
        return data  # type: ignore

    async def get_fixtures(
        self,
        event_id: int | None = None,
        future_only: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Get fixture data.

        Args:
            event_id: Filter by specific gameweek
            future_only: Only return upcoming fixtures
        """
        url = get_fixtures_url(event_id, future_only)
        data = await self.get(url)
        return data  # type: ignore

    async def get_element_summary(self, element_id: int) -> dict[str, Any]:
        """
        Get detailed data for a specific player.

        Includes past gameweek history and upcoming fixtures.
        Use sparingly - don't bulk fetch all players.
        """
        url = get_element_summary_url(element_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_entry(self, manager_id: int) -> dict[str, Any]:
        """Get public manager information."""
        url = get_entry_url(manager_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_entry_history(self, manager_id: int) -> dict[str, Any]:
        """Get manager's season history."""
        url = get_entry_history_url(manager_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_entry_picks(
        self, manager_id: int, event_id: int
    ) -> dict[str, Any]:
        """
        Get manager's team for a specific gameweek.

        Note: Only works if team is public or you're authenticated.
        """
        url = get_entry_picks_url(manager_id, event_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_live_event(self, event_id: int) -> dict[str, Any]:
        """Get live points data for a gameweek."""
        url = get_live_event_url(event_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_event_status(self) -> dict[str, Any]:
        """Get current event status (scoring, processing, etc.)."""
        data = await self.get(EVENT_STATUS)
        return data  # type: ignore

    async def get_set_piece_notes(self) -> list[dict[str, Any]]:
        """Get penalty and free kick takers for all teams."""
        data = await self.get(SET_PIECE_NOTES)
        return data  # type: ignore

    async def get_dream_team(self, event_id: int) -> dict[str, Any]:
        """
        Get dream team (best XI) for a gameweek.

        Returns the top scoring players for consensus picks.
        """
        url = get_dream_team_url(event_id)
        data = await self.get(url)
        return data  # type: ignore

    async def get_classic_league(
        self, league_id: int, page: int = 1
    ) -> dict[str, Any]:
        """
        Get classic league standings.

        Returns league info, standings with manager details.
        """
        url = get_classic_league_url(league_id, page)
        data = await self.get(url)
        return data  # type: ignore

    async def get_h2h_league(
        self, league_id: int, page: int = 1
    ) -> dict[str, Any]:
        """
        Get H2H league standings.

        Returns league info and H2H standings.
        """
        url = get_h2h_league_url(league_id, page)
        data = await self.get(url)
        return data  # type: ignore

    # =========================================================================
    # Authenticated API Methods
    # =========================================================================

    async def get_my_team(self, manager_id: int) -> dict[str, Any]:
        """
        Get user's current squad (requires authentication).

        Returns squad picks, bank, free transfers, and chip status.
        """
        if not self._cookies:
            raise FPLAuthenticationError(
                "Authentication required. Call authenticate() first."
            )

        url = get_my_team_url(manager_id)
        data = await self.get(url)
        return data  # type: ignore

    # =========================================================================
    # Validation
    # =========================================================================

    def _validate_bootstrap(self, data: Any) -> None:
        """Validate bootstrap-static response structure."""
        if not isinstance(data, dict):
            raise FPLAPIError("Invalid bootstrap response: expected dict")

        required_fields = ["elements", "teams", "events", "element_types"]
        missing = [f for f in required_fields if f not in data]

        if missing:
            raise FPLAPIError(
                f"Invalid bootstrap response: missing fields {missing}. "
                "The FPL API structure may have changed."
            )


# =============================================================================
# Synchronous Wrapper
# =============================================================================


class SyncFPLClient:
    """
    Synchronous wrapper for FPLClient.

    Provides a synchronous interface by running the async client
    in an event loop. Useful for CLI and simple scripts.
    """

    def __init__(self, **kwargs: Any):
        """Initialize with same args as FPLClient."""
        self._async_client = FPLClient(**kwargs)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
            return self._loop

    def _run(self, coro):
        """Run a coroutine synchronously."""
        loop = self._get_loop()
        return loop.run_until_complete(coro)

    def close(self) -> None:
        """Close the client."""
        self._run(self._async_client.close())
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    def set_cookies(self, cookies: dict[str, str]) -> None:
        """Set authentication cookies."""
        self._async_client.set_cookies(cookies)

    def get_bootstrap_static(self) -> dict[str, Any]:
        """Get bootstrap-static data."""
        return self._run(self._async_client.get_bootstrap_static())

    def get_fixtures(
        self,
        event_id: int | None = None,
        future_only: bool = False,
    ) -> list[dict[str, Any]]:
        """Get fixture data."""
        return self._run(self._async_client.get_fixtures(event_id, future_only))

    def get_element_summary(self, element_id: int) -> dict[str, Any]:
        """Get player element summary."""
        return self._run(self._async_client.get_element_summary(element_id))

    def get_entry(self, manager_id: int) -> dict[str, Any]:
        """Get manager entry."""
        return self._run(self._async_client.get_entry(manager_id))

    def get_entry_history(self, manager_id: int) -> dict[str, Any]:
        """Get manager history."""
        return self._run(self._async_client.get_entry_history(manager_id))

    def get_entry_picks(self, manager_id: int, event_id: int) -> dict[str, Any]:
        """Get manager's picks."""
        return self._run(self._async_client.get_entry_picks(manager_id, event_id))

    def get_live_event(self, event_id: int) -> dict[str, Any]:
        """Get live event data."""
        return self._run(self._async_client.get_live_event(event_id))

    def get_event_status(self) -> dict[str, Any]:
        """Get event status."""
        return self._run(self._async_client.get_event_status())

    def get_set_piece_notes(self) -> list[dict[str, Any]]:
        """Get set piece notes."""
        return self._run(self._async_client.get_set_piece_notes())

    def get_dream_team(self, event_id: int) -> dict[str, Any]:
        """Get dream team for a gameweek."""
        return self._run(self._async_client.get_dream_team(event_id))

    def get_my_team(self, manager_id: int) -> dict[str, Any]:
        """Get user's team (authenticated)."""
        return self._run(self._async_client.get_my_team(manager_id))

    def get_classic_league(self, league_id: int, page: int = 1) -> dict[str, Any]:
        """Get classic league standings."""
        return self._run(self._async_client.get_classic_league(league_id, page))

    def get_h2h_league(self, league_id: int, page: int = 1) -> dict[str, Any]:
        """Get H2H league standings."""
        return self._run(self._async_client.get_h2h_league(league_id, page))
