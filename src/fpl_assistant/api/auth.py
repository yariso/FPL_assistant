"""
FPL Authentication Handler.

Manages login to the FPL website and session cookie persistence.
"""

import json
import logging
from pathlib import Path
from typing import Any

import httpx

from .endpoints import FPL_LOGIN_URL

logger = logging.getLogger(__name__)


class AuthenticationError(Exception):
    """Raised when authentication fails."""

    pass


class FPLAuthenticator:
    """
    Handles FPL website authentication.

    The FPL API uses session cookies for authentication.
    This class manages the login flow and cookie persistence.
    """

    SESSION_FILE = ".session.json"

    def __init__(self, session_file: str | Path | None = None):
        """
        Initialize the authenticator.

        Args:
            session_file: Path to store session cookies (default: .session.json)
        """
        self.session_file = Path(session_file or self.SESSION_FILE)
        self._cookies: dict[str, str] = {}
        self._manager_id: int | None = None

    @property
    def is_authenticated(self) -> bool:
        """Check if we have valid session cookies."""
        return bool(self._cookies)

    @property
    def cookies(self) -> dict[str, str]:
        """Get current session cookies."""
        return self._cookies

    @property
    def manager_id(self) -> int | None:
        """Get the authenticated manager ID."""
        return self._manager_id

    async def login(self, email: str, password: str) -> dict[str, str]:
        """
        Authenticate with FPL using email and password.

        Args:
            email: FPL account email
            password: FPL account password

        Returns:
            Session cookies dictionary

        Raises:
            AuthenticationError: If login fails
        """
        logger.info(f"Attempting FPL login for {email}")

        async with httpx.AsyncClient(follow_redirects=True) as client:
            # Prepare login data
            login_data = {
                "login": email,
                "password": password,
                "redirect_uri": "https://fantasy.premierleague.com/",
                "app": "plfpl-web",
            }

            try:
                # Make login request
                response = await client.post(
                    FPL_LOGIN_URL,
                    data=login_data,
                    headers={
                        "User-Agent": "FPL-Assistant/1.0",
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                )

                # Check for successful login
                # FPL returns a redirect on success, or stays on login page on failure
                if response.status_code == 200:
                    # Check if we got redirected to FPL (success) or stayed on login (failure)
                    if "fantasy.premierleague.com" in str(response.url):
                        # Extract cookies
                        self._cookies = dict(response.cookies)

                        # Also check for cookies set during redirects
                        for hist_response in response.history:
                            self._cookies.update(dict(hist_response.cookies))

                        if not self._cookies:
                            raise AuthenticationError(
                                "Login appeared successful but no cookies received"
                            )

                        logger.info("FPL login successful")
                        self._save_session()
                        return self._cookies
                    else:
                        raise AuthenticationError(
                            "Login failed - invalid credentials or account issue"
                        )

                elif response.status_code == 400:
                    raise AuthenticationError("Invalid login credentials")

                else:
                    raise AuthenticationError(
                        f"Login failed with status {response.status_code}"
                    )

            except httpx.RequestError as e:
                raise AuthenticationError(f"Login request failed: {e}") from e

    def login_sync(self, email: str, password: str) -> dict[str, str]:
        """
        Synchronous login wrapper.

        Args:
            email: FPL account email
            password: FPL account password

        Returns:
            Session cookies dictionary
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        return loop.run_until_complete(self.login(email, password))

    def load_session(self) -> bool:
        """
        Load session cookies from file.

        Returns:
            True if session loaded successfully, False otherwise
        """
        if not self.session_file.exists():
            logger.debug("No session file found")
            return False

        try:
            with open(self.session_file) as f:
                data = json.load(f)

            self._cookies = data.get("cookies", {})
            self._manager_id = data.get("manager_id")

            if self._cookies:
                logger.info("Loaded session from file")
                return True

            return False

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load session: {e}")
            return False

    def _save_session(self) -> None:
        """Save session cookies to file."""
        try:
            data = {
                "cookies": self._cookies,
                "manager_id": self._manager_id,
            }

            with open(self.session_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Session saved to {self.session_file}")

        except OSError as e:
            logger.warning(f"Failed to save session: {e}")

    def set_manager_id(self, manager_id: int) -> None:
        """Set the manager ID and save to session."""
        self._manager_id = manager_id
        if self._cookies:
            self._save_session()

    def clear_session(self) -> None:
        """Clear session cookies and delete session file."""
        self._cookies = {}
        self._manager_id = None

        if self.session_file.exists():
            try:
                self.session_file.unlink()
                logger.info("Session cleared")
            except OSError as e:
                logger.warning(f"Failed to delete session file: {e}")

    async def validate_session(self) -> bool:
        """
        Validate that the current session is still active.

        Makes a test request to check if cookies are still valid.

        Returns:
            True if session is valid, False otherwise
        """
        if not self._cookies:
            return False

        try:
            async with httpx.AsyncClient() as client:
                # Try to access a page that requires authentication
                response = await client.get(
                    "https://fantasy.premierleague.com/api/me/",
                    cookies=self._cookies,
                    follow_redirects=True,
                )

                # If we get JSON data back, session is valid
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # The /me endpoint returns user info when authenticated
                        if "player" in data:
                            self._manager_id = data["player"]["entry"]
                            self._save_session()
                            return True
                    except json.JSONDecodeError:
                        pass

                return False

        except httpx.RequestError:
            return False

    def validate_session_sync(self) -> bool:
        """Synchronous session validation."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        return loop.run_until_complete(self.validate_session())


# =============================================================================
# Convenience Functions
# =============================================================================


def get_authenticator(session_file: str | Path | None = None) -> FPLAuthenticator:
    """
    Get an authenticator instance, loading existing session if available.

    Args:
        session_file: Optional custom session file path

    Returns:
        FPLAuthenticator instance
    """
    auth = FPLAuthenticator(session_file)
    auth.load_session()
    return auth


async def authenticate(
    email: str,
    password: str,
    session_file: str | Path | None = None,
) -> FPLAuthenticator:
    """
    Authenticate with FPL and return an authenticator.

    Args:
        email: FPL account email
        password: FPL account password
        session_file: Optional custom session file path

    Returns:
        Authenticated FPLAuthenticator instance
    """
    auth = FPLAuthenticator(session_file)

    # Try to load existing session
    if auth.load_session():
        # Validate it's still active
        if await auth.validate_session():
            logger.info("Using existing valid session")
            return auth

    # Need to login fresh
    await auth.login(email, password)
    return auth
