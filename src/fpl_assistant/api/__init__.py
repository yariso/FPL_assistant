"""
FPL API Integration Module.

Provides clients for interacting with the Fantasy Premier League API.
"""

from .auth import FPLAuthenticator, authenticate, get_authenticator
from .cache import APICache, CachedFPLClient, get_cache
from .client import (
    FPLAPIError,
    FPLAuthenticationError,
    FPLClient,
    FPLNotFoundError,
    FPLRateLimitError,
    SyncFPLClient,
)

__all__ = [
    # Client
    "FPLClient",
    "SyncFPLClient",
    # Cache
    "APICache",
    "CachedFPLClient",
    "get_cache",
    # Auth
    "FPLAuthenticator",
    "authenticate",
    "get_authenticator",
    # Errors
    "FPLAPIError",
    "FPLAuthenticationError",
    "FPLNotFoundError",
    "FPLRateLimitError",
]
