"""
API Response Caching Layer.

File-based caching for FPL API responses with configurable TTL.
Provides graceful fallback to cached data on API failures.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CacheEntry:
    """Represents a cached item with metadata."""

    def __init__(
        self,
        data: Any,
        timestamp: float,
        ttl: int,
        key: str,
    ):
        self.data = data
        self.timestamp = timestamp
        self.ttl = ttl
        self.key = key

    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp

    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return self.age > self.ttl

    @property
    def expires_in(self) -> float:
        """Get seconds until expiration (negative if expired)."""
        return self.ttl - self.age

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "data": self.data,
            "timestamp": self.timestamp,
            "ttl": self.ttl,
            "key": self.key,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            data=d["data"],
            timestamp=d["timestamp"],
            ttl=d["ttl"],
            key=d["key"],
        )


class APICache:
    """
    File-based cache for API responses.

    Features:
    - Configurable TTL per cache type
    - Automatic expiration
    - Graceful fallback to stale data
    - Thread-safe file operations
    """

    # Default TTLs in seconds
    DEFAULT_TTLS = {
        "bootstrap": 3600,      # 1 hour
        "fixtures": 86400,       # 24 hours
        "element_summary": 3600, # 1 hour
        "event_status": 300,     # 5 minutes
        "default": 1800,         # 30 minutes
    }

    def __init__(
        self,
        cache_dir: str | Path = ".cache",
        ttls: dict[str, int] | None = None,
    ):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files
            ttls: Custom TTLs for cache types (overrides defaults)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttls = {**self.DEFAULT_TTLS}
        if ttls:
            self.ttls.update(ttls)

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Sanitize key for filesystem
        safe_key = key.replace("/", "_").replace(":", "_").replace("?", "_")
        return self.cache_dir / f"{safe_key}.json"

    def _get_ttl(self, cache_type: str) -> int:
        """Get TTL for cache type."""
        return self.ttls.get(cache_type, self.ttls["default"])

    def get(
        self,
        key: str,
        cache_type: str = "default",
        allow_stale: bool = False,
    ) -> CacheEntry | None:
        """
        Get item from cache.

        Args:
            key: Cache key
            cache_type: Type of cache (for TTL lookup)
            allow_stale: Return expired entries if True

        Returns:
            CacheEntry if found and valid, None otherwise
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            logger.debug(f"Cache miss: {key}")
            return None

        try:
            with open(cache_path) as f:
                entry = CacheEntry.from_dict(json.load(f))

            if entry.is_expired and not allow_stale:
                logger.debug(f"Cache expired: {key} (age: {entry.age:.0f}s)")
                return None

            logger.debug(
                f"Cache hit: {key} "
                f"(age: {entry.age:.0f}s, expired: {entry.is_expired})"
            )
            return entry

        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Cache read error for {key}: {e}")
            return None

    def set(
        self,
        key: str,
        data: Any,
        cache_type: str = "default",
        ttl: int | None = None,
    ) -> None:
        """
        Store item in cache.

        Args:
            key: Cache key
            data: Data to cache
            cache_type: Type of cache (for default TTL)
            ttl: Custom TTL (overrides cache_type default)
        """
        cache_path = self._get_cache_path(key)
        effective_ttl = ttl if ttl is not None else self._get_ttl(cache_type)

        entry = CacheEntry(
            data=data,
            timestamp=time.time(),
            ttl=effective_ttl,
            key=key,
        )

        try:
            with open(cache_path, "w") as f:
                json.dump(entry.to_dict(), f)

            logger.debug(f"Cached: {key} (ttl: {effective_ttl}s)")

        except OSError as e:
            logger.warning(f"Cache write error for {key}: {e}")

    def invalidate(self, key: str) -> bool:
        """
        Invalidate (delete) a cache entry.

        Args:
            key: Cache key to invalidate

        Returns:
            True if entry was deleted, False if not found
        """
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.debug(f"Cache invalidated: {key}")
                return True
            except OSError as e:
                logger.warning(f"Cache invalidation error for {key}: {e}")

        return False

    def invalidate_type(self, cache_type: str) -> int:
        """
        Invalidate all cache entries of a specific type.

        Args:
            cache_type: Type prefix to match

        Returns:
            Number of entries invalidated
        """
        count = 0
        for cache_file in self.cache_dir.glob(f"{cache_type}*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        logger.debug(f"Invalidated {count} entries of type: {cache_type}")
        return count

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except OSError:
                pass

        logger.info(f"Cleared {count} cache entries")
        return count

    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        entries = list(self.cache_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in entries)

        expired_count = 0
        for cache_file in entries:
            try:
                with open(cache_file) as f:
                    entry = CacheEntry.from_dict(json.load(f))
                    if entry.is_expired:
                        expired_count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                expired_count += 1

        return {
            "total_entries": len(entries),
            "expired_entries": expired_count,
            "valid_entries": len(entries) - expired_count,
            "total_size_bytes": total_size,
            "total_size_kb": total_size / 1024,
            "cache_dir": str(self.cache_dir),
        }


class CachedFPLClient:
    """
    FPL Client wrapper with automatic caching.

    Wraps the API client to provide transparent caching of responses.
    """

    def __init__(
        self,
        client: Any,  # FPLClient or SyncFPLClient
        cache: APICache | None = None,
    ):
        """
        Initialize cached client.

        Args:
            client: FPLClient or SyncFPLClient instance
            cache: APICache instance (creates default if None)
        """
        self.client = client
        self.cache = cache or APICache()

    def get_bootstrap_static(
        self,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """
        Get bootstrap-static with caching.

        Args:
            force_refresh: Bypass cache and fetch fresh data
        """
        key = "bootstrap_static"

        if not force_refresh:
            entry = self.cache.get(key, "bootstrap")
            if entry:
                return entry.data

        # Fetch fresh data
        data = self.client.get_bootstrap_static()
        self.cache.set(key, data, "bootstrap")
        return data

    def get_fixtures(
        self,
        event_id: int | None = None,
        future_only: bool = False,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Get fixtures with caching."""
        key = f"fixtures_e{event_id}_f{future_only}"

        if not force_refresh:
            entry = self.cache.get(key, "fixtures")
            if entry:
                return entry.data

        data = self.client.get_fixtures(event_id, future_only)
        self.cache.set(key, data, "fixtures")
        return data

    def get_element_summary(
        self,
        element_id: int,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get element summary with caching."""
        key = f"element_{element_id}"

        if not force_refresh:
            entry = self.cache.get(key, "element_summary")
            if entry:
                return entry.data

        data = self.client.get_element_summary(element_id)
        self.cache.set(key, data, "element_summary")
        return data

    def get_event_status(
        self,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        """Get event status with caching."""
        key = "event_status"

        if not force_refresh:
            entry = self.cache.get(key, "event_status")
            if entry:
                return entry.data

        data = self.client.get_event_status()
        self.cache.set(key, data, "event_status")
        return data

    # Pass-through methods (no caching - always fresh)

    def get_my_team(self, manager_id: int) -> dict[str, Any]:
        """Get user's team (no caching - always fresh)."""
        return self.client.get_my_team(manager_id)

    def get_entry(self, manager_id: int) -> dict[str, Any]:
        """Get manager entry."""
        return self.client.get_entry(manager_id)

    def get_entry_history(self, manager_id: int) -> dict[str, Any]:
        """Get manager history."""
        return self.client.get_entry_history(manager_id)

    def get_entry_picks(self, manager_id: int, event_id: int) -> dict[str, Any]:
        """Get manager's picks."""
        return self.client.get_entry_picks(manager_id, event_id)

    def get_live_event(self, event_id: int) -> dict[str, Any]:
        """Get live event data."""
        return self.client.get_live_event(event_id)

    def set_cookies(self, cookies: dict[str, str]) -> None:
        """Set authentication cookies."""
        self.client.set_cookies(cookies)

    def close(self) -> None:
        """Close the client."""
        if hasattr(self.client, "close"):
            self.client.close()


# =============================================================================
# Module-level cache instance
# =============================================================================

_cache: APICache | None = None


def get_cache(cache_dir: str | Path = ".cache") -> APICache:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = APICache(cache_dir)
    return _cache
