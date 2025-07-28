"""
Multi-tiered persistent kernel cache implementation.
"""

import asyncio
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from ..utils.logging import get_logger
from .cache_backends import CacheEntry
from .cache_backends import PostgreSQLBackend
from .cache_backends import PostgreSQLConfig
from .cache_backends import RedisBackend
from .cache_backends import RedisConfig

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for multi-tiered cache."""

    # L1 Memory cache
    memory_size_mb: int = 512
    memory_ttl_seconds: int = 300  # 5 minutes

    # L2 Redis cache
    redis_config: RedisConfig = field(default_factory=RedisConfig)
    redis_enabled: bool = True

    # L3 PostgreSQL cache
    postgres_config: PostgreSQLConfig = field(default_factory=PostgreSQLConfig)
    postgres_enabled: bool = True

    # Eviction policies
    eviction_policy: str = "lru"  # lru, lfu, or custom
    promotion_threshold: int = 3  # Access count to promote between tiers

    # Performance tuning
    async_writes: bool = True
    write_through: bool = True  # Write to all tiers
    read_fallback: bool = True  # Fall back to lower tiers


class LRUCache:
    """Simple in-memory LRU cache for L1 tier."""

    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry and move to end (most recent)."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.cache[key] = entry
            return entry
        return None

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Add entry, evicting LRU entries if needed."""
        # Remove existing entry if present
        if key in self.cache:
            old_entry = self.cache.pop(key)
            self.current_size_bytes -= old_entry.size_bytes

        # Evict entries until we have space
        while (
            self.current_size_bytes + entry.size_bytes > self.max_size_bytes
            and len(self.cache) > 0
        ):
            lru_key, lru_entry = self.cache.popitem(last=False)
            self.current_size_bytes -= lru_entry.size_bytes
            logger.debug(f"Evicted LRU entry: {lru_key}")

        # Add new entry
        self.cache[key] = entry
        self.current_size_bytes += entry.size_bytes
        return True

    def delete(self, key: str) -> bool:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes
            return True
        return False

    def clear(self):
        """Clear all entries."""
        self.cache.clear()
        self.current_size_bytes = 0

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self.cache),
            "size_bytes": self.current_size_bytes,
            "max_size_bytes": self.max_size_bytes,
            "utilization": self.current_size_bytes / self.max_size_bytes,
        }


class PersistentKernelCache:
    """
    Multi-tiered kernel cache with persistence.

    Tiers:
    - L1: In-memory LRU cache (fastest, smallest)
    - L2: Redis cache (fast, medium size)
    - L3: PostgreSQL (slower, unlimited size)
    - L4: S3 (slowest, archival) - future enhancement
    """

    def __init__(self, config: CacheConfig):
        self.config = config

        # Initialize L1 memory cache
        max_bytes = config.memory_size_mb * 1024 * 1024
        self.l1_cache = LRUCache(max_bytes)

        # Initialize backend connections
        self.redis_backend: Optional[RedisBackend] = None
        self.postgres_backend: Optional[PostgreSQLBackend] = None

        # Statistics
        self.stats = {
            "l1_hits": 0,
            "l2_hits": 0,
            "l3_hits": 0,
            "total_misses": 0,
            "promotions": 0,
            "evictions": 0,
        }

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

    async def initialize(self):
        """Initialize cache backends and start background tasks."""
        # Connect to Redis
        if self.config.redis_enabled:
            self.redis_backend = RedisBackend(self.config.redis_config)
            await self.redis_backend.connect()
            logger.info("Redis backend initialized")

        # Connect to PostgreSQL
        if self.config.postgres_enabled:
            self.postgres_backend = PostgreSQLBackend(self.config.postgres_config)
            await self.postgres_backend.connect()
            logger.info("PostgreSQL backend initialized")

        # Start background tasks
        if self.config.async_writes:
            task = asyncio.create_task(self._background_promotion_task())
            self._background_tasks.append(task)

    async def close(self):
        """Close all connections and stop background tasks."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Close backends
        if self.redis_backend:
            await self.redis_backend.disconnect()
        if self.postgres_backend:
            await self.postgres_backend.disconnect()

    async def get(self, kernel_id: str) -> Optional[bytes]:
        """
        Retrieve kernel from cache, checking all tiers.

        Returns kernel data if found, None otherwise.
        """
        # Check L1 memory cache
        entry = self.l1_cache.get(kernel_id)
        if entry:
            self.stats["l1_hits"] += 1
            logger.debug(f"L1 cache hit for kernel {kernel_id}")
            return entry.kernel_data

        # Check L2 Redis cache
        if self.redis_backend and self.config.read_fallback:
            entry = await self.redis_backend.get(kernel_id)
            if entry:
                self.stats["l2_hits"] += 1
                logger.debug(f"L2 cache hit for kernel {kernel_id}")

                # Promote to L1
                self.l1_cache.put(kernel_id, entry)

                return entry.kernel_data

        # Check L3 PostgreSQL cache
        if self.postgres_backend and self.config.read_fallback:
            entry = await self.postgres_backend.get(kernel_id)
            if entry:
                self.stats["l3_hits"] += 1
                logger.debug(f"L3 cache hit for kernel {kernel_id}")

                # Promote to higher tiers if access count exceeds threshold
                if entry.access_count >= self.config.promotion_threshold:
                    await self._promote_entry(kernel_id, entry)

                return entry.kernel_data

        # Cache miss
        self.stats["total_misses"] += 1
        logger.debug(f"Cache miss for kernel {kernel_id}")
        return None

    async def put(
        self,
        kernel_id: str,
        kernel_data: bytes,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Store kernel in cache.

        Uses write-through or write-back depending on configuration.
        """
        # Create cache entry
        entry = CacheEntry(
            kernel_id=kernel_id,
            kernel_data=kernel_data,
            metadata=metadata or {},
            size_bytes=len(kernel_data),
            tier="L1",
        )

        # Always write to L1
        success = self.l1_cache.put(kernel_id, entry)

        if self.config.write_through:
            # Write to all tiers immediately
            tasks = []

            if self.redis_backend:
                entry.tier = "L2"
                tasks.append(self.redis_backend.put(kernel_id, entry))

            if self.postgres_backend:
                entry.tier = "L3"
                tasks.append(self.postgres_backend.put(kernel_id, entry))

            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to write to tier: {result}")
                        success = False

        return success

    async def delete(self, kernel_id: str) -> bool:
        """Remove kernel from all cache tiers."""
        results = []

        # Delete from L1
        results.append(self.l1_cache.delete(kernel_id))

        # Delete from L2
        if self.redis_backend:
            results.append(await self.redis_backend.delete(kernel_id))

        # Delete from L3
        if self.postgres_backend:
            results.append(await self.postgres_backend.delete(kernel_id))

        return any(results)

    async def exists(self, kernel_id: str) -> bool:
        """Check if kernel exists in any tier."""
        # Check L1
        if kernel_id in self.l1_cache.cache:
            return True

        # Check L2
        if self.redis_backend and await self.redis_backend.exists(kernel_id):
            return True

        # Check L3
        if self.postgres_backend and await self.postgres_backend.exists(kernel_id):
            return True

        return False

    async def warm_cache(self, kernel_ids: List[str]):
        """
        Pre-load frequently used kernels into higher tiers.

        Used during startup to optimize performance.
        """
        if not self.postgres_backend:
            return

        logger.info(f"Warming cache with {len(kernel_ids)} kernels")

        for kernel_id in kernel_ids:
            entry = await self.postgres_backend.get(kernel_id)
            if entry:
                # Load into L1 and L2
                self.l1_cache.put(kernel_id, entry)
                if self.redis_backend:
                    entry.tier = "L2"
                    await self.redis_backend.put(kernel_id, entry)

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {**self.stats, "l1": self.l1_cache.stats()}

        if self.redis_backend:
            stats["l2"] = await self.redis_backend.get_stats()

        if self.postgres_backend:
            stats["l3"] = await self.postgres_backend.get_stats()

        # Calculate hit rates
        total_hits = (
            self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        )
        total_requests = total_hits + self.stats["total_misses"]

        if total_requests > 0:
            stats["overall_hit_rate"] = total_hits / total_requests
            stats["l1_hit_rate"] = self.stats["l1_hits"] / total_requests
            stats["l2_hit_rate"] = self.stats["l2_hits"] / total_requests
            stats["l3_hit_rate"] = self.stats["l3_hits"] / total_requests

        return stats

    async def _promote_entry(self, kernel_id: str, entry: CacheEntry):
        """Promote entry to higher cache tiers based on access patterns."""
        self.stats["promotions"] += 1

        # Promote to L1
        self.l1_cache.put(kernel_id, entry)

        # Promote to L2 if not already there
        if self.redis_backend and entry.tier == "L3":
            entry.tier = "L2"
            await self.redis_backend.put(kernel_id, entry)

    async def _background_promotion_task(self):
        """
        Background task to promote hot entries and demote cold ones.

        Runs periodically to optimize cache tier distribution.
        """
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute

                # This is a placeholder for more sophisticated promotion logic
                # In production, this would analyze access patterns and promote/demote
                # entries between tiers based on usage

                logger.debug("Background promotion task completed")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background promotion task error: {e}")

    def generate_kernel_id(self, kernel_data: bytes) -> str:
        """Generate deterministic kernel ID from kernel data."""
        return hashlib.sha256(kernel_data).hexdigest()[:16]
