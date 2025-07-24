"""
Cache backend implementations for persistent kernel storage.
"""

import asyncio
import json
import pickle
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field

import redis.asyncio as aioredis
import asyncpg
from pydantic import BaseModel

from ..utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "CacheEntry",
    "CacheBackend",
    "RedisConfig",
    "RedisBackend",
    "PostgreSQLConfig", 
    "PostgreSQLBackend",
]


class CacheEntry(BaseModel):
    """Represents a cached kernel entry."""
    
    kernel_id: str
    kernel_data: bytes
    metadata: Dict[str, Any]
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    tier: str = "L2"  # L1=memory, L2=redis, L3=postgres, L4=s3


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry from cache."""
        pass
    
    @abstractmethod
    async def put(self, key: str, entry: CacheEntry) -> bool:
        """Store entry in cache."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Remove entry from cache."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 50
    ttl_seconds: int = 3600  # 1 hour default TTL


class RedisBackend(CacheBackend):
    """Redis-based cache backend for hot kernel data."""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self._pool: Optional[aioredis.ConnectionPool] = None
        self._client: Optional[aioredis.Redis] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "puts": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def connect(self):
        """Establish Redis connection."""
        self._pool = aioredis.ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            max_connections=self.config.max_connections,
        )
        self._client = aioredis.Redis(connection_pool=self._pool)
        
        # Test connection
        await self._client.ping()
        logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve kernel from Redis cache."""
        try:
            data = await self._client.get(f"kernel:{key}")
            if data:
                self._stats["hits"] += 1
                
                # Update access stats
                await self._client.hincrby(f"kernel:stats:{key}", "access_count", 1)
                await self._client.hset(f"kernel:stats:{key}", "last_access", time.time())
                
                # Deserialize entry
                entry_dict = pickle.loads(data)
                return CacheEntry(**entry_dict)
            else:
                self._stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def put(self, key: str, entry: CacheEntry) -> bool:
        """Store kernel in Redis cache."""
        try:
            # Serialize entry
            entry_data = pickle.dumps(entry.model_dump())
            
            # Store with TTL
            success = await self._client.setex(
                f"kernel:{key}",
                self.config.ttl_seconds,
                entry_data
            )
            
            # Store stats separately
            stats = {
                "size_bytes": entry.size_bytes,
                "access_count": entry.access_count,
                "last_access": entry.last_access,
                "created_at": time.time()
            }
            await self._client.hset(f"kernel:stats:{key}", mapping=stats)
            await self._client.expire(f"kernel:stats:{key}", self.config.ttl_seconds)
            
            if success:
                self._stats["puts"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis put error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Remove kernel from Redis cache."""
        try:
            deleted = await self._client.delete(f"kernel:{key}", f"kernel:stats:{key}")
            if deleted > 0:
                self._stats["deletes"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if kernel exists in cache."""
        try:
            return await self._client.exists(f"kernel:{key}") > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self._client.info("memory")
            return {
                **self._stats,
                "memory_used_bytes": info.get("used_memory", 0),
                "memory_peak_bytes": info.get("used_memory_peak", 0),
                "connected_clients": info.get("connected_clients", 0),
                "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return self._stats


@dataclass
class PostgreSQLConfig:
    """PostgreSQL connection configuration."""
    
    host: str = "localhost"
    port: int = 5432
    database: str = "esper"
    user: str = "esper"
    password: str = ""
    min_connections: int = 10
    max_connections: int = 20


class PostgreSQLBackend(CacheBackend):
    """PostgreSQL-based cache backend for cold kernel data and metadata."""
    
    def __init__(self, config: PostgreSQLConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._stats = {
            "hits": 0,
            "misses": 0,
            "puts": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def connect(self):
        """Establish PostgreSQL connection pool."""
        self._pool = await asyncpg.create_pool(
            host=self.config.host,
            port=self.config.port,
            database=self.config.database,
            user=self.config.user,
            password=self.config.password,
            min_size=self.config.min_connections,
            max_size=self.config.max_connections,
        )
        
        # Create tables if not exist
        await self._create_tables()
        logger.info(f"Connected to PostgreSQL at {self.config.host}:{self.config.port}")
    
    async def disconnect(self):
        """Close PostgreSQL connection pool."""
        if self._pool:
            await self._pool.close()
    
    async def _create_tables(self):
        """Create cache tables if they don't exist."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kernel_cache (
                    kernel_id VARCHAR(255) PRIMARY KEY,
                    kernel_data BYTEA NOT NULL,
                    metadata JSONB,
                    size_bytes BIGINT,
                    access_count INTEGER DEFAULT 0,
                    last_access TIMESTAMP DEFAULT NOW(),
                    created_at TIMESTAMP DEFAULT NOW(),
                    tier VARCHAR(10) DEFAULT 'L3'
                );
                
                CREATE INDEX IF NOT EXISTS idx_kernel_last_access 
                ON kernel_cache(last_access);
                
                CREATE INDEX IF NOT EXISTS idx_kernel_access_count 
                ON kernel_cache(access_count);
                
                CREATE INDEX IF NOT EXISTS idx_kernel_metadata 
                ON kernel_cache USING GIN(metadata);
            """)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve kernel from PostgreSQL cache."""
        try:
            async with self._pool.acquire() as conn:
                # Get and update access stats atomically
                row = await conn.fetchrow("""
                    UPDATE kernel_cache 
                    SET access_count = access_count + 1,
                        last_access = NOW()
                    WHERE kernel_id = $1
                    RETURNING kernel_id, kernel_data, metadata, size_bytes, 
                              access_count, last_access, tier
                """, key)
                
                if row:
                    self._stats["hits"] += 1
                    return CacheEntry(
                        kernel_id=row["kernel_id"],
                        kernel_data=row["kernel_data"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                        access_count=row["access_count"],
                        last_access=row["last_access"].timestamp(),
                        size_bytes=row["size_bytes"],
                        tier=row["tier"]
                    )
                else:
                    self._stats["misses"] += 1
                    return None
                    
        except Exception as e:
            logger.error(f"PostgreSQL get error for key {key}: {e}")
            self._stats["errors"] += 1
            return None
    
    async def put(self, key: str, entry: CacheEntry) -> bool:
        """Store kernel in PostgreSQL cache."""
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO kernel_cache 
                    (kernel_id, kernel_data, metadata, size_bytes, access_count, last_access, tier)
                    VALUES ($1, $2, $3, $4, $5, to_timestamp($6), $7)
                    ON CONFLICT (kernel_id) 
                    DO UPDATE SET
                        kernel_data = EXCLUDED.kernel_data,
                        metadata = EXCLUDED.metadata,
                        size_bytes = EXCLUDED.size_bytes,
                        access_count = EXCLUDED.access_count,
                        last_access = EXCLUDED.last_access,
                        tier = EXCLUDED.tier
                """, 
                key, 
                entry.kernel_data,
                json.dumps(entry.metadata),
                entry.size_bytes,
                entry.access_count,
                entry.last_access,
                entry.tier
                )
                
                self._stats["puts"] += 1
                return True
                
        except Exception as e:
            logger.error(f"PostgreSQL put error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Remove kernel from PostgreSQL cache."""
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM kernel_cache WHERE kernel_id = $1",
                    key
                )
                
                if result.split()[-1] != "0":
                    self._stats["deletes"] += 1
                    return True
                return False
                
        except Exception as e:
            logger.error(f"PostgreSQL delete error for key {key}: {e}")
            self._stats["errors"] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if kernel exists in cache."""
        try:
            async with self._pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM kernel_cache WHERE kernel_id = $1)",
                    key
                )
                return result
                
        except Exception as e:
            logger.error(f"PostgreSQL exists error for key {key}: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            async with self._pool.acquire() as conn:
                stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_entries,
                        SUM(size_bytes) as total_bytes,
                        AVG(access_count) as avg_access_count,
                        MAX(last_access) as most_recent_access
                    FROM kernel_cache
                """)
                
                return {
                    **self._stats,
                    "total_entries": stats["total_entries"] or 0,
                    "total_bytes": stats["total_bytes"] or 0,
                    "avg_access_count": float(stats["avg_access_count"] or 0),
                    "most_recent_access": stats["most_recent_access"].isoformat() if stats["most_recent_access"] else None,
                    "hit_rate": self._stats["hits"] / max(1, self._stats["hits"] + self._stats["misses"])
                }
                
        except Exception as e:
            logger.error(f"PostgreSQL stats error: {e}")
            return self._stats
    
    async def cleanup_old_entries(self, days: int = 30) -> int:
        """Remove entries not accessed for specified days."""
        try:
            async with self._pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM kernel_cache 
                    WHERE last_access < NOW() - INTERVAL '%s days'
                """, days)
                
                deleted = int(result.split()[-1])
                logger.info(f"Cleaned up {deleted} old cache entries")
                return deleted
                
        except Exception as e:
            logger.error(f"PostgreSQL cleanup error: {e}")
            return 0