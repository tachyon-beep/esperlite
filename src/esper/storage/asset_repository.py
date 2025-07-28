"""
Asset lifecycle management repository for blueprints and kernels.
"""

import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from uuid import UUID
from uuid import uuid4

import asyncpg
from pydantic import BaseModel

from ..utils.logging import get_logger

logger = get_logger(__name__)


class AssetQuery(BaseModel):
    """Query parameters for asset search."""

    tags: Optional[List[str]] = None
    status: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    performance_threshold: Optional[float] = None
    limit: int = 100
    offset: int = 0


class AssetMetadata(BaseModel):
    """Extended metadata for assets."""

    tags: List[str] = []
    lineage: List[str] = []  # Parent asset IDs
    performance_metrics: Dict[str, float] = {}
    usage_count: int = 0
    last_used: Optional[datetime] = None
    retirement_eligible: bool = False
    storage_tier: str = "hot"  # hot, warm, cold


class RetirementCriteria(BaseModel):
    """Criteria for retiring assets."""

    unused_days: int = 30
    low_performance_threshold: float = 0.5
    max_storage_gb: float = 100.0
    preserve_recent: int = 100  # Keep N most recent


class AssetRepository:
    """
    ACID-compliant asset management with rich querying capabilities.
    
    Provides versioning, lineage tracking, tag-based search, and
    lifecycle management for blueprints and kernels.
    """

    def __init__(self, postgres_config: Dict[str, Any]):
        self.config = postgres_config
        self._pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection and schema."""
        self._pool = await asyncpg.create_pool(
            host=self.config["host"],
            port=self.config["port"],
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"],
            min_size=10,
            max_size=20,
        )

        # Create enhanced schema
        await self._create_schema()
        logger.info("AssetRepository initialized")

    async def close(self):
        """Close database connections."""
        if self._pool:
            await self._pool.close()

    async def _create_schema(self):
        """Create or update database schema for asset management."""
        async with self._pool.acquire() as conn:
            # Enhanced blueprints table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS blueprints_v2 (
                    id UUID PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    architecture_ir TEXT NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    metadata JSONB DEFAULT '{}',
                    tags TEXT[] DEFAULT '{}',
                    lineage UUID[] DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    retired_at TIMESTAMP WITH TIME ZONE,
                    UNIQUE(name, version)
                );
                
                -- Indexes for efficient querying
                CREATE INDEX IF NOT EXISTS idx_blueprints_tags 
                ON blueprints_v2 USING GIN(tags);
                
                CREATE INDEX IF NOT EXISTS idx_blueprints_status 
                ON blueprints_v2(status) WHERE retired_at IS NULL;
                
                CREATE INDEX IF NOT EXISTS idx_blueprints_created 
                ON blueprints_v2(created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_blueprints_metadata 
                ON blueprints_v2 USING GIN(metadata);
            """)

            # Enhanced kernels table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS kernels_v2 (
                    id UUID PRIMARY KEY,
                    blueprint_id UUID NOT NULL,
                    compilation_id UUID NOT NULL,
                    status VARCHAR(50) NOT NULL,
                    compilation_pipeline VARCHAR(50) NOT NULL,
                    artifact_url TEXT,
                    performance_metrics JSONB DEFAULT '{}',
                    metadata JSONB DEFAULT '{}',
                    tags TEXT[] DEFAULT '{}',
                    usage_count INTEGER DEFAULT 0,
                    last_used TIMESTAMP WITH TIME ZONE,
                    storage_tier VARCHAR(20) DEFAULT 'hot',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    retired_at TIMESTAMP WITH TIME ZONE,
                    FOREIGN KEY (blueprint_id) REFERENCES blueprints_v2(id)
                );
                
                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_kernels_blueprint 
                ON kernels_v2(blueprint_id) WHERE retired_at IS NULL;
                
                CREATE INDEX IF NOT EXISTS idx_kernels_tags 
                ON kernels_v2 USING GIN(tags);
                
                CREATE INDEX IF NOT EXISTS idx_kernels_performance 
                ON kernels_v2 USING GIN(performance_metrics);
                
                CREATE INDEX IF NOT EXISTS idx_kernels_usage 
                ON kernels_v2(usage_count DESC, last_used DESC NULLS LAST);
            """)

            # Asset events table for audit trail
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS asset_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    asset_id UUID NOT NULL,
                    asset_type VARCHAR(50) NOT NULL,
                    event_type VARCHAR(50) NOT NULL,
                    event_data JSONB DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    created_by VARCHAR(255)
                );
                
                CREATE INDEX IF NOT EXISTS idx_asset_events_asset 
                ON asset_events(asset_id, created_at DESC);
                
                CREATE INDEX IF NOT EXISTS idx_asset_events_type 
                ON asset_events(event_type, created_at DESC);
            """)

    # Blueprint operations
    async def store_blueprint(
        self,
        blueprint_id: str,
        name: str,
        architecture_ir: str,
        metadata: Optional[AssetMetadata] = None
    ) -> UUID:
        """Store a new blueprint with versioning."""
        async with self._pool.acquire() as conn:
            # Get next version number
            version = await conn.fetchval("""
                SELECT COALESCE(MAX(version), 0) + 1 
                FROM blueprints_v2 
                WHERE name = $1
            """, name)

            # Insert blueprint
            blueprint_uuid = UUID(blueprint_id) if blueprint_id else uuid4()

            meta = metadata or AssetMetadata()

            await conn.execute("""
                INSERT INTO blueprints_v2 
                (id, name, version, architecture_ir, status, metadata, tags, lineage)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            blueprint_uuid, name, version, architecture_ir, "unvalidated",
            json.dumps(meta.model_dump()), meta.tags, meta.lineage
            )

            # Record event
            await self._record_event(
                conn, blueprint_uuid, "blueprint", "created",
                {"name": name, "version": version}
            )

            logger.info(f"Stored blueprint {name} v{version} with ID {blueprint_uuid}")
            return blueprint_uuid

    async def find_blueprints(
        self,
        query: AssetQuery
    ) -> List[Dict[str, Any]]:
        """Find blueprints matching query criteria."""
        async with self._pool.acquire() as conn:
            # Build dynamic query
            conditions = ["retired_at IS NULL"]
            params = []
            param_count = 0

            if query.tags:
                param_count += 1
                conditions.append(f"tags && ${param_count}")
                params.append(query.tags)

            if query.status:
                param_count += 1
                conditions.append(f"status = ${param_count}")
                params.append(query.status)

            if query.created_after:
                param_count += 1
                conditions.append(f"created_at >= ${param_count}")
                params.append(query.created_after)

            if query.created_before:
                param_count += 1
                conditions.append(f"created_at <= ${param_count}")
                params.append(query.created_before)

            where_clause = " AND ".join(conditions)

            # Add offset and limit
            param_count += 1
            limit_param = param_count
            param_count += 1
            offset_param = param_count

            query_sql = f"""
                SELECT id, name, version, status, tags, metadata,
                       created_at, updated_at
                FROM blueprints_v2
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${limit_param} OFFSET ${offset_param}
            """

            params.extend([query.limit, query.offset])

            rows = await conn.fetch(query_sql, *params)

            return [
                {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "version": row["version"],
                    "status": row["status"],
                    "tags": row["tags"],
                    "metadata": json.loads(row["metadata"]),
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
                for row in rows
            ]

    # Kernel operations
    async def track_kernel_lineage(
        self,
        kernel_id: UUID,
        parent_kernel_ids: List[UUID]
    ):
        """Track kernel lineage for evolution history."""
        async with self._pool.acquire() as conn:
            # Update kernel metadata with lineage
            await conn.execute("""
                UPDATE kernels_v2
                SET metadata = jsonb_set(
                    metadata, 
                    '{lineage}', 
                    to_jsonb($2::uuid[])
                ),
                updated_at = NOW()
                WHERE id = $1
            """, kernel_id, parent_kernel_ids)

            # Record lineage event
            await self._record_event(
                conn, kernel_id, "kernel", "lineage_updated",
                {"parent_ids": [str(pid) for pid in parent_kernel_ids]}
            )

    async def update_kernel_performance(
        self,
        kernel_id: UUID,
        performance_metrics: Dict[str, float]
    ):
        """Update kernel performance metrics."""
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE kernels_v2
                SET performance_metrics = $2,
                    usage_count = usage_count + 1,
                    last_used = NOW(),
                    updated_at = NOW()
                WHERE id = $1
            """, kernel_id, json.dumps(performance_metrics))

    # Lifecycle management
    async def retire_assets(
        self,
        criteria: RetirementCriteria
    ) -> Tuple[int, int]:
        """Retire assets based on criteria."""
        async with self._pool.acquire() as conn:
            # Find retirement candidates
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=criteria.unused_days)

            # Retire old unused kernels
            kernel_result = await conn.execute("""
                UPDATE kernels_v2
                SET retired_at = NOW(),
                    storage_tier = 'archive'
                WHERE retired_at IS NULL
                AND (
                    (last_used < $1 OR last_used IS NULL)
                    OR (performance_metrics->>'accuracy' < $2::text)::boolean
                )
                AND created_at NOT IN (
                    SELECT created_at 
                    FROM kernels_v2 
                    WHERE retired_at IS NULL
                    ORDER BY created_at DESC 
                    LIMIT $3
                )
            """, cutoff_date, str(criteria.low_performance_threshold), criteria.preserve_recent)

            kernels_retired = int(kernel_result.split()[-1])

            # Retire old unvalidated blueprints
            blueprint_result = await conn.execute("""
                UPDATE blueprints_v2
                SET retired_at = NOW()
                WHERE retired_at IS NULL
                AND status = 'unvalidated'
                AND created_at < $1
                AND id NOT IN (
                    SELECT DISTINCT blueprint_id 
                    FROM kernels_v2 
                    WHERE retired_at IS NULL
                )
            """, cutoff_date)

            blueprints_retired = int(blueprint_result.split()[-1])

            logger.info(
                f"Retired {kernels_retired} kernels and "
                f"{blueprints_retired} blueprints"
            )

            return kernels_retired, blueprints_retired

    async def optimize_storage(self) -> Dict[str, Any]:
        """Optimize storage by moving cold assets to archive tiers."""
        async with self._pool.acquire() as conn:
            # Move cold kernels to archive tier
            cold_threshold = datetime.now(timezone.utc) - timedelta(days=7)

            result = await conn.execute("""
                UPDATE kernels_v2
                SET storage_tier = CASE
                    WHEN last_used < $1 - INTERVAL '30 days' THEN 'archive'
                    WHEN last_used < $1 THEN 'cold'
                    ELSE 'warm'
                END
                WHERE retired_at IS NULL
                AND storage_tier != 'archive'
            """, cold_threshold)

            updated = int(result.split()[-1])

            # Get storage statistics
            stats = await conn.fetchrow("""
                SELECT 
                    COUNT(*) FILTER (WHERE storage_tier = 'hot') as hot_count,
                    COUNT(*) FILTER (WHERE storage_tier = 'warm') as warm_count,
                    COUNT(*) FILTER (WHERE storage_tier = 'cold') as cold_count,
                    COUNT(*) FILTER (WHERE storage_tier = 'archive') as archive_count,
                    SUM(COALESCE((metadata->>'size_bytes')::bigint, 0)) as total_bytes
                FROM kernels_v2
                WHERE retired_at IS NULL
            """)

            return {
                "updated_count": updated,
                "storage_distribution": {
                    "hot": stats["hot_count"],
                    "warm": stats["warm_count"],
                    "cold": stats["cold_count"],
                    "archive": stats["archive_count"]
                },
                "total_size_gb": stats["total_bytes"] / (1024**3) if stats["total_bytes"] else 0,
                "optimized_at": datetime.now(timezone.utc).isoformat()
            }

    async def _record_event(
        self,
        conn: asyncpg.Connection,
        asset_id: UUID,
        asset_type: str,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Record an asset event for audit trail."""
        await conn.execute("""
            INSERT INTO asset_events 
            (asset_id, asset_type, event_type, event_data, created_by)
            VALUES ($1, $2, $3, $4, $5)
        """, asset_id, asset_type, event_type, json.dumps(event_data), "system")

    async def get_asset_history(
        self,
        asset_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get event history for an asset."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT event_type, event_data, created_at, created_by
                FROM asset_events
                WHERE asset_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, asset_id, limit)

            return [
                {
                    "event_type": row["event_type"],
                    "event_data": json.loads(row["event_data"]),
                    "created_at": row["created_at"].isoformat(),
                    "created_by": row["created_by"]
                }
                for row in rows
            ]
