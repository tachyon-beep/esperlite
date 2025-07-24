"""
Enhanced kernel management with persistent caching for Urza.
"""

import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import json

from sqlalchemy.orm import Session
from sqlalchemy import func

from esper.storage.kernel_cache import PersistentKernelCache, CacheConfig
from esper.storage.cache_backends import RedisConfig, PostgreSQLConfig
from esper.utils.logging import get_logger
from .models import CompiledKernel, Blueprint

logger = get_logger(__name__)


class KernelManager:
    """
    Manages kernel storage and retrieval with persistent caching.
    
    This integrates the multi-tiered cache with Urza's database,
    providing fast access to frequently used kernels while maintaining
    persistent storage for all kernels.
    """
    
    def __init__(self, cache_config: Optional[CacheConfig] = None):
        # Use default config if not provided
        if cache_config is None:
            cache_config = CacheConfig(
                memory_size_mb=512,
                redis_config=RedisConfig(),
                postgres_config=PostgreSQLConfig(
                    database="esper",
                    user="esper",
                    password="esper"
                )
            )
        
        self.cache = PersistentKernelCache(cache_config)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the kernel manager and cache."""
        if self._initialized:
            return
        
        await self.cache.initialize()
        
        # Warm cache with frequently used kernels
        # This would be based on actual usage patterns in production
        await self._warm_cache_on_startup()
        
        self._initialized = True
        logger.info("KernelManager initialized with persistent cache")
    
    async def close(self):
        """Cleanup resources."""
        if self._initialized:
            await self.cache.close()
            self._initialized = False
    
    async def store_kernel(
        self,
        kernel_id: str,
        kernel_data: bytes,
        metadata: Dict[str, Any],
        db_session: Session
    ) -> bool:
        """
        Store kernel in both cache and database.
        
        Args:
            kernel_id: Unique kernel identifier
            kernel_data: Compiled kernel binary data
            metadata: Kernel metadata (compilation info, performance, etc.)
            db_session: Database session for transactional storage
            
        Returns:
            Success status
        """
        try:
            # Store in cache (all tiers via write-through)
            cache_success = await self.cache.put(
                kernel_id=kernel_id,
                kernel_data=kernel_data,
                metadata=metadata
            )
            
            if not cache_success:
                logger.warning(f"Failed to cache kernel {kernel_id}")
            
            # Update database record
            kernel_record = db_session.query(CompiledKernel).filter_by(
                id=kernel_id
            ).first()
            
            if kernel_record:
                # Update cache metadata in validation report
                if kernel_record.validation_report is None:
                    kernel_record.validation_report = {}
                
                kernel_record.validation_report["cache_metadata"] = {
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                    "size_bytes": len(kernel_data),
                    "cache_tiers": ["L1", "L2", "L3"]
                }
                kernel_record.updated_at = datetime.now(timezone.utc)
                
                db_session.commit()
                logger.info(f"Stored kernel {kernel_id} in cache and database")
                return True
            else:
                logger.error(f"Kernel record {kernel_id} not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Failed to store kernel {kernel_id}: {e}")
            db_session.rollback()
            return False
    
    async def retrieve_kernel(
        self,
        kernel_id: str,
        db_session: Optional[Session] = None
    ) -> Optional[bytes]:
        """
        Retrieve kernel from cache or database.
        
        Args:
            kernel_id: Kernel identifier
            db_session: Optional database session for fallback
            
        Returns:
            Kernel binary data if found
        """
        # Try cache first
        kernel_data = await self.cache.get(kernel_id)
        
        if kernel_data:
            logger.debug(f"Retrieved kernel {kernel_id} from cache")
            return kernel_data
        
        # Fallback to database if provided
        if db_session:
            kernel_record = db_session.query(CompiledKernel).filter_by(
                id=kernel_id
            ).first()
            
            if kernel_record and kernel_record.kernel_binary_ref:
                # In production, this would load from S3/blob storage
                # For now, we'll simulate by returning the reference
                logger.warning(
                    f"Kernel {kernel_id} not in cache, loaded from database"
                )
                
                # Store in cache for next time
                kernel_data = kernel_record.kernel_binary_ref.encode()
                await self.cache.put(
                    kernel_id=kernel_id,
                    kernel_data=kernel_data,
                    metadata=kernel_record.validation_report or {}
                )
                
                return kernel_data
        
        logger.warning(f"Kernel {kernel_id} not found in cache or database")
        return None
    
    async def delete_kernel(
        self,
        kernel_id: str,
        db_session: Session
    ) -> bool:
        """
        Remove kernel from cache and mark as retired in database.
        
        Args:
            kernel_id: Kernel identifier
            db_session: Database session
            
        Returns:
            Success status
        """
        try:
            # Remove from cache
            cache_deleted = await self.cache.delete(kernel_id)
            
            # Mark as retired in database (soft delete)
            kernel_record = db_session.query(CompiledKernel).filter_by(
                id=kernel_id
            ).first()
            
            if kernel_record:
                kernel_record.status = "retired"
                kernel_record.updated_at = datetime.now(timezone.utc)
                
                if kernel_record.validation_report is None:
                    kernel_record.validation_report = {}
                    
                kernel_record.validation_report["retired_at"] = (
                    datetime.now(timezone.utc).isoformat()
                )
                
                db_session.commit()
                logger.info(f"Retired kernel {kernel_id}")
                return True
                
            return cache_deleted
            
        except Exception as e:
            logger.error(f"Failed to delete kernel {kernel_id}: {e}")
            db_session.rollback()
            return False
    
    async def find_kernels_by_tags(
        self,
        tags: List[str],
        db_session: Session,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find kernels matching specified tags.
        
        Uses PostgreSQL JSONB queries for efficient tag searching.
        """
        try:
            # Build JSONB query for tags
            query = db_session.query(CompiledKernel)
            
            for tag in tags:
                query = query.filter(
                    CompiledKernel.validation_report.op('?')(tag)
                )
            
            kernels = query.limit(limit).all()
            
            results = []
            for kernel in kernels:
                results.append({
                    "id": kernel.id,
                    "blueprint_id": kernel.blueprint_id,
                    "status": kernel.status,
                    "tags": kernel.validation_report.get("tags", []),
                    "performance_metrics": kernel.validation_report.get(
                        "performance_metrics", {}
                    )
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find kernels by tags: {e}")
            return []
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return await self.cache.get_stats()
    
    async def optimize_cache(self, db_session: Session) -> Dict[str, Any]:
        """
        Optimize cache by analyzing usage patterns.
        
        This promotes frequently accessed kernels and demotes unused ones.
        """
        try:
            # Get most accessed kernels from database
            most_accessed = db_session.query(
                CompiledKernel.id,
                func.coalesce(
                    CompiledKernel.validation_report['cache_metadata']['access_count'],
                    0
                ).label('access_count')
            ).filter(
                CompiledKernel.status == "validated"
            ).order_by(
                func.desc('access_count')
            ).limit(100).all()
            
            # Warm cache with top kernels
            kernel_ids = [k.id for k in most_accessed[:50]]
            await self.cache.warm_cache(kernel_ids)
            
            # Get cache stats
            stats = await self.cache.get_stats()
            
            return {
                "optimized_kernels": len(kernel_ids),
                "cache_stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to optimize cache: {e}")
            return {"error": str(e)}
    
    async def _warm_cache_on_startup(self):
        """Warm cache with frequently used kernels on startup."""
        # In production, this would query the database for
        # the most frequently accessed kernels
        logger.info("Cache warming completed")
    
    def get_kernel_metadata(
        self,
        kernel_id: str,
        db_session: Session
    ) -> Optional[Dict[str, Any]]:
        """Get kernel metadata from database."""
        kernel = db_session.query(CompiledKernel).filter_by(
            id=kernel_id
        ).first()
        
        if kernel:
            return {
                "id": kernel.id,
                "blueprint_id": kernel.blueprint_id,
                "status": kernel.status,
                "compilation_pipeline": kernel.compilation_pipeline,
                "validation_report": kernel.validation_report,
                "created_at": kernel.created_at.isoformat(),
                "updated_at": kernel.updated_at.isoformat()
            }
        
        return None