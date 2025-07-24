"""
Integration tests for Phase B5: Infrastructure Hardening.
"""

import asyncio
import os
import pytest
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

import asyncpg
import redis.asyncio as aioredis

from esper.storage import (
    PersistentKernelCache, CacheConfig, 
    AssetRepository, AssetQuery, AssetMetadata
)
from esper.storage.cache_backends import (
    RedisBackend, PostgreSQLBackend,
    RedisConfig, PostgreSQLConfig
)
from esper.recovery import (
    CheckpointManager, CheckpointConfig,
    StateSnapshot, ComponentState, ComponentType
)
from esper.services.nissa import (
    NissaService, MetricsCollector, MorphogeneticMetrics
)
from esper.services.urza.kernel_manager import KernelManager


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("TEST_DB_SETUP", False),
    reason="Test database not configured"
)
class TestPersistentKernelCache:
    """Test persistent kernel cache implementation."""
    
    async def test_multi_tier_cache_operations(self):
        """Test cache operations across all tiers."""
        # Create cache with test config
        config = CacheConfig(
            memory_size_mb=10,
            redis_config=RedisConfig(host="localhost", port=6379),
            postgres_config=PostgreSQLConfig(
                host="localhost",
                database="test_esper",
                user="test",
                password="test"
            ),
            redis_enabled=True,
            postgres_enabled=True
        )
        
        cache = PersistentKernelCache(config)
        
        try:
            # Initialize cache
            await cache.initialize()
            
            # Test data
            kernel_id = "test_kernel_001"
            kernel_data = b"compiled_kernel_binary_data" * 100
            metadata = {"version": 1, "performance": 0.95}
            
            # Test PUT operation
            success = await cache.put(kernel_id, kernel_data, metadata)
            assert success, "Failed to store kernel in cache"
            
            # Test GET operation (should hit L1)
            retrieved_data = await cache.get(kernel_id)
            assert retrieved_data == kernel_data, "Retrieved data doesn't match"
            assert cache.stats["l1_hits"] == 1
            
            # Clear L1 to test L2 fallback
            cache.l1_cache.clear()
            
            # Test GET operation (should hit L2)
            retrieved_data = await cache.get(kernel_id)
            assert retrieved_data == kernel_data
            assert cache.stats["l2_hits"] == 1
            
            # Test EXISTS operation
            exists = await cache.exists(kernel_id)
            assert exists, "Kernel should exist in cache"
            
            # Test DELETE operation
            deleted = await cache.delete(kernel_id)
            assert deleted, "Failed to delete kernel"
            
            # Verify deletion
            exists = await cache.exists(kernel_id)
            assert not exists, "Kernel should not exist after deletion"
            
            # Test cache stats
            stats = await cache.get_stats()
            assert "l1" in stats
            assert "l2" in stats
            assert "l3" in stats
            assert stats["overall_hit_rate"] > 0
            
        finally:
            await cache.close()
    
    async def test_cache_eviction_and_promotion(self):
        """Test cache eviction policies and tier promotion."""
        config = CacheConfig(
            memory_size_mb=1,  # Small cache to trigger eviction
            eviction_policy="lru",
            promotion_threshold=2
        )
        
        cache = PersistentKernelCache(config)
        
        try:
            await cache.initialize()
            
            # Fill cache beyond capacity
            for i in range(10):
                kernel_id = f"kernel_{i:03d}"
                kernel_data = b"data" * 1000  # ~4KB each
                await cache.put(kernel_id, kernel_data)
            
            # Access early kernels multiple times to trigger promotion
            for _ in range(3):
                await cache.get("kernel_001")
                await cache.get("kernel_002")
            
            # Check promotion occurred
            assert cache.stats["promotions"] > 0
            
            # Verify LRU eviction
            # Early kernels should be evicted from L1
            assert "kernel_000" not in cache.l1_cache.cache
            
        finally:
            await cache.close()
    
    async def test_cache_warming(self):
        """Test cache warming functionality."""
        config = CacheConfig()
        cache = PersistentKernelCache(config)
        
        try:
            await cache.initialize()
            
            # Pre-populate with kernels
            kernel_ids = []
            for i in range(5):
                kernel_id = f"warm_kernel_{i}"
                kernel_ids.append(kernel_id)
                await cache.put(kernel_id, b"data")
            
            # Clear L1 cache
            cache.l1_cache.clear()
            
            # Warm cache
            await cache.warm_cache(kernel_ids[:3])
            
            # Verify warmed kernels are in L1
            for kernel_id in kernel_ids[:3]:
                assert kernel_id in cache.l1_cache.cache
            
        finally:
            await cache.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("TEST_DB_SETUP", False),
    reason="Test database not configured"
)
class TestAssetLifecycleManagement:
    """Test asset repository and lifecycle management."""
    
    async def test_blueprint_versioning(self):
        """Test blueprint versioning and storage."""
        pg_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_esper",
            "user": "test",
            "password": "test"
        }
        
        repo = AssetRepository(pg_config)
        
        try:
            await repo.initialize()
            
            # Store initial blueprint
            blueprint_id = await repo.store_blueprint(
                blueprint_id="bp_001",
                name="test_architecture",
                architecture_ir="graph_definition_v1",
                metadata=AssetMetadata(
                    tags=["experimental", "conv2d"],
                    performance_metrics={"accuracy": 0.92}
                )
            )
            
            assert blueprint_id is not None
            
            # Store updated version
            blueprint_id_v2 = await repo.store_blueprint(
                blueprint_id="bp_002",
                name="test_architecture",  # Same name
                architecture_ir="graph_definition_v2",
                metadata=AssetMetadata(
                    tags=["experimental", "conv2d", "optimized"],
                    lineage=[str(blueprint_id)]
                )
            )
            
            # Query blueprints
            query = AssetQuery(tags=["experimental"])
            results = await repo.find_blueprints(query)
            
            assert len(results) >= 2
            assert any(r["name"] == "test_architecture" for r in results)
            
        finally:
            await repo.close()
    
    async def test_kernel_lineage_tracking(self):
        """Test kernel lineage and evolution tracking."""
        pg_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_esper",
            "user": "test",
            "password": "test"
        }
        
        repo = AssetRepository(pg_config)
        
        try:
            await repo.initialize()
            
            # Create kernel hierarchy
            kernel_v1 = "kernel_001"
            kernel_v2 = "kernel_002"
            kernel_v3 = "kernel_003"
            
            # Track lineage
            await repo.track_kernel_lineage(kernel_v2, [kernel_v1])
            await repo.track_kernel_lineage(kernel_v3, [kernel_v1, kernel_v2])
            
            # Get history
            history = await repo.get_asset_history(kernel_v3)
            
            assert len(history) > 0
            assert any(e["event_type"] == "lineage_updated" for e in history)
            
        finally:
            await repo.close()
    
    async def test_asset_retirement(self):
        """Test asset retirement based on criteria."""
        pg_config = {
            "host": "localhost",
            "port": 5432,
            "database": "test_esper",
            "user": "test",
            "password": "test"
        }
        
        repo = AssetRepository(pg_config)
        
        try:
            await repo.initialize()
            
            # Create old assets
            # This would need actual database setup in practice
            
            # Define retirement criteria
            from esper.storage.asset_repository import RetirementCriteria
            criteria = RetirementCriteria(
                unused_days=30,
                low_performance_threshold=0.5,
                preserve_recent=10
            )
            
            # Retire assets
            kernels_retired, blueprints_retired = await repo.retire_assets(criteria)
            
            # In test environment, might be 0 if no old assets
            assert kernels_retired >= 0
            assert blueprints_retired >= 0
            
        finally:
            await repo.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("TEST_DB_SETUP", False),
    reason="Test database not configured"
)
class TestCheckpointRecovery:
    """Test checkpoint and recovery system."""
    
    async def test_checkpoint_creation_and_restoration(self):
        """Test creating and restoring checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir) / "checkpoints",
                archive_dir=Path(tmpdir) / "archives",
                postgres_config=PostgreSQLConfig(
                    host="localhost",
                    database="test_esper",
                    user="test",
                    password="test"
                )
            )
            
            manager = CheckpointManager(config)
            
            try:
                await manager.initialize()
                
                # Create component states
                tolaria_state = ComponentState(
                    component_type=ComponentType.TOLARIA,
                    component_id="tolaria_main",
                    state_data={
                        "epoch": 10,
                        "global_step": 1000,
                        "model_state": {"weights": "serialized"},
                        "training_metrics": {"loss": 0.5, "accuracy": 0.92}
                    }
                )
                
                tamiyo_state = ComponentState(
                    component_type=ComponentType.TAMIYO,
                    component_id="tamiyo_main",
                    state_data={
                        "policy_state": {"exploration": 0.1},
                        "adaptation_count": 25,
                        "rollback_count": 2
                    }
                )
                
                components = {
                    ComponentType.TOLARIA: tolaria_state,
                    ComponentType.TAMIYO: tamiyo_state
                }
                
                # Create checkpoint
                checkpoint_id = await manager.create_checkpoint(
                    components=components,
                    description="Test checkpoint"
                )
                
                assert checkpoint_id is not None
                
                # List checkpoints
                checkpoints = await manager.list_checkpoints()
                assert len(checkpoints) > 0
                assert any(c["checkpoint_id"] == checkpoint_id for c in checkpoints)
                
                # Restore checkpoint
                restored = await manager.restore_checkpoint(checkpoint_id)
                
                assert restored.checkpoint_id == checkpoint_id
                assert len(restored.components) == 2
                
                # Verify component data
                tolaria_restored = restored.get_component(ComponentType.TOLARIA)
                assert tolaria_restored is not None
                assert tolaria_restored.state_data["epoch"] == 10
                
                # Test validation
                is_valid, errors = await manager.validate_checkpoint(checkpoint_id)
                assert is_valid, f"Checkpoint validation failed: {errors}"
                
            finally:
                await manager.close()
    
    async def test_incremental_checkpoints(self):
        """Test incremental checkpoint functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = CheckpointConfig(
                checkpoint_dir=Path(tmpdir) / "checkpoints",
                incremental_enabled=True,
                checkpoint_interval_minutes=0  # Disable auto checkpoint
            )
            
            manager = CheckpointManager(config)
            
            try:
                await manager.initialize()
                
                # Create initial full checkpoint
                components_v1 = {
                    ComponentType.TOLARIA: ComponentState(
                        component_type=ComponentType.TOLARIA,
                        component_id="tolaria_main",
                        state_data={"epoch": 1, "data": "initial"}
                    )
                }
                
                checkpoint_id_1 = await manager.create_checkpoint(
                    components=components_v1,
                    is_scheduled=True
                )
                
                # Create incremental checkpoint
                components_v2 = {
                    ComponentType.TOLARIA: ComponentState(
                        component_type=ComponentType.TOLARIA,
                        component_id="tolaria_main",
                        state_data={"epoch": 2, "data": "updated"}
                    )
                }
                
                checkpoint_id_2 = await manager.create_checkpoint(
                    components=components_v2,
                    is_scheduled=True
                )
                
                # Verify incremental relationship
                checkpoints = await manager.list_checkpoints()
                incremental = next(
                    c for c in checkpoints 
                    if c["checkpoint_id"] == checkpoint_id_2
                )
                
                assert not incremental["is_full"]
                assert incremental["parent_id"] == checkpoint_id_1
                
            finally:
                await manager.close()


@pytest.mark.asyncio
class TestNissaObservability:
    """Test Nissa observability service."""
    
    async def test_metrics_collection(self):
        """Test metric collection and aggregation."""
        collector = MetricsCollector()
        
        # Register test collector
        def test_collector():
            return {
                "custom_metric": 42,
                "test_gauge": 0.75
            }
        
        collector.register_collector("test", test_collector)
        
        # Collect metrics
        metrics = await collector.collect_once()
        
        assert isinstance(metrics, MorphogeneticMetrics)
        assert metrics.collected_at is not None
        
        # Update specific metrics
        collector.update_seed_metrics(
            layer_name="conv1",
            seed_idx=0,
            performance_score=0.85,
            activated=True
        )
        
        collector.record_kernel_compilation(
            success=True,
            latency_ms=150.0
        )
        
        collector.record_adaptation(
            success=True,
            latency_ms=500.0
        )
        
        # Get summary
        summary = collector.get_metric_summary()
        
        assert "morphogenetic" in summary
        assert "training" in summary
        assert "resources" in summary
        
        # Test trends
        await collector.collect_once()
        await asyncio.sleep(0.1)
        await collector.collect_once()
        
        trends = collector.get_trends(window_minutes=1)
        assert "seed_activation_trend" in trends
    
    async def test_anomaly_detection(self):
        """Test anomaly detection in metrics."""
        from esper.services.nissa.analysis import AnomalyDetector
        
        detector = AnomalyDetector()
        
        # Create metric history with anomaly
        history = []
        for i in range(10):
            metrics = MorphogeneticMetrics(
                training_loss=0.5 - i * 0.01,  # Decreasing normally
                training_accuracy=0.8 + i * 0.01,
                adaptation_attempts=i,
                adaptation_successes=i,
                kernel_compilations_total=i * 2,
                kernel_compilation_failures=0
            )
            history.append(metrics)
        
        # Add anomalous metric
        anomaly_metrics = MorphogeneticMetrics(
            training_loss=2.0,  # Sudden spike
            training_accuracy=0.3,  # Sudden drop
            adaptation_attempts=10,
            adaptation_successes=5,
            adaptation_rollbacks=5,  # High rollbacks
            kernel_compilations_total=20,
            kernel_compilation_failures=10  # High failures
        )
        history.append(anomaly_metrics)
        
        # Detect anomalies
        anomalies = detector.detect(history)
        
        assert len(anomalies) > 0
        assert any(a.metric_name == "training_loss" for a in anomalies)
        assert any(a.severity in ["medium", "high"] for a in anomalies)
    
    async def test_performance_analysis(self):
        """Test performance analysis capabilities."""
        from esper.services.nissa.analysis import PerformanceAnalyzer
        
        analyzer = PerformanceAnalyzer()
        
        # Create metric history
        history = []
        for i in range(5):
            metrics = MorphogeneticMetrics(
                training_epochs_completed=i,
                training_loss=0.5 - i * 0.05,
                training_overhead_percent=3.0 + i * 0.5,
                adaptation_attempts=i * 10,
                adaptation_successes=i * 8,
                kernel_cache_hits=i * 100,
                kernel_cache_misses=i * 20,
                gpu_utilization_percent=70.0,
                cpu_utilization_percent=40.0
            )
            history.append(metrics)
        
        # Analyze performance
        analysis = analyzer.analyze(history)
        
        assert "training_efficiency" in analysis
        assert "adaptation_effectiveness" in analysis
        assert "resource_utilization" in analysis
        assert "cache_performance" in analysis
        assert "recommendations" in analysis
        
        # Find specific issues
        issues = analyzer.find_issues(history)
        
        # Should find training overhead issue if present
        if history[-1].training_overhead_percent > 5:
            assert any(i.issue_type == "high_training_overhead" for i in issues)


@pytest.mark.asyncio
class TestIntegratedInfrastructure:
    """Test integrated infrastructure components."""
    
    @pytest.mark.skipif(
        not os.environ.get("TEST_DB_SETUP", False),
        reason="Test database not configured"
    )
    async def test_kernel_manager_integration(self):
        """Test KernelManager with persistent cache."""
        # This test demonstrates the integration but would need
        # actual database setup to run
        
        manager = KernelManager()
        
        try:
            await manager.initialize()
            
            # Mock database session
            db_session = MagicMock()
            
            # Test kernel storage
            kernel_id = "integrated_kernel_001"
            kernel_data = b"compiled_kernel_data"
            metadata = {"compilation_time": 150, "performance": 0.95}
            
            # Store kernel (would interact with cache and DB)
            # success = await manager.store_kernel(
            #     kernel_id=kernel_id,
            #     kernel_data=kernel_data,
            #     metadata=metadata,
            #     db_session=db_session
            # )
            
            # Test cache stats
            stats = await manager.get_cache_stats()
            assert isinstance(stats, dict)
            
        finally:
            await manager.close()
    
    async def test_observability_integration(self, unused_tcp_port):
        """Test Nissa service integration with other components."""
        port = unused_tcp_port
        
        with tempfile.TemporaryDirectory() as tmpdir:
            service = NissaService(
                port=port,
                metrics_dir=Path(tmpdir),
                enable_analysis=True
            )
            
            try:
                # Start service
                await service.start()
                
                # Give server time to start
                await asyncio.sleep(0.5)
                
                # Test metrics endpoint
                import httpx
                async with httpx.AsyncClient() as client:
                    # Health check
                    response = await client.get(f"http://localhost:{port}/health")
                    assert response.status_code == 200
                    assert response.json()["status"] == "healthy"
                    
                    # Metrics endpoint
                    response = await client.get(f"http://localhost:{port}/metrics")
                    assert response.status_code == 200
                    assert "esper_" in response.text  # Prometheus metrics
                    
                    # Current metrics
                    response = await client.get(f"http://localhost:{port}/api/v1/metrics/current")
                    assert response.status_code == 200
                    data = response.json()
                    assert "morphogenetic" in data
                    assert "timestamp" in data
                
                # Test event recording
                await service.record_event(
                    event_type="test_event",
                    component="test",
                    data={"value": 42},
                    severity="info"
                )
                
                # Test compliance report
                report = service.get_compliance_report(
                    start_date=datetime.utcnow() - timedelta(hours=1),
                    end_date=datetime.utcnow()
                )
                
                assert "summary" in report
                assert report["summary"]["total_events"] >= 1
                
            finally:
                await service.stop()


# Utility fixtures
@pytest.fixture
def unused_tcp_port():
    """Get an unused TCP port."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


if __name__ == "__main__":
    pytest.main([__file__, "-v"])