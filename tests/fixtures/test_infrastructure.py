"""
Test infrastructure components that use real implementations instead of mocks.

These components provide in-memory implementations suitable for testing
without external dependencies.
"""

import asyncio
from typing import Dict, Any, List, Optional
from collections import defaultdict
import time

import pytest

from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.blueprints.registry import BlueprintRegistry
from esper.blueprints.metadata import BlueprintMetadata, BlueprintCategory


class InMemoryPerformanceTracker(PerformanceTracker):
    """In-memory implementation of PerformanceTracker for testing."""
    
    def __init__(self):
        super().__init__()
        self.metrics_store: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.history: List[Dict[str, Any]] = []
    
    async def get_seed_metrics(
        self, layer_name: str, seed_idx: int
    ) -> Dict[str, float]:
        """Get performance metrics for a specific seed."""
        return self.metrics_store.get(layer_name, {}).get(seed_idx, {
            "accuracy_trend": 0.5,
            "loss_trend": 0.5,
            "efficiency": 0.5,
        })
    
    async def record_seed_metrics(
        self, layer_name: str, seed_idx: int, metrics: Dict[str, float]
    ) -> None:
        """Record performance metrics for a seed."""
        self.metrics_store[layer_name][seed_idx].update(metrics)
        self.history.append({
            "timestamp": time.time(),
            "layer_name": layer_name,
            "seed_idx": seed_idx,
            "metrics": metrics
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get metric history for testing."""
        return self.history.copy()
    
    def clear(self):
        """Clear all metrics for test isolation."""
        self.metrics_store.clear()
        self.history.clear()


class InMemoryBlueprintRegistry(BlueprintRegistry):
    """In-memory implementation of BlueprintRegistry for testing."""
    
    def __init__(self):
        # Don't call super().__init__() as it expects manifest file
        self.templates: Dict[str, Dict[str, Any]] = {}
        self.metadata_cache: Dict[str, BlueprintMetadata] = {}
        self._initialized = True
    
    def register_template(
        self, name: str, template: Dict[str, Any], metadata: Optional[BlueprintMetadata] = None
    ) -> None:
        """Register a blueprint template in memory."""
        self.templates[name] = template
        if metadata:
            self.metadata_cache[name] = metadata
    
    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get blueprint template from memory."""
        return self.templates.get(name)
    
    def list_templates(self, category: Optional[BlueprintCategory] = None) -> List[str]:
        """List all registered template names."""
        if category:
            return [
                name for name, meta in self.metadata_cache.items()
                if meta and meta.category == category
            ]
        return list(self.templates.keys())
    
    def clear(self):
        """Clear all blueprints for test isolation."""
        self.templates.clear()
        self.metadata_cache.clear()


def create_test_blueprint_template(
    name: str = "test_blueprint",
    input_size: int = 64,
    output_size: int = 32
) -> Dict[str, Any]:
    """Create a simple test blueprint template."""
    return {
        "name": name,
        "version": "1.0",
        "architecture": {
            "type": "sequential",
            "layers": [
                {
                    "name": "linear1",
                    "type": "linear",
                    "params": {
                        "in_features": input_size,
                        "out_features": output_size
                    }
                }
            ]
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001
        }
    }


@pytest.fixture
def real_performance_tracker():
    """Create a real performance tracker for testing."""
    tracker = InMemoryPerformanceTracker()
    yield tracker
    tracker.clear()


@pytest.fixture
def real_blueprint_registry():
    """Create a real blueprint registry for testing."""
    registry = InMemoryBlueprintRegistry()
    
    # Pre-populate with some test blueprints
    for i in range(3):
        template = create_test_blueprint_template(
            name=f"test_blueprint_{i}",
            input_size=64 * (i + 1),
            output_size=32 * (i + 1)
        )
        metadata = BlueprintMetadata(
            blueprint_id=f"blueprint_{i}",
            name=f"test_blueprint_{i}",
            version="1.0",
            category=BlueprintCategory.EFFICIENCY,
            description=f"Test blueprint {i}",
            param_delta=100,
            flop_delta=1000,
            memory_footprint_kb=10,
            expected_latency_ms=1.0,
            past_accuracy_gain_estimate=0.05,
            stability_improvement_estimate=0.1,
            speed_improvement_estimate=0.0,
            risk_score=0.1,
            is_safe_action=True,
            requires_capability=[],
            compatible_layers=["linear", "conv2d"],
            incompatible_with=[],
            mergeable=True,
            warmup_steps=10,
            peak_benefit_window=100
        )
        registry.register_template(f"blueprint_{i}", template, metadata)
    
    yield registry
    registry.clear()


@pytest.fixture
def real_seed_orchestrator_components(real_performance_tracker, real_blueprint_registry, mock_oona_client):
    """Create real components for seed orchestrator testing."""
    return {
        "performance_tracker": real_performance_tracker,
        "blueprint_registry": real_blueprint_registry,
        "oona_client": mock_oona_client,  # Still mock OonaClient as it's external
        "urza_url": "http://test-urza:8000"  # Will be mocked at HTTP level if needed
    }


class MockUrzaServer:
    """Mock Urza server for testing without real HTTP."""
    
    def __init__(self):
        self.compiled_kernels = {}
        self.compilation_count = 0
    
    async def compile_kernel(self, blueprint_id: str) -> Dict[str, Any]:
        """Simulate kernel compilation."""
        self.compilation_count += 1
        kernel_id = f"kernel_{blueprint_id}_{self.compilation_count}"
        
        # Simulate compilation result
        result = {
            "kernel_id": kernel_id,
            "status": "success",
            "compilation_time_ms": 150,
            "optimization_level": "O2"
        }
        
        self.compiled_kernels[kernel_id] = result
        return result
    
    def get_kernel(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """Get compiled kernel info."""
        return self.compiled_kernels.get(kernel_id)
    
    def clear(self):
        """Clear for test isolation."""
        self.compiled_kernels.clear()
        self.compilation_count = 0


@pytest.fixture
def mock_urza_server():
    """Create a mock Urza server for testing."""
    server = MockUrzaServer()
    yield server
    server.clear()