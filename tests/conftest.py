"""
Global test configuration and fixtures for Esper testing infrastructure.

This module provides common fixtures, utilities, and configuration
for comprehensive testing across the Esper platform.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Generator
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from esper.contracts.operational import HealthSignal
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.state_layout import KasminaStateLayout
from esper.services.oona_client import OonaClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def health_signal() -> HealthSignal:
    """Sample health signal for testing."""
    return HealthSignal(
        layer_id=1,
        seed_id=0,
        chunk_id=0,
        epoch=10,
        activation_variance=0.05,
        dead_neuron_ratio=0.02,
        avg_correlation=0.85,
        health_score=0.9
    )


@pytest.fixture
def sample_tensor() -> torch.Tensor:
    """Sample tensor for testing."""
    return torch.randn(4, 128)


@pytest.fixture
def sample_batch_tensor() -> torch.Tensor:
    """Sample batch tensor for testing."""
    return torch.randn(8, 256, 512)


@pytest.fixture
def sample_conv_tensor() -> torch.Tensor:
    """Sample convolutional tensor for testing."""
    return torch.randn(2, 3, 32, 32)


@pytest.fixture
def sample_transformer_tensor() -> torch.Tensor:
    """Sample transformer input tensor."""
    return torch.randn(4, 20, 256)


@pytest.fixture
def mock_oona_client() -> MagicMock:
    """Mock OonaClient for testing."""
    client = MagicMock(spec=OonaClient)
    client.publish_health_signal = AsyncMock()
    client.publish_adaptation_event = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
async def mock_kasmina_layer(mock_oona_client) -> KasminaLayer:
    """Mock KasminaLayer for testing."""
    layer = KasminaLayer(
        input_size=512,
        output_size=256,
        num_seeds=4,
        cache_size_mb=16,
        telemetry_enabled=False,
        layer_name="test_layer"
    )
    
    # Mock external dependencies
    layer.kernel_cache.load_kernel = AsyncMock(return_value=torch.randn(256, 512))
    layer.kernel_cache.unload_kernel = AsyncMock(return_value=True)
    layer.oona_client = mock_oona_client
    
    return layer


@pytest.fixture
def mock_state_layout() -> KasminaStateLayout:
    """Mock KasminaStateLayout for testing."""
    # Create a mock state layout for 4 seeds
    num_seeds = 4
    layout = KasminaStateLayout(
        lifecycle_states=torch.zeros(num_seeds, dtype=torch.uint8),
        active_kernel_id=torch.zeros(num_seeds, dtype=torch.int64),
        alpha_blend=torch.zeros(num_seeds, dtype=torch.float16),
        health_accumulator=torch.zeros(num_seeds, dtype=torch.float32),
        last_update_epoch=torch.zeros(num_seeds, dtype=torch.int32),
        exec_latency_us=torch.zeros(num_seeds, dtype=torch.int16),
        error_count=torch.zeros(num_seeds, dtype=torch.int16),
        fallback_active=torch.zeros(num_seeds, dtype=torch.bool)
    )
    return layout


@pytest.fixture
def simple_linear_model() -> nn.Module:
    """Simple linear model for testing."""
    return nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )


@pytest.fixture
def simple_conv_model() -> nn.Module:
    """Simple convolutional model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


@pytest.fixture
def simple_transformer_model() -> nn.Module:
    """Simple transformer model for testing."""
    class SimpleTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Linear(256, 256)
            self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
            self.norm = nn.LayerNorm(256)
            self.ffn = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256)
            )
            self.classifier = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.embedding(x)
            attn_out, _ = self.attention(x, x, x)
            x = self.norm(x + attn_out)
            ffn_out = self.ffn(x)
            x = self.norm(x + ffn_out)
            return self.classifier(x.mean(dim=1))
    
    return SimpleTransformer()


@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def performance_config() -> Dict[str, Any]:
    """Configuration for performance testing."""
    return {
        "max_overhead_percent": 5.0,
        "min_iterations": 100,
        "warmup_iterations": 10,
        "acceptable_transformer_overhead": 200.0,
        "target_cache_hit_rate": 0.95,
        "max_latency_ms": 10.0
    }


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """General test configuration."""
    return {
        "tolerance": {
            "atol": 1e-5,
            "rtol": 1e-5,
            "transformer_atol": 1e-3,  # More lenient for transformers
            "performance_threshold": 10.0
        },
        "dimensions": {
            "batch_size": 4,
            "seq_len": 20,
            "embed_dim": 256,
            "num_heads": 8,
            "conv_channels": [3, 32, 64],
            "linear_sizes": [128, 64, 32, 10]
        },
        "seeds": {
            "default_per_layer": 4,
            "test_seed": 42
        }
    }


@pytest.fixture(autouse=True)
def setup_logging():
    """Setup logging for tests."""
    logging.getLogger("esper").setLevel(logging.WARNING)
    # Reduce noise during testing


@pytest.fixture
def disable_telemetry():
    """Disable telemetry for testing."""
    return {"telemetry_enabled": False}


class TestModelFactory:
    """Factory for creating test models with various configurations."""
    
    @staticmethod
    def create_linear_model(layers: list) -> nn.Module:
        """Create a linear model with specified layer sizes."""
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:  # No activation after last layer
                modules.append(nn.ReLU())
        return nn.Sequential(*modules)
    
    @staticmethod
    def create_conv_model(channels: list, kernel_size: int = 3) -> nn.Module:
        """Create a convolutional model with specified channels."""
        modules = []
        for i in range(len(channels) - 1):
            modules.append(nn.Conv2d(
                channels[i], channels[i + 1],
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            if i < len(channels) - 2:
                modules.append(nn.ReLU())
                modules.append(nn.MaxPool2d(2))
        return nn.Sequential(*modules)
    
    @staticmethod
    def create_mixed_model(embed_dim: int = 256, num_classes: int = 10) -> nn.Module:
        """Create a mixed CNN + Transformer model."""
        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.proj = nn.Linear(128 * 8 * 8, embed_dim)
                self.attention = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
                self.norm = nn.LayerNorm(embed_dim)
                self.classifier = nn.Linear(embed_dim, num_classes)
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.flatten(1)
                x = self.proj(x).unsqueeze(1)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm(x + attn_out)
                return self.classifier(x.squeeze(1))
        
        return MixedModel()


@pytest.fixture
def test_model_factory() -> TestModelFactory:
    """Test model factory fixture."""
    return TestModelFactory()


# Performance testing utilities
class PerformanceMonitor:
    """Utility for monitoring performance during tests."""
    
    def __init__(self):
        self.measurements = []
    
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function."""
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        self.measurements.append(execution_time)
        return result, execution_time
    
    def get_average_time(self) -> float:
        """Get average execution time."""
        return sum(self.measurements) / len(self.measurements) if self.measurements else 0.0
    
    def reset(self):
        """Reset measurements."""
        self.measurements.clear()


@pytest.fixture
def performance_monitor() -> PerformanceMonitor:
    """Performance monitor fixture."""
    return PerformanceMonitor()


# Assertion helpers
def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    check_shape: bool = True
):
    """Assert that two tensors are close within tolerance."""
    if check_shape:
        assert tensor1.shape == tensor2.shape, f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    
    assert torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol), \
        f"Tensors not close. Max diff: {torch.max(torch.abs(tensor1 - tensor2)).item():.6f}"


def assert_no_nans_or_infs(tensor: torch.Tensor):
    """Assert tensor contains no NaNs or infinities."""
    assert torch.isfinite(tensor).all(), "Tensor contains NaNs or infinities"


def assert_performance_overhead(
    baseline_time: float,
    measured_time: float,
    max_overhead_percent: float
):
    """Assert that performance overhead is within acceptable limits."""
    overhead = (measured_time - baseline_time) / baseline_time * 100
    assert overhead <= max_overhead_percent, \
        f"Performance overhead {overhead:.1f}% exceeds limit {max_overhead_percent}%"


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow