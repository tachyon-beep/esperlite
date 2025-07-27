"""
Real component fixtures for integration testing.

This module provides fixtures that use actual components instead of mocks,
ensuring tests verify real functionality.
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

from esper.contracts.assets import KernelMetadata
from esper.execution.enhanced_kernel_cache import EnhancedKernelCache
from esper.execution.kasmina_layer import KasminaLayer
from esper.execution.kernel_executor import RealKernelExecutor
from esper.services.oona_client import OonaClient


class TestKernelFactory:
    """Factory for creating real test kernels."""
    
    @staticmethod
    def create_real_kernel(input_size: int, output_size: int) -> Tuple[bytes, KernelMetadata]:
        """Create a real kernel artifact with metadata."""
        # Create a simple linear transformation module
        module = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )
        
        # Trace the module to create TorchScript
        example_input = torch.randn(1, input_size)
        traced_module = torch.jit.trace(module, example_input)
        
        # Serialize to bytes
        buffer = io.BytesIO()
        torch.jit.save(traced_module, buffer)
        kernel_bytes = buffer.getvalue()
        
        # Create metadata
        metadata = KernelMetadata(
            kernel_id=f"test_kernel_{input_size}x{output_size}",
            blueprint_id="test_blueprint",
            name=f"Test Kernel {input_size}x{output_size}",
            input_shape=[input_size],
            output_shape=[output_size],
            parameter_count=input_size * output_size + output_size,
            device_requirements=["cpu"],
            memory_footprint_mb=len(kernel_bytes) / (1024 * 1024),
            compilation_target="torchscript",
            compatibility_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            checksum="test_checksum",
            performance_profile={"confidence": 0.8}
        )
        
        return kernel_bytes, metadata


@pytest.fixture
def real_kernel_cache():
    """Create a real kernel cache for testing."""
    cache = EnhancedKernelCache(
        max_cache_size_mb=64,
        max_entries=32
    )
    return cache


@pytest.fixture
def real_kernel_executor():
    """Create a real kernel executor for testing."""
    executor = RealKernelExecutor(
        device=torch.device("cpu"),
        max_kernel_cache_size=16,
        enable_validation=True,
        execution_timeout=5.0
    )
    return executor


@pytest.fixture
def real_kasmina_layer():
    """Create a real KasminaLayer for testing."""
    layer = KasminaLayer(
        input_size=64,
        output_size=32,
        num_seeds=4,
        cache_size_mb=32,
        telemetry_enabled=False,  # Disable by default for tests
        layer_name="test_layer"
    )
    return layer


@pytest.fixture
async def populated_kernel_cache(real_kernel_cache):
    """Create a kernel cache populated with test kernels."""
    factory = TestKernelFactory()
    
    # Add several test kernels
    sizes = [(64, 32), (32, 16), (128, 64)]
    for input_size, output_size in sizes:
        kernel_bytes, metadata = factory.create_real_kernel(input_size, output_size)
        
        # Manually add to cache
        kernel_id = metadata.kernel_id
        
        # Deserialize kernel to get tensor representation
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        
        # Get state dict as tensor representation
        state_dict = module.state_dict()
        # Flatten state dict to single tensor for cache storage
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        
        # Add to cache with metadata
        real_kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )
        
        # No need to store raw bytes - EnhancedKernelCache doesn't have kernel_bytes_cache
    
    return real_kernel_cache


@pytest.fixture
def real_test_model():
    """Create a real test model without mocking."""
    class RealTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(64)
            self.pool = nn.AdaptiveAvgPool2d((8, 8))
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return RealTestModel()


@pytest.fixture
def temp_kernel_storage():
    """Create temporary storage for kernel artifacts."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        
        # Create kernel storage structure
        (storage_path / "kernels").mkdir()
        (storage_path / "metadata").mkdir()
        
        yield storage_path


@pytest.fixture
def real_oona_client_optional():
    """
    Create a real OonaClient if Redis is available, otherwise None.
    
    This allows tests to run with real telemetry when infrastructure is available.
    """
    try:
        # Try to create real client
        client = OonaClient()
        # Just try to instantiate, don't test connection
        # Connection test would require Redis to be running
        return client
    except (ImportError, ConnectionError, RuntimeError) as e:
        # Redis not available, return None
        return None
    except Exception as e:
        # Catch any other Redis connection errors
        if "redis" in str(type(e)).lower() or "connection" in str(e).lower():
            return None
        raise


class RealComponentTestBase:
    """Base class for tests using real components."""
    
    @staticmethod
    async def execute_real_kernel(
        layer: KasminaLayer,
        kernel_bytes: bytes,
        metadata: KernelMetadata,
        input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """Execute a real kernel through the layer."""
        # Manually set up the kernel in the cache
        kernel_id = metadata.kernel_id
        
        # Skip storing raw bytes - EnhancedKernelCache doesn't have kernel_bytes_cache
        
        # Create tensor representation for cache
        buffer = io.BytesIO(kernel_bytes)
        module = torch.jit.load(buffer)
        state_dict = module.state_dict()
        tensors = []
        for param in state_dict.values():
            tensors.append(param.flatten())
        kernel_tensor = torch.cat(tensors)
        
        layer.kernel_cache._add_to_cache_with_metadata(
            kernel_id, kernel_tensor, metadata
        )
        
        # Load kernel into seed
        success = await layer.load_kernel(0, kernel_id)
        assert success, "Kernel loading should succeed"
        
        # Execute forward pass
        output = layer(input_tensor)
        return output
    
    @staticmethod
    def verify_kernel_execution(
        original_output: torch.Tensor,
        kernel_output: torch.Tensor,
        alpha: float
    ) -> bool:
        """Verify that kernel execution actually affected the output."""
        # If alpha > 0, output should be different from original
        if alpha > 0:
            return not torch.allclose(original_output, kernel_output, atol=1e-6)
        else:
            return torch.allclose(original_output, kernel_output, atol=1e-6)