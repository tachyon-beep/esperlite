"""
Tests for ChunkManager module.

Validates tensor splitting, concatenation, and chunk operations.
"""

import pytest
import torch
import torch.nn as nn
from typing import List  # noqa: F401

from esper.morphogenetic_v2.kasmina.chunk_manager import ChunkManager


class TestChunkManager:
    """Test suite for ChunkManager functionality."""
    
    @pytest.fixture
    def chunk_manager(self):
        """Create a ChunkManager instance for testing."""
        layer_dim = 1024
        num_chunks = 16
        return ChunkManager(layer_dim, num_chunks)
    
    @pytest.fixture
    def sample_tensor(self):
        """Create sample input tensor."""
        batch_size = 32
        layer_dim = 1024
        return torch.randn(batch_size, layer_dim)
    
    def test_initialization(self):
        """Test ChunkManager initialization."""
        layer_dim = 1000
        num_chunks = 10
        
        cm = ChunkManager(layer_dim, num_chunks)
        
        assert cm.layer_dim == layer_dim
        assert cm.num_chunks == num_chunks
        assert cm.chunk_size == 100  # 1000 / 10
        assert cm.remainder == 0
        
        # Test with remainder
        cm2 = ChunkManager(1003, 10)
        assert cm2.chunk_size == 100
        assert cm2.remainder == 3
    
    def test_chunk_boundaries(self):
        """Test chunk boundary computation."""
        # Equal chunks
        cm1 = ChunkManager(100, 10)
        assert len(cm1.chunk_starts) == 10
        assert len(cm1.chunk_ends) == 10
        assert cm1.chunk_starts[0] == 0
        assert cm1.chunk_ends[-1] == 100
        
        # Unequal chunks (with remainder)
        cm2 = ChunkManager(103, 10)
        # First 3 chunks should have size 11, rest have size 10
        assert cm2.get_chunk_size(0) == 11
        assert cm2.get_chunk_size(1) == 11
        assert cm2.get_chunk_size(2) == 11
        assert cm2.get_chunk_size(3) == 10
        assert cm2.get_chunk_size(9) == 10
    
    def test_split_tensor(self, chunk_manager, sample_tensor):
        """Test tensor splitting into chunks."""
        chunks = chunk_manager.split_tensor(sample_tensor)
        
        # Check number of chunks
        assert len(chunks) == chunk_manager.num_chunks
        
        # Check chunk dimensions
        total_size = 0
        for i, chunk in enumerate(chunks):
            expected_size = chunk_manager.get_chunk_size(i)
            assert chunk.shape == (sample_tensor.shape[0], expected_size)
            total_size += expected_size
        
        # Check total size matches
        assert total_size == chunk_manager.layer_dim
        
        # Check that chunks are views (no copy)
        chunks[0][0, 0] = 999.0
        start = chunk_manager.chunk_starts[0].item()
        assert sample_tensor[0, start] == 999.0
    
    def test_concatenate_chunks(self, chunk_manager, sample_tensor):
        """Test chunk concatenation."""
        # Split and concatenate
        chunks = chunk_manager.split_tensor(sample_tensor)
        reconstructed = chunk_manager.concatenate_chunks(chunks)
        
        # Check shape
        assert reconstructed.shape == sample_tensor.shape
        
        # Check values match
        assert torch.allclose(reconstructed, sample_tensor)
    
    def test_split_concatenate_identity(self, chunk_manager):
        """Test that split->concatenate is an identity operation."""
        # Test with different batch sizes
        for batch_size in [1, 16, 64]:
            x = torch.randn(batch_size, chunk_manager.layer_dim)
            chunks = chunk_manager.split_tensor(x)
            reconstructed = chunk_manager.concatenate_chunks(chunks)
            assert torch.allclose(x, reconstructed, atol=1e-6)
    
    def test_apply_to_chunks(self, chunk_manager, sample_tensor):
        """Test applying function to chunks."""
        # Define a simple transformation
        def scale_chunk(chunk, chunk_idx, scale=2.0):
            return chunk * scale
        
        # Apply transformation
        result = chunk_manager.apply_to_chunks(sample_tensor, scale_chunk, scale=3.0)
        
        # Check result
        assert result.shape == sample_tensor.shape
        assert torch.allclose(result, sample_tensor * 3.0)
    
    def test_create_chunk_mask(self, chunk_manager):
        """Test chunk mask creation."""
        # Create mask for specific chunks
        active_chunks = torch.zeros(chunk_manager.num_chunks, dtype=torch.bool)
        active_chunks[0] = True
        active_chunks[5] = True
        active_chunks[10] = True
        
        mask = chunk_manager.create_chunk_mask(active_chunks)
        
        # Check mask shape
        assert mask.shape == (chunk_manager.layer_dim,)
        
        # Check active regions
        assert mask[chunk_manager.chunk_starts[0]:chunk_manager.chunk_ends[0]].all()
        assert mask[chunk_manager.chunk_starts[5]:chunk_manager.chunk_ends[5]].all()
        assert mask[chunk_manager.chunk_starts[10]:chunk_manager.chunk_ends[10]].all()
        
        # Check inactive regions
        assert not mask[chunk_manager.chunk_starts[1]:chunk_manager.chunk_ends[1]].any()
    
    def test_parallel_chunk_operation(self, chunk_manager, sample_tensor):
        """Test parallel operations on chunks."""
        # Create different operations for different chunks
        operations = []
        for i in range(chunk_manager.num_chunks):
            if i % 2 == 0:
                # Scale by 2 for even chunks
                chunk_size = chunk_manager.get_chunk_size(i)
                op = nn.Sequential(
                    nn.Linear(chunk_size, chunk_size, bias=False)
                )
                # Set weight to identity * 2
                with torch.no_grad():
                    op[0].weight.data = torch.eye(chunk_size) * 2.0
                operations.append(op)
            else:
                # Identity for odd chunks
                operations.append(None)
        
        # Apply operations
        result = chunk_manager.parallel_chunk_operation(sample_tensor, operations)
        
        # Check result
        assert result.shape == sample_tensor.shape
        
        # Verify even chunks are scaled
        chunks_in = chunk_manager.split_tensor(sample_tensor)
        chunks_out = chunk_manager.split_tensor(result)
        
        for i in range(chunk_manager.num_chunks):
            if i % 2 == 0:
                # Should be scaled by 2
                assert torch.allclose(chunks_out[i], chunks_in[i] * 2.0, atol=1e-5)
            else:
                # Should be unchanged
                assert torch.allclose(chunks_out[i], chunks_in[i])
    
    def test_get_chunk_statistics(self, chunk_manager):
        """Test chunk statistics computation."""
        # Create tensor with known statistics
        batch_size = 4
        x = torch.zeros(batch_size, chunk_manager.layer_dim)
        
        # Set different values for different chunks
        for i in range(chunk_manager.num_chunks):
            start = chunk_manager.chunk_starts[i].item()
            end = chunk_manager.chunk_ends[i].item()
            x[:, start:end] = i  # Constant value = chunk index
        
        # Get statistics
        means, variances = chunk_manager.get_chunk_statistics(x)
        
        # Check shapes
        assert means.shape == (batch_size, chunk_manager.num_chunks)
        assert variances.shape == (batch_size, chunk_manager.num_chunks)
        
        # Check values
        for i in range(chunk_manager.num_chunks):
            assert torch.allclose(means[:, i], torch.tensor(float(i)))
            assert torch.allclose(variances[:, i], torch.tensor(0.0))  # Constant = 0 variance
    
    def test_device_handling(self):
        """Test device movement."""
        if torch.cuda.is_available():
            # Create on CPU
            cm = ChunkManager(100, 10, device=torch.device("cpu"))
            assert cm.device.type == "cpu"
            assert cm.chunk_starts.device.type == "cpu"
            
            # Move to GPU
            cm_gpu = cm.to(torch.device("cuda"))
            assert cm_gpu.device.type == "cuda"
            assert cm_gpu.chunk_starts.device.type == "cuda"
            
            # Test operations on GPU
            x_gpu = torch.randn(16, 100, device="cuda")
            chunks = cm_gpu.split_tensor(x_gpu)
            reconstructed = cm_gpu.concatenate_chunks(chunks)
            assert reconstructed.device.type == "cuda"
            assert torch.allclose(x_gpu, reconstructed)
    
    def test_error_handling(self, chunk_manager):
        """Test error handling."""
        # Wrong tensor dimension
        wrong_dim = torch.randn(32, 500)  # Wrong last dimension
        with pytest.raises(ValueError, match="Expected tensor with last dimension"):
            chunk_manager.split_tensor(wrong_dim)
        
        # Wrong number of chunks for concatenation
        chunks = [torch.randn(32, 64) for _ in range(5)]  # Wrong number
        with pytest.raises(ValueError, match="Expected .* chunks"):
            chunk_manager.concatenate_chunks(chunks)
        
        # Invalid chunk index
        with pytest.raises(ValueError, match="Chunk index .* out of range"):
            chunk_manager.get_chunk_size(100)
    
    def test_edge_cases(self):
        """Test edge cases."""
        # Single chunk
        cm1 = ChunkManager(100, 1)
        x = torch.randn(16, 100)
        chunks = cm1.split_tensor(x)
        assert len(chunks) == 1
        assert torch.equal(chunks[0], x)
        
        # Many small chunks
        cm2 = ChunkManager(100, 100)
        x = torch.randn(16, 100)
        chunks = cm2.split_tensor(x)
        assert len(chunks) == 100
        assert all(chunk.shape[1] == 1 for chunk in chunks)
        
        # Large remainder
        cm3 = ChunkManager(107, 10)
        x = torch.randn(16, 107)
        chunks = cm3.split_tensor(x)
        reconstructed = cm3.concatenate_chunks(chunks)
        assert torch.equal(x, reconstructed)
        assert reconstructed is not None  # Use the variable
    
    def test_performance_characteristics(self, chunk_manager):
        """Test that operations are efficient."""
        import time
        
        # Large tensor
        x = torch.randn(128, chunk_manager.layer_dim)
        
        # Time splitting (should be fast due to views)
        start = time.perf_counter()
        chunks = chunk_manager.split_tensor(x)
        split_time = time.perf_counter() - start
        
        # Time concatenation
        start = time.perf_counter()
        reconstructed = chunk_manager.concatenate_chunks(chunks)
        concat_time = time.perf_counter() - start
        
        # Both should be fast (< 1ms on most hardware)
        assert split_time < 0.001
        assert concat_time < 0.01  # Concatenation is slightly slower
        
        # Verify no memory copies in splitting
        chunks[0][0, 0] = 12345.0
        assert x[0, 0] == 12345.0  # Original tensor should be modified


class TestChunkManagerIntegration:
    """Integration tests with neural network layers."""
    
    def test_with_linear_layer(self):
        """Test chunk processing with linear layers."""
        # Setup
        layer_dim = 256
        num_chunks = 8
        batch_size = 16
        
        cm = ChunkManager(layer_dim, num_chunks)
        x = torch.randn(batch_size, layer_dim)
        
        # Create per-chunk linear layers
        chunk_layers = []
        for i in range(num_chunks):
            chunk_size = cm.get_chunk_size(i)
            layer = nn.Linear(chunk_size, chunk_size)
            chunk_layers.append(layer)
        
        # Process chunks independently
        chunks = cm.split_tensor(x)
        output_chunks = []
        for i, (chunk, layer) in enumerate(zip(chunks, chunk_layers)):
            output_chunks.append(layer(chunk))
        
        # Reconstruct output
        output = cm.concatenate_chunks(output_chunks)
        
        # Verify output shape
        assert output.shape == x.shape
        
        # Verify gradients flow correctly
        loss = output.sum()
        loss.backward()
        
        for layer in chunk_layers:
            assert layer.weight.grad is not None
            assert layer.bias.grad is not None