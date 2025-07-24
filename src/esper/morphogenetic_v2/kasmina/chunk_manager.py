"""
ChunkManager: Efficient tensor splitting and concatenation for parallel seed processing.

This module provides the core functionality for dividing activation tensors into
chunks that can be processed independently by logical seeds.
"""

import torch
from typing import List, Tuple, Optional
import logging
import math

logger = logging.getLogger(__name__)


class ChunkManager:
    """
    Manages the splitting of activation tensors into chunks and their reconstruction.
    
    This class provides efficient tensor operations for the chunked architecture,
    ensuring that activations can be divided among thousands of logical seeds
    without memory copies or performance degradation.
    """
    
    def __init__(self, layer_dim: int, num_chunks: int, device: Optional[torch.device] = None):
        """
        Initialize ChunkManager.
        
        Args:
            layer_dim: Dimension of the layer (number of features)
            num_chunks: Number of chunks to split the layer into
            device: Device for tensor operations (default: cuda if available)
        """
        self.layer_dim = layer_dim
        self.num_chunks = num_chunks
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Calculate chunk sizes
        self.chunk_size = layer_dim // num_chunks
        self.remainder = layer_dim % num_chunks
        
        # Pre-compute chunk boundaries for efficiency
        self._compute_chunk_boundaries()
        
        # Create chunk indices for gather/scatter operations
        self._create_chunk_indices()
        
        logger.info(
            "ChunkManager initialized: layer_dim=%d, num_chunks=%d, chunk_size=%d, remainder=%d",
            layer_dim, num_chunks, self.chunk_size, self.remainder
        )
    
    def _compute_chunk_boundaries(self):
        """Pre-compute chunk start and end indices."""
        self.chunk_starts = []
        self.chunk_ends = []
        
        current_pos = 0
        for i in range(self.num_chunks):
            # Distribute remainder evenly among first chunks
            chunk_size = self.chunk_size + (1 if i < self.remainder else 0)
            self.chunk_starts.append(current_pos)
            current_pos += chunk_size
            self.chunk_ends.append(current_pos)
        
        # Convert to tensors for GPU operations
        self.chunk_starts = torch.tensor(self.chunk_starts, device=self.device)
        self.chunk_ends = torch.tensor(self.chunk_ends, device=self.device)
    
    def _create_chunk_indices(self):
        """Create index tensors for efficient gather/scatter operations."""
        # Create a mapping from chunk index to feature indices
        self.chunk_indices = []
        
        for i in range(self.num_chunks):
            start = self.chunk_starts[i]
            end = self.chunk_ends[i]
            indices = torch.arange(start, end, device=self.device)
            self.chunk_indices.append(indices)
    
    def split_tensor(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Split input tensor into chunks.
        
        Args:
            x: Input tensor of shape (batch_size, layer_dim)
            
        Returns:
            List of chunk tensors, each of shape (batch_size, chunk_size_i)
        """
        if x.size(-1) != self.layer_dim:
            raise ValueError(f"Expected tensor with last dimension {self.layer_dim}, got {x.size(-1)}")
        
        # Use views for zero-copy splitting when possible
        chunks = []
        for i in range(self.num_chunks):
            start = self.chunk_starts[i].item()
            end = self.chunk_ends[i].item()
            
            # Extract chunk using slicing (creates a view, not a copy)
            chunk = x[..., start:end]
            chunks.append(chunk)
        
        return chunks
    
    def concatenate_chunks(self, chunks: List[torch.Tensor]) -> torch.Tensor:
        """
        Concatenate chunks back into a full tensor.
        
        Args:
            chunks: List of chunk tensors
            
        Returns:
            Reconstructed tensor of shape (batch_size, layer_dim)
        """
        if len(chunks) != self.num_chunks:
            raise ValueError(f"Expected {self.num_chunks} chunks, got {len(chunks)}")
        
        # Use torch.cat for efficient concatenation
        return torch.cat(chunks, dim=-1)
    
    def get_chunk_size(self, chunk_idx: int) -> int:
        """Get the size of a specific chunk."""
        if chunk_idx >= self.num_chunks:
            raise ValueError(f"Chunk index {chunk_idx} out of range [0, {self.num_chunks})")
        
        return (self.chunk_ends[chunk_idx] - self.chunk_starts[chunk_idx]).item()
    
    def apply_to_chunks(self, x: torch.Tensor, chunk_fn, *args, **kwargs) -> torch.Tensor:
        """
        Apply a function to each chunk and reconstruct the result.
        
        This is an optimized path that avoids creating intermediate lists.
        
        Args:
            x: Input tensor
            chunk_fn: Function to apply to each chunk
            *args, **kwargs: Additional arguments for chunk_fn
            
        Returns:
            Reconstructed output tensor
        """
        output_chunks = []
        
        for i in range(self.num_chunks):
            start = self.chunk_starts[i].item()
            end = self.chunk_ends[i].item()
            
            # Extract chunk
            chunk = x[..., start:end]
            
            # Apply function
            output_chunk = chunk_fn(chunk, i, *args, **kwargs)
            output_chunks.append(output_chunk)
        
        # Concatenate results
        return torch.cat(output_chunks, dim=-1)
    
    def create_chunk_mask(self, active_chunks: torch.Tensor) -> torch.Tensor:
        """
        Create a mask for selective chunk processing.
        
        Args:
            active_chunks: Boolean tensor of shape (num_chunks,) indicating active chunks
            
        Returns:
            Feature mask of shape (layer_dim,)
        """
        mask = torch.zeros(self.layer_dim, dtype=torch.bool, device=self.device)
        
        for i in range(self.num_chunks):
            if active_chunks[i]:
                start = self.chunk_starts[i].item()
                end = self.chunk_ends[i].item()
                mask[start:end] = True
        
        return mask
    
    def parallel_chunk_operation(self, x: torch.Tensor, operations: List[Optional[torch.nn.Module]]) -> torch.Tensor:
        """
        Apply different operations to different chunks in parallel.
        
        Args:
            x: Input tensor
            operations: List of operations (None for identity)
            
        Returns:
            Output tensor with operations applied
        """
        if len(operations) != self.num_chunks:
            raise ValueError(f"Expected {self.num_chunks} operations, got {len(operations)}")
        
        # Pre-allocate output tensor
        output = torch.zeros_like(x)
        
        for i in range(self.num_chunks):
            start = self.chunk_starts[i].item()
            end = self.chunk_ends[i].item()
            
            chunk = x[..., start:end]
            
            if operations[i] is not None:
                # Apply operation
                output[..., start:end] = operations[i](chunk)
            else:
                # Identity operation
                output[..., start:end] = chunk
        
        return output
    
    def get_chunk_statistics(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute statistics for each chunk (mean and variance).
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (means, variances) each of shape (batch_size, num_chunks)
        """
        batch_size = x.size(0)
        means = torch.zeros(batch_size, self.num_chunks, device=x.device)
        variances = torch.zeros(batch_size, self.num_chunks, device=x.device)
        
        for i in range(self.num_chunks):
            start = self.chunk_starts[i].item()
            end = self.chunk_ends[i].item()
            
            chunk = x[:, start:end]
            means[:, i] = chunk.mean(dim=1)
            variances[:, i] = chunk.var(dim=1, unbiased=False)
        
        return means, variances
    
    def to(self, device: torch.device) -> "ChunkManager":
        """Move ChunkManager to a different device."""
        self.device = device
        self.chunk_starts = self.chunk_starts.to(device)
        self.chunk_ends = self.chunk_ends.to(device)
        self.chunk_indices = [idx.to(device) for idx in self.chunk_indices]
        return self