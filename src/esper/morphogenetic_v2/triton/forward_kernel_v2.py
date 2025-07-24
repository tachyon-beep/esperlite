"""
Improved Triton forward kernel for Kasmina morphogenetic layer.

This version fixes type conversion issues and improves performance.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kasmina_forward_kernel_v2(
    # Input pointers
    activations_ptr,
    lifecycle_states_ptr,  # Separate arrays for better type handling
    blueprint_ids_ptr,
    grafting_strategies_ptr,
    blend_factors_ptr,
    blueprint_weights_ptr,
    # Output pointers
    output_ptr,
    telemetry_sum_ptr,
    telemetry_sumsq_ptr,
    telemetry_count_ptr,
    # Dimensions
    batch_size: int,
    hidden_dim: int,
    num_seeds: int,
    chunk_size: int,
    # Constants
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized GPU kernel for forward pass through morphogenetic layer.
    """
    # Program ID determines which batch/seed combination we process
    pid = tl.program_id(0)
    
    # Decode batch and seed indices
    batch_id = pid // num_seeds
    seed_id = pid % num_seeds
    
    # Bounds check
    if batch_id >= batch_size:
        return
    
    # Load seed state (now from separate arrays)
    lifecycle_state = tl.load(lifecycle_states_ptr + seed_id)
    blueprint_id = tl.load(blueprint_ids_ptr + seed_id)
    grafting_strategy = tl.load(grafting_strategies_ptr + seed_id)
    
    # Calculate chunk boundaries
    chunk_start = seed_id * chunk_size
    chunk_end = tl.minimum(chunk_start + chunk_size, hidden_dim)
    
    # Initialize accumulator for telemetry
    acc_sum = 0.0
    acc_sumsq = 0.0
    acc_count = 0
    
    # Process chunk in blocks
    for offset in range(0, chunk_size, BLOCK_SIZE):
        # Calculate block boundaries
        block_start = chunk_start + offset
        block_end = tl.minimum(block_start + BLOCK_SIZE, chunk_end)
        
        # Create mask for valid elements
        indices = block_start + tl.arange(0, BLOCK_SIZE)
        mask = indices < block_end
        
        # Load activations
        input_offset = batch_id * hidden_dim + indices
        activations = tl.load(activations_ptr + input_offset, mask=mask, other=0.0)
        
        # Default output is identity
        output = activations
        
        # Check if seed is active (lifecycle states 3-6)
        is_active = (lifecycle_state >= 3) & (lifecycle_state <= 6)
        
        if is_active:
            # Load blueprint weights
            blueprint_offset = blueprint_id * hidden_dim + indices
            weights = tl.load(blueprint_weights_ptr + blueprint_offset, mask=mask, other=1.0)
            
            # Apply transformation based on strategy
            if grafting_strategy == 0:  # Linear blending
                alpha = tl.load(blend_factors_ptr + seed_id)
                output = alpha * weights * activations + (1.0 - alpha) * activations
            elif grafting_strategy == 1:  # Multiplicative
                output = weights * activations
            else:  # Additive (default)
                output = activations + weights
        
        # Store output
        tl.store(output_ptr + input_offset, output, mask=mask)
        
        # Accumulate telemetry
        masked_sum = tl.sum(tl.where(mask, activations, 0.0))
        masked_sumsq = tl.sum(tl.where(mask, activations * activations, 0.0))
        masked_count = tl.sum(tl.where(mask, 1, 0))
        
        acc_sum += masked_sum
        acc_sumsq += masked_sumsq
        acc_count += masked_count
    
    # Store telemetry using atomic operations
    tl.atomic_add(telemetry_sum_ptr + seed_id, acc_sum)
    tl.atomic_add(telemetry_sumsq_ptr + seed_id, acc_sumsq)
    # Cast count to float for atomic add
    count_float = acc_count.to(tl.float32) if hasattr(acc_count, 'to') else tl.float32(acc_count)
    tl.atomic_add(telemetry_count_ptr + seed_id, count_float)


class TritonKasminaLayer(torch.nn.Module):
    """
    Triton-optimized Kasmina layer implementation.
    """
    
    def __init__(self, hidden_dim: int, num_seeds: int, chunk_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.chunk_size = chunk_size
        self.BLOCK_SIZE = 256  # Optimized for most GPUs
        
        # State tensors (Structure of Arrays)
        self.register_buffer('lifecycle_states', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('blueprint_ids', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('grafting_strategies', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('blend_factors', torch.ones(num_seeds, dtype=torch.float32) * 0.5)
        
        # Blueprint weights (10 blueprints by default)
        self.register_buffer('blueprint_weights', torch.randn(10, hidden_dim))
        
        # Telemetry buffers
        self.register_buffer('telemetry_sum', torch.zeros(num_seeds))
        self.register_buffer('telemetry_sumsq', torch.zeros(num_seeds))
        self.register_buffer('telemetry_count', torch.zeros(num_seeds))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using Triton kernel.
        
        Args:
            x: Input tensor [batch_size, hidden_dim]
            
        Returns:
            Output tensor [batch_size, hidden_dim]
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Ensure all tensors are on the same device
        if self.lifecycle_states.device != device:
            self._to_device(device)
        
        # Pre-allocate output
        output = torch.empty_like(x)
        
        # Reset telemetry
        self.telemetry_sum.zero_()
        self.telemetry_sumsq.zero_()
        self.telemetry_count.zero_()
        
        # Configure grid
        grid = (batch_size * self.num_seeds,)
        
        # Launch kernel
        kasmina_forward_kernel_v2[grid](
            x,
            self.lifecycle_states,
            self.blueprint_ids,
            self.grafting_strategies,
            self.blend_factors,
            self.blueprint_weights,
            output,
            self.telemetry_sum,
            self.telemetry_sumsq,
            self.telemetry_count,
            batch_size,
            self.hidden_dim,
            self.num_seeds,
            self.chunk_size,
            BLOCK_SIZE=self.BLOCK_SIZE
        )
        
        return output
    
    def _to_device(self, device):
        """Move all buffers to device."""
        self.lifecycle_states = self.lifecycle_states.to(device)
        self.blueprint_ids = self.blueprint_ids.to(device)
        self.grafting_strategies = self.grafting_strategies.to(device)
        self.blend_factors = self.blend_factors.to(device)
        self.blueprint_weights = self.blueprint_weights.to(device)
        self.telemetry_sum = self.telemetry_sum.to(device)
        self.telemetry_sumsq = self.telemetry_sumsq.to(device)
        self.telemetry_count = self.telemetry_count.to(device)
    
    def set_seed_state(self, seed_id: int, lifecycle_state: int, blueprint_id: int = 0, 
                      grafting_strategy: int = 0, blend_factor: float = 0.5):
        """Update state for a specific seed."""
        self.lifecycle_states[seed_id] = lifecycle_state
        self.blueprint_ids[seed_id] = blueprint_id
        self.grafting_strategies[seed_id] = grafting_strategy
        self.blend_factors[seed_id] = blend_factor
    
    def get_telemetry(self) -> dict:
        """Get telemetry statistics."""
        count = self.telemetry_count.cpu().numpy()
        sum_vals = self.telemetry_sum.cpu().numpy()
        sumsq_vals = self.telemetry_sumsq.cpu().numpy()
        
        # Compute mean and variance
        with torch.no_grad():
            mean = sum_vals / (count + 1e-6)
            variance = (sumsq_vals / (count + 1e-6)) - mean**2
            variance = np.maximum(variance, 0)  # Ensure non-negative
            
        return {
            'mean': mean,
            'variance': variance,
            'std': np.sqrt(variance),
            'count': count
        }


def benchmark_triton_kernel():
    """Benchmark the Triton kernel performance."""
    import time
    
    # Test configurations
    configs = [
        {'batch_size': 32, 'hidden_dim': 512, 'num_seeds': 100, 'chunk_size': 64},
        {'batch_size': 64, 'hidden_dim': 1024, 'num_seeds': 500, 'chunk_size': 128},
        {'batch_size': 128, 'hidden_dim': 2048, 'num_seeds': 1000, 'chunk_size': 256},
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config in configs:
        print(f"\nBenchmarking: {config}")
        
        # Create layer
        layer = TritonKasminaLayer(
            hidden_dim=config['hidden_dim'],
            num_seeds=config['num_seeds'],
            chunk_size=config['chunk_size']
        ).to(device)
        
        # Set some seeds as active
        for i in range(0, config['num_seeds'], 3):
            layer.set_seed_state(i, lifecycle_state=3, blueprint_id=i % 10)
        
        # Create input
        x = torch.randn(config['batch_size'], config['hidden_dim'], device=device)
        
        # Warmup
        for _ in range(10):
            _ = layer(x)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(100):
            _ = layer(x)
            
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed = (end - start) / 100
        throughput = config['batch_size'] * config['hidden_dim'] * 4 * 2 / (elapsed * 1e9)  # GB/s
        
        print(f"  Time per forward pass: {elapsed*1000:.2f} ms")
        print(f"  Memory throughput: {throughput:.2f} GB/s")
        
        # Get telemetry
        telemetry = layer.get_telemetry()
        active_seeds = (layer.lifecycle_states >= 3) & (layer.lifecycle_states <= 6)
        print(f"  Active seeds: {active_seeds.sum().item()}/{config['num_seeds']}")


if __name__ == '__main__':
    import numpy as np
    benchmark_triton_kernel()