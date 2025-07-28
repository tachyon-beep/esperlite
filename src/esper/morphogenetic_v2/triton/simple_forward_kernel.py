"""
Simplified Triton forward kernel that avoids type conversion issues.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def simple_kasmina_kernel(
    # Pointers
    x_ptr, output_ptr,
    lifecycle_ptr, blueprint_ptr, strategy_ptr, weights_ptr,
    # Shapes
    batch_size: int, hidden_dim: int, num_seeds: int, chunk_size: int,
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    """Simplified kernel focusing on core functionality."""
    # Get program id
    pid = tl.program_id(0)

    # Decode indices
    bid = pid // num_seeds  # batch index
    sid = pid % num_seeds   # seed index

    if bid >= batch_size:
        return

    # Load seed state
    lifecycle = tl.load(lifecycle_ptr + sid)
    blueprint_id = tl.load(blueprint_ptr + sid)
    strategy = tl.load(strategy_ptr + sid)

    # Calculate chunk range
    start = sid * chunk_size
    end = tl.minimum(start + chunk_size, hidden_dim)

    # Process chunk
    for i in range(start, end, BLOCK_SIZE):
        # Calculate offsets
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < end

        # Load input
        x_offsets = bid * hidden_dim + offsets
        x = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)

        # Default: identity
        y = x

        # Apply transformation if active (states 3-6)
        if lifecycle >= 3 and lifecycle <= 6:
            # Load weights
            w_offsets = blueprint_id * hidden_dim + offsets
            w = tl.load(weights_ptr + w_offsets, mask=mask, other=1.0)

            # Apply strategy
            if strategy == 0:  # Multiplicative
                y = x * w
            elif strategy == 1:  # Additive
                y = x + w
            else:  # Mixed
                y = x * 0.5 + w * 0.5

        # Store output
        tl.store(output_ptr + x_offsets, y, mask=mask)


class SimpleTritonLayer(torch.nn.Module):
    """Simplified Triton layer for testing."""

    def __init__(self, hidden_dim: int, num_seeds: int, chunk_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_seeds = num_seeds
        self.chunk_size = chunk_size

        # State arrays
        self.register_buffer('lifecycle', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('blueprint', torch.zeros(num_seeds, dtype=torch.int32))
        self.register_buffer('strategy', torch.zeros(num_seeds, dtype=torch.int32))

        # Weight bank (10 blueprints)
        self.register_buffer('weights', torch.randn(10, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        output = torch.empty_like(x)

        # Grid configuration
        grid = lambda meta: (batch_size * self.num_seeds,)

        # Launch kernel
        simple_kasmina_kernel[grid](
            x, output,
            self.lifecycle, self.blueprint, self.strategy, self.weights,
            batch_size, self.hidden_dim, self.num_seeds, self.chunk_size,
            BLOCK_SIZE=256
        )

        return output

    def activate_seed(self, seed_id: int, blueprint_id: int = 0, strategy: int = 0):
        """Activate a seed with given configuration."""
        self.lifecycle[seed_id] = 3  # GRAFTING state
        self.blueprint[seed_id] = blueprint_id
        self.strategy[seed_id] = strategy
