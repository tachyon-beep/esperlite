"""
Triton forward kernel for Kasmina morphogenetic layer.

This kernel processes multiple seeds in parallel, applying blueprints
to active seeds while maintaining identity operations for dormant seeds.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def kasmina_forward_kernel(
    # Input pointers
    activations_ptr,
    state_tensor_ptr,
    blueprint_weights_ptr,
    # Output pointers
    output_ptr,
    telemetry_ptr,
    # Dimensions
    batch_size: int,
    hidden_dim: int,
    num_seeds: int,
    chunk_size: int,
    # Stride information
    stride_batch: int,
    stride_hidden: int,
    stride_seed: int,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    GPU kernel for forward pass through morphogenetic layer.

    Each thread block processes one seed's chunk of the activation tensor.
    """
    # Get program ID - determines which seed/batch this thread block handles
    pid = tl.program_id(0)

    # Calculate which seed and batch element we're processing
    total_chunks = num_seeds * batch_size
    if pid >= total_chunks:
        return

    seed_id = pid % num_seeds
    batch_id = pid // num_seeds

    # Load seed state (coalesced memory access)
    state_base = state_tensor_ptr + seed_id * 8  # 8 state variables per seed
    lifecycle_state = tl.load(state_base + 0).to(tl.int32)  # offset 0: lifecycle state
    blueprint_id = tl.load(state_base + 1).to(tl.int32)     # offset 1: blueprint ID
    # epochs_in_state = tl.load(state_base + 2)  # offset 2: epochs (unused in this kernel)
    grafting_strategy = tl.load(state_base + 3).to(tl.int32) # offset 3: strategy

    # Calculate chunk boundaries for this seed
    chunk_start = seed_id * chunk_size
    chunk_end = min(chunk_start + chunk_size, hidden_dim)
    chunk_size_actual = chunk_end - chunk_start

    # Create offset arrays for vectorized loads
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < chunk_size_actual

    # Calculate input tensor offsets
    input_base = activations_ptr + batch_id * stride_batch + chunk_start

    # Load activation chunk
    activations = tl.load(input_base + offsets, mask=mask, other=0.0)

    # Initialize output as identity (default for dormant seeds)
    output = activations

    # Define lifecycle states as constants (only those used)
    GRAFTING = 3
    FINE_TUNING = 6

    # Process based on lifecycle state
    # Use predication instead of branching for better GPU performance
    is_active = (lifecycle_state >= GRAFTING) & (lifecycle_state <= FINE_TUNING)

    # Apply blueprint transformation for active seeds
    if is_active:
        # Load blueprint weights
        blueprint_base = blueprint_weights_ptr + blueprint_id * hidden_dim + chunk_start
        weights = tl.load(blueprint_base + offsets, mask=mask, other=1.0)

        # Apply transformation based on grafting strategy
        if grafting_strategy == 0:  # Linear blending
            alpha = tl.load(state_base + 4)  # Load blend factor
            output = alpha * weights * activations + (1 - alpha) * activations
        elif grafting_strategy == 1:  # Multiplicative
            output = weights * activations
        else:  # Default: additive
            output = activations + weights

    # Store output
    output_base = output_ptr + batch_id * stride_batch + chunk_start
    tl.store(output_base + offsets, output, mask=mask)

    # Accumulate telemetry (always, for monitoring)
    # Compute statistics in-kernel to reduce memory bandwidth
    sum_val = tl.sum(activations, axis=0)
    sum_sq = tl.sum(activations * activations, axis=0)

    # Atomic updates to telemetry buffer
    telemetry_base = telemetry_ptr + seed_id * 4  # 4 telemetry values per seed
    tl.atomic_add(telemetry_base + 0, sum_val)      # Sum
    tl.atomic_add(telemetry_base + 1, sum_sq)       # Sum of squares
    tl.atomic_add(telemetry_base + 2, float(chunk_size_actual))  # Count
    tl.atomic_add(telemetry_base + 3, 1.0)          # Batch count


class KasminaForwardKernel:
    """Wrapper class for the Triton kernel with PyTorch integration."""

    def __init__(self, num_seeds: int, chunk_size: int, hidden_dim: int):
        self.num_seeds = num_seeds
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.BLOCK_SIZE = min(1024, chunk_size)  # Optimize block size

    def __call__(
        self,
        activations: torch.Tensor,
        state_tensor: torch.Tensor,
        blueprint_weights: torch.Tensor,
        output: torch.Tensor,
        telemetry: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute the forward kernel.

        Args:
            activations: Input tensor [batch_size, hidden_dim]
            state_tensor: Seed states [num_seeds, 8]
            blueprint_weights: Blueprint weight matrix [num_blueprints, hidden_dim]
            output: Pre-allocated output tensor [batch_size, hidden_dim]
            telemetry: Telemetry accumulation buffer [num_seeds, 4]

        Returns:
            Output tensor (same as output argument, for convenience)
        """
        batch_size = activations.shape[0]

        # Ensure tensors are contiguous and on GPU
        activations = activations.contiguous()
        state_tensor = state_tensor.contiguous()
        blueprint_weights = blueprint_weights.contiguous()

        # Calculate strides
        stride_batch = activations.stride(0)
        stride_hidden = activations.stride(1) if len(activations.shape) > 1 else 1
        stride_seed = self.chunk_size

        # Configure grid
        grid = (batch_size * self.num_seeds,)

        # Launch kernel
        kasmina_forward_kernel[grid](
            activations, state_tensor, blueprint_weights,
            output, telemetry,
            batch_size, self.hidden_dim, self.num_seeds, self.chunk_size,
            stride_batch, stride_hidden, stride_seed,
            BLOCK_SIZE=self.BLOCK_SIZE
        )

        return output