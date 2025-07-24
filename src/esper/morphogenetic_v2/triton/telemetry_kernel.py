"""
Triton kernel for telemetry reduction in morphogenetic layers.

This kernel performs efficient reduction operations on telemetry data.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def telemetry_reduction_kernel(
    # Input/output pointers
    telemetry_ptr,
    output_ptr,
    # Dimensions
    num_seeds: int,
    num_metrics: int,
    # Block configuration
    BLOCK_SIZE: tl.constexpr,
):
    """
    GPU kernel for telemetry reduction.

    Computes mean and variance from accumulated telemetry.
    """
    # Get program ID
    pid = tl.program_id(0)

    # Check bounds
    if pid >= num_seeds:
        return

    # Load telemetry data for this seed
    base_offset = pid * num_metrics
    sum_val = tl.load(telemetry_ptr + base_offset + 0)
    sum_sq = tl.load(telemetry_ptr + base_offset + 1)
    count = tl.load(telemetry_ptr + base_offset + 2)
    batch_count = tl.load(telemetry_ptr + base_offset + 3)

    # Compute statistics
    if count > 0:
        mean = sum_val / count
        variance = (sum_sq / count) - (mean * mean)
        # Ensure non-negative variance (numerical stability)
        variance = tl.maximum(variance, 0.0)
    else:
        mean = 0.0
        variance = 0.0

    # Store results
    output_offset = pid * 2
    tl.store(output_ptr + output_offset + 0, mean)
    tl.store(output_ptr + output_offset + 1, variance)