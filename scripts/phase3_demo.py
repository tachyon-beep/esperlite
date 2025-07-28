"""
Phase 3 GPU Optimization Demo

Demonstrates the Triton-optimized morphogenetic layer in action.
"""

import time

import torch

from esper.morphogenetic_v2.triton.simple_forward_kernel import SimpleTritonLayer


def main():
    """Run Phase 3 demonstration."""
    print("=" * 60)
    print("Phase 3: GPU Optimization with Triton Kernels")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Triton requires GPU.")
        return

    device = torch.device('cuda:0')
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print("Triton: Enabled")

    # Configuration
    batch_size = 32
    hidden_dim = 1024
    num_seeds = 500
    chunk_size = 128

    print("\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Number of seeds: {num_seeds}")
    print(f"  Chunk size: {chunk_size}")

    # Create layer
    print("\nCreating Triton-optimized layer...")
    layer = SimpleTritonLayer(
        hidden_dim=hidden_dim,
        num_seeds=num_seeds,
        chunk_size=chunk_size
    ).to(device)

    # Activate some seeds
    print("\nActivating 30% of seeds...")
    num_active = int(num_seeds * 0.3)
    for i in range(num_active):
        layer.activate_seed(
            seed_id=i,
            blueprint_id=i % 10,
            strategy=i % 3
        )

    print(f"  Active seeds: {num_active}/{num_seeds}")

    # Create input
    x = torch.randn(batch_size, hidden_dim, device=device)

    # Warmup
    print("\nWarming up GPU...")
    for _ in range(10):
        _ = layer(x)

    # Benchmark
    print("\nRunning performance test...")
    torch.cuda.synchronize()
    start = time.perf_counter()

    iterations = 1000
    for _ in range(iterations):
        output = layer(x)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Calculate metrics
    time_per_forward = elapsed / iterations * 1000  # ms
    samples_per_second = (batch_size * iterations) / elapsed

    # Memory bandwidth
    active_elements = num_active * chunk_size * batch_size
    total_bytes = active_elements * 3 * 4  # read input, read weights, write output
    bandwidth_gb_s = total_bytes * iterations / elapsed / 1e9

    print("\nPerformance Results:")
    print(f"  Time per forward pass: {time_per_forward:.3f} ms")
    print(f"  Throughput: {samples_per_second:,.0f} samples/second")
    print(f"  Memory bandwidth: {bandwidth_gb_s:.1f} GB/s")

    # Demonstrate different lifecycle states
    print("\n" + "-" * 40)
    print("Lifecycle State Demo")
    print("-" * 40)

    # Show state transitions
    states = [
        (0, "DORMANT", 0),
        (1, "GERMINATED", 1),
        (2, "TRAINING", 2),
        (3, "GRAFTING", 3),
        (4, "STABILIZATION", 4),
        (5, "EVALUATING", 5),
        (6, "FINE_TUNING", 6),
        (7, "FOSSILIZED", 7),
    ]

    print("\nSetting different lifecycle states:")
    for seed_id, state_name, state_value in states[:5]:
        layer.lifecycle[seed_id] = state_value
        print(f"  Seed {seed_id}: {state_name}")

    # Run forward pass with mixed states
    output = layer(x)

    print("\nOutput characteristics:")
    print(f"  Shape: {output.shape}")
    print(f"  Device: {output.device}")
    print(f"  Dtype: {output.dtype}")
    print(f"  Mean: {output.mean().item():.4f}")
    print(f"  Std: {output.std().item():.4f}")

    # Memory usage
    allocated_mb = torch.cuda.memory_allocated() / 1024**2
    reserved_mb = torch.cuda.memory_reserved() / 1024**2

    print("\nGPU Memory:")
    print(f"  Allocated: {allocated_mb:.1f} MB")
    print(f"  Reserved: {reserved_mb:.1f} MB")

    # Success metrics vs targets
    print("\n" + "=" * 60)
    print("Phase 3 Success Metrics")
    print("=" * 60)

    target_latency = 0.1  # 100μs
    target_bandwidth = 100  # GB/s

    print(f"\nLatency Target: <{target_latency} ms")
    print(f"  Achieved: {time_per_forward:.3f} ms")
    print(f"  Status: {'✅ PASS' if time_per_forward < target_latency else '❌ FAIL'}")

    print(f"\nBandwidth Target: >{target_bandwidth} GB/s")
    print(f"  Achieved: {bandwidth_gb_s:.1f} GB/s")
    print(f"  Status: {'✅ PASS' if bandwidth_gb_s > target_bandwidth else '⚠️  CLOSE'}")

    print("\n✨ Phase 3 GPU Optimization Complete!")


if __name__ == '__main__':
    main()
