"""
Focused benchmark for Phase 3 Triton kernels.
"""

import torch
import time
import numpy as np

from esper.morphogenetic_v2.triton.simple_forward_kernel import SimpleTritonLayer


def benchmark_configuration(config):
    """Benchmark a single configuration."""
    device = torch.device('cuda:0')
    
    # Create layer
    layer = SimpleTritonLayer(
        hidden_dim=config['hidden_dim'],
        num_seeds=config['num_seeds'],
        chunk_size=config['chunk_size']
    ).to(device)
    
    # Activate some seeds
    active_ratio = config.get('active_ratio', 0.3)
    num_active = int(config['num_seeds'] * active_ratio)
    
    for i in range(num_active):
        layer.activate_seed(i, blueprint_id=i % 10, strategy=i % 3)
    
    # Create input
    x = torch.randn(config['batch_size'], config['hidden_dim'], device=device)
    
    # Warmup
    for _ in range(10):
        _ = layer(x)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    iterations = 100
    for _ in range(iterations):
        _ = layer(x)
    
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    # Calculate metrics
    time_per_forward = elapsed / iterations * 1000  # ms
    
    # Memory bandwidth calculation
    # Each active seed processes chunk_size elements
    # Read input + write output = 2x bandwidth
    active_elements = num_active * config['chunk_size'] * config['batch_size']
    inactive_elements = (config['num_seeds'] - num_active) * config['chunk_size'] * config['batch_size']
    
    # Active seeds: read input + read weights + write output = 3x
    # Inactive seeds: read input + write output = 2x  
    total_bytes = (active_elements * 3 + inactive_elements * 2) * 4  # float32
    bandwidth_gb_s = total_bytes / (time_per_forward / 1000) / 1e9
    
    # GPU utilization estimate
    # RTX 4060 Ti theoretical bandwidth: ~288 GB/s
    theoretical_bandwidth = 288  # GB/s
    utilization = (bandwidth_gb_s / theoretical_bandwidth) * 100
    
    return {
        'time_ms': time_per_forward,
        'bandwidth_gb_s': bandwidth_gb_s,
        'utilization_pct': utilization,
        'active_seeds': num_active,
        'total_seeds': config['num_seeds']
    }


def main():
    """Run Triton kernel benchmarks."""
    print("Phase 3 Triton Kernel Performance Benchmarks")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Triton: 3.3.1")
    print()
    
    configurations = [
        # Small configurations
        {'name': 'Small-1', 'batch_size': 32, 'hidden_dim': 512, 'num_seeds': 100, 'chunk_size': 64},
        {'name': 'Small-2', 'batch_size': 64, 'hidden_dim': 512, 'num_seeds': 200, 'chunk_size': 64},
        
        # Medium configurations
        {'name': 'Medium-1', 'batch_size': 32, 'hidden_dim': 1024, 'num_seeds': 500, 'chunk_size': 128},
        {'name': 'Medium-2', 'batch_size': 64, 'hidden_dim': 1024, 'num_seeds': 500, 'chunk_size': 128},
        
        # Large configurations
        {'name': 'Large-1', 'batch_size': 16, 'hidden_dim': 2048, 'num_seeds': 1000, 'chunk_size': 256},
        {'name': 'Large-2', 'batch_size': 32, 'hidden_dim': 2048, 'num_seeds': 1000, 'chunk_size': 256},
        
        # Extreme configuration
        {'name': 'Extreme', 'batch_size': 8, 'hidden_dim': 4096, 'num_seeds': 2000, 'chunk_size': 512},
        
        # High seed count
        {'name': 'ManySeeds', 'batch_size': 16, 'hidden_dim': 1024, 'num_seeds': 5000, 'chunk_size': 64},
    ]
    
    print(f"{'Config':<12} {'Batch':<6} {'Hidden':<7} {'Seeds':<6} {'Chunk':<6} {'Time(ms)':<9} {'BW(GB/s)':<10} {'GPU %':<6}")
    print("-" * 80)
    
    results = []
    for config in configurations:
        try:
            result = benchmark_configuration(config)
            
            print(f"{config['name']:<12} {config['batch_size']:<6} {config['hidden_dim']:<7} "
                  f"{config['num_seeds']:<6} {config['chunk_size']:<6} "
                  f"{result['time_ms']:<9.2f} {result['bandwidth_gb_s']:<10.1f} "
                  f"{result['utilization_pct']:<6.1f}")
            
            results.append({**config, **result})
            
        except Exception as e:
            print(f"{config['name']:<12} FAILED: {e}")
    
    # Summary statistics
    if results:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        avg_time = np.mean([r['time_ms'] for r in results])
        min_time = min(r['time_ms'] for r in results)
        max_bandwidth = max(r['bandwidth_gb_s'] for r in results)
        avg_utilization = np.mean([r['utilization_pct'] for r in results])
        
        print(f"Average forward pass time: {avg_time:.2f} ms")
        print(f"Best forward pass time: {min_time:.2f} ms")
        print(f"Peak memory bandwidth: {max_bandwidth:.1f} GB/s")
        print(f"Average GPU utilization: {avg_utilization:.1f}%")
        
        # Performance vs target
        target_time = 0.1  # 100μs target
        achieved = min_time < target_time
        print(f"\nTarget: <{target_time} ms per forward pass")
        print(f"Status: {'✅ ACHIEVED' if achieved else '❌ NOT YET'} ({min_time:.3f} ms)")
        
        # Recommendations
        print("\nRecommendations:")
        if avg_utilization < 50:
            print("- Low GPU utilization. Consider:")
            print("  - Increasing batch size")
            print("  - Using larger chunk sizes")
            print("  - Fusing more operations")
        
        if min_time > target_time:
            print("- Target latency not met. Consider:")
            print("  - Optimizing block sizes")
            print("  - Reducing kernel launch overhead")
            print("  - Using persistent kernels")


if __name__ == '__main__':
    main()