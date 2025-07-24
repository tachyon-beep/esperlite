"""
Phase 3 Triton kernel benchmarks.

Compares performance of:
1. Legacy PyTorch implementation
2. Phase 1/2 chunked implementation 
3. Phase 3 Triton kernels
"""

import torch
import time
import numpy as np
from typing import Dict, List

# Optional plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

from esper.morphogenetic_v2.triton.simple_forward_kernel import SimpleTritonLayer
from esper.morphogenetic_v2.kasmina.chunked_layer_v2 import ChunkedKasminaLayerV2
from esper.morphogenetic_v2.lifecycle.extended_lifecycle import ExtendedLifecycle


class BenchmarkSuite:
    """Comprehensive benchmark suite for morphogenetic layers."""
    
    def __init__(self, device='cuda:0'):
        self.device = torch.device(device)
        self.results = {}
        
    def benchmark_forward_pass(self, layer, x, warmup=10, iterations=100):
        """Benchmark forward pass performance."""
        # Warmup
        for _ in range(warmup):
            _ = layer(x)
            
        torch.cuda.synchronize()
        start = time.time()
        
        for _ in range(iterations):
            _ = layer(x)
            
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        return elapsed / iterations
    
    def profile_memory(self, layer, x):
        """Profile GPU memory usage."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        
        start_mem = torch.cuda.memory_allocated()
        
        # Forward pass
        _ = layer(x)
        
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        
        return (peak_mem - start_mem) / 1024**2  # MB
    
    def run_comparison(self, config: Dict):
        """Run comparative benchmarks."""
        print(f"\nBenchmarking configuration: {config}")
        
        batch_size = config['batch_size']
        hidden_dim = config['hidden_dim']
        num_seeds = config['num_seeds']
        chunk_size = config['chunk_size']
        active_ratio = config.get('active_ratio', 0.3)
        
        # Create layers
        triton_layer = SimpleTritonLayer(
            hidden_dim=hidden_dim,
            num_seeds=num_seeds,
            chunk_size=chunk_size
        ).to(self.device)
        
        # Dummy base layer for Phase 2
        base_layer = torch.nn.Linear(hidden_dim, hidden_dim).to(self.device)
        chunked_layer = ChunkedKasminaLayerV2(
            base_layer=base_layer,
            chunks_per_layer=num_seeds,
            device=self.device
        )
        
        # Activate seeds
        num_active = int(num_seeds * active_ratio)
        for i in range(num_active):
            # Triton layer
            triton_layer.activate_seed(i, blueprint_id=i % 10, strategy=i % 3)
            
            # Chunked layer
            chunked_layer.extended_state.update_state(
                seed_id=i,
                updates={
                    'lifecycle': ExtendedLifecycle.GRAFTING.value,
                    'blueprint': i % 10,
                    'strategy': i % 3
                }
            )
        
        # Create input
        x = torch.randn(batch_size, hidden_dim, device=self.device)
        
        # Benchmark Triton
        triton_time = self.benchmark_forward_pass(triton_layer, x)
        triton_mem = self.profile_memory(triton_layer, x)
        
        # Benchmark Phase 2 
        chunked_time = self.benchmark_forward_pass(chunked_layer, x)
        chunked_mem = self.profile_memory(chunked_layer, x)
        
        # Calculate metrics
        speedup = chunked_time / triton_time
        mem_reduction = (chunked_mem - triton_mem) / chunked_mem * 100
        
        # Throughput calculation
        total_data = batch_size * hidden_dim * 4 * 2  # float32, read+write
        triton_throughput = total_data / triton_time / 1e9  # GB/s
        chunked_throughput = total_data / chunked_time / 1e9
        
        results = {
            'config': config,
            'triton_time_ms': triton_time * 1000,
            'chunked_time_ms': chunked_time * 1000,
            'speedup': speedup,
            'triton_mem_mb': triton_mem,
            'chunked_mem_mb': chunked_mem,
            'mem_reduction_pct': mem_reduction,
            'triton_throughput_gb_s': triton_throughput,
            'chunked_throughput_gb_s': chunked_throughput,
        }
        
        # Print results
        print(f"  Triton time: {results['triton_time_ms']:.2f} ms")
        print(f"  Chunked time: {results['chunked_time_ms']:.2f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
        print(f"  Triton memory: {results['triton_mem_mb']:.1f} MB")
        print(f"  Chunked memory: {results['chunked_mem_mb']:.1f} MB")
        print(f"  Memory reduction: {results['mem_reduction_pct']:.1f}%")
        print(f"  Triton throughput: {results['triton_throughput_gb_s']:.2f} GB/s")
        
        return results
    
    def run_full_suite(self):
        """Run complete benchmark suite."""
        configurations = [
            # Small models
            {'batch_size': 32, 'hidden_dim': 512, 'num_seeds': 100, 'chunk_size': 64},
            {'batch_size': 64, 'hidden_dim': 512, 'num_seeds': 200, 'chunk_size': 64},
            
            # Medium models
            {'batch_size': 32, 'hidden_dim': 1024, 'num_seeds': 500, 'chunk_size': 128},
            {'batch_size': 64, 'hidden_dim': 1024, 'num_seeds': 500, 'chunk_size': 128},
            
            # Large models
            {'batch_size': 16, 'hidden_dim': 2048, 'num_seeds': 1000, 'chunk_size': 256},
            {'batch_size': 32, 'hidden_dim': 2048, 'num_seeds': 1000, 'chunk_size': 256},
            
            # Extreme case
            {'batch_size': 8, 'hidden_dim': 4096, 'num_seeds': 2000, 'chunk_size': 512},
        ]
        
        all_results = []
        for config in configurations:
            try:
                result = self.run_comparison(config)
                all_results.append(result)
            except Exception as e:
                print(f"  Failed: {e}")
                
        return all_results
    
    def plot_results(self, results: List[Dict]):
        """Visualize benchmark results."""
        if not results or not HAS_MATPLOTLIB:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract data
        configs = [f"B{r['config']['batch_size']}_H{r['config']['hidden_dim']}_S{r['config']['num_seeds']}" 
                  for r in results]
        speedups = [r['speedup'] for r in results]
        triton_times = [r['triton_time_ms'] for r in results]
        chunked_times = [r['chunked_time_ms'] for r in results]
        mem_reductions = [r['mem_reduction_pct'] for r in results]
        triton_throughputs = [r['triton_throughput_gb_s'] for r in results]
        
        # Speedup chart
        ax = axes[0, 0]
        x = np.arange(len(configs))
        ax.bar(x, speedups, color='green', alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', label='Break-even')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Speedup')
        ax.set_title('Triton Speedup vs Phase 2')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        
        # Execution time comparison
        ax = axes[0, 1]
        width = 0.35
        ax.bar(x - width/2, triton_times, width, label='Triton', alpha=0.7)
        ax.bar(x + width/2, chunked_times, width, label='Phase 2', alpha=0.7)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Forward Pass Execution Time')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        
        # Memory reduction
        ax = axes[1, 0]
        ax.bar(x, mem_reductions, color='blue', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-')
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Memory Reduction (%)')
        ax.set_title('Memory Usage Improvement')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        
        # Throughput
        ax = axes[1, 1]
        ax.plot(x, triton_throughputs, 'o-', label='Triton', markersize=8)
        ax.set_xlabel('Configuration')
        ax.set_ylabel('Throughput (GB/s)')
        ax.set_title('Memory Bandwidth Utilization')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('phase3_benchmark_results.png', dpi=150)
        print("\nResults saved to phase3_benchmark_results.png")


def main():
    """Run Phase 3 benchmarks."""
    print("Phase 3 GPU Optimization Benchmarks")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_memory:.1f} GB")
    
    # Run benchmarks
    suite = BenchmarkSuite()
    results = suite.run_full_suite()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        max_speedup = max(r['speedup'] for r in results)
        avg_mem_reduction = np.mean([r['mem_reduction_pct'] for r in results])
        
        print(f"Average speedup: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        print(f"Average memory reduction: {avg_mem_reduction:.1f}%")
        
        # Plot results
        suite.plot_results(results)


if __name__ == '__main__':
    main()