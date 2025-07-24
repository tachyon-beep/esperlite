#!/usr/bin/env python3
"""
Phase 1 Performance Benchmarks for Morphogenetic Migration.

This script benchmarks the chunked architecture against the legacy implementation
to validate performance improvements and identify bottlenecks.
"""

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Dict, List, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.esper.morphogenetic_v2.kasmina import (
    ChunkedKasminaLayer,
    ChunkManager,
    StateTensor
)
from src.esper.execution.kasmina_layer import KasminaLayer


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    implementation: str
    config: Dict[str, Any]
    timings: List[float]
    memory_usage: Dict[str, float]
    throughput: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class Phase1Benchmarks:
    """Benchmark suite for Phase 1 morphogenetic features."""
    
    def __init__(self, device: str = "cuda", warmup_runs: int = 10):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.warmup_runs = warmup_runs
        self.results: List[BenchmarkResult] = []
        
        print(f"Benchmarking on device: {self.device}")
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def benchmark_chunk_manager(self):
        """Benchmark ChunkManager operations."""
        print("\n=== ChunkManager Benchmarks ===")
        
        configs = [
            {"layer_dim": 256, "num_chunks": 16, "batch_size": 32},
            {"layer_dim": 1024, "num_chunks": 64, "batch_size": 64},
            {"layer_dim": 4096, "num_chunks": 256, "batch_size": 128},
            {"layer_dim": 16384, "num_chunks": 1024, "batch_size": 256},
        ]
        
        for config in configs:
            print(f"\nConfig: {config}")
            
            # Setup
            cm = ChunkManager(
                config["layer_dim"], 
                config["num_chunks"], 
                device=self.device
            )
            x = torch.randn(
                config["batch_size"], 
                config["layer_dim"], 
                device=self.device
            )
            
            # Warmup
            for _ in range(self.warmup_runs):
                chunks = cm.split_tensor(x)
                _ = cm.concatenate_chunks(chunks)
            
            # Benchmark split
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            split_times = []
            for _ in range(100):
                start = time.perf_counter()
                chunks = cm.split_tensor(x)
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                split_times.append(time.perf_counter() - start)
            
            # Benchmark concatenate
            concat_times = []
            for _ in range(100):
                start = time.perf_counter()
                _ = cm.concatenate_chunks(chunks)
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                concat_times.append(time.perf_counter() - start)
            
            # Record results
            split_result = BenchmarkResult(
                name="chunk_split",
                implementation="ChunkManager",
                config=config,
                timings=split_times,
                memory_usage=self._get_memory_usage(),
                throughput=config["batch_size"] * config["layer_dim"] * 4 / np.mean(split_times) / 1e9,  # GB/s
                latency_p50=np.percentile(split_times, 50) * 1000,
                latency_p95=np.percentile(split_times, 95) * 1000,
                latency_p99=np.percentile(split_times, 99) * 1000
            )
            
            concat_result = BenchmarkResult(
                name="chunk_concat",
                implementation="ChunkManager",
                config=config,
                timings=concat_times,
                memory_usage=self._get_memory_usage(),
                throughput=config["batch_size"] * config["layer_dim"] * 4 / np.mean(concat_times) / 1e9,
                latency_p50=np.percentile(concat_times, 50) * 1000,
                latency_p95=np.percentile(concat_times, 95) * 1000,
                latency_p99=np.percentile(concat_times, 99) * 1000
            )
            
            self.results.extend([split_result, concat_result])
            
            print(f"  Split: {split_result.latency_p50:.3f}ms (p50), {split_result.throughput:.1f} GB/s")
            print(f"  Concat: {concat_result.latency_p50:.3f}ms (p50), {concat_result.throughput:.1f} GB/s")
    
    def benchmark_state_tensor(self):
        """Benchmark StateTensor operations."""
        print("\n=== StateTensor Benchmarks ===")
        
        num_seeds_list = [100, 1000, 5000, 10000]
        
        for num_seeds in num_seeds_list:
            print(f"\nNum seeds: {num_seeds}")
            
            # Setup
            st = StateTensor(num_seeds, device=self.device)
            seed_ids = torch.randint(0, num_seeds, (100,), device=self.device)
            
            # Warmup
            for _ in range(self.warmup_runs):
                st.get_active_seeds()
                st.increment_epochs()
            
            # Benchmark state queries
            query_times = []
            for _ in range(1000):
                start = time.perf_counter()
                _ = st.get_active_seeds()
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                query_times.append(time.perf_counter() - start)
            
            # Benchmark state updates
            update_times = []
            for _ in range(100):
                start = time.perf_counter()
                st.batch_set_lifecycle_state(seed_ids, 2)  # ACTIVE
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                update_times.append(time.perf_counter() - start)
            
            # Record results
            query_result = BenchmarkResult(
                name="state_query",
                implementation="StateTensor",
                config={"num_seeds": num_seeds},
                timings=query_times,
                memory_usage=self._get_memory_usage(),
                throughput=num_seeds / np.mean(query_times),  # seeds/sec
                latency_p50=np.percentile(query_times, 50) * 1000,
                latency_p95=np.percentile(query_times, 95) * 1000,
                latency_p99=np.percentile(query_times, 99) * 1000
            )
            
            update_result = BenchmarkResult(
                name="state_update",
                implementation="StateTensor",
                config={"num_seeds": num_seeds, "batch_size": 100},
                timings=update_times,
                memory_usage=self._get_memory_usage(),
                throughput=100 / np.mean(update_times),  # updates/sec
                latency_p50=np.percentile(update_times, 50) * 1000,
                latency_p95=np.percentile(update_times, 95) * 1000,
                latency_p99=np.percentile(update_times, 99) * 1000
            )
            
            self.results.extend([query_result, update_result])
            
            print(f"  Query: {query_result.latency_p50:.3f}ms (p50), {query_result.throughput:.0f} seeds/s")
            print(f"  Update: {update_result.latency_p50:.3f}ms (p50), {update_result.throughput:.0f} updates/s")
    
    def benchmark_layer_implementations(self):
        """Benchmark legacy vs chunked layer implementations."""
        print("\n=== Layer Implementation Benchmarks ===")
        
        configs = [
            {"input_dim": 256, "output_dim": 256, "batch_size": 32, "num_seeds": 16},
            {"input_dim": 512, "output_dim": 512, "batch_size": 64, "num_seeds": 64},
            {"input_dim": 1024, "output_dim": 1024, "batch_size": 128, "num_seeds": 256},
            {"input_dim": 2048, "output_dim": 2048, "batch_size": 256, "num_seeds": 1024},
        ]
        
        for config in configs:
            print(f"\nConfig: {config}")
            
            # Create base layer
            base_layer = nn.Linear(config["input_dim"], config["output_dim"]).to(self.device)
            
            # Create input
            x = torch.randn(config["batch_size"], config["input_dim"], device=self.device)
            
            # Benchmark legacy (if available)
            try:
                legacy_layer = KasminaLayer(
                    input_size=config["input_dim"],
                    output_size=config["output_dim"]
                ).to(self.device)
                
                # Warmup
                for _ in range(self.warmup_runs):
                    with torch.no_grad():
                        _ = legacy_layer(x)
                
                # Benchmark
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                legacy_times = []
                for _ in range(100):
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = legacy_layer(x)
                    if self.device.type == "cuda":
                torch.cuda.synchronize()
                    legacy_times.append(time.perf_counter() - start)
                
                legacy_result = BenchmarkResult(
                    name="forward_pass",
                    implementation="legacy",
                    config=config,
                    timings=legacy_times,
                    memory_usage=self._get_memory_usage(),
                    throughput=config["batch_size"] / np.mean(legacy_times),
                    latency_p50=np.percentile(legacy_times, 50) * 1000,
                    latency_p95=np.percentile(legacy_times, 95) * 1000,
                    latency_p99=np.percentile(legacy_times, 99) * 1000
                )
                self.results.append(legacy_result)
                print(f"  Legacy: {legacy_result.latency_p50:.3f}ms (p50)")
                
            except (ImportError, AttributeError) as e:
                print(f"  Legacy: Not available ({e})")
            
            # Benchmark chunked
            chunked_layer = ChunkedKasminaLayer(
                base_layer=base_layer,
                num_seeds=config["num_seeds"],
                device=self.device
            )
            
            # Test with different activation levels
            for active_pct in [0, 25, 50, 100]:
                num_active = int(config["num_seeds"] * active_pct / 100)
                
                # Activate seeds
                for i in range(num_active):
                    chunked_layer.state_tensor.set_lifecycle_state(i, 2)  # ACTIVE
                
                # Warmup
                for _ in range(self.warmup_runs):
                    with torch.no_grad():
                        _ = chunked_layer(x)
                
                # Benchmark
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                chunked_times = []
                for _ in range(100):
                    start = time.perf_counter()
                    with torch.no_grad():
                        _ = chunked_layer(x)
                    if self.device.type == "cuda":
                torch.cuda.synchronize()
                    chunked_times.append(time.perf_counter() - start)
                
                chunked_config = config.copy()
                chunked_config["active_seeds_pct"] = active_pct
                
                chunked_result = BenchmarkResult(
                    name="forward_pass",
                    implementation=f"chunked_{active_pct}pct",
                    config=chunked_config,
                    timings=chunked_times,
                    memory_usage=self._get_memory_usage(),
                    throughput=config["batch_size"] / np.mean(chunked_times),
                    latency_p50=np.percentile(chunked_times, 50) * 1000,
                    latency_p95=np.percentile(chunked_times, 95) * 1000,
                    latency_p99=np.percentile(chunked_times, 99) * 1000
                )
                self.results.append(chunked_result)
                print(f"  Chunked ({active_pct}% active): {chunked_result.latency_p50:.3f}ms (p50)")
                
                # Reset for next iteration
                for i in range(config["num_seeds"]):
                    chunked_layer.state_tensor.set_lifecycle_state(i, 0)  # DORMANT
    
    def benchmark_scaling(self):
        """Benchmark scaling characteristics."""
        print("\n=== Scaling Benchmarks ===")
        
        # Test scaling with number of seeds
        base_config = {
            "input_dim": 1024,
            "output_dim": 1024,
            "batch_size": 64
        }
        
        seed_counts = [10, 50, 100, 500, 1000, 2000, 5000]
        
        for num_seeds in seed_counts:
            print(f"\nSeeds: {num_seeds}")
            
            base_layer = nn.Linear(
                base_config["input_dim"], 
                base_config["output_dim"]
            ).to(self.device)
            
            layer = ChunkedKasminaLayer(
                base_layer=base_layer,
                num_seeds=num_seeds,
                device=self.device
            )
            
            x = torch.randn(
                base_config["batch_size"], 
                base_config["input_dim"], 
                device=self.device
            )
            
            # Warmup
            for _ in range(self.warmup_runs):
                with torch.no_grad():
                    _ = layer(x)
            
            # Benchmark
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(50):
                start = time.perf_counter()
                with torch.no_grad():
                    _ = layer(x)
                if self.device.type == "cuda":
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
            
            config = base_config.copy()
            config["num_seeds"] = num_seeds
            
            result = BenchmarkResult(
                name="scaling_test",
                implementation="chunked",
                config=config,
                timings=times,
                memory_usage=self._get_memory_usage(),
                throughput=base_config["batch_size"] / np.mean(times),
                latency_p50=np.percentile(times, 50) * 1000,
                latency_p95=np.percentile(times, 95) * 1000,
                latency_p99=np.percentile(times, 99) * 1000
            )
            self.results.append(result)
            
            print(f"  Latency: {result.latency_p50:.3f}ms (p50)")
            print(f"  Throughput: {result.throughput:.1f} samples/s")
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if self.device.type == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
            }
        else:
            return {
                "allocated_gb": 0,
                "reserved_gb": 0,
                "max_allocated_gb": 0
            }
    
    def save_results(self, output_path: Path):
        """Save benchmark results to JSON."""
        results_dict = {
            "timestamp": time.time(),
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name() if self.device.type == "cuda" else "CPU",
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
    
    def print_summary(self):
        """Print summary of benchmark results."""
        print("\n=== Benchmark Summary ===")
        
        # Group by benchmark name
        by_name = {}
        for result in self.results:
            if result.name not in by_name:
                by_name[result.name] = []
            by_name[result.name].append(result)
        
        for name, results in by_name.items():
            print(f"\n{name}:")
            for r in results:
                print(f"  {r.implementation}: {r.latency_p50:.3f}ms (p50), {r.latency_p95:.3f}ms (p95)")


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 morphogenetic benchmarks")
    parser.add_argument("--device", default="cuda", help="Device to run on (cuda/cpu)")
    parser.add_argument("--output", type=Path, default="benchmark_results.json", help="Output file")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup runs")
    
    args = parser.parse_args()
    
    # Run benchmarks
    suite = Phase1Benchmarks(device=args.device, warmup_runs=args.warmup)
    
    print("Running Phase 1 Morphogenetic Benchmarks...")
    suite.benchmark_chunk_manager()
    suite.benchmark_state_tensor()
    suite.benchmark_layer_implementations()
    suite.benchmark_scaling()
    
    # Save and summarize
    suite.save_results(args.output)
    suite.print_summary()


if __name__ == "__main__":
    main()