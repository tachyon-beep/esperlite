#!/usr/bin/env python3
"""
Phase 2 Performance Benchmarks for Extended Lifecycle System.

This module provides comprehensive benchmarks for:
- State transition latency
- Checkpoint save/restore times
- Memory usage with extended state tensor
- Grafting strategy performance
- Overall system throughput
"""

import gc
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict

import numpy as np
import psutil
import torch
import torch.nn as nn

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

# Add parent directory to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.esper.morphogenetic_v2.grafting import GraftingConfig
from src.esper.morphogenetic_v2.grafting import GraftingContext
from src.esper.morphogenetic_v2.grafting import create_grafting_strategy
from src.esper.morphogenetic_v2.kasmina.chunked_layer_v2 import ChunkedKasminaLayerV2
from src.esper.morphogenetic_v2.lifecycle import CheckpointManager
from src.esper.morphogenetic_v2.lifecycle import ExtendedLifecycle
from src.esper.morphogenetic_v2.lifecycle import ExtendedStateTensor
from src.esper.morphogenetic_v2.lifecycle import LifecycleManager


class Phase2Benchmarks:
    """Comprehensive benchmark suite for Phase 2 components."""

    def __init__(self, device: torch.device = None):
        """Initialize benchmark suite."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

        # Standard benchmark sizes
        self.seed_counts = [100, 1000, 5000, 10000]
        self.batch_sizes = [32, 64, 128, 256]
        self.dimensions = [512, 1024, 2048]

        print(f"Phase 2 Benchmarks initialized on {self.device}")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    def benchmark_state_transitions(self) -> Dict[str, Any]:
        """Benchmark state transition performance."""
        print("\n" + "="*60)
        print("Benchmarking State Transitions")
        print("="*60)

        results = {
            "seed_counts": self.seed_counts,
            "single_transition_times": [],
            "bulk_transition_times": [],
            "validation_overhead": []
        }

        for num_seeds in self.seed_counts:
            print(f"\nTesting with {num_seeds} seeds...")

            # Initialize components
            state_tensor = ExtendedStateTensor(num_seeds, self.device)
            lifecycle_manager = LifecycleManager(num_seeds)

            # Single transition benchmark
            seed_id = num_seeds // 2  # Middle seed

            # Warm up
            for _ in range(10):
                lifecycle_manager.request_transition(
                    seed_id,
                    ExtendedLifecycle.DORMANT,
                    ExtendedLifecycle.GERMINATED
                )

            # Measure single transition
            times = []
            for _ in range(100):
                start = time.perf_counter()
                result = lifecycle_manager.request_transition(
                    seed_id,
                    ExtendedLifecycle.GERMINATED,
                    ExtendedLifecycle.TRAINING
                )
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            single_time = np.mean(times)
            results["single_transition_times"].append(single_time)
            print(f"  Single transition: {single_time:.3f} ms")

            # Bulk transition benchmark
            seed_indices = torch.arange(num_seeds, device=self.device)

            # Set all to GERMINATED
            state_tensor.set_state(seed_indices, torch.full((num_seeds,), ExtendedLifecycle.GERMINATED.value))

            # Measure bulk transition
            times = []
            for _ in range(10):
                start = time.perf_counter()
                # Simulate bulk transition with state tensor
                new_states = torch.full((num_seeds,), ExtendedLifecycle.TRAINING.value, device=self.device)
                state_tensor.set_state(seed_indices, new_states)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            bulk_time = np.mean(times)
            results["bulk_transition_times"].append(bulk_time)
            print(f"  Bulk transition: {bulk_time:.3f} ms for {num_seeds} seeds")
            print(f"  Per-seed cost: {bulk_time/num_seeds:.6f} ms")

            # Measure validation overhead
            validation_pct = (single_time - bulk_time/num_seeds) / single_time * 100
            results["validation_overhead"].append(validation_pct)
            print(f"  Validation overhead: {validation_pct:.1f}%")

            # Clean up
            del state_tensor, lifecycle_manager
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self.results["state_transitions"] = results
        return results

    def benchmark_checkpoint_operations(self) -> Dict[str, Any]:
        """Benchmark checkpoint save/restore performance."""
        print("\n" + "="*60)
        print("Benchmarking Checkpoint Operations")
        print("="*60)

        results = {
            "seed_counts": self.seed_counts,
            "save_times": [],
            "restore_times": [],
            "checkpoint_sizes_mb": [],
            "compression_ratios": []
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for num_seeds in self.seed_counts:
                print(f"\nTesting with {num_seeds} seeds...")

                # Initialize components
                checkpoint_manager = CheckpointManager(Path(tmpdir) / f"test_{num_seeds}")
                state_tensor = ExtendedStateTensor(num_seeds, self.device)

                # Create sample blueprint
                blueprint = nn.Sequential(
                    nn.Linear(1024, 2048),
                    nn.ReLU(),
                    nn.Linear(2048, 1024)
                ).to(self.device)

                # Prepare state data
                state_data = {
                    'lifecycle_state': ExtendedLifecycle.TRAINING.value,
                    'epochs_in_state': 100,
                    'performance_metrics': {
                        'loss': 0.234,
                        'accuracy': 0.956,
                        'stability': 0.887,
                        'efficiency': 0.912
                    },
                    'error_count': 0
                }

                blueprint_state = blueprint.state_dict()

                # Benchmark save operation
                save_times = []
                checkpoint_ids = []

                for i in range(10):
                    start = time.perf_counter()
                    checkpoint_id = checkpoint_manager.save_checkpoint(
                        layer_id="test_layer",
                        seed_id=i,
                        state_data=state_data,
                        blueprint_state=blueprint_state,
                        priority='normal'
                    )
                    end = time.perf_counter()
                    save_times.append((end - start) * 1000)
                    checkpoint_ids.append(checkpoint_id)

                avg_save_time = np.mean(save_times)
                results["save_times"].append(avg_save_time)
                print(f"  Average save time: {avg_save_time:.3f} ms")

                # Measure checkpoint size
                checkpoint_files = list(checkpoint_manager.checkpoint_dir.glob("*.pth"))
                if checkpoint_files:
                    avg_size_bytes = np.mean([f.stat().st_size for f in checkpoint_files])
                    avg_size_mb = avg_size_bytes / (1024 * 1024)
                    results["checkpoint_sizes_mb"].append(avg_size_mb)

                    # Calculate compression ratio
                    raw_size = (
                        state_tensor.state_tensor.element_size() * state_tensor.state_tensor.nelement() +
                        sum(p.element_size() * p.nelement() for p in blueprint.parameters())
                    )
                    compression_ratio = raw_size / avg_size_bytes
                    results["compression_ratios"].append(compression_ratio)

                    print(f"  Average checkpoint size: {avg_size_mb:.2f} MB")
                    print(f"  Compression ratio: {compression_ratio:.2f}x")
                else:
                    results["checkpoint_sizes_mb"].append(0)
                    results["compression_ratios"].append(0)

                # Benchmark restore operation
                restore_times = []

                for checkpoint_id in checkpoint_ids[:5]:  # Test first 5
                    start = time.perf_counter()
                    restored = checkpoint_manager.restore_checkpoint(checkpoint_id)
                    end = time.perf_counter()
                    restore_times.append((end - start) * 1000)

                avg_restore_time = np.mean(restore_times)
                results["restore_times"].append(avg_restore_time)
                print(f"  Average restore time: {avg_restore_time:.3f} ms")

                # Clean up
                del checkpoint_manager, state_tensor, blueprint
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self.results["checkpoint_operations"] = results
        return results

    def benchmark_extended_state_memory(self) -> Dict[str, Any]:
        """Benchmark memory usage of extended state tensor."""
        print("\n" + "="*60)
        print("Benchmarking Extended State Memory Usage")
        print("="*60)

        results = {
            "seed_counts": self.seed_counts,
            "state_memory_mb": [],
            "perf_metrics_memory_mb": [],
            "telemetry_memory_mb": [],
            "history_memory_mb": [],
            "total_memory_mb": [],
            "memory_per_seed_kb": []
        }

        for num_seeds in self.seed_counts:
            print(f"\nTesting with {num_seeds} seeds...")

            # Force garbage collection
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
            else:
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss

            # Create extended state tensor
            state_tensor = ExtendedStateTensor(num_seeds, self.device)

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
            else:
                end_memory = process.memory_info().rss

            # Calculate memory usage
            total_memory_bytes = end_memory - start_memory
            total_memory_mb = total_memory_bytes / (1024 * 1024)

            # Calculate component sizes
            state_size = state_tensor.state_tensor.element_size() * state_tensor.state_tensor.nelement()
            perf_size = state_tensor.performance_metrics.element_size() * state_tensor.performance_metrics.nelement()
            telemetry_size = state_tensor.telemetry_buffer.element_size() * state_tensor.telemetry_buffer.nelement()
            history_size = state_tensor.transition_history.element_size() * state_tensor.transition_history.nelement()

            results["state_memory_mb"].append(state_size / (1024 * 1024))
            results["perf_metrics_memory_mb"].append(perf_size / (1024 * 1024))
            results["telemetry_memory_mb"].append(telemetry_size / (1024 * 1024))
            results["history_memory_mb"].append(history_size / (1024 * 1024))
            results["total_memory_mb"].append(total_memory_mb)
            results["memory_per_seed_kb"].append(total_memory_bytes / num_seeds / 1024)

            print(f"  State tensor: {state_size / (1024 * 1024):.2f} MB")
            print(f"  Performance metrics: {perf_size / (1024 * 1024):.2f} MB")
            print(f"  Telemetry buffer: {telemetry_size / (1024 * 1024):.2f} MB")
            print(f"  Transition history: {history_size / (1024 * 1024):.2f} MB")
            print(f"  Total memory: {total_memory_mb:.2f} MB")
            print(f"  Memory per seed: {total_memory_bytes / num_seeds / 1024:.2f} KB")

            # Clean up
            del state_tensor
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        self.results["memory_usage"] = results
        return results

    def benchmark_grafting_strategies(self) -> Dict[str, Any]:
        """Benchmark performance of different grafting strategies."""
        print("\n" + "="*60)
        print("Benchmarking Grafting Strategies")
        print("="*60)

        strategies = ['linear', 'drift_controlled', 'momentum', 'adaptive', 'stability']
        results = {
            "strategies": strategies,
            "compute_times_us": {s: [] for s in strategies},  # microseconds
            "memory_usage_kb": {s: [] for s in strategies},
            "alpha_convergence": {s: [] for s in strategies}
        }

        config = GraftingConfig(ramp_duration=100)

        for strategy_name in strategies:
            print(f"\nTesting {strategy_name} strategy...")

            strategy = create_grafting_strategy(strategy_name, config)

            # Prepare context
            context = GraftingContext(
                seed_id=0,
                current_epoch=50,
                total_epochs=100,
                current_alpha=0.5,
                metrics={
                    'loss': 0.234,
                    'accuracy': 0.956,
                    'stability': 0.887,
                    'efficiency': 0.912
                }
            )

            # Warm up
            for _ in range(100):
                _ = strategy.compute_alpha(context)

            # Benchmark compute time
            times = []
            for epoch in range(100):
                context.current_epoch = epoch
                context.current_alpha = epoch / 100.0

                start = time.perf_counter()
                alpha = strategy.compute_alpha(context)
                end = time.perf_counter()

                times.append((end - start) * 1e6)  # Convert to microseconds

            avg_time = np.mean(times)
            results["compute_times_us"][strategy_name] = avg_time
            print(f"  Average compute time: {avg_time:.2f} µs")

            # Measure memory usage
            if hasattr(strategy, '__sizeof__'):
                size_bytes = sys.getsizeof(strategy)
            else:
                # Estimate based on attributes
                size_bytes = sum(sys.getsizeof(getattr(strategy, attr))
                               for attr in dir(strategy)
                               if not attr.startswith('_'))

            size_kb = size_bytes / 1024
            results["memory_usage_kb"][strategy_name] = size_kb
            print(f"  Memory usage: {size_kb:.2f} KB")

            # Test alpha convergence pattern
            alphas = []
            for epoch in range(config.ramp_duration):
                context.current_epoch = epoch
                alpha = strategy.compute_alpha(context)
                alphas.append(alpha)

            results["alpha_convergence"][strategy_name] = alphas
            print(f"  Final alpha: {alphas[-1]:.3f}")

        self.results["grafting_strategies"] = results
        return results

    def benchmark_full_layer_throughput(self) -> Dict[str, Any]:
        """Benchmark full ChunkedKasminaLayerV2 throughput."""
        print("\n" + "="*60)
        print("Benchmarking Full Layer Throughput")
        print("="*60)

        results = {
            "configurations": [],
            "forward_times_ms": [],
            "throughput_samples_per_sec": [],
            "memory_usage_mb": [],
            "state_distribution": []
        }

        # Test different configurations
        configs = [
            (1000, 32, 1024),   # seeds, batch_size, dim
            (1000, 128, 1024),
            (5000, 32, 1024),
            (5000, 128, 1024),
            (10000, 32, 512),
            (10000, 64, 512)
        ]

        for num_seeds, batch_size, dim in configs:
            config_str = f"{num_seeds} seeds, batch {batch_size}, dim {dim}"
            print(f"\nTesting: {config_str}")
            results["configurations"].append(config_str)

            # Create base layer
            base_layer = nn.Linear(dim, dim).to(self.device)

            # Create ChunkedKasminaLayerV2
            with tempfile.TemporaryDirectory() as tmpdir:
                layer = ChunkedKasminaLayerV2(
                    base_layer=base_layer,
                    num_seeds=num_seeds,
                    device=self.device,
                    checkpoint_dir=Path(tmpdir)
                )

                # Activate some seeds with different states
                active_ratio = 0.3  # 30% active
                num_active = int(num_seeds * active_ratio)

                for i in range(num_active):
                    if i < num_active // 3:
                        layer.request_germination(i, grafting_strategy='linear')
                        layer._request_transition(i, ExtendedLifecycle.TRAINING)
                    elif i < 2 * num_active // 3:
                        layer.request_germination(i, grafting_strategy='adaptive')
                        layer._request_transition(i, ExtendedLifecycle.GRAFTING)
                    else:
                        layer.request_germination(i, grafting_strategy='momentum')
                        layer._request_transition(i, ExtendedLifecycle.FINE_TUNING)

                # Prepare input
                x = torch.randn(batch_size, dim, device=self.device)

                # Warm up
                for _ in range(10):
                    _ = layer(x)

                if self.device.type == 'cuda':
                    torch.cuda.synchronize()

                # Benchmark forward pass
                times = []
                num_iterations = 100

                start_total = time.perf_counter()
                for _ in range(num_iterations):
                    start = time.perf_counter()
                    output = layer(x)
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                end_total = time.perf_counter()

                avg_time = np.mean(times)
                total_time = end_total - start_total
                throughput = (batch_size * num_iterations) / total_time

                results["forward_times_ms"].append(avg_time)
                results["throughput_samples_per_sec"].append(throughput)

                print(f"  Forward pass: {avg_time:.3f} ms")
                print(f"  Throughput: {throughput:.0f} samples/sec")

                # Get state distribution
                state_summary = layer.extended_state.get_state_summary()
                results["state_distribution"].append(state_summary)

                # Measure memory
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    memory_bytes = torch.cuda.memory_allocated()
                    memory_mb = memory_bytes / (1024 * 1024)
                    results["memory_usage_mb"].append(memory_mb)
                    print(f"  GPU memory: {memory_mb:.2f} MB")
                else:
                    results["memory_usage_mb"].append(0)

                # Clean up
                del layer, base_layer, x
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        self.results["full_layer_throughput"] = results
        return results

    def run_all_benchmarks(self):
        """Run all benchmarks and generate report."""
        print("\n" + "="*80)
        print("PHASE 2 PERFORMANCE BENCHMARKS")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Run benchmarks
        self.benchmark_state_transitions()
        self.benchmark_checkpoint_operations()
        self.benchmark_extended_state_memory()
        self.benchmark_grafting_strategies()
        self.benchmark_full_layer_throughput()

        # Generate visualizations
        self._generate_plots()

        # Save results
        self._save_results()

        # Generate summary
        self._print_summary()

        print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _generate_plots(self):
        """Generate visualization plots."""
        output_dir = Path("benchmark_results/phase2")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. State Transition Performance
        if "state_transitions" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            data = self.results["state_transitions"]
            x = data["seed_counts"]

            # Single vs Bulk transitions
            ax1.plot(x, data["single_transition_times"], 'o-', label='Single Transition', markersize=8)
            ax1.plot(x, data["bulk_transition_times"], 's-', label='Bulk Transition', markersize=8)
            ax1.set_xlabel("Number of Seeds")
            ax1.set_ylabel("Time (ms)")
            ax1.set_title("State Transition Performance")
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Validation overhead
            ax2.bar(range(len(x)), data["validation_overhead"],
                   tick_label=[str(s) for s in x])
            ax2.set_xlabel("Number of Seeds")
            ax2.set_ylabel("Validation Overhead (%)")
            ax2.set_title("Transition Validation Cost")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "state_transitions.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 2. Checkpoint Performance
        if "checkpoint_operations" in self.results:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

            data = self.results["checkpoint_operations"]
            x = data["seed_counts"]

            # Save/Restore times
            ax1.plot(x, data["save_times"], 'o-', label='Save', markersize=8)
            ax1.plot(x, data["restore_times"], 's-', label='Restore', markersize=8)
            ax1.set_xlabel("Number of Seeds")
            ax1.set_ylabel("Time (ms)")
            ax1.set_title("Checkpoint Operation Times")
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Checkpoint sizes
            ax2.plot(x, data["checkpoint_sizes_mb"], 'o-', color='green', markersize=8)
            ax2.set_xlabel("Number of Seeds")
            ax2.set_ylabel("Size (MB)")
            ax2.set_title("Average Checkpoint Size")
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)

            # Compression ratios
            ax3.plot(x, data["compression_ratios"], 'o-', color='orange', markersize=8)
            ax3.set_xlabel("Number of Seeds")
            ax3.set_ylabel("Compression Ratio")
            ax3.set_title("Checkpoint Compression")
            ax3.set_xscale('log')
            ax3.grid(True, alpha=0.3)

            # Save vs Restore ratio
            ratios = [s/r for s, r in zip(data["save_times"], data["restore_times"])]
            ax4.bar(range(len(x)), ratios, tick_label=[str(s) for s in x])
            ax4.set_xlabel("Number of Seeds")
            ax4.set_ylabel("Save/Restore Time Ratio")
            ax4.set_title("Relative Operation Cost")
            ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5)
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "checkpoint_operations.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Memory Usage
        if "memory_usage" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            data = self.results["memory_usage"]
            x = data["seed_counts"]

            # Stacked bar chart of memory components
            width = 0.35
            indices = np.arange(len(x))

            p1 = ax1.bar(indices, data["state_memory_mb"], width, label='State Tensor')
            p2 = ax1.bar(indices, data["perf_metrics_memory_mb"], width,
                        bottom=data["state_memory_mb"], label='Perf Metrics')
            p3 = ax1.bar(indices, data["telemetry_memory_mb"], width,
                        bottom=[s+p for s,p in zip(data["state_memory_mb"],
                                                   data["perf_metrics_memory_mb"])],
                        label='Telemetry')
            p4 = ax1.bar(indices, data["history_memory_mb"], width,
                        bottom=[s+p+t for s,p,t in zip(data["state_memory_mb"],
                                                       data["perf_metrics_memory_mb"],
                                                       data["telemetry_memory_mb"])],
                        label='History')

            ax1.set_xlabel("Number of Seeds")
            ax1.set_ylabel("Memory (MB)")
            ax1.set_title("Memory Usage Breakdown")
            ax1.set_xticks(indices)
            ax1.set_xticklabels([str(s) for s in x])
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Memory per seed
            ax2.plot(x, data["memory_per_seed_kb"], 'o-', color='purple', markersize=8)
            ax2.set_xlabel("Number of Seeds")
            ax2.set_ylabel("Memory per Seed (KB)")
            ax2.set_title("Memory Efficiency")
            ax2.set_xscale('log')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "memory_usage.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Grafting Strategy Performance
        if "grafting_strategies" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            data = self.results["grafting_strategies"]
            strategies = data["strategies"]

            # Compute times
            times = [data["compute_times_us"][s] for s in strategies]
            ax1.bar(strategies, times)
            ax1.set_ylabel("Compute Time (µs)")
            ax1.set_title("Grafting Strategy Performance")
            ax1.set_xticklabels(strategies, rotation=45)
            ax1.grid(True, alpha=0.3)

            # Alpha convergence patterns
            epochs = range(100)
            for strategy in strategies:
                alphas = data["alpha_convergence"][strategy]
                ax2.plot(epochs, alphas, label=strategy, linewidth=2)

            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Alpha Value")
            ax2.set_title("Grafting Alpha Convergence")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "grafting_strategies.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 5. Full Layer Throughput
        if "full_layer_throughput" in self.results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            data = self.results["full_layer_throughput"]
            configs = data["configurations"]
            x_pos = np.arange(len(configs))

            # Forward times
            ax1.bar(x_pos, data["forward_times_ms"])
            ax1.set_xlabel("Configuration")
            ax1.set_ylabel("Forward Time (ms)")
            ax1.set_title("Layer Forward Pass Performance")
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(configs, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)

            # Throughput
            ax2.bar(x_pos, data["throughput_samples_per_sec"], color='green')
            ax2.set_xlabel("Configuration")
            ax2.set_ylabel("Throughput (samples/sec)")
            ax2.set_title("Processing Throughput")
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(configs, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "layer_throughput.png", dpi=300, bbox_inches='tight')
            plt.close()

        print(f"\nPlots saved to: {output_dir}")

    def _save_results(self):
        """Save benchmark results to JSON."""
        output_dir = Path("benchmark_results/phase2")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, dict):
                        json_results[key][k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                                              for kk, vv in v.items()}
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value

        # Add metadata
        json_results["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A"
        }

        with open(output_dir / "benchmark_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to: {output_dir / 'benchmark_results.json'}")

    def _print_summary(self):
        """Print executive summary of benchmark results."""
        print("\n" + "="*80)
        print("EXECUTIVE SUMMARY")
        print("="*80)

        # State Transitions
        if "state_transitions" in self.results:
            data = self.results["state_transitions"]
            print("\n📊 State Transition Performance:")
            print(f"  • Single transition: {data['single_transition_times'][-1]:.3f} ms (10K seeds)")
            print(f"  • Bulk transition: {data['bulk_transition_times'][-1]:.3f} ms for 10K seeds")
            print(f"  • Per-seed cost: {data['bulk_transition_times'][-1]/10000:.6f} ms")
            print(f"  • Validation overhead: {data['validation_overhead'][-1]:.1f}%")

        # Checkpoints
        if "checkpoint_operations" in self.results:
            data = self.results["checkpoint_operations"]
            print("\n💾 Checkpoint Performance:")
            print(f"  • Save time: {data['save_times'][-1]:.3f} ms")
            print(f"  • Restore time: {data['restore_times'][-1]:.3f} ms")
            print(f"  • Checkpoint size: {data['checkpoint_sizes_mb'][-1]:.2f} MB")
            print(f"  • Compression ratio: {data['compression_ratios'][-1]:.2f}x")

        # Memory
        if "memory_usage" in self.results:
            data = self.results["memory_usage"]
            print("\n🧠 Memory Usage (10K seeds):")
            print(f"  • Total memory: {data['total_memory_mb'][-1]:.2f} MB")
            print(f"  • Per-seed memory: {data['memory_per_seed_kb'][-1]:.2f} KB")
            print(f"  • State tensor: {data['state_memory_mb'][-1]:.2f} MB")
            print(f"  • Performance metrics: {data['perf_metrics_memory_mb'][-1]:.2f} MB")

        # Grafting
        if "grafting_strategies" in self.results:
            data = self.results["grafting_strategies"]
            fastest = min(data["compute_times_us"].items(), key=lambda x: x[1])
            slowest = max(data["compute_times_us"].items(), key=lambda x: x[1])
            print("\n🔄 Grafting Strategy Performance:")
            print(f"  • Fastest: {fastest[0]} ({fastest[1]:.2f} µs)")
            print(f"  • Slowest: {slowest[0]} ({slowest[1]:.2f} µs)")
            print(f"  • Performance ratio: {slowest[1]/fastest[1]:.2f}x")

        # Throughput
        if "full_layer_throughput" in self.results:
            data = self.results["full_layer_throughput"]
            best_idx = np.argmax(data["throughput_samples_per_sec"])
            print("\n🚀 Layer Throughput:")
            print(f"  • Best config: {data['configurations'][best_idx]}")
            print(f"  • Forward time: {data['forward_times_ms'][best_idx]:.3f} ms")
            print(f"  • Throughput: {data['throughput_samples_per_sec'][best_idx]:,.0f} samples/sec")
            if data["memory_usage_mb"][best_idx] > 0:
                print(f"  • GPU memory: {data['memory_usage_mb'][best_idx]:.2f} MB")

        print("\n✅ All benchmarks completed successfully!")


def main():
    """Run Phase 2 benchmarks."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2 Performance Benchmarks")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"], help="Device to run benchmarks on")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmarks with reduced iterations")
    args = parser.parse_args()

    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Create and run benchmarks
    benchmarks = Phase2Benchmarks(device)

    if args.quick:
        # Reduce sizes for quick test
        benchmarks.seed_counts = [100, 1000]
        benchmarks.batch_sizes = [32, 64]
        benchmarks.dimensions = [512]

    benchmarks.run_all_benchmarks()


if __name__ == "__main__":
    main()
