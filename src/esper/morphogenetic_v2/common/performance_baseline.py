"""
Performance baseline measurement for morphogenetic migration.

Captures current system performance to track improvements and prevent regressions.
"""

import json
import logging
import time
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import GPUtil
import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance measurements."""
    # Latency metrics (in milliseconds)
    forward_pass_latency_ms: float = 0.0
    forward_pass_p50_ms: float = 0.0
    forward_pass_p95_ms: float = 0.0
    forward_pass_p99_ms: float = 0.0

    # Throughput metrics
    throughput_samples_per_sec: float = 0.0
    batch_size: int = 0

    # Memory metrics (in MB)
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    cpu_memory_mb: float = 0.0

    # GPU utilization
    gpu_utilization_percent: float = 0.0
    gpu_compute_percent: float = 0.0

    # Morphogenetic specific
    seeds_per_layer: int = 1
    adaptation_success_rate: float = 0.0
    seed_lifecycle_duration_sec: float = 0.0

    # System info
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    phase: int = 0
    implementation: str = "legacy"


class PerformanceBaseline:
    """Establishes and tracks performance baselines for the migration."""

    def __init__(self, output_dir: Path = Path("benchmarks/morphogenetic")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.measurements: List[PerformanceMetrics] = []
        self.cuda_available = torch.cuda.is_available()

    def measure_forward_pass(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """Measure forward pass performance."""
        device = input_tensor.device
        latencies = []

        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = model(input_tensor)

        # Synchronize before measurement
        if self.cuda_available and device.type == 'cuda':
            torch.cuda.synchronize(device)

        # Measure
        for _ in range(num_iterations):
            if self.cuda_available and device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                start_event.record()
                with torch.no_grad():
                    _ = model(input_tensor)
                end_event.record()

                torch.cuda.synchronize()
                latency_ms = start_event.elapsed_time(end_event)
            else:
                start = time.perf_counter()
                with torch.no_grad():
                    _ = model(input_tensor)
                latency_ms = (time.perf_counter() - start) * 1000

            latencies.append(latency_ms)

        # Calculate statistics
        latencies = np.array(latencies)
        return {
            "mean": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "std": float(np.std(latencies))
        }

    def measure_memory(self, device: Optional[torch.device] = None) -> Dict[str, float]:
        """Measure current memory usage."""
        metrics = {}

        # CPU memory
        process = psutil.Process()
        metrics["cpu_memory_mb"] = process.memory_info().rss / 1024 / 1024

        # GPU memory
        if self.cuda_available and device and device.type == 'cuda':
            metrics["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated(device) / 1024 / 1024
            metrics["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved(device) / 1024 / 1024

            # Peak memory
            torch.cuda.reset_peak_memory_stats(device)
            metrics["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated(device) / 1024 / 1024

            # GPU utilization
            try:
                gpus = GPUtil.getGPUs()
                if gpus and device.index < len(gpus):
                    gpu = gpus[device.index]
                    metrics["gpu_utilization_percent"] = gpu.load * 100
                    metrics["gpu_compute_percent"] = gpu.memoryUtil * 100
            except:
                pass

        return metrics

    def measure_morphogenetic_metrics(
        self,
        model: torch.nn.Module,
        test_duration_sec: int = 60
    ) -> Dict[str, float]:
        """Measure morphogenetic-specific metrics."""
        metrics = {
            "seeds_per_layer": 1,  # Current implementation
            "adaptation_success_rate": 0.0,
            "seed_lifecycle_duration_sec": 0.0
        }

        # Count morphogenetic layers
        morphogenetic_layers = 0
        total_seeds = 0

        for name, module in model.named_modules():
            if hasattr(module, 'seed_manager'):  # KasminaLayer
                morphogenetic_layers += 1
                if hasattr(module.seed_manager, 'num_seeds'):
                    total_seeds += module.seed_manager.num_seeds

        if morphogenetic_layers > 0:
            metrics["seeds_per_layer"] = total_seeds / morphogenetic_layers

        # TODO: Measure actual adaptation metrics during test run
        # This would require running actual adaptations and tracking success

        return metrics

    def run_baseline_benchmark(
        self,
        model: torch.nn.Module,
        input_shape: tuple,
        batch_size: int = 32,
        device: Optional[torch.device] = None,
        implementation: str = "legacy",
        phase: int = 0
    ) -> PerformanceMetrics:
        """Run complete baseline benchmark."""
        if device is None:
            device = torch.device("cuda" if self.cuda_available else "cpu")

        model = model.to(device)
        model.eval()

        # Create test input
        input_tensor = torch.randn(batch_size, *input_shape[1:], device=device)

        # Measure forward pass
        logger.info("Measuring forward pass performance...")
        forward_metrics = self.measure_forward_pass(model, input_tensor)

        # Measure throughput
        start_time = time.perf_counter()
        num_samples = 1000
        for _ in range(num_samples // batch_size):
            with torch.no_grad():
                _ = model(input_tensor)
        if self.cuda_available and device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        throughput = num_samples / elapsed

        # Measure memory
        logger.info("Measuring memory usage...")
        memory_metrics = self.measure_memory(device)

        # Measure morphogenetic metrics
        logger.info("Measuring morphogenetic metrics...")
        morph_metrics = self.measure_morphogenetic_metrics(model)

        # Compile results
        metrics = PerformanceMetrics(
            forward_pass_latency_ms=forward_metrics["mean"],
            forward_pass_p50_ms=forward_metrics["p50"],
            forward_pass_p95_ms=forward_metrics["p95"],
            forward_pass_p99_ms=forward_metrics["p99"],
            throughput_samples_per_sec=throughput,
            batch_size=batch_size,
            implementation=implementation,
            phase=phase,
            **memory_metrics,
            **morph_metrics
        )

        self.measurements.append(metrics)
        return metrics

    def save_baseline(self, name: str = "baseline"):
        """Save baseline measurements to file."""
        output_file = self.output_dir / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        data = {
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "cuda_available": self.cuda_available,
                "torch_version": torch.__version__,
                "num_measurements": len(self.measurements)
            },
            "measurements": [
                {k: v for k, v in m.__dict__.items()}
                for m in self.measurements
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved baseline to {output_file}")
        return output_file

    def load_baseline(self, filepath: Path) -> List[PerformanceMetrics]:
        """Load baseline measurements from file."""
        with open(filepath) as f:
            data = json.load(f)

        measurements = []
        for m in data["measurements"]:
            measurements.append(PerformanceMetrics(**m))

        return measurements

    def compare_with_baseline(
        self,
        current: PerformanceMetrics,
        baseline: PerformanceMetrics,
        tolerance: float = 0.05  # 5% tolerance
    ) -> Dict[str, Any]:
        """Compare current metrics with baseline."""
        comparison = {}

        key_metrics = [
            ("forward_pass_p99_ms", False),  # Lower is better
            ("throughput_samples_per_sec", True),  # Higher is better
            ("gpu_memory_peak_mb", False),  # Lower is better
            ("adaptation_success_rate", True),  # Higher is better
        ]

        for metric, higher_is_better in key_metrics:
            baseline_val = getattr(baseline, metric)
            current_val = getattr(current, metric)

            if baseline_val > 0:
                ratio = current_val / baseline_val
                percent_change = (ratio - 1) * 100

                if higher_is_better:
                    regression = ratio < (1 - tolerance)
                    improvement = ratio > (1 + tolerance)
                else:
                    regression = ratio > (1 + tolerance)
                    improvement = ratio < (1 - tolerance)

                comparison[metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "percent_change": percent_change,
                    "regression": regression,
                    "improvement": improvement
                }

        # Overall assessment
        regressions = [m for m, v in comparison.items() if v["regression"]]
        improvements = [m for m, v in comparison.items() if v["improvement"]]

        comparison["summary"] = {
            "regressions": regressions,
            "improvements": improvements,
            "pass": len(regressions) == 0
        }

        return comparison


def establish_baseline(model: torch.nn.Module, config: dict) -> Path:
    """Convenience function to establish performance baseline."""
    baseline = PerformanceBaseline()

    # Run baseline with different configurations
    for batch_size in [1, 16, 32, 64]:
        logger.info(f"Running baseline with batch_size={batch_size}")
        metrics = baseline.run_baseline_benchmark(
            model=model,
            input_shape=config.get("input_shape", (3, 224, 224)),
            batch_size=batch_size,
            implementation="legacy",
            phase=0
        )

        # Log key metrics
        logger.info(f"Forward pass P99: {metrics.forward_pass_p99_ms:.2f}ms")
        logger.info(f"Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
        logger.info(f"GPU Memory: {metrics.gpu_memory_peak_mb:.2f}MB")

    # Save baseline
    return baseline.save_baseline("phase0_baseline")
