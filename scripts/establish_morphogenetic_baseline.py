#!/usr/bin/env python3
"""
Establish performance baseline for morphogenetic system.

This script measures current system performance to track
improvements and prevent regressions during migration.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.esper.execution.kasmina_config import KasminaConfig
from src.esper.execution.kasmina_layer import KasminaLayer
from src.esper.morphogenetic_v2.common.performance_baseline import PerformanceBaseline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_model(num_layers: int = 12, hidden_dim: int = 768) -> torch.nn.Module:
    """Create a test model with morphogenetic layers."""

    class TestModel(torch.nn.Module):
        def __init__(self, num_layers: int, hidden_dim: int):
            super().__init__()

            # Create transformer-like architecture
            self.layers = torch.nn.ModuleList()

            for i in range(num_layers):
                # Base transformer layer
                base_layer = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim * 4, hidden_dim)
                )

                # Wrap with KasminaLayer
                config = KasminaConfig(
                    num_seeds=1,  # Current implementation
                    health_threshold=0.7,
                    enable_async=False
                )

                kasmina_layer = KasminaLayer(base_layer, config)
                self.layers.append(kasmina_layer)

            self.output = torch.nn.Linear(hidden_dim, 1000)  # Classification head

        def forward(self, x):
            for layer in self.layers:
                x = layer(x) + x  # Residual connection
            return self.output(x)

    return TestModel(num_layers, hidden_dim)


def main():
    parser = argparse.ArgumentParser(description="Establish morphogenetic baseline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/morphogenetic"),
        help="Output directory for baseline results"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=12,
        help="Number of morphogenetic layers"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=768,
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 16, 32, 64],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )

    args = parser.parse_args()

    logger.info("Establishing morphogenetic performance baseline")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.num_layers} layers, {args.hidden_dim} hidden dim")

    # Create baseline measurement system
    baseline = PerformanceBaseline(output_dir=args.output_dir)

    # Create test model
    logger.info("Creating test model...")
    model = create_test_model(args.num_layers, args.hidden_dim)
    device = torch.device(args.device)

    # Run baseline for each batch size
    all_metrics = []

    for batch_size in args.batch_sizes:
        logger.info(f"\nRunning baseline with batch_size={batch_size}")

        try:
            metrics = baseline.run_baseline_benchmark(
                model=model,
                input_shape=(batch_size, args.hidden_dim),
                batch_size=batch_size,
                device=device,
                implementation="legacy",
                phase=0
            )

            all_metrics.append(metrics)

            # Log key metrics
            logger.info(f"Results for batch_size={batch_size}:")
            logger.info(f"  Forward pass P99: {metrics.forward_pass_p99_ms:.2f}ms")
            logger.info(f"  Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
            logger.info(f"  GPU Memory Peak: {metrics.gpu_memory_peak_mb:.2f}MB")
            logger.info(f"  Seeds per layer: {metrics.seeds_per_layer}")

        except Exception as e:
            logger.error(f"Failed to run baseline for batch_size={batch_size}: {e}")
            continue

    # Save baseline
    if all_metrics:
        baseline_file = baseline.save_baseline("phase0_baseline")
        logger.info(f"\nBaseline saved to: {baseline_file}")

        # Summary statistics
        logger.info("\nBaseline Summary:")
        logger.info("=" * 60)

        avg_latency = sum(m.forward_pass_p99_ms for m in all_metrics) / len(all_metrics)
        max_memory = max(m.gpu_memory_peak_mb for m in all_metrics)

        logger.info(f"Average P99 Latency: {avg_latency:.2f}ms")
        logger.info(f"Max GPU Memory: {max_memory:.2f}MB")
        logger.info("Current Implementation: Single seed per layer")
        logger.info("Target (Phase 5): <100ms latency, thousands of seeds")
        logger.info("=" * 60)

        # Check against targets
        if avg_latency > 100:
            logger.warning(f"⚠️  Current latency ({avg_latency:.2f}ms) exceeds target (100ms)")
        else:
            logger.info(f"✓ Current latency ({avg_latency:.2f}ms) meets target")

        return 0
    else:
        logger.error("No baseline metrics collected")
        return 1


if __name__ == "__main__":
    sys.exit(main())
