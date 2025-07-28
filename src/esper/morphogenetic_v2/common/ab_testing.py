"""
A/B testing framework for morphogenetic migration.

Enables safe comparison of legacy and new implementations.
"""

import json
import logging
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""
    test_id: str
    variant_a: str
    variant_b: str
    metrics: Dict[str, Dict[str, float]]  # metric -> {a: value, b: value}
    statistical_significance: Dict[str, float]  # metric -> p-value
    winner: Optional[str] = None
    confidence: float = 0.0
    sample_size: int = 0
    duration_sec: float = 0.0
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()


class ABTestMetric(ABC):
    """Base class for A/B test metrics."""

    @abstractmethod
    def compute(self, output: Any, target: Optional[Any] = None) -> float:
        """Compute metric value from model output."""
        pass

    @abstractmethod
    def is_better(self, value_a: float, value_b: float) -> bool:
        """Determine if value_b is better than value_a."""
        pass


class LatencyMetric(ABTestMetric):
    """Measures forward pass latency."""

    def compute(self, output: Any, target: Optional[Any] = None) -> float:
        # Latency is measured externally, not from output
        return 0.0

    def is_better(self, value_a: float, value_b: float) -> bool:
        return value_b < value_a  # Lower is better


class AccuracyMetric(ABTestMetric):
    """Measures prediction accuracy."""

    def compute(self, output: torch.Tensor, target: torch.Tensor) -> float:
        if target is None:
            return 0.0
        predictions = output.argmax(dim=-1)
        correct = (predictions == target).float()
        return correct.mean().item()

    def is_better(self, value_a: float, value_b: float) -> bool:
        return value_b > value_a  # Higher is better


class OutputSimilarityMetric(ABTestMetric):
    """Measures similarity between outputs (for regression testing)."""

    def compute(self, output_a: torch.Tensor, output_b: torch.Tensor) -> float:
        # Compute cosine similarity
        output_a_flat = output_a.flatten()
        output_b_flat = output_b.flatten()

        similarity = torch.nn.functional.cosine_similarity(
            output_a_flat.unsqueeze(0),
            output_b_flat.unsqueeze(0)
        )
        return similarity.item()

    def is_better(self, value_a: float, value_b: float) -> bool:
        return value_b > value_a  # Higher similarity is better


class ABTestRunner:
    """Runs A/B tests between implementations."""

    def __init__(
        self,
        metrics: Optional[Dict[str, ABTestMetric]] = None,
        output_dir: Path = Path("benchmarks/ab_tests")
    ):
        self.metrics = metrics or {
            "latency": LatencyMetric(),
            "accuracy": AccuracyMetric(),
            "output_similarity": OutputSimilarityMetric()
        }
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_test(
        self,
        model_a: torch.nn.Module,
        model_b: torch.nn.Module,
        test_data: torch.utils.data.DataLoader,
        test_id: Optional[str] = None,
        variant_names: Tuple[str, str] = ("legacy", "new"),
        sample_size: Optional[int] = None,
        measure_latency: bool = True
    ) -> ABTestResult:
        """
        Run A/B test comparing two model implementations.
        
        Args:
            model_a: First model variant (usually legacy)
            model_b: Second model variant (usually new)
            test_data: DataLoader with test samples
            test_id: Unique test identifier
            variant_names: Names for the variants
            sample_size: Max samples to test (None for all)
            measure_latency: Whether to measure latency
            
        Returns:
            ABTestResult with comparison metrics
        """
        if test_id is None:
            test_id = f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting A/B test {test_id}")
        logger.info(f"Variant A: {variant_names[0]}, Variant B: {variant_names[1]}")

        # Put models in eval mode
        model_a.eval()
        model_b.eval()

        # Collect results
        results_a = {metric: [] for metric in self.metrics}
        results_b = {metric: [] for metric in self.metrics}
        outputs_similarity = []

        device = next(model_a.parameters()).device
        samples_tested = 0

        import time
        start_time = time.perf_counter()

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_data):
                if sample_size and samples_tested >= sample_size:
                    break

                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(device)
                    targets = batch[1].to(device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(device)
                    targets = None

                # Run both models
                if measure_latency:
                    # Measure variant A latency
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_a = time.perf_counter()
                    output_a = model_a(inputs)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    latency_a = (time.perf_counter() - start_a) * 1000  # ms

                    # Measure variant B latency
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    start_b = time.perf_counter()
                    output_b = model_b(inputs)
                    torch.cuda.synchronize() if torch.cuda.is_available() else None
                    latency_b = (time.perf_counter() - start_b) * 1000  # ms

                    results_a["latency"].append(latency_a)
                    results_b["latency"].append(latency_b)
                else:
                    output_a = model_a(inputs)
                    output_b = model_b(inputs)

                # Compute metrics
                for metric_name, metric in self.metrics.items():
                    if metric_name == "latency":
                        continue  # Already measured
                    elif metric_name == "output_similarity":
                        similarity = metric.compute(output_a, output_b)
                        outputs_similarity.append(similarity)
                    else:
                        if targets is not None:
                            value_a = metric.compute(output_a, targets)
                            value_b = metric.compute(output_b, targets)
                            results_a[metric_name].append(value_a)
                            results_b[metric_name].append(value_b)

                samples_tested += inputs.size(0)

        duration = time.perf_counter() - start_time

        # Compute aggregate metrics
        aggregate_metrics = {}
        significance_tests = {}

        for metric_name in results_a:
            if results_a[metric_name]:  # Has results
                values_a = np.array(results_a[metric_name])
                values_b = np.array(results_b[metric_name])

                aggregate_metrics[metric_name] = {
                    variant_names[0]: {
                        "mean": float(np.mean(values_a)),
                        "std": float(np.std(values_a)),
                        "p50": float(np.percentile(values_a, 50)),
                        "p95": float(np.percentile(values_a, 95)),
                        "p99": float(np.percentile(values_a, 99))
                    },
                    variant_names[1]: {
                        "mean": float(np.mean(values_b)),
                        "std": float(np.std(values_b)),
                        "p50": float(np.percentile(values_b, 50)),
                        "p95": float(np.percentile(values_b, 95)),
                        "p99": float(np.percentile(values_b, 99))
                    }
                }

                # Statistical significance test (t-test)
                from scipy import stats
                _, p_value = stats.ttest_rel(values_a, values_b)
                significance_tests[metric_name] = float(p_value)

        # Add output similarity if measured
        if outputs_similarity:
            aggregate_metrics["output_similarity"] = {
                "mean": float(np.mean(outputs_similarity)),
                "std": float(np.std(outputs_similarity)),
                "min": float(np.min(outputs_similarity))
            }

        # Determine winner based on key metrics
        winner, confidence = self._determine_winner(
            aggregate_metrics, significance_tests, variant_names
        )

        # Create result
        result = ABTestResult(
            test_id=test_id,
            variant_a=variant_names[0],
            variant_b=variant_names[1],
            metrics=aggregate_metrics,
            statistical_significance=significance_tests,
            winner=winner,
            confidence=confidence,
            sample_size=samples_tested,
            duration_sec=duration
        )

        # Save result
        self._save_result(result)

        # Log summary
        self._log_summary(result)

        return result

    def _determine_winner(
        self,
        metrics: Dict[str, Dict[str, Dict[str, float]]],
        p_values: Dict[str, float],
        variant_names: Tuple[str, str],
        significance_threshold: float = 0.05
    ) -> Tuple[Optional[str], float]:
        """Determine winner based on metrics and significance."""
        wins_a = 0
        wins_b = 0
        total_metrics = 0

        for metric_name, metric_values in metrics.items():
            if metric_name == "output_similarity":
                continue  # Not a comparison metric

            if variant_names[0] in metric_values and variant_names[1] in metric_values:
                mean_a = metric_values[variant_names[0]]["mean"]
                mean_b = metric_values[variant_names[1]]["mean"]

                # Check if statistically significant
                p_value = p_values.get(metric_name, 1.0)
                if p_value < significance_threshold:
                    metric_obj = self.metrics.get(metric_name)
                    if metric_obj and metric_obj.is_better(mean_a, mean_b):
                        wins_b += 1
                    else:
                        wins_a += 1
                    total_metrics += 1

        if total_metrics == 0:
            return None, 0.0

        if wins_b > wins_a:
            confidence = wins_b / total_metrics
            return variant_names[1], confidence
        elif wins_a > wins_b:
            confidence = wins_a / total_metrics
            return variant_names[0], confidence
        else:
            return None, 0.0

    def _save_result(self, result: ABTestResult):
        """Save test result to file."""
        output_file = self.output_dir / f"{result.test_id}.json"

        with open(output_file, 'w') as f:
            json.dump(result.__dict__, f, indent=2)

        logger.info(f"Saved A/B test result to {output_file}")

    def _log_summary(self, result: ABTestResult):
        """Log test result summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"A/B Test Summary: {result.test_id}")
        logger.info(f"{'='*60}")
        logger.info(f"Samples tested: {result.sample_size}")
        logger.info(f"Duration: {result.duration_sec:.2f}s")

        for metric_name, metric_values in result.metrics.items():
            logger.info(f"\n{metric_name.upper()}:")
            if isinstance(metric_values, dict) and result.variant_a in metric_values:
                # Comparison metric
                for variant in [result.variant_a, result.variant_b]:
                    if variant in metric_values:
                        values = metric_values[variant]
                        logger.info(f"  {variant}:")
                        logger.info(f"    Mean: {values['mean']:.4f}")
                        if 'p99' in values:
                            logger.info(f"    P99: {values['p99']:.4f}")

                # Significance
                p_value = result.statistical_significance.get(metric_name, 1.0)
                logger.info(f"  P-value: {p_value:.4f}")
                if p_value < 0.05:
                    logger.info("  ✓ Statistically significant")
            else:
                # Single metric (like output similarity)
                logger.info(f"  {metric_values}")

        if result.winner:
            logger.info(f"\n🏆 Winner: {result.winner} (confidence: {result.confidence:.2%})")
        else:
            logger.info("\n🤝 No clear winner")
        logger.info(f"{'='*60}\n")


class ModelComparator:
    """High-level interface for model comparison."""

    def __init__(self, ab_runner: Optional[ABTestRunner] = None):
        self.ab_runner = ab_runner or ABTestRunner()

    def compare_implementations(
        self,
        legacy_model: torch.nn.Module,
        new_model: torch.nn.Module,
        test_data: torch.utils.data.DataLoader,
        test_name: str = "implementation_comparison"
    ) -> Dict[str, Any]:
        """
        Compare legacy and new implementations.
        
        Returns:
            Comparison results and recommendations
        """
        # Run A/B test
        result = self.ab_runner.run_test(
            model_a=legacy_model,
            model_b=new_model,
            test_data=test_data,
            test_id=test_name,
            variant_names=("legacy", "new")
        )

        # Analyze results
        analysis = {
            "test_result": result,
            "recommendations": [],
            "risks": [],
            "rollout_safe": False
        }

        # Check latency regression
        if "latency" in result.metrics:
            legacy_p99 = result.metrics["latency"]["legacy"]["p99"]
            new_p99 = result.metrics["latency"]["new"]["p99"]

            if new_p99 > legacy_p99 * 1.1:  # >10% regression
                analysis["risks"].append(
                    f"Latency regression detected: {new_p99:.2f}ms vs {legacy_p99:.2f}ms"
                )
            elif new_p99 < legacy_p99 * 0.9:  # >10% improvement
                analysis["recommendations"].append(
                    f"Latency improved: {new_p99:.2f}ms vs {legacy_p99:.2f}ms"
                )

        # Check output similarity
        if "output_similarity" in result.metrics:
            similarity = result.metrics["output_similarity"]["mean"]
            if similarity < 0.99:
                analysis["risks"].append(
                    f"Output divergence detected: {similarity:.4f} similarity"
                )
            else:
                analysis["recommendations"].append(
                    f"Output consistency maintained: {similarity:.4f} similarity"
                )

        # Overall recommendation
        if not analysis["risks"]:
            analysis["rollout_safe"] = True
            analysis["recommendations"].append("Safe to proceed with rollout")
        else:
            analysis["recommendations"].append("Address risks before rollout")

        return analysis
