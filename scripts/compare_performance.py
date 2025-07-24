#!/usr/bin/env python3
"""
Compare performance metrics between baseline and current measurements.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

def load_metrics(filepath: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    # Extract measurements
    if "measurements" in data:
        measurements = data["measurements"]
    else:
        measurements = [data]
    
    # Average across measurements
    metrics = {}
    for m in measurements:
        for key, value in m.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)
    
    # Compute averages
    avg_metrics = {}
    for key, values in metrics.items():
        avg_metrics[key] = sum(values) / len(values)
    
    return avg_metrics

def compare_metrics(baseline: Dict[str, Any], current: Dict[str, Any], tolerance: float) -> Dict[str, Any]:
    """Compare baseline and current metrics."""
    comparison = {
        "passed": True,
        "metrics": {},
        "regressions": [],
        "improvements": []
    }
    
    key_metrics = [
        ("forward_pass_p99_ms", False),  # Lower is better
        ("throughput_samples_per_sec", True),  # Higher is better
        ("gpu_memory_peak_mb", False),  # Lower is better
        ("gpu_utilization_percent", True),  # Higher is better
    ]
    
    for metric, higher_is_better in key_metrics:
        if metric in baseline and metric in current:
            baseline_val = baseline[metric]
            current_val = current[metric]
            
            if baseline_val > 0:
                ratio = current_val / baseline_val
                percent_change = (ratio - 1) * 100
                
                if higher_is_better:
                    regression = ratio < (1 - tolerance)
                    improvement = ratio > (1 + tolerance)
                else:
                    regression = ratio > (1 + tolerance)
                    improvement = ratio < (1 - tolerance)
                
                comparison["metrics"][metric] = {
                    "baseline": baseline_val,
                    "current": current_val,
                    "percent_change": percent_change,
                    "regression": regression,
                    "improvement": improvement
                }
                
                if regression:
                    comparison["regressions"].append(metric)
                    comparison["passed"] = False
                elif improvement:
                    comparison["improvements"].append(metric)
    
    return comparison

def generate_markdown_report(comparison: Dict[str, Any]) -> str:
    """Generate markdown report for PR comments."""
    lines = ["## Performance Comparison Report\n"]
    
    if comparison["passed"]:
        lines.append("✅ **All performance metrics within acceptable range**\n")
    else:
        lines.append("❌ **Performance regressions detected**\n")
    
    lines.append("### Metrics Summary\n")
    lines.append("| Metric | Baseline | Current | Change | Status |")
    lines.append("|--------|----------|---------|--------|--------|")
    
    for metric, data in comparison["metrics"].items():
        status = "🔴" if data["regression"] else ("🟢" if data["improvement"] else "🟡")
        change = f"{data['percent_change']:+.1f}%"
        
        lines.append(
            f"| {metric} | {data['baseline']:.2f} | {data['current']:.2f} | {change} | {status} |"
        )
    
    if comparison["regressions"]:
        lines.append("\n### ⚠️ Regressions")
        for metric in comparison["regressions"]:
            lines.append(f"- {metric}")
    
    if comparison["improvements"]:
        lines.append("\n### 🎉 Improvements")
        for metric in comparison["improvements"]:
            lines.append(f"- {metric}")
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Compare performance metrics")
    parser.add_argument("--baseline", required=True, help="Baseline metrics file")
    parser.add_argument("--current", required=True, help="Current metrics file")
    parser.add_argument("--tolerance", type=float, default=0.1, help="Tolerance for regression (0.1 = 10%)")
    parser.add_argument("--output", help="Output comparison file")
    
    args = parser.parse_args()
    
    # Load metrics
    baseline_metrics = load_metrics(Path(args.baseline))
    current_metrics = load_metrics(Path(args.current))
    
    # Compare
    comparison = compare_metrics(baseline_metrics, current_metrics, args.tolerance)
    
    # Generate report
    report = generate_markdown_report(comparison)
    print(report)
    
    # Save comparison if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON comparison
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save markdown report
        with open(output_path.with_suffix('.md'), 'w') as f:
            f.write(report)
    
    # Exit with error if regressions found
    sys.exit(0 if comparison["passed"] else 1)

if __name__ == "__main__":
    main()