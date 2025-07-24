#!/bin/bash
# Run Phase 2 Performance Benchmarks

echo "Running Phase 2 Performance Benchmarks..."
echo "========================================"

# Create results directory
mkdir -p benchmark_results/phase2

# Run benchmarks
python benchmarks/morphogenetic_v2/phase2_benchmarks.py

# Check if plots were generated
if [ -d "benchmark_results/phase2" ] && [ "$(ls -A benchmark_results/phase2/*.png 2>/dev/null)" ]; then
    echo ""
    echo "Benchmark plots generated in: benchmark_results/phase2/"
    ls -la benchmark_results/phase2/*.png
fi

# Check if results were saved
if [ -f "benchmark_results/phase2/benchmark_results.json" ]; then
    echo ""
    echo "Benchmark results saved to: benchmark_results/phase2/benchmark_results.json"
fi

echo ""
echo "Benchmarks completed!"