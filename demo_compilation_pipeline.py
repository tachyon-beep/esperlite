#!/usr/bin/env python3
"""
Demo script showing the real kernel compilation pipeline in action.

This demonstrates Phase B1 of the Remediation Plan Beta.
"""

from datetime import datetime

import torch

from esper.contracts.assets import Blueprint
from esper.contracts.enums import BlueprintState
from esper.services.tezzeret.compiler import BlueprintCompiler
from esper.services.tezzeret.optimizer import KernelOptimizer
from esper.services.tezzeret.validator import KernelValidator


def main():
    """Demonstrate the compilation pipeline."""
    print("=" * 80)
    print("Esper Kernel Compilation Pipeline Demo")
    print("Phase B1: Real Kernel Compilation")
    print("=" * 80)

    # Create a sample blueprint
    blueprint = Blueprint(
        blueprint_id="demo_001",
        name="Demo Neural Network",
        description="A demonstration neural network for kernel compilation",
        state=BlueprintState.PROPOSED,
        architecture={
            "type": "linear",
            "config": {
                "input_size": 784,  # MNIST-like input
                "hidden_sizes": [512, 256, 128],
                "output_size": 10,  # 10 classes
                "activation": "relu",
                "dropout": 0.2,
            }
        },
        hyperparameters={
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        created_by="demo_script",
    )

    print(f"\n1. Created Blueprint: {blueprint.name}")
    print(f"   Architecture: {blueprint.architecture['type']}")
    print(f"   Input size: {blueprint.architecture['config']['input_size']}")
    print(f"   Hidden layers: {blueprint.architecture['config']['hidden_sizes']}")
    print(f"   Output size: {blueprint.architecture['config']['output_size']}")

    # Initialize compilation pipeline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n2. Initializing compilation pipeline on device: {device}")

    compiler = BlueprintCompiler(device=device)
    optimizer = KernelOptimizer(device=device)
    validator = KernelValidator(device=device)

    # Step 1: Compile the blueprint
    print("\n3. Compiling blueprint to kernel...")
    start_time = datetime.now()

    try:
        compiled_kernel = compiler.compile_blueprint(blueprint)
        compilation_time = (datetime.now() - start_time).total_seconds()

        print(f"   ✓ Compilation successful in {compilation_time:.2f} seconds")
        print(f"   Kernel ID: {compiled_kernel.kernel_id}")
        print(f"   Parameter count: {compiled_kernel.metadata.parameter_count:,}")
        print(f"   Memory footprint: {compiled_kernel.metadata.memory_footprint_mb:.2f} MB")
        print(f"   Target: {compiled_kernel.metadata.compilation_target}")

    except Exception as e:
        print(f"   ✗ Compilation failed: {e}")
        return

    # Step 2: Optimize the kernel
    print("\n4. Optimizing kernel for performance...")

    # Create a dummy module for optimization demo
    # In production, this would load the actual compiled kernel
    dummy_module = torch.nn.Sequential(
        torch.nn.Linear(784, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 10),
    )
    scripted_module = torch.jit.script(dummy_module)

    try:
        target_device = "cuda" if device.type == "cuda" else "cpu"
        optimized_module = optimizer.optimize_kernel(scripted_module, target_device=target_device)

        # Profile and compare performance
        perf_comparison = optimizer.compare_performance(scripted_module, optimized_module)

        print("   ✓ Optimization successful")
        print(f"   Speedup: {perf_comparison['speedup']:.2f}x")
        print(f"   Memory reduction: {perf_comparison['memory_reduction']:.2f}x")
        print(f"   Throughput improvement: {perf_comparison['throughput_improvement']:.2f}x")

    except Exception as e:
        print(f"   ✗ Optimization failed: {e}")
        optimized_module = scripted_module

    # Step 3: Validate the kernel
    print("\n5. Validating kernel correctness and safety...")

    try:
        validation_result = validator.validate_kernel(
            compiled_kernel, blueprint, reference_module=None
        )

        print(f"   Overall validation: {'PASSED' if validation_result.is_valid else 'FAILED'}")
        print(f"   - Functional correctness: {'✓' if validation_result.functional_correctness else '✗'}")
        print(f"   - Performance acceptable: {'✓' if validation_result.performance_acceptable else '✗'}")
        print(f"   - Memory safe: {'✓' if validation_result.memory_safe else '✗'}")
        print(f"   - Gradient correct: {'✓' if validation_result.gradient_correct else '✗'}")

        if validation_result.metrics:
            print(f"   - Execution time: {validation_result.metrics.get('avg_execution_time_ms', 0):.2f} ms")
            print(f"   - Gradient norm: {validation_result.metrics.get('input_gradient_norm', 0):.2f}")

        if validation_result.warnings:
            print(f"   Warnings: {', '.join(validation_result.warnings)}")

        if validation_result.errors:
            print(f"   Errors: {', '.join(validation_result.errors)}")

    except Exception as e:
        print(f"   ✗ Validation failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"- Blueprint '{blueprint.name}' successfully compiled to kernel {compiled_kernel.kernel_id}")
    print(f"- Total parameter count: {compiled_kernel.metadata.parameter_count:,}")
    print(f"- Compilation pipeline status: {'Production Ready' if validation_result.is_valid else 'Needs Improvement'}")
    print("=" * 80)

    # Show statistics
    print("\nPipeline Statistics:")
    print(f"- Compiler stats: {compiler.compilation_cache}")
    print(f"- Optimizer stats: {optimizer.get_optimization_stats()}")
    print(f"- Validator stats: {validator.get_validation_stats()}")


if __name__ == "__main__":
    main()
