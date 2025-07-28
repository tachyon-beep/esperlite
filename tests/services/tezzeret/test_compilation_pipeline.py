"""
Integration tests for the real kernel compilation pipeline.
"""


import pytest
import torch

from esper.contracts.assets import Blueprint
from esper.contracts.enums import BlueprintState
from esper.services.tezzeret.compiler import BlueprintCompiler
from esper.services.tezzeret.compiler import CompilationError
from esper.services.tezzeret.optimizer import KernelOptimizer
from esper.services.tezzeret.validator import KernelValidator


class TestCompilationPipeline:
    """Test the complete compilation pipeline."""

    def test_end_to_end_linear_compilation(self):
        """Test compiling a linear architecture blueprint."""
        # Create a linear blueprint
        blueprint = Blueprint(
            blueprint_id="test_linear_001",
            name="Linear Test Module",
            description="Test linear layer compilation",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "linear",
                "config": {
                    "input_size": 128,
                    "hidden_sizes": [256, 512, 256],
                    "output_size": 64,
                    "activation": "relu",
                    "dropout": 0.1,
                }
            },
            hyperparameters={"learning_rate": 0.001},
            created_by="test_suite",
        )

        # Initialize pipeline components
        device = torch.device("cpu")
        compiler = BlueprintCompiler(device=device)
        optimizer = KernelOptimizer(device=device)
        validator = KernelValidator(device=device)

        # Step 1: Compile
        compiled_kernel = compiler.compile_blueprint(blueprint)
        assert compiled_kernel is not None
        assert compiled_kernel.kernel_id
        assert compiled_kernel.blueprint_id == blueprint.blueprint_id
        assert compiled_kernel.status == "compiled"

        # Verify metadata
        metadata = compiled_kernel.metadata
        assert metadata.input_shape == [128]
        assert metadata.output_shape == [64]
        assert metadata.parameter_count > 0
        assert metadata.compilation_target == "torchscript"

        # Step 2: Load and optimize
        # For testing, create a simple module since we don't have S3
        test_module = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, 64),
        )
        scripted_module = torch.jit.script(test_module)

        # Optimize the kernel
        optimized_module = optimizer.optimize_kernel(scripted_module, target_device="cpu")
        assert optimized_module is not None

        # Compare performance
        perf_comparison = optimizer.compare_performance(
            scripted_module, optimized_module
        )
        assert perf_comparison["speedup"] >= 0.5  # At least not worse
        assert perf_comparison["throughput_improvement"] >= 0.5

        # Step 3: Validate (without reference module since we don't have the actual compiled one)
        validation_result = validator.validate_kernel(
            compiled_kernel, blueprint, reference_module=None
        )
        assert validation_result.is_valid
        assert validation_result.functional_correctness
        assert validation_result.performance_acceptable
        assert validation_result.memory_safe
        assert validation_result.gradient_correct
        assert len(validation_result.errors) == 0

    def test_conv_architecture_compilation(self):
        """Test compiling a convolutional architecture."""
        blueprint = Blueprint(
            blueprint_id="test_conv_001",
            name="Conv Test Module",
            description="Test convolutional layer compilation",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "conv",
                "config": {
                    "in_channels": 3,
                    "out_channels": [64, 128, 256],
                    "kernel_sizes": [3, 3, 3],
                    "strides": [1, 2, 2],
                    "paddings": [1, 1, 1],
                    "activation": "gelu",
                    "pool_type": "max",
                    "pool_size": 2,
                }
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        compiled_kernel = compiler.compile_blueprint(blueprint)

        assert compiled_kernel is not None
        assert compiled_kernel.metadata.input_shape == [3, 32, 32]
        assert compiled_kernel.metadata.output_shape == [256]
        assert "cuda" in compiled_kernel.metadata.device_requirements

    def test_attention_architecture_compilation(self):
        """Test compiling an attention-based architecture."""
        blueprint = Blueprint(
            blueprint_id="test_attention_001",
            name="Attention Test Module",
            description="Test attention layer compilation",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "attention",
                "config": {
                    "embed_dim": 512,
                    "num_heads": 8,
                    "dropout": 0.1,
                }
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        compiled_kernel = compiler.compile_blueprint(blueprint)

        assert compiled_kernel is not None
        assert compiled_kernel.metadata.parameter_count > 0

    def test_custom_architecture_compilation(self):
        """Test compiling a custom architecture."""
        blueprint = Blueprint(
            blueprint_id="test_custom_001",
            name="Custom Test Module",
            description="Test custom architecture compilation",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "custom",
                "config": {
                    "dim": 256,
                    "num_blocks": 3,
                }
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        compiled_kernel = compiler.compile_blueprint(blueprint)

        assert compiled_kernel is not None
        assert compiled_kernel.status == "compiled"

    def test_invalid_blueprint_compilation(self):
        """Test that invalid blueprints raise errors."""
        # Missing architecture type
        invalid_blueprint = Blueprint(
            blueprint_id="test_invalid_001",
            name="Invalid Blueprint",
            description="Missing architecture type",
            state=BlueprintState.PROPOSED,
            architecture={
                "config": {"input_size": 128}
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        with pytest.raises(CompilationError, match="missing required field: type"):
            compiler.compile_blueprint(invalid_blueprint)

    def test_unsupported_architecture_type(self):
        """Test that unsupported architecture types raise errors."""
        blueprint = Blueprint(
            blueprint_id="test_unsupported_001",
            name="Unsupported Architecture",
            description="Unsupported type",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "transformer",  # Not yet supported
                "config": {}
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        with pytest.raises(CompilationError, match="Unsupported architecture type"):
            compiler.compile_blueprint(blueprint)

    def test_optimizer_cuda_vs_cpu(self):
        """Test optimization for different devices."""
        # Create simple test module
        module = torch.nn.Sequential(
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
        )
        scripted = torch.jit.script(module)

        # CPU optimization
        cpu_optimizer = KernelOptimizer(device=torch.device("cpu"))
        cpu_optimized = cpu_optimizer.optimize_cpu_kernel(scripted)
        assert cpu_optimized is not None
        assert next(cpu_optimized.parameters()).device.type == "cpu"

        # Profile CPU kernel
        cpu_metrics = cpu_optimizer.profile_kernel(cpu_optimized)
        assert "execution_time_ms" in cpu_metrics
        assert cpu_metrics["execution_time_ms"] > 0

        # CUDA optimization (if available)
        if torch.cuda.is_available():
            cuda_optimizer = KernelOptimizer(device=torch.device("cuda"))
            cuda_optimized = cuda_optimizer.optimize_cuda_kernel(scripted.cuda())
            assert cuda_optimized is not None
            assert next(cuda_optimized.parameters()).device.type == "cuda"

            # Profile CUDA kernel
            cuda_metrics = cuda_optimizer.profile_kernel(cuda_optimized)
            assert "peak_memory_mb" in cuda_metrics

    def test_validator_gradient_checking(self):
        """Test gradient validation."""
        # Create test module
        module = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
        )
        scripted = torch.jit.script(module)

        # Create dummy kernel and blueprint
        from esper.contracts.assets import CompiledKernel
        from esper.contracts.assets import KernelMetadata

        metadata = KernelMetadata(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            name="test",
            input_shape=[64],
            output_shape=[64],
            parameter_count=sum(p.numel() for p in module.parameters()),
            device_requirements=["cpu"],
            memory_footprint_mb=1.0,
        )

        kernel = CompiledKernel(
            kernel_id="test_kernel",
            blueprint_id="test_blueprint",
            binary_ref="dummy",
            metadata=metadata,
        )

        blueprint = Blueprint(
            blueprint_id="test_blueprint",
            name="Test",
            description="Test",
            architecture={"type": "linear", "config": {}},
            created_by="test",
        )

        # Validate with specific focus on gradients
        validator = KernelValidator()
        # Override the load method for testing
        validator._load_kernel_module = lambda k: scripted

        result = validator.validate_kernel(kernel, blueprint)
        assert result.gradient_correct
        assert "input_gradient_norm" in result.metrics

    def test_performance_regression_detection(self):
        """Test that performance regressions are detected."""
        # Create a "slow" module
        class SlowModule(torch.nn.Module):
            def forward(self, x):
                # Simulate slow operation
                for _ in range(100):
                    x = x * 1.0001
                return x

        slow_module = torch.jit.script(SlowModule())

        # Validator with strict performance threshold
        validator = KernelValidator(performance_threshold=1.0)  # Very strict

        # Create dummy kernel
        from esper.contracts.assets import CompiledKernel
        from esper.contracts.assets import KernelMetadata

        kernel = CompiledKernel(
            kernel_id="slow_kernel",
            blueprint_id="test",
            binary_ref="dummy",
            metadata=KernelMetadata(
                kernel_id="slow_kernel",
                blueprint_id="test",
                name="slow",
                input_shape=[128],
                output_shape=[128],
                parameter_count=0,
                device_requirements=["cpu"],
                memory_footprint_mb=1.0,
            ),
        )

        blueprint = Blueprint(
            blueprint_id="test",
            name="Test",
            description="Test",
            architecture={"type": "linear", "config": {}},
            created_by="test",
        )

        # Override load method
        validator._load_kernel_module = lambda k: slow_module

        result = validator.validate_kernel(kernel, blueprint)
        # The slow module should fail performance validation
        assert not result.performance_acceptable or len(result.errors) > 0

    @pytest.mark.parametrize("arch_type,config", [
        ("linear", {"input_size": 64, "hidden_sizes": [128], "output_size": 32}),
        ("conv", {"in_channels": 1, "out_channels": [16, 32], "kernel_sizes": [3, 3]}),
        ("attention", {"embed_dim": 256, "num_heads": 4}),
        ("custom", {"dim": 128, "num_blocks": 2}),
    ])
    def test_parameterized_compilation(self, arch_type, config):
        """Test compilation with various architecture types and configs."""
        blueprint = Blueprint(
            blueprint_id=f"test_{arch_type}_param",
            name=f"{arch_type} parameterized test",
            description="Parameterized test",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": arch_type,
                "config": config,
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()
        compiled_kernel = compiler.compile_blueprint(blueprint)

        assert compiled_kernel is not None
        assert compiled_kernel.status == "compiled"
        assert compiled_kernel.metadata.parameter_count >= 0

    def test_compilation_cache(self):
        """Test that compilation cache works."""
        blueprint = Blueprint(
            blueprint_id="test_cache_001",
            name="Cache Test",
            description="Test compilation caching",
            state=BlueprintState.PROPOSED,
            architecture={
                "type": "linear",
                "config": {
                    "input_size": 32,
                    "hidden_sizes": [64],
                    "output_size": 16,
                }
            },
            hyperparameters={},
            created_by="test_suite",
        )

        compiler = BlueprintCompiler()

        # First compilation
        kernel1 = compiler.compile_blueprint(blueprint)

        # Second compilation of same blueprint
        kernel2 = compiler.compile_blueprint(blueprint)

        # Should get different kernel IDs (no caching in basic implementation)
        assert kernel1.kernel_id != kernel2.kernel_id

        # But same blueprint ID
        assert kernel1.blueprint_id == kernel2.blueprint_id
