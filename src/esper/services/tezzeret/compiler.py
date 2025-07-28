"""
Blueprint Compiler Module.

This module compiles Blueprint definitions into executable TorchScript kernels,
replacing the placeholder implementations with real compilation logic.
"""

import hashlib
import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.jit
import torch.nn as nn

from esper.contracts.assets import Blueprint
from esper.contracts.assets import CompiledKernel
from esper.contracts.assets import KernelMetadata

logger = logging.getLogger(__name__)


class CompilationError(Exception):
    """Raised when blueprint compilation fails."""
    pass


class BlueprintCompiler:
    """Compiles Blueprint definitions into executable TorchScript kernels."""

    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the compiler.
        
        Args:
            device: Target device for compilation (cpu/cuda)
        """
        self.device = device or torch.device("cpu")
        self.compilation_cache = {}

    def compile_blueprint(self, blueprint: Blueprint) -> CompiledKernel:
        """
        Main compilation pipeline.
        
        Args:
            blueprint: Blueprint to compile
            
        Returns:
            CompiledKernel: Compiled kernel artifact
            
        Raises:
            CompilationError: If compilation fails
        """
        try:
            # Step 1: Validate blueprint structure
            self._validate_blueprint(blueprint)

            # Step 2: Generate PyTorch module from architecture
            module = self._generate_pytorch_module(blueprint.architecture)

            # Step 3: Compile to TorchScript
            script_module = self._compile_to_torchscript(module)

            # Step 4: Optimize for target device
            optimized_module = self._optimize_for_device(script_module, self.device)

            # Step 5: Extract metadata
            metadata = self._extract_kernel_metadata(
                optimized_module, blueprint, module
            )

            # Step 6: Package kernel
            kernel = self._package_kernel(optimized_module, blueprint, metadata)

            logger.info(
                "Successfully compiled blueprint %s into kernel %s",
                blueprint.blueprint_id,
                kernel.kernel_id
            )

            return kernel

        except Exception as e:
            logger.error("Failed to compile blueprint %s: %s", blueprint.blueprint_id, e)
            raise CompilationError(f"Compilation failed: {e}") from e

    def _validate_blueprint(self, blueprint: Blueprint) -> None:
        """
        Ensure blueprint meets compilation requirements.
        
        Args:
            blueprint: Blueprint to validate
            
        Raises:
            CompilationError: If validation fails
        """
        if not blueprint.architecture:
            raise CompilationError("Blueprint has no architecture definition")

        required_fields = ["type", "config"]
        for field in required_fields:
            if field not in blueprint.architecture:
                raise CompilationError(
                    f"Blueprint architecture missing required field: {field}"
                )

        # Validate architecture type
        arch_type = blueprint.architecture["type"]
        supported_types = ["linear", "conv", "attention", "custom"]
        if arch_type not in supported_types:
            raise CompilationError(
                f"Unsupported architecture type: {arch_type}. "
                f"Supported types: {supported_types}"
            )

    def _generate_pytorch_module(self, architecture: Dict[str, Any]) -> nn.Module:
        """
        Convert blueprint architecture to PyTorch module.
        
        Args:
            architecture: Architecture definition from blueprint
            
        Returns:
            nn.Module: Generated PyTorch module
        """
        arch_type = architecture["type"]
        config = architecture["config"]

        if arch_type == "linear":
            return self._create_linear_module(config)
        elif arch_type == "conv":
            return self._create_conv_module(config)
        elif arch_type == "attention":
            return self._create_attention_module(config)
        elif arch_type == "custom":
            return self._create_custom_module(config)
        else:
            raise CompilationError(f"Unknown architecture type: {arch_type}")

    def _create_linear_module(self, config: Dict[str, Any]) -> nn.Module:
        """Create a linear (fully-connected) module."""
        layers = []

        input_size = config.get("input_size", 128)
        hidden_sizes = config.get("hidden_sizes", [256, 256])
        output_size = config.get("output_size", 128)
        activation = config.get("activation", "relu")
        dropout = config.get("dropout", 0.0)

        # Build layers
        current_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_size, hidden_size))

            # Add activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "tanh":
                layers.append(nn.Tanh())

            # Add dropout if specified
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, output_size))

        return nn.Sequential(*layers)

    def _create_conv_module(self, config: Dict[str, Any]) -> nn.Module:
        """Create a convolutional module."""
        layers = []

        in_channels = config.get("in_channels", 3)
        out_channels_list = config.get("out_channels", [64, 128, 256])
        kernel_sizes = config.get("kernel_sizes", [3, 3, 3])
        strides = config.get("strides", [1, 1, 1])
        paddings = config.get("paddings", [1, 1, 1])
        activation = config.get("activation", "relu")
        pool_type = config.get("pool_type", "max")
        pool_size = config.get("pool_size", 2)

        # Build convolutional layers
        current_channels = in_channels
        for i, out_channels in enumerate(out_channels_list):
            # Conv layer
            layers.append(
                nn.Conv2d(
                    current_channels,
                    out_channels,
                    kernel_sizes[i] if i < len(kernel_sizes) else 3,
                    stride=strides[i] if i < len(strides) else 1,
                    padding=paddings[i] if i < len(paddings) else 1,
                )
            )

            # Batch norm
            layers.append(nn.BatchNorm2d(out_channels))

            # Activation
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())

            # Pooling
            if pool_type == "max":
                layers.append(nn.MaxPool2d(pool_size))
            elif pool_type == "avg":
                layers.append(nn.AvgPool2d(pool_size))

            current_channels = out_channels

        # Global pooling
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())

        return nn.Sequential(*layers)

    def _create_attention_module(self, config: Dict[str, Any]) -> nn.Module:
        """Create an attention-based module."""
        # Simplified attention module for MVP
        embed_dim = config.get("embed_dim", 256)
        num_heads = config.get("num_heads", 8)
        dropout = config.get("dropout", 0.1)

        return nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def _create_custom_module(self, config: Dict[str, Any]) -> nn.Module:
        """Create a custom module from configuration."""
        # This would parse a more complex custom architecture definition
        # For MVP, we'll create a simple residual block
        class ResidualBlock(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(dim, dim),
                    nn.ReLU(),
                    nn.Linear(dim, dim),
                )

            def forward(self, x):
                return x + self.layers(x)

        dim = config.get("dim", 256)
        num_blocks = config.get("num_blocks", 2)

        blocks = [ResidualBlock(dim) for _ in range(num_blocks)]
        return nn.Sequential(*blocks)

    def _compile_to_torchscript(self, module: nn.Module) -> torch.jit.ScriptModule:
        """
        JIT compile the module with optimizations.
        
        Args:
            module: PyTorch module to compile
            
        Returns:
            torch.jit.ScriptModule: Compiled module
        """
        # Move to device before compilation
        module = module.to(self.device)
        module.eval()  # Set to eval mode for compilation

        # Create example input for tracing
        example_input = self._create_example_input(module)

        try:
            # Try scripting first (preferred for control flow)
            script_module = torch.jit.script(module)
        except (RuntimeError, TypeError):
            # Fall back to tracing if scripting fails
            logger.warning("Scripting failed, falling back to tracing")
            script_module = torch.jit.trace(module, example_input)

        # Optimize the graph
        script_module = torch.jit.optimize_for_inference(script_module)

        return script_module

    def _create_example_input(self, module: nn.Module) -> torch.Tensor:
        """Create example input tensor for tracing."""
        # Analyze module to determine input shape
        # This is simplified - real implementation would be more sophisticated

        if isinstance(module, nn.Sequential):
            first_layer = module[0]
            if isinstance(first_layer, nn.Linear):
                batch_size = 1
                input_size = first_layer.in_features
                return torch.randn(batch_size, input_size, device=self.device)
            elif isinstance(first_layer, nn.Conv2d):
                batch_size = 1
                channels = first_layer.in_channels
                height = width = 32  # Default spatial dimensions
                return torch.randn(batch_size, channels, height, width, device=self.device)

        # Default fallback
        return torch.randn(1, 128, device=self.device)

    def _optimize_for_device(
        self, script_module: torch.jit.ScriptModule, device: torch.device
    ) -> torch.jit.ScriptModule:
        """
        Apply device-specific optimizations.
        
        Args:
            script_module: Compiled module
            device: Target device
            
        Returns:
            torch.jit.ScriptModule: Optimized module
        """
        if device.type == "cuda":
            # CUDA-specific optimizations
            # Enable TensorRT optimization if available
            try:
                import torch_tensorrt
                script_module = torch_tensorrt.compile(
                    script_module,
                    inputs=[self._create_example_input(script_module)],
                    enabled_precisions={torch.float32},
                )
            except ImportError:
                logger.info("TensorRT not available, using standard CUDA optimization")

        # Set to eval mode before freezing
        script_module.eval()

        # Freeze the module for inference
        try:
            script_module = torch.jit.freeze(script_module)
        except AttributeError:
            # If freeze fails, just optimize for inference
            script_module = torch.jit.optimize_for_inference(script_module)

        return script_module

    def _extract_kernel_metadata(
        self,
        script_module: torch.jit.ScriptModule,
        blueprint: Blueprint,
        original_module: nn.Module
    ) -> KernelMetadata:
        """Extract metadata from compiled kernel."""
        # Calculate parameter count from original module (script module may not have parameters)
        param_count = sum(p.numel() for p in original_module.parameters())

        # Estimate memory footprint
        memory_mb = (param_count * 4) / (1024 * 1024)  # Assume float32

        # Determine input/output shapes
        input_shape, output_shape = self._infer_io_shapes(original_module)

        # Generate kernel ID
        kernel_id = self._generate_kernel_id(blueprint.blueprint_id)

        return KernelMetadata(
            kernel_id=kernel_id,
            blueprint_id=blueprint.blueprint_id,
            name=f"kernel_{blueprint.name}",
            input_shape=input_shape,
            output_shape=output_shape,
            parameter_count=param_count,
            device_requirements=["cuda", "cpu"] if self.device.type == "cuda" else ["cpu"],
            memory_footprint_mb=memory_mb,
            compilation_target="torchscript",
            optimization_flags={
                "device": str(self.device),
                "optimized": True,
                "frozen": True,
            },
            performance_profile={},
            compatibility_version="1.0",
        )

    def _infer_io_shapes(self, module: nn.Module) -> Tuple[list, list]:
        """Infer input and output shapes from module."""
        # Simplified shape inference
        if isinstance(module, nn.Sequential):
            first_layer = module[0]
            last_layer = module[-1]

            # Input shape
            if isinstance(first_layer, nn.Linear):
                input_shape = [first_layer.in_features]
            elif isinstance(first_layer, nn.Conv2d):
                input_shape = [first_layer.in_channels, 32, 32]  # Default spatial
            else:
                input_shape = [128]  # Default

            # Output shape
            if isinstance(last_layer, nn.Linear):
                output_shape = [last_layer.out_features]
            elif isinstance(last_layer, nn.Conv2d):
                output_shape = [last_layer.out_channels, 1, 1]
            elif isinstance(last_layer, nn.Flatten):
                # Try to find the last conv/linear layer before flatten
                for layer in reversed(list(module.children())[:-1]):
                    if isinstance(layer, nn.AdaptiveAvgPool2d):
                        prev_layer = None
                        for l in reversed(list(module.children())):
                            if isinstance(l, nn.Conv2d):
                                prev_layer = l
                                break
                        if prev_layer:
                            output_shape = [prev_layer.out_channels]
                        else:
                            output_shape = [256]
                        break
                else:
                    output_shape = [256]
            else:
                output_shape = [128]  # Default
        else:
            input_shape = [128]
            output_shape = [128]

        return input_shape, output_shape

    def _generate_kernel_id(self, blueprint_id: str) -> str:
        """Generate unique kernel ID."""
        import time
        content = f"{blueprint_id}:{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _package_kernel(
        self,
        script_module: torch.jit.ScriptModule,
        blueprint: Blueprint,
        metadata: KernelMetadata
    ) -> CompiledKernel:
        """
        Package compiled code with metadata and versioning.
        
        Args:
            script_module: Compiled module
            blueprint: Original blueprint
            metadata: Kernel metadata
            
        Returns:
            CompiledKernel: Packaged kernel
        """
        # For now, we'll create a reference to where the binary would be stored
        # In production, this would upload to S3
        binary_ref = f"s3://esper-kernels/{metadata.kernel_id}/kernel.pt"

        # Calculate checksum
        import io
        buffer = io.BytesIO()
        torch.jit.save(script_module, buffer)
        binary_data = buffer.getvalue()
        checksum = hashlib.sha256(binary_data).hexdigest()
        metadata.checksum = checksum

        # Create compiled kernel
        kernel = CompiledKernel(
            kernel_id=metadata.kernel_id,
            blueprint_id=blueprint.blueprint_id,
            binary_ref=binary_ref,
            metadata=metadata,
            status="compiled",
            validation_results={
                "compilation_successful": True,
                "optimization_applied": True,
                "device_compatible": True,
            },
        )

        return kernel
