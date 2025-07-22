"""
Core asset models for the Esper system.
These models define the primary entities: Seeds, Blueprints, and related structures.
"""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import BlueprintState, SeedState


def _empty_dict() -> Dict[str, Any]:
    """Factory function for empty dictionaries to help pylint understand types."""
    return {}


def _empty_float_dict() -> Dict[str, float]:
    """Factory function for empty float dictionaries to help pylint understand types."""
    return {}


class Seed(BaseModel):
    """A morphogenetic seed embedded in a neural network layer."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        # Enable field caching for computed properties
        extra="forbid",  # Prevent extra fields for better performance
    )

    seed_id: str = Field(default_factory=lambda: str(uuid4()))
    layer_id: int = Field(ge=0)  # Non-negative layer ID
    position: int = Field(ge=0)  # Non-negative position
    state: SeedState = SeedState.DORMANT
    blueprint_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = Field(default_factory=_empty_dict)

    def state_display(self) -> str:
        """Display-friendly state name."""
        return self.state.title()

    def is_active(self) -> bool:
        """Check if seed is in an active training state."""
        return self.state in {SeedState.TRAINING, SeedState.GRAFTING}


class Blueprint(BaseModel):
    """An architectural blueprint for neural network components."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        extra="forbid",  # Prevent extra fields for better performance
    )

    blueprint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=255)  # Bounded string length
    description: str = Field(min_length=1, max_length=1000)
    state: BlueprintState = BlueprintState.PROPOSED
    architecture: Dict[str, Any]  # Architecture definition
    hyperparameters: Dict[str, Any] = Field(default_factory=_empty_dict)
    performance_metrics: Dict[str, float] = Field(default_factory=_empty_float_dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_by: str = Field(
        min_length=1, max_length=100
    )  # Component that created this blueprint

    def state_display(self) -> str:
        """Display-friendly state name."""
        return self.state.title()

    def is_ready_for_deployment(self) -> bool:
        """Check if blueprint is ready for deployment."""
        return self.state == BlueprintState.CHARACTERIZED

    def get_performance_summary(self) -> str:
        """Get a summary of performance metrics."""
        if not self.performance_metrics:
            return "No metrics available"

        # Explicit type hint forces Pylint to recognize this as a dict
        metrics: Dict[str, float] = self.performance_metrics
        metrics_str = ", ".join(
            f"{k}: {v:.3f}" for k, v in metrics.items()  # pylint: disable=no-member
        )
        return f"Performance: {metrics_str}"


class KernelMetadata(BaseModel):
    """Metadata for compiled kernel artifacts."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid",
    )

    kernel_id: str
    blueprint_id: str
    name: str = Field(min_length=1, max_length=255)
    input_shape: List[int]  # Expected input tensor shape (excluding batch)
    output_shape: List[int]  # Expected output tensor shape (excluding batch)
    parameter_count: int = Field(ge=0)
    device_requirements: List[str] = Field(default_factory=list)  # ["cuda", "cpu"]
    memory_footprint_mb: float = Field(ge=0.0)
    compilation_target: str = Field(default="torchscript")  # "torchscript" or "pickle"
    optimization_flags: Dict[str, Any] = Field(default_factory=_empty_dict)
    performance_profile: Dict[str, float] = Field(default_factory=_empty_float_dict)
    compatibility_version: str = Field(default="1.0")
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    checksum: Optional[str] = None  # SHA256 of kernel binary

    def is_compatible_with_shape(self, input_shape: List[int]) -> bool:
        """Check if kernel is compatible with given input shape."""
        if len(input_shape) != len(self.input_shape):
            return False

        # Check all dimensions (input_shape is already excluding batch)
        for i in range(len(input_shape)):
            if input_shape[i] != self.input_shape[i]:
                return False

        return True

    def get_memory_estimate(self, batch_size: int) -> float:
        """Estimate memory usage in MB for given batch size."""
        base_memory = self.memory_footprint_mb
        # Rough estimate: linear scaling with batch size
        return base_memory * (batch_size / 32.0)  # Assume baseline is batch=32


class CompiledKernel(BaseModel):
    """A compiled kernel artifact ready for execution."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        extra="forbid",
    )

    kernel_id: str = Field(default_factory=lambda: str(uuid4()))
    blueprint_id: str
    binary_ref: str  # S3 reference to kernel binary
    metadata: KernelMetadata
    status: str = Field(
        default="compiled", pattern=r"^(compiled|validated|deployed|deprecated)$"
    )
    validation_results: Dict[str, Any] = Field(default_factory=_empty_dict)
    deployment_stats: Dict[str, float] = Field(default_factory=_empty_float_dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_used_at: Optional[datetime] = None

    def is_ready_for_deployment(self) -> bool:
        """Check if kernel is ready for deployment."""
        return self.status in {"validated", "deployed"}

    def update_usage_stats(self):
        """Update usage statistics."""
        self.last_used_at = datetime.now(UTC)


class TrainingSession(BaseModel):
    """A training session for a morphogenetic system."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        extra="forbid",  # Prevent extra fields for better performance
    )

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(min_length=1, max_length=1000)
    training_model_config: Dict[str, Any]  # Renamed from model_config
    training_config: Dict[str, Any]
    seeds: List[Seed] = Field(default_factory=list)
    blueprints: List[Blueprint] = Field(default_factory=list)
    status: str = Field(
        default="initialized",
        pattern=r"^(initialized|running|paused|completed|failed)$",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def get_active_seed_count(self) -> int:
        """Get count of active seeds."""
        return sum(1 for seed in self.seeds if seed.is_active())

    def get_session_summary(self) -> str:
        """Get a summary of the training session."""
        total_seeds = len(self.seeds)
        active_seeds = self.get_active_seed_count()
        total_blueprints = len(self.blueprints)

        return (
            f"Session '{self.name}': {total_seeds} seeds "
            f"({active_seeds} active), {total_blueprints} blueprints, "
            f"status: {self.status}"
        )
