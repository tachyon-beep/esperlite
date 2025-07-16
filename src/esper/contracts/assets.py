"""
Core asset models for the Esper system.
These models define the primary entities: Seeds, Blueprints, and related structures.
"""

from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from .enums import BlueprintState
from .enums import SeedState


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
            f"{k}: {v:.3f}"
            for k, v in metrics.items()  # pylint: disable=no-member
        )
        return f"Performance: {metrics_str}"


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
