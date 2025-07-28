"""
Operational data models for runtime system monitoring and control.
"""

import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Set

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator


class HealthSignal(BaseModel):
    """High-frequency health signal from a KasminaSeed."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        extra="forbid",  # Prevent extra fields for better performance
    )

    layer_id: int = Field(ge=0)  # Non-negative layer ID
    seed_id: int = Field(ge=0)  # Non-negative seed ID
    chunk_id: int = Field(ge=0)  # Non-negative chunk ID
    epoch: int = Field(ge=0)  # Non-negative epoch
    activation_variance: float = Field(ge=0.0)  # Non-negative variance
    dead_neuron_ratio: float = Field(ge=0.0, le=1.0)  # Ratio between 0-1
    avg_correlation: float = Field(ge=-1.0, le=1.0)  # Correlation between -1 and 1
    is_ready_for_transition: bool = False  # Critical for state machine sync

    # Additional fields for Tamiyo analysis
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_latency: float = Field(default=0.0, ge=0.0)
    error_count: int = Field(default=0, ge=0)
    active_seeds: int = Field(default=1, ge=0)
    total_seeds: int = Field(default=1, ge=1)
    timestamp: float = Field(default_factory=time.time)

    # Training dynamics for morphogenetic adaptation
    gradient_norm: float = Field(default=0.0, ge=0.0)
    gradient_variance: float = Field(default=0.0, ge=0.0)
    gradient_sign_stability: float = Field(default=0.0, ge=0.0, le=1.0)
    param_norm_ratio: float = Field(default=1.0, ge=0.0)

    # Performance metrics for system efficiency
    total_executions: int = Field(default=0, ge=0)
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    @field_validator("seed_id", "layer_id", "chunk_id", "epoch")
    @classmethod
    def _validate_ids(cls, v):
        """Validate that IDs are non-negative integers."""
        if v < 0:
            raise ValueError("IDs must be non-negative")
        return v

    def health_status(self) -> str:
        """Get health status as human-readable string."""
        if self.health_score >= 0.8:
            return "Healthy"
        elif self.health_score >= 0.6:
            return "Warning"
        else:
            return "Critical"

    def is_healthy(self) -> bool:
        """Check if seed is in healthy state."""
        return self.health_score >= 0.7 and self.error_count < 5


@dataclass
class ModelGraphState:
    """Represents the current state of the model graph for Tamiyo analysis."""

    topology: Any  # GraphTopology from analyzer
    health_signals: Dict[str, HealthSignal]
    health_trends: Dict[str, float]
    problematic_layers: Set[str]
    overall_health: float
    analysis_timestamp: float


class SystemStatePacket(BaseModel):
    """System-wide state information for the Strategic Controller."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        extra="forbid",  # Prevent extra fields for better performance
    )

    epoch: int = Field(ge=0)  # Non-negative epoch
    total_seeds: int = Field(ge=0)  # Non-negative seed count
    active_seeds: int = Field(ge=0)  # Non-negative active seed count
    training_loss: float = Field(ge=0.0)  # Non-negative loss
    validation_loss: float = Field(ge=0.0)  # Non-negative validation loss
    system_load: float = Field(ge=0.0, le=1.0)  # Load percentage 0-100%
    memory_usage: float = Field(ge=0.0, le=1.0)  # Memory usage percentage 0-100%

    @field_validator("epoch")
    @classmethod
    def _validate_epoch(cls, v):
        """Validate epoch is non-negative."""
        if v < 0:
            raise ValueError("Epoch must be non-negative")
        return v

    def system_health(self) -> str:
        """Get overall system health status."""
        if self.system_load > 0.9 or self.memory_usage > 0.9:
            return "Critical"
        elif self.system_load > 0.7 or self.memory_usage > 0.7:
            return "Warning"
        else:
            return "Normal"

    def is_overloaded(self) -> bool:
        """Check if system is overloaded."""
        return self.system_load > 0.8 or self.memory_usage > 0.8


class AdaptationDecision(BaseModel):
    """Represents a strategic adaptation decision from Tamiyo."""

    model_config = ConfigDict(
        # Performance optimizations
        arbitrary_types_allowed=True,
        use_enum_values=True,
        validate_assignment=True,
        str_to_lower=False,
        str_strip_whitespace=True,
        extra="forbid",  # Prevent extra fields for better performance
    )

    layer_name: str = Field(min_length=1, max_length=255)  # Bounded layer name
    adaptation_type: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^(add_seed|remove_seed|modify_architecture|optimize_parameters)$",
    )  # Controlled adaptation types
    confidence: float = Field(ge=0.0, le=1.0)  # Confidence level 0-1
    urgency: float = Field(ge=0.0, le=1.0)  # Urgency level 0-1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)

    def decision_priority(self) -> str:
        """Get decision priority based on confidence and urgency."""
        priority_score = (self.confidence + self.urgency) / 2
        if priority_score >= 0.8:
            return "High"
        elif priority_score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def should_execute_immediately(self) -> bool:
        """Check if decision should be executed immediately."""
        return self.urgency > 0.7 and self.confidence > 0.6
