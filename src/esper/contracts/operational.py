"""
Operational data models for runtime system monitoring and control.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
import time


class HealthSignal(BaseModel):
    """High-frequency health signal from a KasminaSeed."""

    layer_id: int
    seed_id: int
    chunk_id: int
    epoch: int
    activation_variance: float
    dead_neuron_ratio: float = Field(..., ge=0.0, le=1.0)
    avg_correlation: float = Field(..., ge=-1.0, le=1.0)
    is_ready_for_transition: bool = False  # Critical for state machine sync
    
    # Additional fields for Tamiyo analysis
    health_score: float = Field(default=1.0, ge=0.0, le=1.0)
    execution_latency: float = Field(default=0.0, ge=0.0)
    error_count: int = Field(default=0, ge=0)
    active_seeds: int = Field(default=1, ge=0)
    total_seeds: int = Field(default=1, ge=1)
    timestamp: float = Field(default_factory=time.time)

    @field_validator("seed_id", "layer_id")
    @classmethod
    def _validate_ids(cls, v):
        if v < 0:
            raise ValueError("IDs must be non-negative")
        return v


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

    epoch: int
    total_seeds: int
    active_seeds: int
    training_loss: float
    validation_loss: float
    system_load: float = Field(..., ge=0.0, le=1.0)
    memory_usage: float = Field(..., ge=0.0, le=1.0)

    @field_validator("epoch")
    @classmethod
    def _validate_epoch(cls, v):
        if v < 0:
            raise ValueError("Epoch must be non-negative")
        return v


class AdaptationDecision(BaseModel):
    """Represents a strategic adaptation decision from Tamiyo."""
    
    layer_name: str
    adaptation_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    urgency: float = Field(..., ge=0.0, le=1.0) 
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)
