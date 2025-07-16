"""
Core asset models for the Esper system.
These models define the primary entities: Seeds, Blueprints, and related structures.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .enums import BlueprintState, SeedState


class Seed(BaseModel):
    """A morphogenetic seed embedded in a neural network layer."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    seed_id: str = Field(default_factory=lambda: str(uuid4()))
    layer_id: int
    position: int  # Position within the layer
    state: SeedState = SeedState.DORMANT
    blueprint_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Blueprint(BaseModel):
    """An architectural blueprint for neural network components."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    blueprint_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    state: BlueprintState = BlueprintState.PROPOSED
    architecture: Dict[str, Any]  # Architecture definition
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: str  # Component that created this blueprint


class TrainingSession(BaseModel):
    """A training session for a morphogenetic system."""

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat()})

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    training_model_config: Dict[str, Any]  # Renamed from model_config
    training_config: Dict[str, Any]
    seeds: List[Seed] = Field(default_factory=list)
    blueprints: List[Blueprint] = Field(default_factory=list)
    status: str = "initialized"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
