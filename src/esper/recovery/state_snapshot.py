"""
State snapshot models for checkpoint and recovery.
"""

import json
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pydantic import BaseModel


class ComponentType(str, Enum):
    """Types of components that can be checkpointed."""

    TOLARIA = "tolaria"  # Training orchestrator
    TAMIYO = "tamiyo"   # Strategic controller
    KASMINA = "kasmina" # Execution layer
    URZA = "urza"       # Asset hub
    KARN = "karn"       # Blueprint generator
    SIMIC = "simic"     # Policy engine


@dataclass
class ComponentState:
    """State snapshot for a single component."""

    component_type: ComponentType
    component_id: str
    state_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "component_type": self.component_type.value,
            "component_id": self.component_id,
            "state_data": self.state_data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        })

    @classmethod
    def from_json(cls, data: str) -> "ComponentState":
        """Deserialize from JSON."""
        obj = json.loads(data)
        return cls(
            component_type=ComponentType(obj["component_type"]),
            component_id=obj["component_id"],
            state_data=obj["state_data"],
            metadata=obj["metadata"],
            timestamp=datetime.fromisoformat(obj["timestamp"])
        )


class CheckpointMetadata(BaseModel):
    """Metadata for a checkpoint."""

    checkpoint_id: str
    created_at: datetime
    created_by: str = "system"
    description: Optional[str] = None
    component_count: int = 0
    total_size_bytes: int = 0
    is_full_checkpoint: bool = True
    parent_checkpoint_id: Optional[str] = None  # For incremental checkpoints
    tags: List[str] = []


@dataclass
class StateSnapshot:
    """Complete system state snapshot."""

    checkpoint_id: str
    components: Dict[str, ComponentState]  # component_id -> state
    metadata: CheckpointMetadata
    created_at: datetime = field(default_factory=datetime.utcnow)

    def add_component(self, component: ComponentState):
        """Add a component state to the snapshot."""
        self.components[component.component_id] = component
        self.metadata.component_count = len(self.components)

    def get_component(
        self,
        component_type: ComponentType,
        component_id: Optional[str] = None
    ) -> Optional[ComponentState]:
        """Get component state by type and optional ID."""
        if component_id:
            return self.components.get(component_id)

        # Find first component of given type
        for comp in self.components.values():
            if comp.component_type == component_type:
                return comp

        return None

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate snapshot integrity."""
        errors = []

        # Check required components
        required_types = {ComponentType.TOLARIA, ComponentType.TAMIYO, ComponentType.URZA}
        present_types = {comp.component_type for comp in self.components.values()}

        missing = required_types - present_types
        if missing:
            errors.append(f"Missing required components: {missing}")

        # Validate component states
        for comp_id, comp in self.components.items():
            if not comp.state_data:
                errors.append(f"Component {comp_id} has empty state")

            # Component-specific validation
            if comp.component_type == ComponentType.TOLARIA:
                if "model_state" not in comp.state_data:
                    errors.append("Tolaria missing model_state")
            elif comp.component_type == ComponentType.TAMIYO:
                if "policy_state" not in comp.state_data:
                    errors.append("Tamiyo missing policy_state")

        return len(errors) == 0, errors

    def to_json(self) -> str:
        """Serialize entire snapshot to JSON."""
        return json.dumps({
            "checkpoint_id": self.checkpoint_id,
            "components": {
                comp_id: json.loads(comp.to_json())
                for comp_id, comp in self.components.items()
            },
            "metadata": self.metadata.model_dump_json(),
            "created_at": self.created_at.isoformat()
        })

    @classmethod
    def from_json(cls, data: str) -> "StateSnapshot":
        """Deserialize snapshot from JSON."""
        obj = json.loads(data)

        components = {}
        for comp_id, comp_data in obj["components"].items():
            components[comp_id] = ComponentState(
                component_type=ComponentType(comp_data["component_type"]),
                component_id=comp_data["component_id"],
                state_data=comp_data["state_data"],
                metadata=comp_data["metadata"],
                timestamp=datetime.fromisoformat(comp_data["timestamp"])
            )

        return cls(
            checkpoint_id=obj["checkpoint_id"],
            components=components,
            metadata=CheckpointMetadata.model_validate_json(obj["metadata"]),
            created_at=datetime.fromisoformat(obj["created_at"])
        )


# Component-specific state models
class TolariaState(BaseModel):
    """Training orchestrator state."""

    epoch: int
    global_step: int
    model_state_dict: Dict[str, Any]  # Serialized model weights
    optimizer_state_dict: Dict[str, Any]
    learning_rate: float
    training_metrics: Dict[str, float]
    active_seeds: Dict[str, List[int]]  # layer -> active seed indices


class TamiyoState(BaseModel):
    """Strategic controller state."""

    policy_state: Dict[str, Any]
    performance_history: List[Dict[str, float]]
    adaptation_count: int
    rollback_count: int
    seed_selection_stats: Dict[str, Any]


class KasminaState(BaseModel):
    """Execution layer state."""

    seed_states: Dict[str, Dict[str, Any]]  # layer -> seed states
    kernel_mappings: Dict[str, str]  # seed_id -> kernel_id
    blend_factors: Dict[str, List[float]]  # layer -> blend factors
    execution_stats: Dict[str, Any]


class UrzaState(BaseModel):
    """Asset hub state."""

    blueprint_count: int
    kernel_count: int
    cache_stats: Dict[str, Any]
    recent_compilations: List[str]
    storage_usage_gb: float
