"""
Blueprint metadata schemas for Tamiyo integration.

Provides comprehensive metadata for each blueprint template to enable
intelligent decision making by the GNN policy network.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class BlueprintCategory(str, Enum):
    """Categories of blueprint templates."""
    
    TRANSFORMER = "transformer"
    MIXTURE_OF_EXPERTS = "moe"
    EFFICIENCY = "efficiency"
    ROUTING = "routing"
    DIAGNOSTICS = "diagnostics"
    CUSTOM = "custom"


@dataclass
class BlueprintMetadata:
    """
    Comprehensive metadata for blueprint templates.
    
    This metadata enables Tamiyo's GNN to reason about trade-offs
    and make informed adaptation decisions.
    """
    
    # Core identification
    blueprint_id: str
    name: str
    version: str
    category: BlueprintCategory
    description: str
    
    # Cost analysis (for Tamiyo decision making)
    param_delta: int  # Change in parameter count
    flop_delta: int  # Change in FLOPs per forward pass
    memory_footprint_kb: int  # Additional memory required
    expected_latency_ms: float  # Expected execution latency
    
    # Benefit estimation (learned from experience)
    past_accuracy_gain_estimate: float  # Historical accuracy improvement
    stability_improvement_estimate: float  # Historical stability gain
    speed_improvement_estimate: float  # Historical speed improvement
    
    # Safety and constraints
    risk_score: float  # 0.0 (safe) to 1.0 (risky)
    is_safe_action: bool  # Can be applied without supervision
    requires_capability: List[str]  # Required hardware/software capabilities
    
    # Integration hints
    compatible_layers: List[str]  # Layer types this can be applied to
    incompatible_with: List[str]  # Blueprint IDs that conflict
    mergeable: bool  # Can be merged with existing adaptations
    
    # Temporal characteristics
    warmup_steps: int  # Steps before benefits manifest
    peak_benefit_window: int  # Steps of peak performance
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "version": self.version,
            "category": self.category.value,
            "description": self.description,
            "param_delta": self.param_delta,
            "flop_delta": self.flop_delta,
            "memory_footprint_kb": self.memory_footprint_kb,
            "expected_latency_ms": self.expected_latency_ms,
            "past_accuracy_gain_estimate": self.past_accuracy_gain_estimate,
            "stability_improvement_estimate": self.stability_improvement_estimate,
            "speed_improvement_estimate": self.speed_improvement_estimate,
            "risk_score": self.risk_score,
            "is_safe_action": self.is_safe_action,
            "requires_capability": self.requires_capability,
            "compatible_layers": self.compatible_layers,
            "incompatible_with": self.incompatible_with,
            "mergeable": self.mergeable,
            "warmup_steps": self.warmup_steps,
            "peak_benefit_window": self.peak_benefit_window,
        }


class BlueprintArchitecture(BaseModel):
    """
    Architecture definition for a blueprint.
    
    This is the actual neural network architecture that will be
    compiled by Tezzeret into a kernel.
    """
    
    module_type: str = Field(..., description="PyTorch module class name")
    config: Dict = Field(default_factory=dict, description="Module configuration")
    init_params: Dict = Field(default_factory=dict, description="Initialization parameters")
    forward_logic: Optional[str] = Field(None, description="Custom forward pass logic")
    
    class Config:
        extra = "forbid"


class BlueprintManifest(BaseModel):
    """
    Complete blueprint definition including metadata and architecture.
    """
    
    metadata: Dict = Field(..., description="Blueprint metadata")
    architecture: BlueprintArchitecture = Field(..., description="Neural architecture")
    validation: Dict = Field(default_factory=dict, description="Validation criteria")
    
    class Config:
        extra = "forbid"