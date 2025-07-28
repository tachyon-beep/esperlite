"""
Model Surgery Infrastructure for Dynamic Architecture Modification.

This module provides safe runtime modifications to PyTorch models,
enabling morphogenetic neural networks to evolve their architecture
during training.
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SurgeryResult:
    """Result of a model surgery operation."""
    success: bool
    operation: str
    details: Dict[str, Any]
    error: Optional[str] = None
    rollback_state: Optional[Dict] = None
    duration_ms: float = 0.0


class LayerWrapper(nn.Module):
    """Wrapper for layers with different connection types."""

    def __init__(self, original_layer: nn.Module, new_layer: nn.Module,
                 connection_type: str = "sequential"):
        super().__init__()
        self.original_layer = original_layer
        self.new_layer = new_layer
        self.connection_type = connection_type

    def forward(self, x):
        if self.connection_type == "sequential":
            x = self.original_layer(x)
            x = self.new_layer(x)
            return x
        elif self.connection_type == "residual":
            orig_out = self.original_layer(x)
            new_out = self.new_layer(orig_out)
            # Handle dimension mismatch for residual connections
            if orig_out.shape == new_out.shape:
                return orig_out + new_out
            else:
                return new_out
        else:
            raise ValueError(f"Unknown connection type: {self.connection_type}")


class DimensionAdapter(nn.Module):
    """Adapts dimensions between layers when bridging removed layers."""

    def __init__(self, in_shape: torch.Size, out_shape: torch.Size):
        super().__init__()
        self.in_shape = in_shape
        self.out_shape = out_shape

        # Simple linear adapter for now
        # TODO: Support more sophisticated adapters
        if len(in_shape) == 2 and len(out_shape) == 2:
            # Linear case
            self.adapter = nn.Linear(in_shape[-1], out_shape[-1])
        elif len(in_shape) == 4 and len(out_shape) == 4:
            # Conv2d case
            self.adapter = nn.Conv2d(
                in_shape[1], out_shape[1],
                kernel_size=1, stride=1, padding=0
            )
        else:
            # Fallback to flatten + linear
            in_features = torch.prod(torch.tensor(in_shape[1:])).item()
            out_features = torch.prod(torch.tensor(out_shape[1:])).item()
            self.adapter = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, out_features),
                nn.Unflatten(1, out_shape[1:])
            )

    def forward(self, x):
        return self.adapter(x)


class ModelSurgeon:
    """
    Performs safe runtime modifications to PyTorch models.
    
    Key capabilities:
    - Add/remove layers
    - Modify layer parameters
    - Add skip connections
    - Validate changes
    - Rollback on failure
    """

    def __init__(self):
        self.operation_history: List[SurgeryResult] = []
        # Validators will be initialized separately
        self.validators = None

    def set_validators(self, validators):
        """Set the validator chain."""
        self.validators = validators
