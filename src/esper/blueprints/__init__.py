"""
Esper Blueprint Library - Production-ready morphogenetic adaptation templates.

This module provides the blueprint template library required for Tamiyo's
intelligent adaptation decisions. Each blueprint represents a specific
architectural modification that can be dynamically compiled and integrated
into a running neural network.
"""

from esper.blueprints.registry import BlueprintRegistry
from esper.blueprints.metadata import (
    BlueprintMetadata,
    BlueprintCategory,
    BlueprintManifest,
)

__all__ = [
    "BlueprintRegistry",
    "BlueprintMetadata", 
    "BlueprintCategory",
    "BlueprintManifest",
]

# Module version
__version__ = "1.0.0"