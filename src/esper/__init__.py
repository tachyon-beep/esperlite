"""
Esper - Morphogenetic Training Platform
A neural network training system that enables autonomous architectural evolution.
"""

from .configs import EsperConfig
from .core.model_wrapper import MorphableModel
from .core.model_wrapper import unwrap
from .core.model_wrapper import wrap
from .execution.kasmina_layer import KasminaLayer
from .execution.state_layout import SeedLifecycleState

API_VERSION = "v1"

__version__ = "0.2.0"
__author__ = "John Morrissey"

__all__ = [
    "wrap",
    "MorphableModel",
    "unwrap",
    "KasminaLayer",
    "SeedLifecycleState",
    "EsperConfig",
]

# Set up logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
