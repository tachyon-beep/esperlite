"""
Esper - Morphogenetic Training Platform
A neural network training system that enables autonomous architectural evolution.
"""

from .core.model_wrapper import wrap, MorphableModel, unwrap
from .execution.kasmina_layer import KasminaLayer
from .execution.state_layout import SeedLifecycleState
from .configs import EsperConfig

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
