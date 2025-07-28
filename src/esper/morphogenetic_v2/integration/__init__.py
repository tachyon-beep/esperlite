"""
Integration components for morphogenetic v2 system.

This module provides integrated components that combine:
- TritonChunkedKasminaLayer with message bus support
- TamiyoController for high-level orchestration
- Unified lifecycle and telemetry management
"""

from .tamiyo_controller import OptimizationStrategy
from .tamiyo_controller import SeedPopulationMetrics
from .tamiyo_controller import TamiyoConfig
from .tamiyo_controller import TamiyoController
from .triton_message_bus_layer import MessageBusIntegrationConfig
from .triton_message_bus_layer import TritonMessageBusLayer

__all__ = [
    # Layers
    "TritonMessageBusLayer",
    "MessageBusIntegrationConfig",

    # Controller
    "TamiyoController",
    "TamiyoConfig",
    "OptimizationStrategy",
    "SeedPopulationMetrics"
]
