"""
Morphogenetic v2 - Next Generation Neural Architecture Adaptation

This module contains the implementation of the enhanced morphogenetic system
as specified in the Kasmina v0.1a and Tamiyo v0.1a design documents.

Migration Status: Phase 0 - Foundation
"""

__version__ = "2.0.0-alpha"
__migration_phase__ = 0

# Feature flags for gradual rollout
FEATURES = {
    "chunked_architecture": False,
    "triton_kernels": False,
    "extended_lifecycle": False,
    "message_bus": False,
    "neural_controller": False,
    "grafting_strategies": False,
}