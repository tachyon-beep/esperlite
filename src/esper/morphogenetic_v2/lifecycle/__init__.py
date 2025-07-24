"""Morphogenetic lifecycle management components."""

from .extended_lifecycle import (
    ExtendedLifecycle,
    TransitionContext,
    StateTransition,
    LifecycleManager
)
from .state_manager import ExtendedStateTensor
from .checkpoint_manager_v2 import CheckpointManager, CheckpointRecovery

__all__ = [
    'ExtendedLifecycle',
    'TransitionContext', 
    'StateTransition',
    'LifecycleManager',
    'ExtendedStateTensor',
    'CheckpointManager',
    'CheckpointRecovery'
]
