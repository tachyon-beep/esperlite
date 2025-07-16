"""
Custom validators for Pydantic models.

This module provides high-performance validation functions for complex data types
used across the Esper platform contracts.
"""

from typing import Tuple


def validate_seed_id(v: Tuple[int, int]) -> Tuple[int, int]:
    """
    Validates a seed_id tuple ensuring proper format and values.

    A seed_id is a unique identifier tuple consisting of (layer_id, seed_idx)
    used to reference specific neural network seeds within layers.

    Args:
        v: Input value to validate as a seed ID tuple

    Returns:
        Tuple[int, int]: The validated seed ID tuple (layer_id, seed_idx)

    Raises:
        ValueError: If input is not a 2-element tuple
        TypeError: If tuple elements are not integers
        ValueError: If tuple elements are negative

    Example:
        >>> validate_seed_id((0, 1))
        (0, 1)
        >>> validate_seed_id((5, 10))
        (5, 10)
    """
    # Fast path: check if it's a tuple with exactly 2 elements
    if not isinstance(v, tuple) or len(v) != 2:
        raise ValueError("Seed ID must be a tuple of (layer_id, seed_idx)")

    layer_id, seed_idx = v

    # Type validation with early exit
    if not isinstance(layer_id, int) or not isinstance(seed_idx, int):
        raise TypeError("Seed ID elements must be integers")

    # Value validation
    if layer_id < 0 or seed_idx < 0:
        raise ValueError("Seed ID elements must be non-negative")

    return v
