"""
Custom validators for Pydantic models.
"""

from typing import Tuple


def validate_seed_id(v: Tuple[int, int]) -> Tuple[int, int]:
    """Ensures a seed_id tuple (layer_id, seed_idx) is valid."""
    if not isinstance(v, tuple) or len(v) != 2:
        raise ValueError("Seed ID must be a tuple of (layer_id, seed_idx)")

    layer_id, seed_idx = v
    if not isinstance(layer_id, int) or not isinstance(seed_idx, int):
        raise TypeError("Seed ID elements must be integers")

    if layer_id < 0 or seed_idx < 0:
        raise ValueError("Seed ID elements must be non-negative")
    return v
