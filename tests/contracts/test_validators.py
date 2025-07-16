"""
Tests for data validators.
"""

import pytest

from esper.contracts.validators import validate_seed_id


def test_validate_seed_id_valid():
    """Test valid seed ID validation."""
    valid_id = (0, 1)
    result = validate_seed_id(valid_id)
    assert result == valid_id


def test_validate_seed_id_invalid_type():
    """Test invalid seed ID type."""
    with pytest.raises(ValueError, match="Seed ID must be a tuple"):
        validate_seed_id("invalid")


def test_validate_seed_id_invalid_length():
    """Test invalid seed ID length."""
    with pytest.raises(ValueError, match="Seed ID must be a tuple"):
        validate_seed_id((1, 2, 3))


def test_validate_seed_id_invalid_element_type():
    """Test invalid element types in seed ID."""
    with pytest.raises(TypeError, match="Seed ID elements must be integers"):
        validate_seed_id(("a", "b"))


def test_validate_seed_id_negative_values():
    """Test negative values in seed ID."""
    with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
        validate_seed_id((-1, 0))

    with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
        validate_seed_id((0, -1))
