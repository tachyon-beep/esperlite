"""
Comprehensive tests for data validators.
"""

import time

import pytest

from esper.contracts.validators import validate_seed_id


class TestValidateSeedId:
    """Test suite for validate_seed_id validator function."""

    def test_validate_seed_id_valid_cases(self):
        """Test validate_seed_id with valid inputs."""
        # Test basic valid case
        valid_id = (0, 1)
        result = validate_seed_id(valid_id)
        assert result == valid_id

        # Test valid zero values
        result = validate_seed_id((0, 0))
        assert result == (0, 0)

        # Test valid positive values
        result = validate_seed_id((5, 10))
        assert result == (5, 10)

        # Test large values
        result = validate_seed_id((1000, 9999))
        assert result == (1000, 9999)

        # Test asymmetric values
        result = validate_seed_id((1, 100))
        assert result == (1, 100)

        result = validate_seed_id((100, 1))
        assert result == (100, 1)

    def test_validate_seed_id_type_errors(self):
        """Test validate_seed_id with invalid types."""
        # Test invalid seed ID type - string
        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id("invalid")

        # Test non-tuple inputs
        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id([1, 2])  # list instead of tuple

        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id(12)  # int instead of tuple

        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id(None)  # None instead of tuple

    def test_validate_seed_id_length_errors(self):
        """Test validate_seed_id with invalid lengths."""
        # Test invalid seed ID length - too many elements
        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id((1, 2, 3))

        # Test tuple with wrong length
        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id((1,))  # single element

        with pytest.raises(ValueError, match="Seed ID must be a tuple"):
            validate_seed_id(())  # empty tuple

    def test_validate_seed_id_element_type_errors(self):
        """Test validate_seed_id with invalid element types."""
        # Test invalid element types in seed ID - strings
        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id(("a", "b"))

        # Test mixed invalid types
        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id((1.5, 2))  # float in first position

        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id((1, 2.0))  # float in second position

        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id(("1", 2))  # string in first position

        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id((1, "2"))  # string in second position

        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id((None, 2))  # None in first position

        with pytest.raises(TypeError, match="Seed ID elements must be integers"):
            validate_seed_id((1, None))  # None in second position

    def test_validate_seed_id_negative_values(self):
        """Test validate_seed_id with negative values."""
        # Test negative values in seed ID - first element
        with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
            validate_seed_id((-1, 0))

        # Test negative values in seed ID - second element
        with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
            validate_seed_id((0, -1))

        # Test both negative
        with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
            validate_seed_id((-1, -1))

        # Test large negative values
        with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
            validate_seed_id((-10, 0))

        with pytest.raises(ValueError, match="Seed ID elements must be non-negative"):
            validate_seed_id((0, -10))

    def test_validate_seed_id_edge_cases(self):
        """Test validate_seed_id with edge cases."""
        # Test boundary values
        result = validate_seed_id((0, 0))
        assert result == (0, 0)

        # Test very large values (within int range)
        large_val = 2**30
        result = validate_seed_id((large_val, large_val))
        assert result == (large_val, large_val)

        # Test mixed boundary cases
        result = validate_seed_id((0, 1000))
        assert result == (0, 1000)

        result = validate_seed_id((1000, 0))
        assert result == (1000, 0)


class TestValidatorPerformance:
    """Performance tests for validator functions."""

    def test_validate_seed_id_performance(self):
        """Test validate_seed_id performance with batch validation."""
        # Generate test data
        test_cases = [(i, i + 1) for i in range(1000)]

        # Test validation performance
        start_time = time.perf_counter()
        for case in test_cases:
            validate_seed_id(case)
        elapsed = time.perf_counter() - start_time

        # Should be very fast for simple validation
        assert elapsed < 0.01, f"Validation took {elapsed:.3f}s, expected <0.01s"

        # Test throughput
        ops_per_second = len(test_cases) / elapsed
        assert (
            ops_per_second > 50000
        ), f"Validation throughput {ops_per_second:.0f} ops/s, expected >50000 ops/s"

    def test_validate_seed_id_error_performance(self):
        """Test validate_seed_id error handling performance."""
        # Test error cases don't significantly impact performance
        error_cases = [
            ("invalid", 1),
            (1, "invalid"),
            (-1, 1),
            (1, -1),
            [1, 2],  # list instead of tuple
        ]

        start_time = time.perf_counter()
        error_count = 0
        for case in error_cases * 100:  # 500 error cases
            try:
                validate_seed_id(case)
            except (ValueError, TypeError):
                error_count += 1
        elapsed = time.perf_counter() - start_time

        # All should fail validation
        assert error_count == 500

        # Error handling should still be fast
        assert elapsed < 0.1, f"Error handling took {elapsed:.3f}s, expected <0.1s"
