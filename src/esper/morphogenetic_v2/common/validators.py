"""Input validation utilities for security.

This module provides common validation functions to prevent
injection attacks and ensure data integrity.
"""

import re
from typing import Any
from typing import Optional
from typing import Union

import torch


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_identifier(value: Any, name: str, max_length: int = 256) -> str:
    """Validate an identifier (layer_id, checkpoint_id, etc).
    
    Args:
        value: Value to validate
        name: Name of the parameter (for error messages)
        max_length: Maximum allowed length
        
    Returns:
        Validated string
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string, got {type(value).__name__}")

    if not value:
        raise ValidationError(f"{name} cannot be empty")

    if len(value) > max_length:
        raise ValidationError(f"{name} too long (max {max_length} chars)")

    # Only allow alphanumeric, underscore, dash
    if not re.match(r'^[a-zA-Z0-9_-]+$', value):
        raise ValidationError(
            f"{name} can only contain letters, numbers, underscore, and dash"
        )

    # Prevent path traversal
    if '..' in value or '/' in value or '\\' in value:
        raise ValidationError(f"{name} contains invalid path characters")

    return value


def validate_seed_id(value: Any) -> int:
    """Validate a seed ID.
    
    Args:
        value: Value to validate
        
    Returns:
        Validated integer
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        raise ValidationError("seed_id cannot be None")

    if not isinstance(value, int):
        raise ValidationError(f"seed_id must be an integer, got {type(value).__name__}")

    if value < 0:
        raise ValidationError(f"seed_id must be non-negative, got {value}")

    if value > 1_000_000:  # Reasonable upper limit
        raise ValidationError(f"seed_id too large (max 1,000,000), got {value}")

    return value


def validate_priority(value: Any) -> str:
    """Validate checkpoint priority.
    
    Args:
        value: Value to validate
        
    Returns:
        Validated priority string
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return 'normal'  # Default

    if not isinstance(value, str):
        raise ValidationError(f"priority must be a string, got {type(value).__name__}")

    valid_priorities = ['low', 'normal', 'high', 'critical']
    if value not in valid_priorities:
        raise ValidationError(
            f"priority must be one of {valid_priorities}, got '{value}'"
        )

    return value


def validate_json_serializable(value: Any, name: str) -> Any:
    """Validate that a value is JSON-serializable.
    
    Args:
        value: Value to validate
        name: Name of the parameter
        
    Returns:
        The value if valid
        
    Raises:
        ValidationError: If not JSON-serializable
    """
    import json

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError) as e:
        raise ValidationError(f"{name} must be JSON-serializable: {str(e)}")


def validate_tensor_dict(value: Any, name: str) -> Optional[dict]:
    """Validate a dictionary of tensors.
    
    Args:
        value: Value to validate
        name: Name of the parameter
        
    Returns:
        Validated dict or None
        
    Raises:
        ValidationError: If validation fails
    """
    if value is None:
        return None

    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be a dict, got {type(value).__name__}")

    for key, tensor in value.items():
        if not isinstance(key, str):
            raise ValidationError(f"{name} keys must be strings, got {type(key).__name__}")

        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(
                f"{name}['{key}'] must be a torch.Tensor, got {type(tensor).__name__}"
            )

    return value


def validate_file_path(value: Any, name: str, must_exist: bool = False) -> str:
    """Validate a file path for safety.
    
    Args:
        value: Path to validate
        name: Name of the parameter
        must_exist: Whether the path must exist
        
    Returns:
        Validated path string
        
    Raises:
        ValidationError: If validation fails
    """
    import os

    if value is None:
        raise ValidationError(f"{name} cannot be None")

    if not isinstance(value, (str, os.PathLike)):
        raise ValidationError(
            f"{name} must be a string or Path, got {type(value).__name__}"
        )

    path_str = str(value)

    # Check for path traversal attempts
    if '..' in path_str:
        raise ValidationError(f"{name} contains path traversal (..) which is not allowed")

    # Check for absolute paths (security risk)
    if os.path.isabs(path_str):
        raise ValidationError(f"{name} must be a relative path, not absolute")

    # Check length
    if len(path_str) > 4096:  # Linux PATH_MAX
        raise ValidationError(f"{name} path too long")

    if must_exist and not os.path.exists(path_str):
        raise ValidationError(f"{name} path does not exist: {path_str}")

    return path_str


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for safe file operations.
    
    Args:
        filename: Filename to sanitize
        
    Returns:
        Sanitized filename
    """
    # Remove directory separators
    filename = filename.replace('/', '_').replace('\\', '_')

    # Remove other dangerous characters
    filename = re.sub(r'[^\w\s.-]', '_', filename)

    # Remove leading dots (hidden files)
    filename = filename.lstrip('.')

    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        name = name[:255-len(ext)]
        filename = name + ext

    # Ensure not empty
    if not filename:
        filename = 'unnamed'

    return filename


def validate_config_value(value: Any, name: str, value_type: type,
                         min_value: Optional[Union[int, float]] = None,
                         max_value: Optional[Union[int, float]] = None) -> Any:
    """Validate a configuration value.
    
    Args:
        value: Value to validate
        name: Parameter name
        value_type: Expected type
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        
    Returns:
        Validated value
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, value_type):
        raise ValidationError(
            f"{name} must be {value_type.__name__}, got {type(value).__name__}"
        )

    if value_type in (int, float):
        if min_value is not None and value < min_value:
            raise ValidationError(f"{name} must be >= {min_value}, got {value}")

        if max_value is not None and value > max_value:
            raise ValidationError(f"{name} must be <= {max_value}, got {value}")

    return value
