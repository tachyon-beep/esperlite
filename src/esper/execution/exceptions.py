"""
Exception classes for the execution system.

This module defines all exception classes used across the execution system
to avoid circular import issues.
"""


class KernelExecutionError(Exception):
    """Exception raised when kernel execution fails."""
    
    def __init__(self, message: str, layer_name: str = None, error_code: str = None):
        super().__init__(message)
        self.layer_name = layer_name
        self.error_code = error_code


class KernelDeserializationError(Exception):
    """Exception raised when kernel deserialization fails."""
    
    def __init__(self, message: str, kernel_path: str = None):
        super().__init__(message)
        self.kernel_path = kernel_path


class KernelValidationError(Exception):
    """Exception raised when kernel validation fails."""
    
    def __init__(self, message: str, validation_type: str = None):
        super().__init__(message)
        self.validation_type = validation_type


class MemoryOverflowError(Exception):
    """Exception raised when memory limits are exceeded."""
    
    def __init__(self, message: str, allocated_mb: float = None, limit_mb: float = None):
        super().__init__(message)
        self.allocated_mb = allocated_mb
        self.limit_mb = limit_mb


class DeviceError(Exception):
    """Exception raised for device-related errors."""
    
    def __init__(self, message: str, device: str = None):
        super().__init__(message)
        self.device = device