"""
Centralized service configuration management.

This module provides environment-based configuration for all Esper services,
eliminating hard-coded URLs and enabling flexible deployment configurations.
"""

import logging
import os
from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """
    Base configuration for all Esper services.

    All configuration values can be overridden via environment variables,
    enabling flexible deployment across different environments.
    """

    # Core service URLs
    urza_url: str = field(
        default_factory=lambda: os.getenv("URZA_URL", "http://localhost:8000")
    )
    tamiyo_url: str = field(
        default_factory=lambda: os.getenv("TAMIYO_URL", "http://localhost:8001")
    )
    tezzeret_url: str = field(
        default_factory=lambda: os.getenv("TEZZERET_URL", "http://localhost:8002")
    )

    # Redis/Oona configuration
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # S3/Storage configuration
    s3_endpoint: str = field(
        default_factory=lambda: os.getenv("S3_ENDPOINT", "http://localhost:9000")
    )
    s3_bucket: str = field(
        default_factory=lambda: os.getenv("S3_BUCKET", "esper-artifacts")
    )
    s3_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_ACCESS_KEY")
    )
    s3_secret_key: Optional[str] = field(
        default_factory=lambda: os.getenv("S3_SECRET_KEY")
    )

    # Timeout and retry configuration
    http_timeout: int = field(
        default_factory=lambda: int(os.getenv("HTTP_TIMEOUT", "30"))
    )
    retry_attempts: int = field(
        default_factory=lambda: int(os.getenv("RETRY_ATTEMPTS", "3"))
    )

    # Cache and performance settings
    cache_size_mb: int = field(
        default_factory=lambda: int(os.getenv("CACHE_SIZE_MB", "512"))
    )
    max_cache_entries: int = field(
        default_factory=lambda: int(os.getenv("MAX_CACHE_ENTRIES", "128"))
    )

    # Polling and timing configuration
    poll_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
    )

    # Environment and deployment settings
    environment: str = field(
        default_factory=lambda: os.getenv("ESPER_ENVIRONMENT", "development")
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
        logger.info("ServiceConfig initialized for environment: %s", self.environment)

    def validate(self) -> None:
        """
        Validate configuration values and log warnings for potential issues.

        Raises:
            ValueError: If critical configuration is invalid
        """
        # Validate URLs
        for url_name, url_value in [
            ("urza_url", self.urza_url),
            ("tamiyo_url", self.tamiyo_url),
            ("tezzeret_url", self.tezzeret_url),
            ("redis_url", self.redis_url),
            ("s3_endpoint", self.s3_endpoint),
        ]:
            if not self._is_valid_url(url_value):
                raise ValueError(f"Invalid URL for {url_name}: {url_value}")

        # Validate positive integers
        for config_name, config_value in [
            ("http_timeout", self.http_timeout),
            ("retry_attempts", self.retry_attempts),
            ("cache_size_mb", self.cache_size_mb),
            ("max_cache_entries", self.max_cache_entries),
            ("poll_interval_seconds", self.poll_interval_seconds),
        ]:
            if config_value <= 0:
                raise ValueError(f"{config_name} must be positive, got: {config_value}")

        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            logger.warning("Invalid log level '%s', using INFO", self.log_level)
            self.log_level = "INFO"

        # Warn about production readiness
        if self.environment == "production":
            self._validate_production_config()

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def _validate_production_config(self) -> None:
        """Validate production-specific configuration requirements."""
        warnings = []

        # Check for localhost URLs in production
        localhost_urls = []
        for url_name, url_value in [
            ("urza_url", self.urza_url),
            ("tamiyo_url", self.tamiyo_url),
            ("tezzeret_url", self.tezzeret_url),
            ("s3_endpoint", self.s3_endpoint),
        ]:
            if "localhost" in url_value or "127.0.0.1" in url_value:
                localhost_urls.append(url_name)

        if localhost_urls:
            warnings.append(
                f"Localhost URLs in production: {', '.join(localhost_urls)}"
            )

        # Check for missing S3 credentials
        if not self.s3_access_key or not self.s3_secret_key:
            warnings.append(
                "S3 credentials not configured - may cause authentication failures"
            )

        # Check for low timeout values
        if self.http_timeout < 30:
            warnings.append(
                f"HTTP timeout ({self.http_timeout}s) may be too low for production"
            )

        # Check cache size
        if self.cache_size_mb < 256:
            warnings.append(
                f"Cache size ({self.cache_size_mb}MB) may be too small for production"
            )

        # Log all warnings
        for warning in warnings:
            logger.warning("Production config warning: %s", warning)

    def get_urza_api_url(self, endpoint: str = "") -> str:
        """Get full Urza API URL with optional endpoint."""
        base_url = self.urza_url.rstrip("/")
        if endpoint:
            endpoint = endpoint.lstrip("/")
            return f"{base_url}/api/v1/{endpoint}"
        return f"{base_url}/api/v1"

    def get_tamiyo_api_url(self, endpoint: str = "") -> str:
        """Get full Tamiyo API URL with optional endpoint."""
        base_url = self.tamiyo_url.rstrip("/")
        if endpoint:
            endpoint = endpoint.lstrip("/")
            return f"{base_url}/api/v1/{endpoint}"
        return f"{base_url}/api/v1"

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging/debugging."""
        config_dict = {}
        for key, value in self.__dict__.items():
            # Mask sensitive values
            if (
                "secret" in key.lower()
                or "password" in key.lower()
                or "key" in key.lower()
            ) and value:
                config_dict[key] = "***MASKED***"
            else:
                config_dict[key] = value
        return config_dict


# Global configuration instance
_service_config = None


def get_service_config() -> ServiceConfig:
    """
    Get the global service configuration instance.

    Returns:
        ServiceConfig: Singleton configuration instance
    """
    global _service_config
    if _service_config is None:
        _service_config = ServiceConfig()
    return _service_config


def reset_service_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _service_config
    _service_config = None


def init_service_config(**overrides) -> ServiceConfig:
    """
    Initialize service configuration with optional overrides.

    Args:
        **overrides: Configuration values to override

    Returns:
        ServiceConfig: Initialized configuration instance
    """
    global _service_config

    # Create new config
    config = ServiceConfig()

    # Apply any overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            logger.warning("Unknown configuration key: %s", key)

    # Validate after overrides
    config.validate()

    _service_config = config
    return config
