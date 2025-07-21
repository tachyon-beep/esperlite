"""
Tests for the configuration management system.
"""

import os
from unittest.mock import patch

import pytest

from esper.utils.config import ServiceConfig
from esper.utils.config import get_service_config
from esper.utils.config import init_service_config
from esper.utils.config import reset_service_config


class TestServiceConfig:
    """Test the ServiceConfig class."""

    def setup_method(self):
        """Reset configuration before each test to ensure clean state."""
        reset_service_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_service_config()

    def test_default_configuration(self):
        """Test that default configuration values are correct."""
        config = ServiceConfig()

        assert config.urza_url == "http://localhost:8000"
        assert config.tamiyo_url == "http://localhost:8001"
        assert config.redis_url == "redis://localhost:6379/0"
        assert config.s3_endpoint == "http://localhost:9000"
        assert config.s3_bucket == "esper-artifacts"
        assert config.http_timeout == 30
        assert config.retry_attempts == 3
        assert config.cache_size_mb == 512
        assert config.max_cache_entries == 128
        assert config.poll_interval_seconds == 10
        assert config.environment == "development"
        assert config.log_level == "INFO"

    @patch.dict(
        os.environ,
        {
            "URZA_URL": "https://urza.production.com",
            "TAMIYO_URL": "https://tamiyo.production.com",
            "REDIS_URL": "redis://redis-cluster:6379/0",
            "S3_ENDPOINT": "https://s3.amazonaws.com",
            "S3_BUCKET": "production-artifacts",
            "HTTP_TIMEOUT": "60",
            "RETRY_ATTEMPTS": "5",
            "CACHE_SIZE_MB": "1024",
            "MAX_CACHE_ENTRIES": "256",
            "POLL_INTERVAL_SECONDS": "15",
            "ESPER_ENVIRONMENT": "production",
            "LOG_LEVEL": "WARNING",
        },
    )
    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        config = ServiceConfig()

        assert config.urza_url == "https://urza.production.com"
        assert config.tamiyo_url == "https://tamiyo.production.com"
        assert config.redis_url == "redis://redis-cluster:6379/0"
        assert config.s3_endpoint == "https://s3.amazonaws.com"
        assert config.s3_bucket == "production-artifacts"
        assert config.http_timeout == 60
        assert config.retry_attempts == 5
        assert config.cache_size_mb == 1024
        assert config.max_cache_entries == 256
        assert config.poll_interval_seconds == 15
        assert config.environment == "production"
        assert config.log_level == "WARNING"

    def test_validation_valid_config(self):
        """Test that valid configuration passes validation."""
        config = ServiceConfig()
        # Should not raise any exceptions
        config.validate()

    def test_validation_invalid_url(self):
        """Test that invalid URLs raise validation errors."""
        with patch.dict(os.environ, {"URZA_URL": "not-a-valid-url"}):
            with pytest.raises(ValueError, match="Invalid URL for urza_url"):
                ServiceConfig()

    def test_validation_negative_values(self):
        """Test that negative values raise validation errors."""
        with patch.dict(os.environ, {"HTTP_TIMEOUT": "-1"}):
            with pytest.raises(ValueError, match="http_timeout must be positive"):
                ServiceConfig()

    def test_validation_invalid_log_level(self):
        """Test that invalid log levels are corrected with warning."""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}):
            config = ServiceConfig()
            assert config.log_level == "INFO"  # Should be corrected

    @patch.dict(os.environ, {"ESPER_ENVIRONMENT": "production"})
    def test_production_validation_warnings(self):
        """Test that production environment generates appropriate warnings."""
        # This should log warnings but not raise exceptions
        config = ServiceConfig()
        assert config.environment == "production"

    def test_get_urza_api_url(self):
        """Test URL construction methods."""
        config = ServiceConfig()

        # Base URL without endpoint
        assert config.get_urza_api_url() == "http://localhost:8000/api/v1"

        # URL with endpoint
        assert (
            config.get_urza_api_url("kernels") == "http://localhost:8000/api/v1/kernels"
        )
        assert (
            config.get_urza_api_url("/kernels")
            == "http://localhost:8000/api/v1/kernels"
        )

    def test_get_tamiyo_api_url(self):
        """Test Tamiyo URL construction."""
        config = ServiceConfig()

        # Base URL without endpoint
        assert config.get_tamiyo_api_url() == "http://localhost:8001/api/v1"

        # URL with endpoint
        assert (
            config.get_tamiyo_api_url("analyze")
            == "http://localhost:8001/api/v1/analyze"
        )

    def test_to_dict_masks_sensitive_data(self):
        """Test that sensitive configuration is masked in dictionary output."""
        with patch.dict(
            os.environ,
            {
                "S3_ACCESS_KEY": "secret_access_key",
                "S3_SECRET_KEY": "secret_secret_key",
            },
        ):
            config = ServiceConfig()
            config_dict = config.to_dict()

            assert config_dict["s3_access_key"] == "***MASKED***"
            assert config_dict["s3_secret_key"] == "***MASKED***"
            assert config_dict["urza_url"] == "http://localhost:8000"  # Not masked


class TestConfigurationSingleton:
    """Test the global configuration singleton functionality."""

    def setup_method(self):
        """Reset configuration before each test to ensure clean state."""
        reset_service_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_service_config()

    def test_get_service_config_creates_singleton(self):
        """Test that get_service_config creates and returns singleton."""
        config1 = get_service_config()
        config2 = get_service_config()

        assert config1 is config2  # Same instance

    def test_reset_service_config(self):
        """Test that reset_service_config clears singleton."""
        config1 = get_service_config()
        reset_service_config()
        config2 = get_service_config()

        assert config1 is not config2  # Different instances

    def test_init_service_config_with_overrides(self):
        """Test configuration initialization with overrides."""
        config = init_service_config(
            urza_url="https://custom-urza.com", cache_size_mb=2048
        )

        assert config.urza_url == "https://custom-urza.com"
        assert config.cache_size_mb == 2048
        assert config.tamiyo_url == "http://localhost:8001"  # Default value preserved

    def test_init_service_config_unknown_key_warning(self):
        """Test that unknown configuration keys generate warnings."""
        # This should log a warning but not raise an exception
        config = init_service_config(unknown_key="value")
        assert not hasattr(config, "unknown_key")


class TestConfigurationIntegration:
    """Test configuration integration with other components."""

    def setup_method(self):
        """Reset configuration before each test to ensure clean state."""
        reset_service_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_service_config()

    @patch.dict(
        os.environ,
        {
            "URZA_URL": "https://test-urza.com",
            "HTTP_TIMEOUT": "45",
            "CACHE_SIZE_MB": "256",
        },
    )
    def test_kernel_cache_uses_configuration(self):
        """Test that KernelCache properly uses configuration."""
        from esper.execution.kernel_cache import KernelCache

        config = get_service_config()
        cache = KernelCache(config=config)

        assert cache.config.urza_url == "https://test-urza.com"
        assert cache.config.http_timeout == 45
        assert cache.max_cache_size_mb == 256

    @patch.dict(
        os.environ,
        {
            "URZA_URL": "https://test-urza.com",
            "HTTP_TIMEOUT": "20",
            "S3_BUCKET": "test-bucket",
        },
    )
    def test_tezzeret_worker_uses_configuration(self):
        """Test that TezzeretWorker properly uses configuration."""
        from esper.services.tezzeret.worker import TezzeretWorker

        config = get_service_config()
        worker = TezzeretWorker("test-worker", config=config)

        assert worker.config.urza_url == "https://test-urza.com"
        assert worker.config.http_timeout == 20
        assert worker.bucket_name == "test-bucket"
        assert worker.urza_base_url == "https://test-urza.com"


class TestEnvironmentFiles:
    """Test that environment file examples are valid."""

    def setup_method(self):
        """Reset configuration before each test to ensure clean state."""
        reset_service_config()

    def teardown_method(self):
        """Reset configuration after each test."""
        reset_service_config()

    def test_development_environment_values(self):
        """Test development environment configuration values."""
        env_vars = {
            "ESPER_ENVIRONMENT": "development",
            "LOG_LEVEL": "DEBUG",
            "URZA_URL": "http://localhost:8000",
            "CACHE_SIZE_MB": "128",
            "POLL_INTERVAL_SECONDS": "5",
        }

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig()
            assert config.environment == "development"
            assert config.log_level == "DEBUG"
            assert config.cache_size_mb == 128
            assert config.poll_interval_seconds == 5

    def test_production_environment_values(self):
        """Test production environment configuration values."""
        env_vars = {
            "ESPER_ENVIRONMENT": "production",
            "LOG_LEVEL": "INFO",
            "URZA_URL": "https://urza.production.esper.ai",
            "CACHE_SIZE_MB": "1024",
            "HTTP_TIMEOUT": "60",
            "RETRY_ATTEMPTS": "5",
        }

        with patch.dict(os.environ, env_vars):
            config = ServiceConfig()
            assert config.environment == "production"
            assert config.log_level == "INFO"
            assert config.cache_size_mb == 1024
            assert config.http_timeout == 60
            assert config.retry_attempts == 5
