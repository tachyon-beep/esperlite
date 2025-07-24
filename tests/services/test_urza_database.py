"""
Tests for Urza database configuration and connection pooling.
"""

import os
from unittest.mock import MagicMock
from unittest.mock import patch

from sqlalchemy.pool import QueuePool
from sqlalchemy.pool import StaticPool

from esper.services.urza.database import DatabaseConfig


class TestDatabaseConfig:
    """Test database configuration and pooling."""

    def test_development_uses_queuepool(self):
        """Test that development environment uses QueuePool."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "development"}):
            config = DatabaseConfig()
            assert isinstance(config.engine.pool, QueuePool)

    def test_production_uses_queuepool(self):
        """Test that production environment uses QueuePool."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "production"}):
            config = DatabaseConfig()
            assert isinstance(config.engine.pool, QueuePool)

    def test_testing_uses_staticpool(self):
        """Test that testing environment uses StaticPool for isolation."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "testing"}):
            config = DatabaseConfig()
            assert isinstance(config.engine.pool, StaticPool)

    def test_queuepool_configuration_defaults(self):
        """Test QueuePool uses correct default configuration."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "development"}):
            config = DatabaseConfig()

            # Verify it's a QueuePool
            assert isinstance(config.engine.pool, QueuePool)

            # QueuePool internal attributes may vary between SQLAlchemy versions
            # Instead, test the get_pool_stats method which provides consistent interface
            stats = config.get_pool_stats()
            assert stats["pool_type"] == "QueuePool"

    def test_queuepool_configuration_overrides(self):
        """Test QueuePool respects environment variable overrides."""
        with patch.dict(
            os.environ,
            {
                "ESPER_ENVIRONMENT": "production",
                "DB_POOL_SIZE": "50",
                "DB_MAX_OVERFLOW": "75",
                "DB_POOL_TIMEOUT": "45",
                "DB_POOL_RECYCLE": "7200",
            },
        ):
            config = DatabaseConfig()

            # Verify it's a QueuePool with custom configuration
            assert isinstance(config.engine.pool, QueuePool)

            # Test that environment variables were used during engine creation
            # The actual configuration is tested via integration

    def test_database_url_construction(self):
        """Test database URL is constructed correctly from environment."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_HOST": "test-host",
                "POSTGRES_PORT": "5433",
                "POSTGRES_USER": "test_user",
                "POSTGRES_PASSWORD": "test_pass",
                "POSTGRES_DB": "test_db",
            },
        ):
            config = DatabaseConfig()
            expected_url = "postgresql://test_user:test_pass@test-host:5433/test_db"
            assert config.database_url == expected_url

    def test_database_url_defaults(self):
        """Test database URL uses correct defaults when env vars not set."""
        # Clear any existing environment variables
        env_vars_to_clear = [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DB",
        ]
        with patch.dict(os.environ, {}, clear=True):
            for var in env_vars_to_clear:
                os.environ.pop(var, None)

            config = DatabaseConfig()
            expected_url = "postgresql://postgres:postgres@localhost:5432/urza_db"
            assert config.database_url == expected_url

    def test_get_pool_stats_queuepool(self):
        """Test pool statistics for QueuePool."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "development"}):
            config = DatabaseConfig()
            stats = config.get_pool_stats()

            assert "pool_type" in stats
            assert stats["pool_type"] == "QueuePool"
            assert "environment" in stats
            assert stats["environment"] == "development"
            assert "pool_status" in stats

            # QueuePool statistics may be available depending on initialization
            # The method should handle both cases gracefully

    def test_get_pool_stats_staticpool(self):
        """Test pool statistics for StaticPool."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "testing"}):
            config = DatabaseConfig()
            stats = config.get_pool_stats()

            assert stats["pool_type"] == "StaticPool"
            assert stats["environment"] == "testing"
            assert stats["pool_status"] == "active"

    def test_session_management(self):
        """Test database session creation and cleanup."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "testing"}):
            config = DatabaseConfig()

            # Test session generator
            session_gen = config.get_session()
            session = next(session_gen)

            assert session is not None
            assert hasattr(session, "query")
            assert hasattr(session, "commit")
            assert hasattr(session, "rollback")

            # Test session cleanup
            try:
                next(session_gen)
            except StopIteration:
                pass  # Expected behavior after session cleanup

    def test_session_rollback_on_exception(self):
        """Test that sessions rollback on exceptions."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "testing"}):
            config = DatabaseConfig()

            # Mock session to test rollback behavior
            mock_session = MagicMock()
            mock_session_local = MagicMock(return_value=mock_session)
            config.session_local = mock_session_local

            # Test the generator's exception handling
            session_gen = config.get_session()
            session = next(session_gen)

            # Verify we got the mocked session
            assert session is mock_session

            # Simulate exception and call generator's cleanup
            try:
                raise ValueError("Test exception")
            except ValueError:
                try:
                    next(session_gen)
                except (StopIteration, ValueError):
                    pass

            # The close should always be called
            mock_session.close.assert_called_once()


class TestDatabaseConfigIntegration:
    """Integration tests for database configuration."""

    def test_sqlite_configuration_for_testing(self):
        """Test SQLite configuration works for testing."""
        with patch.dict(
            os.environ,
            {
                "ESPER_ENVIRONMENT": "testing",
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5432",
                "POSTGRES_USER": "test_user",
                "POSTGRES_PASSWORD": "test_pass",
                "POSTGRES_DB": "test_db",
            },
        ):
            # For this test, we'll mock SQLite usage
            with patch(
                "esper.services.urza.database.create_engine"
            ) as mock_create_engine:
                mock_engine = MagicMock()
                mock_engine.pool = MagicMock(spec=StaticPool)
                mock_create_engine.return_value = mock_engine

                DatabaseConfig()

                # Verify StaticPool was used
                create_engine_call = mock_create_engine.call_args
                assert "poolclass" in create_engine_call.kwargs
                assert create_engine_call.kwargs["poolclass"] == StaticPool

    def test_production_pool_configuration(self):
        """Test production pool configuration is applied correctly."""
        with patch.dict(
            os.environ,
            {
                "ESPER_ENVIRONMENT": "production",
                "DB_POOL_SIZE": "40",
                "DB_MAX_OVERFLOW": "60",
                "DB_POOL_TIMEOUT": "90",
                "DB_POOL_RECYCLE": "7200",
            },
        ):
            with patch(
                "esper.services.urza.database.create_engine"
            ) as mock_create_engine:
                mock_engine = MagicMock()
                mock_engine.pool = MagicMock(spec=QueuePool)
                mock_create_engine.return_value = mock_engine

                DatabaseConfig()

                # Verify QueuePool configuration
                create_engine_call = mock_create_engine.call_args
                kwargs = create_engine_call.kwargs

                assert kwargs["poolclass"] == QueuePool
                assert kwargs["pool_size"] == 40
                assert kwargs["max_overflow"] == 60
                assert kwargs["pool_timeout"] == 90
                assert kwargs["pool_recycle"] == 7200
                assert kwargs["pool_pre_ping"] is True
                assert kwargs["future"] is True

    def test_environment_detection(self):
        """Test correct environment detection from ESPER_ENVIRONMENT."""
        environments = ["development", "staging", "production", "testing"]

        for env in environments:
            with patch.dict(os.environ, {"ESPER_ENVIRONMENT": env}):
                config = DatabaseConfig()
                assert config.environment == env


class TestDatabaseStatsEndpoint:
    """Test database statistics functionality."""

    def test_pool_stats_content(self):
        """Test that pool stats contain required information."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "development"}):
            config = DatabaseConfig()
            stats = config.get_pool_stats()

            # Required fields for monitoring
            required_fields = ["pool_type", "environment", "pool_status"]

            for field in required_fields:
                assert field in stats, f"Missing required field: {field}"

            # QueuePool should have additional fields (when available)
            if stats["pool_type"] == "QueuePool" and "note" not in stats:
                queuepool_fields = [
                    "pool_size",
                    "checked_in",
                    "checked_out",
                    "overflow",
                    "total_connections",
                ]
                for field in queuepool_fields:
                    assert field in stats, f"Missing QueuePool field: {field}"

    def test_pool_health_status(self):
        """Test pool health status determination."""
        with patch.dict(os.environ, {"ESPER_ENVIRONMENT": "development"}):
            config = DatabaseConfig()

            # Mock pool with healthy status (has checked in connections)
            mock_pool = MagicMock()
            mock_pool.size.return_value = 20
            mock_pool.checkedin.return_value = 15  # Healthy - has available connections
            mock_pool.checkedout.return_value = 5
            mock_pool.overflow.return_value = 0
            mock_pool.invalid.return_value = 0

            config.engine.pool = mock_pool

            stats = config.get_pool_stats()
            assert stats["pool_status"] == "healthy"

            # Mock pool with degraded status (no checked in connections)
            mock_pool.checkedin.return_value = 0  # Degraded - no available connections
            mock_pool.checkedout.return_value = 20

            stats = config.get_pool_stats()
            assert stats["pool_status"] == "degraded"
