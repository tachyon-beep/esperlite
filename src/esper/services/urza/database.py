"""
Database configuration for Urza service.

This module provides database connection and session management with production-ready
connection pooling.
"""

import os
from typing import Any
from typing import Dict
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseConfig:
    """Database configuration and session management."""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.environment = os.getenv("ESPER_ENVIRONMENT", "development")
        self.engine = self._create_engine()
        self.session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def _create_engine(self):
        """Create SQLAlchemy engine with appropriate pooling for environment."""
        if self.environment == "testing":
            # Use StaticPool for testing to ensure test isolation
            return create_engine(
                self.database_url,
                poolclass=StaticPool,
                pool_pre_ping=True,
                echo=False,
                # Testing-specific settings
                connect_args=(
                    {"check_same_thread": False}
                    if "sqlite" in self.database_url
                    else {}
                ),
            )
        else:
            # Use QueuePool for development and production
            pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
            max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "30"))
            pool_timeout = int(os.getenv("DB_POOL_TIMEOUT", "30"))
            pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # 1 hour

            return create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=pool_size,  # Base connection pool size
                max_overflow=max_overflow,  # Additional connections under load
                pool_timeout=pool_timeout,  # Timeout waiting for connection
                pool_recycle=pool_recycle,  # Recycle connections after 1 hour
                pool_pre_ping=True,  # Verify connections before use
                echo=False,  # Set to True for SQL debugging
                future=True,  # Use SQLAlchemy 2.0 style
            )

    def _get_database_url(self) -> str:
        """Constructs database URL from environment variables."""
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = os.getenv("POSTGRES_PORT", "5432")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        database = os.getenv("POSTGRES_DB", "urza_db")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def create_tables(self) -> None:
        """Creates all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Generator[Session, None, None]:
        """
        Dependency function that provides a database session.

        Yields:
            Session: SQLAlchemy database session
        """
        session = self.session_local()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_pool_stats(self) -> Dict[str, Any]:
        """
        Get database connection pool statistics.

        Returns:
            Dict containing pool statistics
        """
        pool = self.engine.pool

        if hasattr(pool, "size"):
            # QueuePool statistics
            try:
                pool_size = pool.size()
                checked_in = pool.checkedin()
                checked_out = pool.checkedout()
                overflow = pool.overflow()

                return {
                    "pool_type": pool.__class__.__name__,
                    "pool_size": pool_size,
                    "checked_in": checked_in,
                    "checked_out": checked_out,
                    "overflow": overflow,
                    "total_connections": pool_size + overflow,
                    "pool_status": "healthy" if checked_in > 0 else "degraded",
                    "environment": self.environment,
                }
            except AttributeError as e:
                # Fallback for pools that don't support all statistics
                return {
                    "pool_type": pool.__class__.__name__,
                    "environment": self.environment,
                    "pool_status": "active",
                    "note": f"Limited statistics available: {e}",
                }
        else:
            # StaticPool or other pool types
            return {
                "pool_type": pool.__class__.__name__,
                "environment": self.environment,
                "pool_status": "active",
            }


# Global database configuration instance
db_config = DatabaseConfig()


# Convenience function for FastAPI dependency injection
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    yield from db_config.get_session()
