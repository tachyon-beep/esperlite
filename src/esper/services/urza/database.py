"""
Database configuration for Urza service.

This module provides database connection and session management.
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .models import Base


class DatabaseConfig:
    """Database configuration and session management."""

    def __init__(self):
        self.database_url = self._get_database_url()
        self.engine = create_engine(
            self.database_url,
            poolclass=StaticPool,
            pool_pre_ping=True,
            echo=False,  # Set to True for SQL debugging
        )
        self.session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
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


# Global database configuration instance
db_config = DatabaseConfig()


# Convenience function for FastAPI dependency injection
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    yield from db_config.get_session()
