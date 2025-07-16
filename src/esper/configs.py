"""
Configuration models for parsing YAML configuration files.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "urza_db"
    username: str
    password: str
    ssl_mode: str = "prefer"


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None


class StorageConfig(BaseModel):
    """Object storage configuration."""

    endpoint_url: str = "http://localhost:9000"
    access_key: str
    secret_key: str
    bucket_name: str = "esper-artifacts"
    region: str = "us-east-1"


class ComponentConfig(BaseModel):
    """Configuration for individual system components."""

    enabled: bool = True
    replicas: int = 1
    resources: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class EsperConfig(BaseModel):
    """Main configuration model for the Esper system."""

    name: str
    version: str = "0.1.0"
    environment: str = "development"

    database: DatabaseConfig
    redis: RedisConfig
    storage: StorageConfig

    components: Dict[str, ComponentConfig] = Field(default_factory=dict)

    # Training configuration
    training: Dict[str, Any] = Field(default_factory=dict)

    # Logging configuration
    logging: Dict[str, Any] = Field(
        default_factory=lambda: {"level": "INFO", "format": "structured"}
    )
