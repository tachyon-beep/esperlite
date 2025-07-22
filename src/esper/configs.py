"""
Configuration models for parsing YAML configuration files.
"""

from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel
from pydantic import Field


class DatabaseConfig(BaseModel):
    """Database connection configuration."""

    host: str = Field(default="localhost", description="Database host address")
    port: int = Field(default=5432, description="Database port number", ge=1, le=65535)
    database: str = Field(default="urza_db", description="Database name")
    username: str = Field(description="Database username")
    password: str = Field(description="Database password")
    ssl_mode: str = Field(default="prefer", description="SSL connection mode")


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = Field(default="localhost", description="Redis host address")
    port: int = Field(default=6379, description="Redis port number", ge=1, le=65535)
    database: int = Field(default=0, description="Redis database number", ge=0)
    password: Optional[str] = Field(default=None, description="Redis password")


class StorageConfig(BaseModel):
    """Object storage configuration."""

    endpoint_url: str = Field(
        default="http://localhost:9000",
        description="S3-compatible storage endpoint URL",
    )
    access_key: str = Field(description="Storage access key")
    secret_key: str = Field(description="Storage secret key")
    bucket_name: str = Field(
        default="esper-artifacts", description="Storage bucket name"
    )
    region: str = Field(default="us-east-1", description="Storage region")


class ComponentConfig(BaseModel):
    """Configuration for individual system components."""

    enabled: bool = True
    replicas: int = 1
    resources: Dict[str, Any] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class EsperConfig(BaseModel):
    """Main configuration model for the Esper system."""

    name: str = Field(description="Configuration name")
    version: str = Field(default="0.1.0", description="Configuration version")
    environment: str = Field(
        default="development",
        description="Deployment environment",
        pattern="^(development|staging|production)$",
    )

    database: DatabaseConfig = Field(description="Database configuration")
    redis: RedisConfig = Field(description="Redis configuration")
    storage: StorageConfig = Field(description="Object storage configuration")

    components: Dict[str, ComponentConfig] = Field(
        default_factory=dict, description="Component-specific configurations"
    )

    # Training configuration
    training: Dict[str, Any] = Field(
        default_factory=dict, description="Training-specific settings"
    )

    # Logging configuration
    logging: Dict[str, Any] = Field(
        default_factory=lambda: {"level": "INFO", "format": "structured"},
        description="Logging configuration",
    )
