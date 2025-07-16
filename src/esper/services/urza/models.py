"""
Database models for Urza service.

This module defines SQLAlchemy models for blueprints and compiled kernels.
"""

from typing import Any
from typing import Dict

from sqlalchemy import JSON
from sqlalchemy import Column
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    """Base class for all database models."""


class Blueprint(Base):
    """Blueprint database model."""

    __tablename__ = "blueprints"

    id = Column(String(64), primary_key=True)
    status = Column(String(32), nullable=False)
    architecture_ir = Column(Text, nullable=False)
    blueprint_metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # Relationship to compiled kernels
    compiled_kernels = relationship(
        "CompiledKernel", back_populates="blueprint", cascade="all, delete-orphan"
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "status": self.status,
            "architecture_ir": self.architecture_ir,
            "blueprint_metadata": self.blueprint_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class CompiledKernel(Base):
    """Compiled kernel database model."""

    __tablename__ = "compiled_kernels"

    id = Column(String(128), primary_key=True)
    blueprint_id = Column(
        String(64), ForeignKey("blueprints.id", ondelete="CASCADE"), nullable=False
    )
    status = Column(String(32), nullable=False)
    compilation_pipeline = Column(String(32), nullable=False)
    kernel_binary_ref = Column(Text, nullable=False)
    validation_report = Column(JSON)
    created_at = Column(DateTime(timezone=True), nullable=False)
    updated_at = Column(DateTime(timezone=True), nullable=False)

    # Relationship to blueprint
    blueprint = relationship("Blueprint", back_populates="compiled_kernels")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "id": self.id,
            "blueprint_id": self.blueprint_id,
            "status": self.status,
            "compilation_pipeline": self.compilation_pipeline,
            "kernel_binary_ref": self.kernel_binary_ref,
            "validation_report": self.validation_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
