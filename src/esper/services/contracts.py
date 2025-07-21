"""
Phase 1 specific contracts for API interfaces.

These contracts are simplified versions for the MVP implementation
and may differ from the full Esper contracts.
"""

from typing import Any
from typing import Dict
from typing import Optional

from pydantic import BaseModel

from esper.contracts.enums import KernelStatus


class SimpleBlueprintContract(BaseModel):
    """Simplified blueprint contract for Phase 1 API."""

    id: str
    architecture_ir: str  # JSON string of the architecture
    metadata: Optional[Dict[str, Any]] = None


class SimpleCompiledKernelContract(BaseModel):
    """Simplified compiled kernel contract for Phase 1 API."""

    id: str
    blueprint_id: str
    status: KernelStatus
    compilation_pipeline: str
    kernel_binary_ref: str  # S3 URI to the binary artifact
    validation_report: Optional[Dict[str, Any]] = None


class BlueprintSubmissionRequest(BaseModel):
    """Contract for blueprint submission API."""

    id: str
    architecture_ir: str  # JSON string of the architecture
    metadata: Optional[Dict[str, Any]] = None


class BlueprintResponse(BaseModel):
    """Contract for blueprint API responses."""

    id: str
    architecture_ir: str
    metadata: Optional[Dict[str, Any]] = None


class CompiledKernelArtifact(BaseModel):
    """Contract for compiled kernel artifacts."""

    id: str
    blueprint_id: str
    status: str  # Will be converted to KernelStatus enum
    compilation_pipeline: str
    kernel_binary_ref: str
    validation_report: Optional[Dict[str, Any]] = None
