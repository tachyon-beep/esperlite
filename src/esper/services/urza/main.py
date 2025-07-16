"""
Urza Asset Hub - FastAPI Service.

This module provides the REST API for managing blueprints and compiled kernels.
Urza acts as the central repository for all architectural assets in the system.
"""

import logging
import os
from datetime import datetime
from datetime import timezone
from typing import List
from typing import Optional

from fastapi import Depends
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import status
from fastapi.responses import JSONResponse
from sqlalchemy import desc
from sqlalchemy.orm import Session

from esper.contracts.enums import BlueprintStatus
from esper.services.contracts import SimpleBlueprintContract
from esper.services.contracts import SimpleCompiledKernelContract

from .database import get_db
from .models import Blueprint
from .models import CompiledKernel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Urza Asset Hub",
    description="Central repository for architectural assets",
    version="1.0.0",
)


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "urza",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Public API endpoints
@app.post("/api/v1/blueprints", response_model=dict, tags=["blueprints"])
async def create_blueprint(
    blueprint: SimpleBlueprintContract, db: Session = Depends(get_db)
):
    """Create a new blueprint."""
    try:
        # Create database record
        db_blueprint = Blueprint(
            id=blueprint.id,
            architecture_ir=blueprint.architecture_ir,
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(db_blueprint)
        db.commit()
        db.refresh(db_blueprint)

        logger.info("Created blueprint: %s", blueprint.id)
        return {"id": db_blueprint.id, "status": "created"}

    except Exception as e:
        logger.error("Failed to create blueprint: %s", e)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create blueprint",
        ) from e


@app.get("/api/v1/blueprints", response_model=List[dict], tags=["blueprints"])
async def list_blueprints(
    status_filter: Optional[str] = None, limit: int = 100, db: Session = Depends(get_db)
):
    """List blueprints with optional status filtering."""
    try:
        query = db.query(Blueprint)

        if status_filter:
            query = query.filter(Blueprint.status == status_filter)

        blueprints = query.order_by(desc(Blueprint.created_at)).limit(limit).all()

        return [
            {
                "id": bp.id,
                "status": bp.status,
                "created_at": bp.created_at.isoformat(),
                "updated_at": bp.updated_at.isoformat(),
            }
            for bp in blueprints
        ]

    except Exception as e:
        logger.error("Failed to list blueprints: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list blueprints",
        ) from e


@app.get("/api/v1/blueprints/{blueprint_id}", response_model=dict, tags=["blueprints"])
async def get_blueprint(blueprint_id: str, db: Session = Depends(get_db)):
    """Get a specific blueprint."""
    try:
        blueprint = db.query(Blueprint).filter(Blueprint.id == blueprint_id).first()

        if not blueprint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Blueprint not found"
            )

        return {
            "id": blueprint.id,
            "architecture_ir": blueprint.architecture_ir,
            "status": blueprint.status,
            "created_at": blueprint.created_at.isoformat(),
            "updated_at": blueprint.updated_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get blueprint: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get blueprint",
        ) from e


@app.get(
    "/api/v1/kernels/{kernel_id}", response_model=dict, tags=["kernels"]
)
async def get_kernel(kernel_id: str, db: Session = Depends(get_db)):
    """Get a compiled kernel by ID."""
    try:
        kernel = db.query(CompiledKernel).filter(CompiledKernel.id == kernel_id).first()

        if not kernel:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Kernel {kernel_id} not found"
            )

        logger.info("Retrieved kernel: %s", kernel_id)
        return {
            "id": kernel.id,
            "blueprint_id": kernel.blueprint_id,
            "status": kernel.status,
            "compilation_pipeline": kernel.compilation_pipeline,
            "kernel_binary_ref": kernel.kernel_binary_ref,
            "validation_report": kernel.validation_report,
            "created_at": kernel.created_at.isoformat(),
            "updated_at": kernel.updated_at.isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get kernel %s: %s", kernel_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get kernel"
        ) from e


# Internal API endpoints (used by Tezzeret)
@app.get(
    "/internal/v1/blueprints/unvalidated", response_model=List[dict], tags=["internal"]
)
async def get_unvalidated_blueprints(db: Session = Depends(get_db)):
    """Get unvalidated blueprints for compilation."""
    try:
        blueprints = (
            db.query(Blueprint)
            .filter(Blueprint.status == BlueprintStatus.UNVALIDATED.value)
            .order_by(Blueprint.created_at)
            .limit(10)
            .all()
        )

        return [
            {
                "id": bp.id,
                "architecture_ir": bp.architecture_ir,
                "status": bp.status,
                "created_at": bp.created_at.isoformat(),
                "updated_at": bp.updated_at.isoformat(),
            }
            for bp in blueprints
        ]

    except Exception as e:
        logger.error("Failed to get unvalidated blueprints: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get unvalidated blueprints",
        ) from e


@app.put("/internal/v1/blueprints/{blueprint_id}/status", tags=["internal"])
async def update_blueprint_status(
    blueprint_id: str, status_update: dict, db: Session = Depends(get_db)
):
    """Update blueprint status."""
    try:
        blueprint = db.query(Blueprint).filter(Blueprint.id == blueprint_id).first()

        if not blueprint:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Blueprint not found"
            )

        blueprint.status = status_update["status"]
        blueprint.updated_at = datetime.now(timezone.utc)  # type: ignore[assignment]

        db.commit()
        db.refresh(blueprint)

        logger.info(
            "Updated blueprint %s status to %s", blueprint_id, status_update["status"]
        )
        return {"id": blueprint_id, "status": blueprint.status}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update blueprint status: %s", e)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update blueprint status",
        ) from e


@app.post("/internal/v1/kernels", response_model=dict, tags=["internal"])
async def create_kernel(
    kernel: SimpleCompiledKernelContract, db: Session = Depends(get_db)
):
    """Create a new compiled kernel."""
    try:
        # Create database record
        db_kernel = CompiledKernel(
            id=kernel.id,
            blueprint_id=kernel.blueprint_id,
            status=kernel.status,
            compilation_pipeline=kernel.compilation_pipeline,
            kernel_binary_ref=kernel.kernel_binary_ref,
            validation_report=kernel.validation_report,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(db_kernel)
        db.commit()
        db.refresh(db_kernel)

        logger.info(
            "Created kernel: %s for blueprint: %s", kernel.id, kernel.blueprint_id
        )
        return {"id": db_kernel.id, "status": "created"}

    except Exception as e:
        logger.error("Failed to create kernel: %s", e)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create kernel",
        ) from e


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(_request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
