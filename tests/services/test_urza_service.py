"""
Tests for Urza FastAPI service.

This module tests the REST API endpoints for managing blueprints and compiled kernels.
"""

from datetime import datetime
from datetime import timezone
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from esper.contracts.enums import BlueprintStatus
from esper.services.urza.main import app
from esper.services.urza.main import get_db
from esper.services.urza.models import Base
from esper.services.urza.models import Blueprint
from esper.services.urza.models import CompiledKernel


class TestUrzaService:
    """Test suite for Urza service."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        engine = create_engine(
            "sqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=engine)
        testing_session_local = sessionmaker(
            autocommit=False, autoflush=False, bind=engine
        )

        session = testing_session_local()
        try:
            yield session
        finally:
            session.close()

    @pytest.fixture
    def client(self, db_session):
        """Create test client with mocked database and kernel manager."""

        def get_test_db():
            yield db_session

        # Mock the kernel manager to avoid Redis connection
        with patch('esper.services.urza.main.kernel_manager') as mock_km:
            # Mock the async methods
            mock_km.initialize = AsyncMock(return_value=None)
            mock_km.close = AsyncMock(return_value=None)
            mock_km.store_kernel = AsyncMock(return_value={"success": True})
            mock_km.get_kernel = AsyncMock(return_value=None)

            app.dependency_overrides[get_db] = get_test_db
            with TestClient(app) as test_client:
                yield test_client
            app.dependency_overrides.clear()

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "urza"
        assert "timestamp" in data

    def test_create_blueprint_success(self, client, db_session):
        """Test successful blueprint creation."""
        blueprint_data = {
            "id": "test-blueprint-123",
            "architecture_ir": '{"type": "linear", "input_size": 10, "output_size": 5}',
        }

        response = client.post("/api/v1/blueprints", json=blueprint_data)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-blueprint-123"
        assert data["status"] == "created"

        # Verify database record
        db_blueprint = (
            db_session.query(Blueprint)
            .filter(Blueprint.id == "test-blueprint-123")
            .first()
        )
        assert db_blueprint is not None
        assert db_blueprint.status == BlueprintStatus.UNVALIDATED.value

    def test_create_blueprint_invalid_data(self, client):
        """Test blueprint creation with invalid data."""
        response = client.post("/api/v1/blueprints", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    def test_list_blueprints_empty(self, client):
        """Test listing blueprints when none exist."""
        response = client.get("/api/v1/blueprints")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_blueprints_with_data(self, client, db_session):
        """Test listing blueprints with existing data."""
        # Create test blueprints
        blueprint1 = Blueprint(
            id="blueprint-1",
            architecture_ir='{"type": "linear"}',
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        blueprint2 = Blueprint(
            id="blueprint-2",
            architecture_ir='{"type": "conv"}',
            status=BlueprintStatus.VALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db_session.add_all([blueprint1, blueprint2])
        db_session.commit()

        response = client.get("/api/v1/blueprints")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

        # Test with status filter
        response = client.get("/api/v1/blueprints?status_filter=unvalidated")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "blueprint-1"

    def test_get_blueprint_success(self, client, db_session):
        """Test getting a specific blueprint."""
        blueprint = Blueprint(
            id="test-blueprint",
            architecture_ir='{"type": "linear"}',
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(blueprint)
        db_session.commit()

        response = client.get("/api/v1/blueprints/test-blueprint")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-blueprint"
        assert data["architecture_ir"] == '{"type": "linear"}'
        assert data["status"] == "unvalidated"

    def test_get_blueprint_not_found(self, client):
        """Test getting non-existent blueprint."""
        response = client.get("/api/v1/blueprints/non-existent")
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Blueprint not found"

    def test_get_unvalidated_blueprints_empty(self, client):
        """Test getting unvalidated blueprints when none exist."""
        response = client.get("/internal/v1/blueprints/unvalidated")
        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_get_unvalidated_blueprints_with_data(self, client, db_session):
        """Test getting unvalidated blueprints with existing data."""
        # Create test blueprints with different statuses
        blueprint1 = Blueprint(
            id="unvalidated-1",
            architecture_ir='{"type": "linear"}',
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        blueprint2 = Blueprint(
            id="validated-1",
            architecture_ir='{"type": "conv"}',
            status=BlueprintStatus.VALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db_session.add_all([blueprint1, blueprint2])
        db_session.commit()

        response = client.get("/internal/v1/blueprints/unvalidated")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == "unvalidated-1"

    def test_update_blueprint_status_success(self, client, db_session):
        """Test successful blueprint status update."""
        blueprint = Blueprint(
            id="test-blueprint",
            architecture_ir='{"type": "linear"}',
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(blueprint)
        db_session.commit()

        response = client.put(
            "/internal/v1/blueprints/test-blueprint/status",
            json={"status": "COMPILING"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-blueprint"
        assert data["status"] == "COMPILING"

        # Verify database was updated
        db_session.refresh(blueprint)
        assert blueprint.status == "COMPILING"

    def test_update_blueprint_status_not_found(self, client):
        """Test updating status of non-existent blueprint."""
        response = client.put(
            "/internal/v1/blueprints/non-existent/status", json={"status": "COMPILING"}
        )
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Blueprint not found"

    def test_create_kernel_success(self, client, db_session):
        """Test successful kernel creation."""
        # Create parent blueprint first
        blueprint = Blueprint(
            id="parent-blueprint",
            architecture_ir='{"type": "linear"}',
            status=BlueprintStatus.UNVALIDATED.value,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db_session.add(blueprint)
        db_session.commit()

        kernel_data = {
            "id": "test-kernel-123",
            "blueprint_id": "parent-blueprint",
            "status": "validated",
            "compilation_pipeline": "fast",
            "kernel_binary_ref": "s3://bucket/kernel.pt",
            "validation_report": {"passed": True},
        }

        response = client.post("/internal/v1/kernels", json=kernel_data)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "test-kernel-123"
        assert data["status"] == "created"

        # Verify database record
        db_kernel = (
            db_session.query(CompiledKernel)
            .filter(CompiledKernel.id == "test-kernel-123")
            .first()
        )
        assert db_kernel is not None
        assert db_kernel.blueprint_id == "parent-blueprint"
        assert db_kernel.status == "validated"

    def test_create_kernel_invalid_data(self, client):
        """Test kernel creation with invalid data."""
        response = client.post("/internal/v1/kernels", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    @patch("esper.services.urza.main.logger")
    def test_error_handling(self, _mock_logger, client):
        """Test error handling in endpoints."""

        # Force a database error by providing invalid database dependency
        def get_broken_db():
            raise RuntimeError("Database connection failed")

        app.dependency_overrides[get_db] = get_broken_db

        try:
            blueprint_data = {
                "id": "test-blueprint-error",
                "architecture_ir": '{"type": "linear"}',
            }

            # The dependency injection will fail, so we expect this to raise
            with pytest.raises(RuntimeError, match="Database connection failed"):
                client.post("/api/v1/blueprints", json=blueprint_data)

        finally:
            # Clean up dependency override
            if get_db in app.dependency_overrides:
                del app.dependency_overrides[get_db]

    def test_response_models(self, client):
        """Test that response models match expected schema."""
        # Create a blueprint
        blueprint_data = {
            "id": "schema-test-blueprint",
            "architecture_ir": '{"type": "linear", "input_size": 10}',
        }

        response = client.post("/api/v1/blueprints", json=blueprint_data)
        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "id" in data
        assert "status" in data
        assert data["id"] == "schema-test-blueprint"
        assert data["status"] == "created"
