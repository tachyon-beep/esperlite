"""
Unit tests for TezzeretWorker.

Tests the compilation worker functionality.
"""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

from esper.contracts.enums import BlueprintStatus, KernelStatus
from esper.services.tezzeret.worker import TezzeretWorker


class TestTezzeretWorker:
    """Test cases for TezzeretWorker."""

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_init(self, mock_get_s3_client):
        """Test TezzeretWorker initialization."""
        mock_s3_client = Mock()
        mock_get_s3_client.return_value = mock_s3_client

        worker = TezzeretWorker("test-worker")

        assert worker.worker_id == "test-worker"
        assert worker.urza_base_url == "http://localhost:8000"
        assert worker.s3_client == mock_s3_client

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_ir_to_module_linear(self, mock_get_s3_client):
        """Test IR to module conversion for linear type."""
        worker = TezzeretWorker("test-worker")

        ir_data = {
            "type": "linear",
            "input_size": 5,
            "output_size": 3,
            "hidden_size": 10,
        }

        module = worker.ir_to_module(ir_data)

        assert isinstance(module, nn.Sequential)
        assert len(module) == 3  # Linear, ReLU, Linear
        assert isinstance(module[0], nn.Linear)
        assert module[0].in_features == 5
        assert module[0].out_features == 10
        assert isinstance(module[1], nn.ReLU)
        assert isinstance(module[2], nn.Linear)
        assert module[2].in_features == 10
        assert module[2].out_features == 3

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_ir_to_module_conv(self, mock_get_s3_client):
        """Test IR to module conversion for conv type."""
        worker = TezzeretWorker("test-worker")

        ir_data = {
            "type": "conv",
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 5,
        }

        module = worker.ir_to_module(ir_data)

        assert isinstance(module, nn.Sequential)
        assert len(module) == 4  # Conv2d, ReLU, AdaptiveAvgPool2d, Flatten
        assert isinstance(module[0], nn.Conv2d)
        assert module[0].in_channels == 3
        assert module[0].out_channels == 16
        assert module[0].kernel_size == (5, 5)

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_ir_to_module_default(self, mock_get_s3_client):
        """Test IR to module conversion for unknown type (default fallback)."""
        worker = TezzeretWorker("test-worker")

        ir_data = {"type": "unknown"}

        module = worker.ir_to_module(ir_data)

        assert isinstance(module, nn.Sequential)
        assert len(module) == 3  # Default: Linear(10,20), ReLU, Linear(20,10)

    @patch("esper.services.tezzeret.worker.get_s3_client")
    @patch("torch.compile")
    @patch("torch.save")
    def test_run_fast_compilation_success(
        self, mock_torch_save, mock_torch_compile, mock_get_s3_client
    ):
        """Test successful compilation."""
        worker = TezzeretWorker("test-worker")

        # Mock torch.compile
        mock_compiled_module = Mock()
        mock_compiled_module.state_dict.return_value = {"weight": torch.tensor([1.0])}
        mock_torch_compile.return_value = mock_compiled_module

        ir_data = {"type": "linear", "input_size": 10, "output_size": 5}

        result = worker.run_fast_compilation(ir_data)

        assert isinstance(result, bytes)
        mock_torch_compile.assert_called_once()
        mock_torch_save.assert_called_once()

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_generate_kernel_id(self, mock_get_s3_client):
        """Test kernel ID generation."""
        worker = TezzeretWorker("test-worker")

        kernel_id = worker.generate_kernel_id("blueprint-123", "fast")

        assert isinstance(kernel_id, str)
        assert len(kernel_id) == 64  # SHA256 hex string length

        # Should be deterministic for same inputs at same time
        kernel_id2 = worker.generate_kernel_id("blueprint-123", "fast")
        # Note: These will be different due to timestamp, but format should be same
        assert len(kernel_id2) == 64

    @patch("esper.services.tezzeret.worker.get_s3_client")
    @patch("esper.services.tezzeret.worker.requests.get")
    def test_fetch_unvalidated_blueprints_success(
        self, mock_requests_get, mock_get_s3_client
    ):
        """Test successful fetching of unvalidated blueprints."""
        worker = TezzeretWorker("test-worker")

        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "bp1", "architecture_ir": "{}"},
            {"id": "bp2", "architecture_ir": "{}"},
        ]
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        blueprints = worker.fetch_unvalidated_blueprints()

        assert len(blueprints) == 2
        assert blueprints[0]["id"] == "bp1"
        mock_requests_get.assert_called_once_with(
            "http://localhost:8000/internal/v1/blueprints/unvalidated", timeout=30
        )

    @patch("esper.services.tezzeret.worker.get_s3_client")
    @patch("esper.services.tezzeret.worker.requests.get")
    def test_fetch_unvalidated_blueprints_error(
        self, mock_requests_get, mock_get_s3_client
    ):
        """Test fetching blueprints with error."""
        import requests

        worker = TezzeretWorker("test-worker")

        mock_requests_get.side_effect = requests.RequestException("Network error")

        blueprints = worker.fetch_unvalidated_blueprints()

        assert blueprints == []

    @patch("esper.services.tezzeret.worker.get_s3_client")
    @patch("esper.services.tezzeret.worker.requests.put")
    def test_update_blueprint_status_success(
        self, mock_requests_put, mock_get_s3_client
    ):
        """Test successful blueprint status update."""
        worker = TezzeretWorker("test-worker")

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_requests_put.return_value = mock_response

        result = worker.update_blueprint_status("bp1", BlueprintStatus.COMPILING)

        assert result is True
        mock_requests_put.assert_called_once()

    @patch("esper.services.tezzeret.worker.get_s3_client")
    @patch("esper.services.tezzeret.worker.requests.post")
    def test_submit_compiled_kernel_success(
        self, mock_requests_post, mock_get_s3_client
    ):
        """Test successful kernel submission."""
        from esper.services.contracts import CompiledKernelArtifact

        worker = TezzeretWorker("test-worker")

        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_requests_post.return_value = mock_response

        kernel = CompiledKernelArtifact(
            id="kernel-123",
            blueprint_id="bp-123",
            status=KernelStatus.VALIDATED,
            compilation_pipeline="fast",
            kernel_binary_ref="s3://bucket/kernel.pt",
            validation_report={},
        )

        result = worker.submit_compiled_kernel(kernel)

        assert result is True
        mock_requests_post.assert_called_once()
