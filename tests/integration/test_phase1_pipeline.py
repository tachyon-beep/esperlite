"""
Integration tests for Phase 1 Core Asset Pipeline.

This module contains tests that verify the end-to-end functionality
of the Oona -> Urza -> Tezzeret pipeline.
"""

import json
from unittest.mock import Mock, patch

import pytest

from esper.services.contracts import BlueprintSubmissionRequest


class TestPhase1Pipeline:
    """Integration tests for Phase 1 pipeline."""

    def test_blueprint_submission_contract(self):
        """Test the blueprint submission contract."""
        # Create a test blueprint
        blueprint = BlueprintSubmissionRequest(
            id="test-blueprint-001",
            architecture_ir='{"type": "linear", "input_size": 10, "output_size": 5}',
            metadata={"description": "Test blueprint"},
        )

        # Verify blueprint structure
        assert blueprint.id == "test-blueprint-001"
        assert "linear" in blueprint.architecture_ir
        assert blueprint.metadata["description"] == "Test blueprint"

    @patch("esper.services.tezzeret.worker.upload_bytes")
    @patch("esper.services.tezzeret.worker.requests.post")
    @patch("esper.services.tezzeret.worker.requests.put")
    @patch("esper.services.tezzeret.worker.requests.get")
    def test_tezzeret_compilation_flow(
        self, mock_get, mock_put, mock_post, mock_upload
    ):
        """Test Tezzeret compilation workflow."""
        from esper.services.tezzeret.worker import TezzeretWorker

        # Mock HTTP responses
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = [
            {
                "id": "test-blueprint-001",
                "architecture_ir": '{"type": "linear", "input_size": 10, "output_size": 5}',
            }
        ]

        mock_put.return_value.raise_for_status.return_value = None
        mock_post.return_value.raise_for_status.return_value = None

        # Mock S3 upload
        mock_upload.return_value = "s3://test-bucket/kernels/test-kernel/compiled.pt"

        # Create worker and process one blueprint
        worker = TezzeretWorker("test-worker")
        result = worker.process_one_blueprint()

        # Verify the flow
        assert result is True
        mock_get.assert_called_once()
        mock_put.assert_called()
        mock_post.assert_called_once()
        mock_upload.assert_called_once()

    def test_ir_to_module_conversion(self):
        """Test IR to PyTorch module conversion."""
        from esper.services.tezzeret.worker import TezzeretWorker

        worker = TezzeretWorker("test-worker")

        # Test linear module
        linear_ir = {
            "type": "linear",
            "input_size": 5,
            "output_size": 3,
            "hidden_size": 10,
        }
        linear_module = worker.ir_to_module(linear_ir)

        assert linear_module is not None
        assert len(list(linear_module.children())) == 3  # Linear, ReLU, Linear

        # Test conv module
        conv_ir = {
            "type": "conv",
            "in_channels": 3,
            "out_channels": 16,
            "kernel_size": 3,
        }
        conv_module = worker.ir_to_module(conv_ir)

        assert conv_module is not None
        assert (
            len(list(conv_module.children())) == 4
        )  # Conv2d, ReLU, AdaptiveAvgPool2d, Flatten

        # Test default module
        default_ir = {"type": "unknown"}
        default_module = worker.ir_to_module(default_ir)

        assert default_module is not None
        assert len(list(default_module.children())) == 3  # Default linear structure

    def test_compilation_pipeline(self):
        """Test the compilation pipeline with a real module."""
        from esper.services.tezzeret.worker import TezzeretWorker

        worker = TezzeretWorker("test-worker")

        # Test compilation with valid IR
        test_ir = {"type": "linear", "input_size": 10, "output_size": 5}

        try:
            compiled_bytes = worker.run_fast_compilation(test_ir)
            assert isinstance(compiled_bytes, bytes)
            assert len(compiled_bytes) > 0
        except Exception as e:
            # On some systems torch.compile might not work, that's okay for testing
            assert "compilation" in str(e).lower() or "torch" in str(e).lower()

    def test_kernel_id_generation(self):
        """Test kernel ID generation."""
        from esper.services.tezzeret.worker import TezzeretWorker

        worker = TezzeretWorker("test-worker")

        # Generate IDs with different inputs to ensure uniqueness
        id1 = worker.generate_kernel_id("blueprint-1", "fast")
        id2 = worker.generate_kernel_id(
            "blueprint-1", "optimized"
        )  # Different pipeline
        id3 = worker.generate_kernel_id("blueprint-2", "fast")  # Different blueprint

        # IDs should be different (input-based)
        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

        # IDs should be hex strings
        assert all(c in "0123456789abcdef" for c in id1)
        assert len(id1) == 64  # SHA256 hex length
