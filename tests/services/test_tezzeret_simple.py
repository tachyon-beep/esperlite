"""
Simple test for TezzeretWorker.
"""

import os
from unittest.mock import Mock
from unittest.mock import patch


class TestTezzeretWorkerSimple:
    """Simple test cases for TezzeretWorker."""

    @patch.dict(os.environ, {}, clear=True)
    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_worker_creation(self, mock_get_s3_client):
        """Test basic TezzeretWorker creation."""
        from esper.services.tezzeret.worker import TezzeretWorker
        from esper.utils.config import reset_service_config

        # Reset configuration to ensure clean test environment
        reset_service_config()

        mock_s3_client = Mock()
        mock_get_s3_client.return_value = mock_s3_client

        worker = TezzeretWorker("test-worker")

        assert worker.worker_id == "test-worker"
        assert worker.urza_base_url == "http://localhost:8000"

    @patch("esper.services.tezzeret.worker.get_s3_client")
    def test_ir_to_module_simple(self, mock_get_s3_client):
        """Test simple IR to module conversion."""
        import torch.nn as nn

        from esper.services.tezzeret.worker import TezzeretWorker

        worker = TezzeretWorker("test-worker")

        ir_data = {"type": "linear"}
        module = worker.ir_to_module(ir_data)

        assert isinstance(module, nn.Sequential)
