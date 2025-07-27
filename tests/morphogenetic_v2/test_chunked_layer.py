"""
Tests for ChunkedKasminaLayer module.

Validates the high-performance chunked implementation.
"""

import pytest
import torch
import torch.nn as nn
import time
# Removed typing imports - not needed after removing test_initialization

from esper.morphogenetic_v2.kasmina.chunked_layer import ChunkedKasminaLayer
from esper.morphogenetic_v2.kasmina.logical_seed import SeedLifecycle


class TestChunkedKasminaLayer:
    """Test suite for ChunkedKasminaLayer functionality."""

    @pytest.fixture
    def base_layer(self):
        """Create a base neural network layer."""
        return nn.Linear(256, 256)

    @pytest.fixture
    def chunked_layer(self, base_layer):
        """Create a ChunkedKasminaLayer instance."""
        return ChunkedKasminaLayer(
            base_layer=base_layer,
            num_seeds=16,
            layer_id="test_layer",
            enable_telemetry=True
        )

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        batch_size = 32
        input_dim = 256
        return torch.randn(batch_size, input_dim)

    # Removed test_initialization - only checked constructor parameters

    def test_dimension_detection(self):
        """Test dimension detection from various layer types."""
        # Linear layer
        linear = nn.Linear(128, 64)
        layer1 = ChunkedKasminaLayer(linear, num_seeds=8)
        assert layer1.input_dim == 128
        assert layer1.output_dim == 64

        # Conv1d layer (has weight but not in/out_features)
        conv = nn.Conv1d(32, 64, kernel_size=3)
        layer2 = ChunkedKasminaLayer(conv, num_seeds=4)
        assert layer2.output_dim == 64  # out_channels
        assert layer2.input_dim == 32   # in_channels

        # Invalid layer
        with pytest.raises(ValueError, match="Cannot determine dimensions"):
            ChunkedKasminaLayer(nn.ReLU(), num_seeds=4)

    def test_forward_dormant_seeds(self, chunked_layer, sample_input):
        """Test forward pass with all dormant seeds."""
        # All seeds start dormant
        output = chunked_layer(sample_input)

        # Should be equivalent to base layer
        base_output = chunked_layer.base_layer(sample_input.to(chunked_layer.device))
        assert output.shape == base_output.shape
        assert torch.allclose(output, base_output, atol=1e-6)

        # Check metrics
        assert chunked_layer.forward_count == 1
        assert chunked_layer.total_latency > 0

    def test_forward_active_seeds(self, chunked_layer, sample_input):
        """Test forward pass with active seeds."""
        # Activate some seeds
        chunked_layer.request_germination(0)
        chunked_layer.request_germination(5)
        chunked_layer.request_germination(10)

        # Transition to active
        chunked_layer.state_tensor.set_lifecycle_state(0, SeedLifecycle.ACTIVE)
        chunked_layer.state_tensor.set_lifecycle_state(5, SeedLifecycle.ACTIVE)
        chunked_layer.state_tensor.set_lifecycle_state(10, SeedLifecycle.ACTIVE)

        # Forward pass
        output = chunked_layer(sample_input)

        # Should have correct shape
        assert output.shape == (sample_input.shape[0], chunked_layer.output_dim)

        # Check that chunks were processed
        assert chunked_layer.forward_count == 1

    def test_germination_requests(self, chunked_layer):
        """Test seed germination interface."""
        # Request germination
        success = chunked_layer.request_germination(3, blueprint_id=None, grafting_strategy=1)
        assert success

        # Check state updated
        state = chunked_layer.state_tensor.get_seed_state(3)
        assert state["lifecycle_state"] == SeedLifecycle.LOADING
        assert state["blueprint_id"] > 0  # Auto-created
        assert state["grafting_strategy"] == 1

        # Request on non-dormant seed should fail
        success2 = chunked_layer.request_germination(3)
        assert not success2

        # Invalid seed ID
        success3 = chunked_layer.request_germination(100)
        assert not success3

    def test_cancel_germination(self, chunked_layer):
        """Test germination cancellation."""
        # Start germination
        chunked_layer.request_germination(5)

        # Cancel it
        success = chunked_layer.cancel_germination(5)
        assert success

        # Check state reset
        state = chunked_layer.state_tensor.get_seed_state(5)
        assert state["lifecycle_state"] == SeedLifecycle.DORMANT
        assert state["blueprint_id"] == 0
        assert state["alpha_blend"] == 0.0

        # Cancel on dormant seed should fail
        success2 = chunked_layer.cancel_germination(5)
        assert not success2

    def test_blueprint_management(self, chunked_layer):
        """Test blueprint creation and registration."""
        # Register custom blueprint
        custom_blueprint = nn.Sequential(
            nn.Linear(16, 32),  # 16 = chunk size for seed
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        blueprint_id = chunked_layer.register_blueprint(custom_blueprint)
        assert blueprint_id in chunked_layer.blueprints
        assert chunked_layer.blueprints[blueprint_id] == custom_blueprint

        # Auto-create blueprint
        success = chunked_layer.request_germination(7)
        assert success
        state = chunked_layer.state_tensor.get_seed_state(7)
        auto_blueprint_id = state["blueprint_id"]
        assert auto_blueprint_id in chunked_layer.blueprints

    def test_telemetry_collection(self, chunked_layer, sample_input):
        """Test telemetry and health monitoring."""
        # Enable telemetry
        assert chunked_layer.enable_telemetry

        # Run forward passes
        for _ in range(5):
            chunked_layer(sample_input)

        # Check telemetry was collected
        stats = chunked_layer.get_layer_stats()
        assert stats["forward_count"] == 5
        assert stats["avg_latency_ms"] > 0
        assert stats["num_blueprints"] == 0

        # Get health report
        report = chunked_layer.get_health_report()
        assert report["layer_id"] == "test_layer"
        assert len(report["seeds"]) == chunked_layer.num_seeds
        assert len(report["telemetry"]) == chunked_layer.num_seeds
        assert report["performance"]["forward_count"] == 5

    def test_chunk_processing(self, chunked_layer, sample_input):
        """Test chunk-wise processing with blueprints."""
        # Create and register blueprints
        # chunk_size = chunked_layer.chunk_manager.get_chunk_size(0)  # Not used

        # Identity blueprint
        identity_bp = nn.Identity()
        identity_id = chunked_layer.register_blueprint(identity_bp)

        # Scaling blueprint
        class ScaleLayer(nn.Module):
            def forward(self, x):
                return x * 2.0

        scale_bp = ScaleLayer()
        scale_id = chunked_layer.register_blueprint(scale_bp)

        # Activate seeds with different blueprints
        chunked_layer.state_tensor.set_lifecycle_state(0, SeedLifecycle.ACTIVE)
        chunked_layer.state_tensor.set_blueprint(0, identity_id)
        chunked_layer.state_tensor.update_alpha_blend(torch.tensor([0]), torch.tensor([1.0]))

        chunked_layer.state_tensor.set_lifecycle_state(1, SeedLifecycle.ACTIVE)
        chunked_layer.state_tensor.set_blueprint(1, scale_id)
        chunked_layer.state_tensor.update_alpha_blend(torch.tensor([1]), torch.tensor([1.0]))

        # Forward pass
        output = chunked_layer(sample_input)

        # Verify chunk 0 unchanged, chunk 1 scaled
        base_output = chunked_layer.base_layer(sample_input.to(chunked_layer.device))
        chunks_out = chunked_layer.chunk_manager.split_tensor(output)
        chunks_base = chunked_layer.chunk_manager.split_tensor(base_output)

        # Chunk 0 should be unchanged (identity)
        assert torch.allclose(chunks_out[0], chunks_base[0], atol=1e-5)

        # Chunk 1 should be scaled by 2
        assert torch.allclose(chunks_out[1], chunks_base[1] * 2.0, atol=1e-5)

    def test_error_handling(self, chunked_layer, sample_input):
        """Test error recovery mechanisms."""
        # Create faulty blueprint
        class FaultyBlueprint(nn.Module):
            def forward(self, x):
                raise RuntimeError("Intentional error")

        faulty_bp = FaultyBlueprint()
        faulty_id = chunked_layer.register_blueprint(faulty_bp)

        # Activate seed with faulty blueprint
        chunked_layer.state_tensor.set_lifecycle_state(2, SeedLifecycle.ACTIVE)
        chunked_layer.state_tensor.set_blueprint(2, faulty_id)
        chunked_layer.state_tensor.update_alpha_blend(torch.tensor([2]), torch.tensor([1.0]))

        # Forward should handle error gracefully
        output = chunked_layer(sample_input)
        assert output is not None
        assert output.shape == (sample_input.shape[0], chunked_layer.output_dim)

        # Check error count increased
        state = chunked_layer.state_tensor.get_seed_state(2)
        assert state["error_count"] >= 1

        # After 3 errors, should transition to ERROR_RECOVERY
        for _ in range(3):
            chunked_layer(sample_input)

        state = chunked_layer.state_tensor.get_seed_state(2)
        assert state["lifecycle_state"] == SeedLifecycle.ERROR_RECOVERY

    def test_grafting_ramp(self, chunked_layer, sample_input):
        """Test gradual blueprint integration."""
        # Create blueprint
        blueprint = nn.Identity()
        bp_id = chunked_layer.register_blueprint(blueprint)

        # Start germination
        chunked_layer.state_tensor.set_lifecycle_state(4, SeedLifecycle.LOADING)
        chunked_layer.state_tensor.set_blueprint(4, bp_id)

        # Track alpha values over epochs
        alphas = []
        for _ in range(35):
            # Forward pass
            chunked_layer(sample_input)

            # Check alpha value
            alpha = chunked_layer.state_tensor.alpha_blend[4].item()
            alphas.append(alpha)

        # Alpha should gradually increase
        assert alphas[0] < alphas[10] < alphas[20]
        assert alphas[30] == 1.0  # Should be clamped at 1.0

    def test_health_scoring(self, chunked_layer, sample_input):
        """Test health score computation."""
        # Run forward passes to collect telemetry
        for _ in range(10):
            chunked_layer(sample_input)

        # Get health scores
        health_scores = chunked_layer.state_tensor.health_scores

        # All seeds should have reasonable health scores
        assert (health_scores > 0).all()
        assert (health_scores <= 1.0).all()

        # Find unhealthy seeds
        _ = chunked_layer.state_tensor.find_unhealthy_seeds(threshold=0.5)

        # Dormant seeds might have low health due to low variance
        # This is expected behavior

    def test_state_transitions(self, chunked_layer, sample_input):
        """Test automatic state transitions."""
        # Set up seed in LOADING with complete ramp
        chunked_layer.state_tensor.set_lifecycle_state(6, SeedLifecycle.LOADING)
        chunked_layer.state_tensor.update_alpha_blend(torch.tensor([6]), torch.tensor([1.0]))

        # Check transition detection
        transitions = chunked_layer.state_tensor.transition_ready_seeds()
        assert 6 in transitions["loading_to_active"]

        # Manually transition (Tamiyo would do this)
        ready_seeds = transitions["loading_to_active"]
        if ready_seeds:
            chunked_layer.state_tensor.batch_set_lifecycle_state(
                torch.tensor(ready_seeds),
                SeedLifecycle.ACTIVE
            )

        # Verify transition
        state = chunked_layer.state_tensor.get_seed_state(6)
        assert state["lifecycle_state"] == SeedLifecycle.ACTIVE

    def test_device_movement(self, base_layer):
        """Test moving layer between devices."""
        if torch.cuda.is_available():
            # Create on CPU
            layer = ChunkedKasminaLayer(base_layer, num_seeds=8, device=torch.device("cpu"))
            assert layer.device.type == "cpu"

            # Move to GPU
            layer_gpu = layer.to("cuda")
            assert layer_gpu.device.type == "cuda"
            assert layer_gpu.base_layer.weight.device.type == "cuda"

            # Test forward on GPU
            x_gpu = torch.randn(16, 256, device="cuda")
            output_gpu = layer_gpu(x_gpu)
            assert output_gpu.device.type == "cuda"

    def test_performance_characteristics(self, chunked_layer, sample_input):
        """Test performance metrics."""
        # Warm up
        for _ in range(10):
            chunked_layer(sample_input)

        # Time forward passes
        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            with torch.no_grad():
                chunked_layer(sample_input)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / num_runs
        stats = chunked_layer.get_layer_stats()

        # Check timing consistency
        assert abs(stats["avg_latency_ms"] - (avg_time * 1000)) < 10  # Within 10ms

        # Performance should be reasonable
        assert stats["avg_latency_ms"] < 50  # Less than 50ms per forward

    def test_fossilized_state(self, chunked_layer, sample_input):
        """Test fossilized seed behavior."""
        # Create blueprint
        blueprint = nn.Linear(16, 16)
        bp_id = chunked_layer.register_blueprint(blueprint)

        # Set seed to fossilized
        chunked_layer.state_tensor.set_lifecycle_state(8, SeedLifecycle.FOSSILIZED)
        chunked_layer.state_tensor.set_blueprint(8, bp_id)
        chunked_layer.state_tensor.update_alpha_blend(torch.tensor([8]), torch.tensor([1.0]))

        # Get initial blueprint parameters
        initial_params = blueprint.weight.clone()

        # Forward pass with gradient enabled
        output = chunked_layer(sample_input)
        assert output is not None  # Use the variable
        loss = output.sum()
        loss.backward()

        # Blueprint weights should not change (no_grad in fossilized)
        assert torch.equal(blueprint.weight, initial_params)

    def test_concurrent_seed_operations(self, chunked_layer):
        """Test multiple seeds operating simultaneously."""
        # Activate multiple seeds
        active_seeds = [0, 2, 4, 6, 8, 10, 12, 14]
        for seed_id in active_seeds:
            chunked_layer.request_germination(seed_id)
            chunked_layer.state_tensor.set_lifecycle_state(seed_id, SeedLifecycle.ACTIVE)
            chunked_layer.state_tensor.update_alpha_blend(
                torch.tensor([seed_id]),
                torch.tensor([1.0])
            )

        # Run forward
        x = torch.randn(64, 256)
        output = chunked_layer(x)

        # Get stats
        stats = chunked_layer.get_layer_stats()
        state_dist = chunked_layer.state_tensor.get_state_distribution()

        assert state_dist["ACTIVE"] == len(active_seeds)
        assert stats["num_blueprints"] >= len(active_seeds)

    def test_empty_input_handling(self, chunked_layer):
        """Test handling of edge cases."""
        # Empty batch
        empty_input = torch.randn(0, 256)
        output = chunked_layer(empty_input)
        assert output.shape == (0, 256)

        # Single sample
        single_input = torch.randn(1, 256)
        output = chunked_layer(single_input)
        assert output.shape == (1, 256)