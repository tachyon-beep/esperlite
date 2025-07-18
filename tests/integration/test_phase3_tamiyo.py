"""
Integration tests for Phase 3 - Tamiyo Strategic Controller.

This module contains tests that verify the end-to-end functionality
of the Tamiyo strategic controller and policy training system.
"""

import pytest
import torch
import tempfile
import os
import warnings
from pathlib import Path
from unittest.mock import Mock

# Suppress torch_geometric warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")

from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from esper.services.tamiyo.training import TamiyoTrainer, TrainingConfig
from esper.services.tamiyo.analyzer import ModelGraphAnalyzer
from esper.services.tamiyo.main import TamiyoService
from esper.contracts.operational import HealthSignal, AdaptationDecision


class TestTamiyoPolicyGNN:
    """Test the GNN policy model."""
    
    def test_policy_initialization(self):
        """Test policy model initialization."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        
        assert policy.config == config
        assert len(list(policy.parameters())) > 0
    
    def test_policy_forward_pass(self):
        """Test policy forward pass."""
        config = PolicyConfig(node_feature_dim=64, hidden_dim=64)
        policy = TamiyoPolicyGNN(config)
        
        # Create test graph data
        num_nodes = 3
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        adaptation_prob, layer_priorities, urgency_score, value_estimate = policy(
            node_features, edge_index
        )
        
        # Verify output shapes and ranges
        assert adaptation_prob.shape == torch.Size([1])
        assert 0 <= adaptation_prob.item() <= 1
        assert layer_priorities.shape == torch.Size([1, 1])
        assert 0 <= urgency_score.item() <= 1
        assert value_estimate.shape == torch.Size([1, 1])
    
    def test_policy_acceleration_status(self):
        """Verify policy reports acceleration status correctly."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        
        status = policy.acceleration_status
        assert isinstance(status["torch_scatter_available"], bool)
        assert isinstance(status["acceleration_enabled"], bool)
        assert isinstance(status["fallback_mode"], bool)
        
        # Logical consistency
        assert status["acceleration_enabled"] == status["torch_scatter_available"]
        assert status["fallback_mode"] == (not status["torch_scatter_available"])
    
    def test_make_decision(self):
        """Test decision making functionality."""
        # Use lower confidence threshold for testing with untrained network
        config = PolicyConfig(adaptation_confidence_threshold=0.4)
        policy = TamiyoPolicyGNN(config)
        
        # Mock model state
        mock_state = Mock()
        layer_health = {"layer1": 0.2, "layer2": 0.8}  # layer1 is unhealthy
        
        decision = policy.make_decision(mock_state, layer_health)
        
        # Should make a decision for unhealthy layer
        assert decision is not None
        assert decision.layer_name == "layer1"
        assert 0 <= decision.confidence <= 1
        assert 0 <= decision.urgency <= 1


class TestModelGraphAnalyzer:
    """Test the model graph analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        analyzer = ModelGraphAnalyzer()
        
        assert analyzer.health_history_window == 10
        assert len(analyzer.health_history) == 0
    
    def test_health_trend_calculation(self):
        """Test health trend calculation."""
        analyzer = ModelGraphAnalyzer()
        
        # Add some health history
        analyzer.health_history["layer1"] = [0.8, 0.7, 0.6, 0.5, 0.4]
        
        trend = analyzer._calculate_layer_trend("layer1")
        assert trend < 0  # Should be negative (declining health)
    
    def test_problematic_layer_identification(self):
        """Test identification of problematic layers."""
        analyzer = ModelGraphAnalyzer()
        
        # Create health signals
        health_signals = {
            "layer1": HealthSignal(
                layer_id=1, seed_id=1, chunk_id=1, epoch=1,
                activation_variance=0.1, dead_neuron_ratio=0.1, avg_correlation=0.5,
                health_score=0.2,  # Low health
                error_count=10     # High errors
            ),
            "layer2": HealthSignal(
                layer_id=2, seed_id=2, chunk_id=1, epoch=1,
                activation_variance=0.2, dead_neuron_ratio=0.05, avg_correlation=0.7,
                health_score=0.9,  # High health
                error_count=0      # No errors
            )
        }
        
        problematic = analyzer._identify_problematic_layers(health_signals)
        
        assert "layer1" in problematic
        assert "layer2" not in problematic


class TestTamiyoTrainer:
    """Test the policy trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        training_config = TrainingConfig()
        
        trainer = TamiyoTrainer(policy, training_config)
        
        assert trainer.policy == policy
        assert trainer.config == training_config
        assert trainer.training_step == 0
    
    def test_experience_data_save_load(self):
        """Test saving and loading experience data."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = TrainingConfig(
                training_data_path=str(Path(temp_dir) / "test_data.pkl")
            )
            trainer = TamiyoTrainer(policy, training_config)
            
            # Create test experience data
            test_data = [
                {"state": {"health": 0.5}, "action": True, "reward": 1.0},
                {"state": {"health": 0.8}, "action": False, "reward": 0.1}
            ]
            
            # Save and load
            trainer.save_experience_data(test_data)
            loaded_data = trainer.load_experience_data()
            
            assert len(loaded_data) == 2
            assert loaded_data == test_data
    
    def test_checkpoint_save_load(self):
        """Test model checkpoint save/load."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = TrainingConfig(
                model_save_path=str(Path(temp_dir) / "test_model.pt")
            )
            trainer = TamiyoTrainer(policy, training_config)
            
            # Get initial state
            initial_state = {k: v.clone() for k, v in policy.state_dict().items()}
            
            # Save checkpoint
            trainer._save_checkpoint(0, 0.5)
            
            # Modify model weights
            with torch.no_grad():
                for param in policy.parameters():
                    param.fill_(0.999)
            
            # Load checkpoint
            checkpoint_path = training_config.model_save_path
            trainer.load_checkpoint(checkpoint_path)
            
            # Verify weights restored
            loaded_state = policy.state_dict()
            for key in initial_state:
                assert torch.allclose(initial_state[key], loaded_state[key], atol=1e-6)


class TestTamiyoService:
    """Test the Tamiyo service."""
    
    def test_service_initialization(self):
        """Test service initialization."""
        mock_oona = Mock()
        
        service = TamiyoService(
            analysis_interval=5.0,
            adaptation_cooldown=30.0,
            policy_config=PolicyConfig(),
            oona_client=mock_oona,
            urza_client=None
        )
        
        assert service.is_running is False
        assert abs(service.analysis_interval - 5.0) < 1e-6
        assert abs(service.adaptation_cooldown - 30.0) < 1e-6
        assert service.policy is not None
        assert service.analyzer is not None
    
    def test_service_status(self):
        """Test service status reporting."""
        mock_oona = Mock()
        
        service = TamiyoService(
            analysis_interval=5.0,
            adaptation_cooldown=30.0,
            policy_config=PolicyConfig(),
            oona_client=mock_oona,
            urza_client=None
        )
        
        status = service.get_status()
        
        assert "is_running" in status
        assert "current_health_signals" in status
        assert "total_adaptations" in status
        assert status["is_running"] is False
        assert status["total_adaptations"] == 0


class TestIntegrationWorkflow:
    """Test end-to-end integration workflows."""
    
    def test_synthetic_training_workflow(self):
        """Test the complete training workflow with synthetic data."""
        # This test replicates what the training script does
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            training_config = TrainingConfig(
                num_epochs=2,
                batch_size=4,
                model_save_path=str(Path(temp_dir) / "test_policy.pt"),
                training_data_path=str(Path(temp_dir) / "test_data.pkl")
            )
            
            trainer = TamiyoTrainer(policy, training_config)
            
            # Create synthetic experience data
            experience_data = []
            for i in range(20):  # Need at least batch_size samples
                experience_data.append({
                    "state": {"health_score": 0.5, "latency": 0.01, "error_count": 1},
                    "action": i % 2 == 0,  # Alternate actions
                    "reward": 0.5 if i % 2 == 0 else 0.1,
                    "next_state": {"health_score": 0.6, "latency": 0.01, "error_count": 0},
                    "timestamp": float(i)
                })
            
            # Train the policy
            metrics = trainer.train_from_experience(experience_data)
            
            # Verify training completed
            assert "final_train_loss" in metrics
            assert "final_val_loss" in metrics
            assert "total_epochs" in metrics
            assert metrics["total_epochs"] == 2
            
            # Verify model was saved
            assert Path(training_config.model_save_path).exists()
    
    def test_policy_inference_performance(self):
        """Test policy inference performance."""
        config = PolicyConfig()
        policy = TamiyoPolicyGNN(config)
        policy.eval()
        
        # Create test data
        num_nodes = 10
        node_features = torch.randn(num_nodes, config.node_feature_dim)
        edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t()
        
        # Measure inference time
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                policy(node_features, edge_index)
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / 100
        
        # Should be fast enough for real-time decision making
        assert avg_inference_time < 0.1  # Less than 100ms per inference
        print(f"Average inference time: {avg_inference_time*1000:.2f}ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
