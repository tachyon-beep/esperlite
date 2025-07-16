"""
Integration tests for Phase 4: Full System Orchestration

These tests validate the complete morphogenetic training system,
including the Tolaria trainer, service orchestration, and end-to-end workflows.
"""

import asyncio
import pytest
import tempfile
import yaml
import warnings
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Suppress torchvision deprecation warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

from esper.services.tolaria.config import TolariaConfig
from esper.services.tolaria.trainer import TolariaTrainer, TrainingState
from esper.services.tolaria.main import TolariaService


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for test configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def minimal_config(temp_config_dir):
    """Create a minimal configuration for testing."""
    config_data = {
        "run_id": "test-run",
        "max_epochs": 2,
        "learning_rate": 0.01,
        "batch_size": 32,
        "device": "cpu",
        "compile_model": False,
        "model": {
            "architecture": "resnet18",
            "num_classes": 10,
            "pretrained": False,
            "target_layers": ["linear"],  # Use standard layer type name
            "seeds_per_layer": 2,
            "seed_cache_size_mb": 64
        },
        "dataset": {
            "name": "cifar10",
            "data_dir": str(temp_config_dir / "data"),
            "download": True,  # Enable download for integration tests
            "val_split": 0.2
        },
        "optimizer": "adam",
        "weight_decay": 0.001,
        "scheduler": "none",
        "checkpoint_dir": str(temp_config_dir / "checkpoints"),
        "checkpoint_frequency": 1,
        "adaptation_frequency": 5,
        "adaptation_cooldown": 1,
        "max_adaptations_per_epoch": 1,
        "oona": {
            "url": "http://localhost:8001",
            "timeout": 5.0
        }
    }
    
    config_path = temp_config_dir / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    return TolariaConfig.from_yaml(config_path)


class TestTolariaTrainer:
    """Test the core TolariaTrainer functionality."""
    
    def test_trainer_initialization(self, minimal_config):
        """Test that trainer initializes correctly."""
        trainer = TolariaTrainer(minimal_config)
        
        assert trainer.config == minimal_config
        assert trainer.run_id is not None
        assert trainer.device.type == "cpu"
        assert not trainer.running
        assert trainer.state.epoch == 0
    
    def test_device_setup(self, minimal_config):
        """Test device setup logic."""
        # Test explicit CPU device
        minimal_config.device = "cpu"
        trainer = TolariaTrainer(minimal_config)
        assert trainer.device.type == "cpu"
        
        # Test auto device selection
        minimal_config.device = "auto"
        trainer = TolariaTrainer(minimal_config)
        assert trainer.device.type in ["cpu", "cuda"]
    
    @patch('torch.utils.data.DataLoader')
    @patch('torchvision.datasets.CIFAR10')
    def test_data_loader_setup(self, mock_cifar10, mock_dataloader, minimal_config):
        """Test data loader setup."""
        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=1000)
        mock_cifar10.return_value = mock_dataset
        
        trainer = TolariaTrainer(minimal_config)
        trainer._setup_data_loaders()
        
        # Verify CIFAR-10 dataset was created
        assert mock_cifar10.call_count == 2  # Train and val datasets
        
        # Verify data loaders were created
        assert trainer.train_loader is not None
        assert trainer.val_loader is not None
    
    @patch('esper.services.tolaria.trainer.wrap')
    @patch('torchvision.models.resnet18')
    def test_model_setup(self, mock_resnet18, mock_wrap, minimal_config):
        """Test model setup and wrapping."""
        # Mock base model with proper PyTorch module interface
        mock_base_model = Mock()
        mock_base_model.conv1 = Mock()
        mock_base_model.maxpool = Mock()
        mock_base_model.fc = Mock()
        mock_base_model.fc.in_features = 512
        # Mock named_children to return empty iterable (no children to iterate)
        mock_base_model.named_children = Mock(return_value=iter([]))
        mock_resnet18.return_value = mock_base_model
        
        # Mock wrapped model
        mock_wrapped_model = Mock()
        mock_wrapped_model.to = Mock(return_value=mock_wrapped_model)
        mock_wrapped_model.parameters = Mock(return_value=[])
        mock_wrap.return_value = mock_wrapped_model
        
        trainer = TolariaTrainer(minimal_config)
        trainer._setup_model()
        
        # Verify model was created and wrapped
        mock_resnet18.assert_called_once()
        mock_wrap.assert_called_once()
        assert trainer.model is not None
    
    def test_optimizer_setup(self, minimal_config):
        """Test optimizer setup."""
        import torch
        trainer = TolariaTrainer(minimal_config)
        
        # Mock model with real tensor parameters
        mock_model = Mock()
        # Create real tensor parameters for the optimizer
        dummy_param = torch.tensor([1.0], requires_grad=True)
        mock_model.parameters = Mock(return_value=[dummy_param])
        trainer.model = mock_model
        
        trainer._setup_optimizer()
        
        assert trainer.optimizer is not None
        assert trainer.optimizer.__class__.__name__ == "Adam"
    
    def test_scheduler_setup(self, minimal_config):
        """Test learning rate scheduler setup."""
        import torch
        trainer = TolariaTrainer(minimal_config)
        
        # Create real optimizer for scheduler testing
        dummy_param = torch.tensor([1.0], requires_grad=True)
        trainer.optimizer = torch.optim.Adam([dummy_param], lr=0.01, weight_decay=0.0001)
        
        # Test no scheduler
        minimal_config.scheduler = "none"
        trainer._setup_scheduler()
        assert trainer.scheduler is None
        
        # Test cosine scheduler
        minimal_config.scheduler = "cosine"
        trainer._setup_scheduler()
        assert trainer.scheduler is not None
    
    def test_health_signal_collection(self, minimal_config):
        """Test health signal collection."""
        trainer = TolariaTrainer(minimal_config)
        
        # Mock layer with health metrics
        mock_layer = Mock()
        mock_layer.get_health_metrics = Mock(return_value={
            'activation_variance': 0.5,
            'dead_neuron_ratio': 0.1,
            'avg_correlation': 0.3,
            'health_score': 0.9
        })
        
        # Create a simple mock model with kasmina_layers attribute
        from types import SimpleNamespace
        mock_model = SimpleNamespace()
        mock_model.kasmina_layers = {'layer1': mock_layer}
        
        trainer.model = mock_model
        trainer.state.epoch = 10
        
        signals = trainer._collect_health_signals()
        
        assert len(signals) == 1
        assert signals[0].layer_id == 0  # Enumeration index from items()
        assert signals[0].epoch == 10
        assert abs(signals[0].activation_variance - 0.5) < 1e-6


class TestTolariaService:
    """Test the TolariaService orchestration."""
    
    @patch('esper.services.tolaria.main.TolariaTrainer')
    def test_service_initialization(self, mock_trainer_class, minimal_config):
        """Test service initialization."""
        service = TolariaService(minimal_config)
        
        assert service.config == minimal_config
        assert not service.running
        assert service.trainer is None
    
    @patch('esper.services.tolaria.main.TolariaTrainer')
    def test_health_check(self, mock_trainer_class, minimal_config):
        """Test service health check."""
        service = TolariaService(minimal_config)
        
        # Test when not running
        status = service.health_check()
        assert status['service'] == 'tolaria'
        assert status['status'] == 'stopped'
        assert not status['trainer_running']
        
        # Test when running
        service.running = True
        mock_trainer_instance = mock_trainer_class.return_value
        mock_trainer_instance.running = True
        mock_trainer_instance.get_training_state.return_value = TrainingState(
            epoch=5,
            total_adaptations=2,
            best_val_accuracy=0.85
        )
        service.trainer = mock_trainer_instance
        
        status = service.health_check()
        assert status['status'] == 'healthy'
        assert status['trainer_running']
        assert status['current_epoch'] == 5
        assert status['total_adaptations'] == 2
        assert abs(status['best_val_accuracy'] - 0.85) < 1e-6
    
    @pytest.mark.asyncio
    @patch('esper.services.tolaria.main.TolariaTrainer')
    async def test_service_shutdown(self, mock_trainer_class, minimal_config):
        """Test service shutdown."""
        service = TolariaService(minimal_config)
        
        # Mock trainer
        mock_trainer_instance = AsyncMock()
        service.trainer = mock_trainer_instance
        service.running = True
        
        await service.shutdown()
        
        assert not service.running
        mock_trainer_instance.shutdown.assert_called_once()


class TestFullSystemIntegration:
    """Test complete system integration scenarios."""
    
    @pytest.mark.asyncio
    @patch('esper.services.tolaria.trainer.TolariaTrainer.train')
    @patch('esper.services.tolaria.trainer.TolariaTrainer.initialize')
    async def test_complete_training_cycle(self, mock_initialize, mock_train, minimal_config):
        """Test a complete training cycle from start to finish."""
        # Mock trainer methods with REAL data structures that logging can handle
        mock_initialize.return_value = None
        
        # Create real TrainingMetrics objects instead of Mock objects
        from types import SimpleNamespace
        epoch0_metrics = SimpleNamespace(
            epoch=0, train_loss=1.0, train_accuracy=0.6, val_loss=0.9, val_accuracy=0.7
        )
        epoch1_metrics = SimpleNamespace(
            epoch=1, train_loss=0.8, train_accuracy=0.75, val_loss=0.7, val_accuracy=0.8
        )
        mock_train.return_value = [epoch0_metrics, epoch1_metrics]
        
        service = TolariaService(minimal_config)
        
        # This should complete without errors
        await service.start()
        
        # Verify initialization and training were called
        mock_initialize.assert_called_once()
        mock_train.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('esper.services.tolaria.trainer.TolariaTrainer.initialize')
    async def test_training_failure_handling(self, mock_initialize, minimal_config):
        """Test handling of training failures."""
        # Mock initialization failure
        mock_initialize.side_effect = RuntimeError("Initialization failed")
        
        service = TolariaService(minimal_config)
        
        with pytest.raises(RuntimeError, match="Initialization failed"):
            await service.start()
    
    @pytest.mark.asyncio
    @patch('esper.services.tolaria.trainer.TolariaTrainer.train')
    @patch('esper.services.tolaria.trainer.TolariaTrainer._setup_data_loaders')
    async def test_graceful_shutdown_on_signal(self, mock_setup_data, mock_train, minimal_config):
        """Test that service handles cancellation gracefully with real components."""
        # Mock expensive operations but use real service and trainer instances
        async def hang_forever():
            await asyncio.sleep(100)  # Long-running task
            
        mock_train.side_effect = hang_forever
        mock_setup_data.return_value = None  # Skip dataset download
        
        service = TolariaService(minimal_config)
        
        # Start service with real TolariaTrainer instance
        service_task = asyncio.create_task(service.start())
        await asyncio.sleep(0.1)  # Let service start
        
        # Verify real integration: service created real trainer
        assert service.trainer is not None
        assert isinstance(service.trainer, TolariaTrainer)
        assert service.running is True
        
        # Cancel the task
        service_task.cancel()
        
        with pytest.raises(asyncio.CancelledError):
            await service_task
        
        # Verify real cleanup occurred
        assert not service.running


class TestConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_config_from_yaml(self, temp_config_dir):
        """Test loading configuration from YAML."""
        config_data = {
            "run_id": "test",
            "max_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 64,
            "device": "cpu",
            "model": {
                "architecture": "resnet18",
                "num_classes": 10
            },
            "dataset": {
                "name": "cifar10",
                "data_dir": "/tmp"
            },
            "optimizer": "adam"
        }
        
        config_path = temp_config_dir / "valid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        config = TolariaConfig.from_yaml(config_path)
        
        assert config.run_id == "test"
        assert config.max_epochs == 10
        assert config.model.architecture == "resnet18"
    
    def test_invalid_config_validation(self, temp_config_dir):
        """Test validation of invalid configurations."""
        # Invalid adaptation_frequency (must be positive)
        invalid_config = {
            "adaptation_frequency": -1  # This should trigger validation error
        }
        
        config_path = temp_config_dir / "invalid_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValueError, match="adaptation_frequency must be positive"):
            TolariaConfig.from_yaml(config_path)


class TestMorphogeneticAdaptation:
    """Test morphogenetic adaptation workflows."""
    
    def test_adaptation_decision_processing(self, minimal_config):
        """Test processing of adaptation decisions."""
        trainer = TolariaTrainer(minimal_config)
        
        # Create model with kasmina layers containing the target layer
        mock_layer = Mock()
        from types import SimpleNamespace
        mock_model = SimpleNamespace()
        mock_model.kasmina_layers = {'layer1': mock_layer}
        trainer.model = mock_model
        
        # Mock adaptation decision
        from esper.contracts.operational import AdaptationDecision
        decision = AdaptationDecision(
            layer_name="layer1",
            adaptation_type="kernel_replacement",
            confidence=0.8,
            urgency=0.6
        )
        
        # This should work without patching since we're using SimpleNamespace
        result = asyncio.run(trainer._apply_adaptation(decision))
        
        # With our current implementation, this should return True (simulated success)
        assert result is True
    
    def test_adaptation_frequency_control(self, minimal_config):
        """Test adaptation frequency and cooldown controls."""
        trainer = TolariaTrainer(minimal_config)
        
        # Set up state
        trainer.state.epoch = 10
        trainer.state.last_adaptation_epoch = 8
        trainer.state.adaptations_this_epoch = 0
        
        # Mock methods
        trainer._collect_health_signals = Mock(return_value=[])
        trainer._consult_tamiyo = AsyncMock(return_value=None)
        
        # Should not trigger adaptation due to cooldown
        minimal_config.adaptation_cooldown = 5
        asyncio.run(trainer._handle_end_of_epoch())
        
        trainer._collect_health_signals.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
