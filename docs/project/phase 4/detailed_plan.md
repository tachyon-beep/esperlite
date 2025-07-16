# **Phase 4 Implementation Plan: Full System Orchestration**

**Objective:** Implement the final integration layer that coordinates all Esper components into a cohesive autonomous morphogenetic training system.

**Key Components to Implement:** Tolaria Training Orchestrator, Main System Entrypoint, End-to-End Integration Tests, System Configuration Framework

**Timeline:** 3 weeks

-----

## **1. Tolaria Training Orchestrator: The Master Conductor**

**Task:** Implement the training orchestrator that coordinates model training with morphogenetic adaptations.

### **1.1. Core Trainer Implementation (`src/esper/services/tolaria/trainer.py`)**

Implement the main training orchestrator class that manages the complete training lifecycle.

```python
"""
Tolaria Training Orchestrator - Master training loop coordinator.

This module implements the core training orchestrator that manages model training,
coordinates with Tamiyo for strategic decisions, and handles morphogenetic adaptations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from esper.services.oona_client import OonaClient
from esper.contracts.messages import OonaMessage, TopicNames
from esper.contracts.operational import HealthSignal, AdaptationDecision, TrainingMetrics
from esper.contracts.enums import SeedLifecycleState
from esper.core.model_wrapper import MorphableModel
from esper.services.tamiyo.main import TamiyoService
from esper.configs import TolariaConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    """Current state of the training process."""
    epoch: int
    step: int
    global_step: int
    best_loss: float
    best_accuracy: float
    adaptations_this_epoch: int
    total_adaptations: int
    last_adaptation_epoch: int


class TolariaTrainer:
    """
    Master training orchestrator for morphogenetic models.
    
    This class coordinates all aspects of morphogenetic training:
    - Standard training loop execution
    - Integration with Tamiyo strategic controller
    - Morphogenetic adaptation lifecycle management
    - Model checkpointing and state persistence
    - Performance monitoring and metrics collection
    """

    def __init__(self, config: TolariaConfig):
        self.config = config
        self.training_state = TrainingState(
            epoch=0, step=0, global_step=0,
            best_loss=float('inf'), best_accuracy=0.0,
            adaptations_this_epoch=0, total_adaptations=0,
            last_adaptation_epoch=-1
        )
        
        # Initialize components
        self.model: Optional[MorphableModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Services
        self.oona_client: Optional[OonaClient] = None
        self.tamiyo_service: Optional[TamiyoService] = None
        
        # Training state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.running = False

    async def initialize(self) -> None:
        """Initialize all training components and services."""
        
    async def train(self) -> TrainingMetrics:
        """Execute the complete training process."""
        
    async def train_epoch(self) -> Dict[str, float]:
        """Execute a single training epoch with morphogenetic hooks."""
        
    async def validate_epoch(self) -> Dict[str, float]:
        """Execute validation and return metrics."""
        
    async def handle_end_of_epoch(self) -> None:
        """Handle end-of-epoch processing including Tamiyo consultation."""
        
    async def handle_adaptation_signal(self, signal: AdaptationDecision) -> bool:
        """Process an adaptation decision from Tamiyo."""
        
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """Save training checkpoint including model and Tamiyo state."""
        
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint and restore state."""
        
    async def shutdown(self) -> None:
        """Gracefully shutdown the trainer and all services."""
```

### **1.2. Configuration System (`src/esper/configs/tolaria.py`)**

Implement comprehensive configuration for the training orchestrator.

```python
"""
Tolaria configuration system.

This module defines the configuration structure for the Tolaria training orchestrator,
including training parameters, morphogenetic settings, and service configurations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

from esper.configs import DatabaseConfig, RedisConfig, StorageConfig


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    name: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = False
    
    # Morphogenetic settings
    target_layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"])
    seeds_per_layer: int = 4
    max_kernels_per_seed: int = 3


@dataclass
class DatasetConfig:
    """Configuration for dataset and data loading."""
    name: str = "cifar10"
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = True
    normalize: bool = True
    
    # Validation split
    val_split: float = 0.1
    shuffle: bool = True


@dataclass
class OptimizationConfig:
    """Configuration for optimization and training."""
    optimizer: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    
    # Learning rate scheduling
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    
    # Training parameters
    max_epochs: int = 100
    gradient_clip_norm: float = 1.0
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class MorphogeneticConfig:
    """Configuration for morphogenetic adaptation."""
    
    # Tamiyo integration
    tamiyo_policy_path: str = "models/tamiyo_policy.pt"
    enable_tamiyo: bool = True
    
    # Adaptation control
    adaptation_frequency: int = 1  # epochs between adaptation checks
    max_adaptations_per_epoch: int = 2
    adaptation_cooldown: int = 3  # epochs between adaptations for same layer
    
    # Thresholds
    health_threshold: float = 0.3
    improvement_threshold: float = 0.02
    
    # Blueprint selection
    blueprint_categories: List[str] = field(default_factory=lambda: ["residual", "attention", "normalization"])
    max_blueprint_complexity: int = 5


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Metrics logging
    log_interval: int = 100  # steps
    eval_interval: int = 1000  # steps
    save_interval: int = 5  # epochs
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: str = "esper-morphogenetic"
    wandb_entity: Optional[str] = None


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    checkpoint_dir: str = "./checkpoints"
    save_best: bool = True
    save_last: bool = True
    save_interval: int = 10  # epochs
    
    # Checkpoint retention
    max_checkpoints: int = 5
    
    # Resume settings
    resume_from: Optional[str] = None
    auto_resume: bool = True


@dataclass
class TolariaConfig:
    """
    Complete configuration for Tolaria training orchestrator.
    
    This configuration class brings together all aspects of morphogenetic training,
    from model architecture to service integration to monitoring.
    """
    
    # Core configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    morphogenetic: MorphogeneticConfig = field(default_factory=MorphogeneticConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Service configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    
    # Experiment metadata
    experiment_name: str = "morphogenetic_training"
    description: str = "Esper morphogenetic training experiment"
    tags: List[str] = field(default_factory=list)
    
    # System settings
    device: str = "auto"  # auto, cpu, cuda
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile
    
    # Debugging
    debug_mode: bool = False
    profile_training: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure directories exist
        Path(self.logging.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.dataset.data_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate device setting
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Validate paths
        if self.morphogenetic.enable_tamiyo:
            policy_path = Path(self.morphogenetic.tamiyo_policy_path)
            if not policy_path.exists() and not self.debug_mode:
                logger.warning(f"Tamiyo policy not found at {policy_path}, will create default policy")
```

### **1.3. Service Integration (`src/esper/services/tolaria/main.py`)**

Implement the main service class that orchestrates Tolaria.

```python
"""
Tolaria Service - Main orchestration service.

This module implements the main Tolaria service that coordinates training
with the broader Esper ecosystem.
"""

import asyncio
import logging
import signal
from typing import Optional
from pathlib import Path

from esper.services.tolaria.trainer import TolariaTrainer, TolariaConfig
from esper.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class TolariaService:
    """
    Main Tolaria service for orchestrating morphogenetic training.
    
    This service manages the complete lifecycle of a training run,
    including initialization, execution, and graceful shutdown.
    """

    def __init__(self, config: TolariaConfig):
        self.config = config
        self.trainer: Optional[TolariaTrainer] = None
        self.running = False

    async def start(self) -> None:
        """Start the Tolaria service and begin training."""
        
    async def stop(self) -> None:
        """Stop the service gracefully."""
        
    async def run_training(self) -> None:
        """Execute the complete training run."""
        
    def setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        
    async def health_check(self) -> Dict[str, Any]:
        """Return service health status."""


async def main(config_path: str) -> None:
    """Main entry point for Tolaria service."""
    # Load configuration
    config = load_config(config_path, TolariaConfig)
    
    # Setup logging
    setup_logging(config.logging)
    
    # Create and start service
    service = TolariaService(config)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    finally:
        await service.stop()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python -m esper.services.tolaria.main <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    asyncio.run(main(config_path))
```

## **2. Main System Entrypoint**

**Task:** Create the single command interface for launching complete Esper training runs.

### **2.1. Main Training Script (`train.py`)**

```python
#!/usr/bin/env python3
"""
Esper Morphogenetic Training - Main Entrypoint

This script provides a single command interface for launching complete
morphogenetic training experiments with the Esper platform.

Usage:
    python train.py --config configs/cifar10_experiment.yaml
    python train.py --model resnet18 --dataset cifar10 --epochs 50 --morphogenetic
    python train.py --config configs/baseline.yaml --no-morphogenetic
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from esper.services.tolaria.trainer import TolariaTrainer
from esper.services.tolaria.main import TolariaService
from esper.configs import load_config, TolariaConfig
from esper.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Esper Morphogenetic Training Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with configuration file
  python train.py --config configs/resnet18_cifar10.yaml
  
  # Quick training with command line args
  python train.py --model resnet18 --dataset cifar10 --epochs 50
  
  # Baseline training without morphogenetic features
  python train.py --config configs/baseline.yaml --no-morphogenetic
  
  # Resume from checkpoint
  python train.py --config configs/experiment.yaml --resume checkpoints/latest.pt
  
  # Debug mode with verbose logging
  python train.py --config configs/debug.yaml --verbose --debug
        """
    )
    
    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument(
        '--config', type=str, 
        help='Path to YAML configuration file'
    )
    config_group.add_argument(
        '--output-dir', type=str, default='./experiments',
        help='Output directory for checkpoints and logs'
    )
    
    # Quick setup arguments
    quick_group = parser.add_argument_group('Quick Setup')
    quick_group.add_argument(
        '--model', type=str, choices=['resnet18', 'resnet34', 'resnet50'],
        help='Model architecture'
    )
    quick_group.add_argument(
        '--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet'],
        help='Dataset to use'
    )
    quick_group.add_argument(
        '--epochs', type=int, default=100,
        help='Number of training epochs'
    )
    quick_group.add_argument(
        '--batch-size', type=int, default=128,
        help='Training batch size'
    )
    quick_group.add_argument(
        '--learning-rate', type=float, default=1e-3,
        help='Learning rate'
    )
    
    # Morphogenetic settings
    morph_group = parser.add_argument_group('Morphogenetic Settings')
    morph_group.add_argument(
        '--morphogenetic', action='store_true', default=True,
        help='Enable morphogenetic features (default: True)'
    )
    morph_group.add_argument(
        '--no-morphogenetic', action='store_true',
        help='Disable morphogenetic features'
    )
    morph_group.add_argument(
        '--tamiyo-policy', type=str,
        help='Path to Tamiyo policy model'
    )
    morph_group.add_argument(
        '--max-adaptations', type=int, default=2,
        help='Maximum adaptations per epoch'
    )
    
    # Training control
    training_group = parser.add_argument_group('Training Control')
    training_group.add_argument(
        '--resume', type=str,
        help='Resume from checkpoint'
    )
    training_group.add_argument(
        '--device', type=str, choices=['auto', 'cpu', 'cuda'],
        default='auto', help='Device to use for training'
    )
    training_group.add_argument(
        '--mixed-precision', action='store_true',
        help='Use mixed precision training'
    )
    
    # Logging and debugging
    debug_group = parser.add_argument_group('Logging and Debugging')
    debug_group.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging'
    )
    debug_group.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )
    debug_group.add_argument(
        '--wandb', action='store_true',
        help='Enable Weights & Biases logging'
    )
    debug_group.add_argument(
        '--profile', action='store_true',
        help='Enable training profiling'
    )
    
    # Service management
    service_group = parser.add_argument_group('Service Management')
    service_group.add_argument(
        '--services-only', action='store_true',
        help='Start supporting services only (Urza, Tezzeret, etc.)'
    )
    service_group.add_argument(
        '--check-services', action='store_true',
        help='Check if supporting services are running'
    )
    
    return parser


def build_config_from_args(args: argparse.Namespace) -> TolariaConfig:
    """Build a TolariaConfig from command line arguments."""
    
    
def validate_environment() -> bool:
    """Validate that the training environment is properly configured."""
    
    
async def start_services_if_needed(config: TolariaConfig) -> bool:
    """Start supporting services if they're not already running."""
    
    
async def check_service_health(config: TolariaConfig) -> Dict[str, bool]:
    """Check the health of all required services."""
    

async def main() -> int:
    """Main entry point for Esper training."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle special modes
        if args.check_services:
            return await check_and_report_services()
        
        if args.services_only:
            return await start_services_only()
        
        # Load or build configuration
        if args.config:
            config = load_config(args.config, TolariaConfig)
            # Override with command line arguments if provided
            config = override_config_with_args(config, args)
        else:
            config = build_config_from_args(args)
        
        # Setup logging
        setup_logging(
            level="DEBUG" if args.verbose else "INFO",
            log_dir=Path(config.logging.log_dir)
        )
        
        logger.info("Starting Esper Morphogenetic Training")
        logger.info(f"Configuration: {config.experiment_name}")
        logger.info(f"Model: {config.model.name}, Dataset: {config.dataset.name}")
        logger.info(f"Morphogenetic: {'Enabled' if config.morphogenetic.enable_tamiyo else 'Disabled'}")
        
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            return 1
        
        # Check and start services
        if not await start_services_if_needed(config):
            logger.error("Failed to start required services")
            return 1
        
        # Create and run training service
        service = TolariaService(config)
        await service.start()
        
        logger.info("Training completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

### **2.2. Configuration Templates (`configs/`)**

Create example configuration files for different training scenarios.

```yaml
# configs/resnet18_cifar10_morphogenetic.yaml
experiment_name: "resnet18_cifar10_morphogenetic"
description: "ResNet-18 on CIFAR-10 with morphogenetic adaptations"

model:
  name: "resnet18"
  num_classes: 10
  pretrained: false
  target_layers: ["layer1", "layer2", "layer3", "layer4"]
  seeds_per_layer: 4

dataset:
  name: "cifar10"
  data_dir: "./data"
  batch_size: 128
  use_augmentation: true

optimization:
  optimizer: "adamw"
  learning_rate: 1e-3
  max_epochs: 100
  scheduler: "cosine"

morphogenetic:
  enable_tamiyo: true
  tamiyo_policy_path: "models/tamiyo_policy.pt"
  adaptation_frequency: 1
  max_adaptations_per_epoch: 2

logging:
  log_level: "INFO"
  use_wandb: false
```

## **3. End-to-End Integration Tests**

**Task:** Create comprehensive tests that validate the complete morphogenetic training lifecycle.

### **3.1. Full System Integration Tests (`tests/integration/test_phase4_full_system.py`)**

```python
"""
Full system integration tests for Phase 4.

This module contains comprehensive tests that validate the complete Esper
morphogenetic training system end-to-end.
"""

import asyncio
import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import yaml

from esper.services.tolaria.trainer import TolariaTrainer, TolariaConfig
from esper.services.tolaria.main import TolariaService
from esper.core.model_wrapper import wrap
from esper.contracts.operational import AdaptationDecision


class TestFullSystemIntegration:
    """Integration tests for the complete Esper system."""

    @pytest.fixture
    async def temp_workspace(self):
        """Create a temporary workspace for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def basic_config(self, temp_workspace):
        """Create a basic configuration for testing."""
        return TolariaConfig(
            model=ModelConfig(name="resnet18", num_classes=10),
            dataset=DatasetConfig(name="cifar10", batch_size=32),
            optimization=OptimizationConfig(max_epochs=2, learning_rate=1e-3),
            morphogenetic=MorphogeneticConfig(enable_tamiyo=False),  # Start simple
            logging=LoggingConfig(log_dir=str(temp_workspace / "logs")),
            checkpoint=CheckpointConfig(checkpoint_dir=str(temp_workspace / "checkpoints"))
        )

    async def test_basic_training_loop(self, basic_config):
        """Test that basic training loop works without morphogenetic features."""
        trainer = TolariaTrainer(basic_config)
        await trainer.initialize()
        
        # Should be able to run a few training steps
        metrics = await trainer.train_epoch()
        
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0

    async def test_morphogenetic_training_integration(self, basic_config, temp_workspace):
        """Test complete morphogenetic training integration."""
        # Enable morphogenetic features
        basic_config.morphogenetic.enable_tamiyo = True
        basic_config.morphogenetic.tamiyo_policy_path = str(temp_workspace / "mock_policy.pt")
        
        # Create a mock policy file
        mock_policy = torch.nn.Linear(10, 2)  # Simple mock
        torch.save(mock_policy.state_dict(), basic_config.morphogenetic.tamiyo_policy_path)
        
        trainer = TolariaTrainer(basic_config)
        
        with patch('esper.services.tamiyo.main.TamiyoService') as mock_tamiyo:
            mock_service = AsyncMock()
            mock_service.analyze_and_decide.return_value = AdaptationDecision(
                should_adapt=True,
                target_layer="layer1",
                confidence=0.8,
                reasoning="Test adaptation"
            )
            mock_tamiyo.return_value = mock_service
            
            await trainer.initialize()
            
            # Run one epoch which should trigger Tamiyo
            metrics = await trainer.train_epoch()
            await trainer.handle_end_of_epoch()
            
            # Verify Tamiyo was consulted
            mock_service.analyze_and_decide.assert_called()
            
            assert "loss" in metrics
            assert trainer.training_state.adaptations_this_epoch >= 0

    async def test_complete_adaptation_cycle(self, basic_config, temp_workspace):
        """Test a complete adaptation cycle from detection to implementation."""
        # Setup for full morphogenetic training
        basic_config.morphogenetic.enable_tamiyo = True
        basic_config.model.target_layers = ["layer1"]
        basic_config.model.seeds_per_layer = 1
        
        trainer = TolariaTrainer(basic_config)
        
        # Mock the services
        with patch('esper.services.oona_client.OonaClient') as mock_oona, \
             patch('esper.services.tamiyo.main.TamiyoService') as mock_tamiyo:
            
            # Setup mocks
            mock_oona_instance = AsyncMock()
            mock_oona.return_value = mock_oona_instance
            
            mock_tamiyo_instance = AsyncMock()
            mock_tamiyo_instance.analyze_and_decide.return_value = AdaptationDecision(
                should_adapt=True,
                target_layer="layer1",
                blueprint_id="test_blueprint_123",
                confidence=0.9,
                reasoning="Performance bottleneck detected"
            )
            mock_tamiyo.return_value = mock_tamiyo_instance
            
            await trainer.initialize()
            
            # Simulate training that triggers adaptation
            initial_metrics = await trainer.train_epoch()
            await trainer.handle_end_of_epoch()
            
            # Verify the adaptation flow
            assert trainer.training_state.total_adaptations >= 0
            mock_tamiyo_instance.analyze_and_decide.assert_called()

    async def test_service_lifecycle_management(self, basic_config):
        """Test proper service startup and shutdown."""
        service = TolariaService(basic_config)
        
        # Test startup
        await service.start()
        assert service.running
        assert service.trainer is not None
        
        # Test health check
        health = await service.health_check()
        assert "status" in health
        assert health["status"] in ["healthy", "starting"]
        
        # Test shutdown
        await service.stop()
        assert not service.running

    async def test_checkpoint_save_and_resume(self, basic_config, temp_workspace):
        """Test checkpoint saving and resuming functionality."""
        checkpoint_path = temp_workspace / "test_checkpoint.pt"
        
        # Train for a few steps and save checkpoint
        trainer1 = TolariaTrainer(basic_config)
        await trainer1.initialize()
        
        await trainer1.train_epoch()
        saved_path = trainer1.save_checkpoint(epoch=1, is_best=True)
        
        assert Path(saved_path).exists()
        
        # Create new trainer and resume
        trainer2 = TolariaTrainer(basic_config)
        await trainer2.initialize()
        
        resume_success = trainer2.load_checkpoint(saved_path)
        assert resume_success
        assert trainer2.training_state.epoch == 1

    async def test_error_handling_and_recovery(self, basic_config):
        """Test system behavior under various error conditions."""
        trainer = TolariaTrainer(basic_config)
        
        # Test initialization with invalid config
        invalid_config = basic_config
        invalid_config.model.name = "nonexistent_model"
        
        with pytest.raises(ValueError):
            await trainer.initialize()

    async def test_performance_benchmarking(self, basic_config):
        """Test system performance meets requirements."""
        trainer = TolariaTrainer(basic_config)
        await trainer.initialize()
        
        # Measure training step time
        import time
        start_time = time.time()
        
        metrics = await trainer.train_epoch()
        
        elapsed = time.time() - start_time
        
        # Should complete an epoch in reasonable time (adjust based on requirements)
        assert elapsed < 60  # 60 seconds for small test epoch
        assert "loss" in metrics

    async def test_configuration_validation(self, temp_workspace):
        """Test configuration validation and error handling."""
        # Test valid configuration
        valid_config = TolariaConfig()
        assert valid_config.model.name == "resnet18"
        
        # Test configuration post-init validation
        config_dict = {
            'model': {'name': 'resnet18'},
            'dataset': {'name': 'cifar10'},
            'logging': {'log_dir': str(temp_workspace / 'logs')}
        }
        
        config = TolariaConfig(**config_dict)
        assert Path(config.logging.log_dir).exists()


class TestMainEntrypointIntegration:
    """Integration tests for the main training entrypoint."""

    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a mock configuration file."""
        config_data = {
            'experiment_name': 'test_experiment',
            'model': {'name': 'resnet18', 'num_classes': 10},
            'dataset': {'name': 'cifar10', 'batch_size': 32},
            'optimization': {'max_epochs': 1}
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return config_file

    async def test_config_loading_from_file(self, mock_config_file):
        """Test loading configuration from YAML file."""
        from esper.configs import load_config
        
        config = load_config(str(mock_config_file), TolariaConfig)
        assert config.experiment_name == 'test_experiment'
        assert config.model.name == 'resnet18'

    async def test_command_line_argument_parsing(self):
        """Test command line argument parsing."""
        import train
        
        parser = train.create_parser()
        
        # Test basic arguments
        args = parser.parse_args(['--model', 'resnet18', '--dataset', 'cifar10', '--epochs', '50'])
        assert args.model == 'resnet18'
        assert args.dataset == 'cifar10'
        assert args.epochs == 50

    async def test_service_health_checking(self):
        """Test service health checking functionality."""
        # This would normally check real services
        # For testing, we'll mock the health check
        
        with patch('train.check_service_health') as mock_health:
            mock_health.return_value = {
                'urza': True,
                'tezzeret': True,
                'redis': True,
                'postgres': True
            }
            
            health = await mock_health()
            assert all(health.values())


class TestE2EWorkflows:
    """End-to-end workflow tests."""

    async def test_cifar10_baseline_training(self):
        """Test complete CIFAR-10 baseline training workflow."""
        # This test would run actual training for a few epochs
        # to validate the complete pipeline
        
        config = TolariaConfig(
            model=ModelConfig(name="resnet18", num_classes=10),
            dataset=DatasetConfig(name="cifar10", batch_size=32),
            optimization=OptimizationConfig(max_epochs=1),
            morphogenetic=MorphogeneticConfig(enable_tamiyo=False)
        )
        
        service = TolariaService(config)
        
        # This would be a longer test in practice
        await service.start()
        await asyncio.sleep(1)  # Brief training simulation
        await service.stop()

    async def test_morphogenetic_adaptation_workflow(self):
        """Test complete morphogenetic adaptation workflow."""
        # Test the full pipeline:
        # 1. Model training starts
        # 2. Tamiyo detects bottleneck
        # 3. Blueprint is requested from Urza
        # 4. Kernel is loaded and applied
        # 5. Training continues with adaptation
        
        # This would require more complex setup with actual services
        # For now, we'll create a framework for the test
        
        assert True  # Placeholder until full implementation


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmark tests to validate system efficiency."""

    async def test_training_overhead_measurement(self):
        """Measure overhead introduced by morphogenetic features."""
        # Compare baseline vs morphogenetic training speed
        pass

    async def test_memory_usage_benchmarks(self):
        """Test memory usage stays within acceptable bounds."""
        pass

    async def test_adaptation_latency_benchmarks(self):
        """Test that adaptations complete within target latency."""
        pass
```

## **4. Documentation and User Guide**

**Task:** Create comprehensive documentation for Phase 4 components.

### **4.1. User Guide (`docs/phase 4/user_guide.md`)**

```markdown
# Esper Phase 4 User Guide: Complete System Operation

This guide covers how to use the complete Esper morphogenetic training system.

## Quick Start

### 1. Basic Training

```bash
# Train ResNet-18 on CIFAR-10 with morphogenetic features
python train.py --model resnet18 --dataset cifar10 --epochs 100

# Train with custom configuration
python train.py --config configs/resnet18_cifar10.yaml
```

### 2. Configuration File Structure

```yaml
experiment_name: "my_experiment"

model:
  name: "resnet18"
  num_classes: 10
  target_layers: ["layer1", "layer2", "layer3"]

dataset:
  name: "cifar10"
  batch_size: 128

morphogenetic:
  enable_tamiyo: true
  max_adaptations_per_epoch: 2
```

### 3. Monitoring Training

Training progress is logged to console and optionally to Weights & Biases:

```bash
# Enable W&B logging
python train.py --config configs/experiment.yaml --wandb
```

## Advanced Usage

### Morphogenetic Configuration

Control how the system adapts:

```yaml
morphogenetic:
  enable_tamiyo: true
  adaptation_frequency: 1  # Check every epoch
  max_adaptations_per_epoch: 2
  health_threshold: 0.3
  blueprint_categories: ["residual", "attention"]
```

### Service Management

Check and manage supporting services:

```bash
# Check if all services are running
python train.py --check-services

# Start services only (for development)
python train.py --services-only
```

### Checkpointing and Resume

```bash
# Resume from specific checkpoint
python train.py --config configs/experiment.yaml --resume checkpoints/epoch_50.pt

# Auto-resume from latest checkpoint
python train.py --config configs/experiment.yaml  # auto_resume: true in config
```

## Troubleshooting

### Common Issues

1. **Services not running**: Use `--check-services` to verify
2. **GPU memory issues**: Reduce batch size or enable gradient checkpointing
3. **Slow adaptation**: Check Tamiyo policy is loaded correctly

### Performance Tuning

- Use mixed precision: `--mixed-precision`
- Optimize data loading: Increase `num_workers` in dataset config
- Monitor GPU utilization: Enable profiling with `--profile`

```

## **5. Phase 4 Definition of Done**

Phase 4 is complete when:

- ✅ **Tolaria Training Orchestrator** implemented with full training lifecycle management
- ✅ **Main system entrypoint** (`train.py`) provides single command interface
- ✅ **Configuration system** supports unified multi-service management
- ✅ **End-to-end integration tests** validate complete morphogenetic cycles
- ✅ **Service orchestration** handles startup, shutdown, and health monitoring
- ✅ **Checkpoint management** enables training resume and model persistence
- ✅ **Performance validation** meets <5% overhead requirements
- ✅ **User documentation** provides comprehensive usage guide
- ✅ **All tests passing** with maintained code coverage
- ✅ **Benchmark demonstration** shows autonomous adaptation on CIFAR-10

## **6. Implementation Timeline**

### **Week 1: Core Implementation**
- Days 1-2: TolariaTrainer class and core training loop
- Days 3-4: Configuration system and service integration
- Day 5: Basic integration testing

### **Week 2: System Integration**
- Days 1-2: Main entrypoint and CLI interface
- Days 3-4: Service orchestration and lifecycle management
- Day 5: End-to-end integration testing

### **Week 3: Validation and Documentation**
- Days 1-2: Comprehensive integration tests
- Days 3-4: Performance benchmarking and optimization
- Day 5: Documentation and user guide completion

**Target Completion:** August 4, 2025

This plan provides the complete roadmap for implementing Phase 4 and achieving full Esper system functionality.
