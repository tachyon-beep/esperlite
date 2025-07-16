"""
Tolaria Training Orchestrator - Core trainer implementation.

This module implements the main training orchestrator that coordinates
model training with morphogenetic adaptations through the Esper platform.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision
import torchvision.transforms as transforms

from esper.core.model_wrapper import wrap, MorphableModel
from esper.services.oona_client import OonaClient
from esper.services.tamiyo.main import TamiyoService
from esper.contracts.messages import OonaMessage, TopicNames
from esper.contracts.operational import HealthSignal, AdaptationDecision
from esper.contracts.enums import SeedState
from esper.services.tolaria.config import TolariaConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics for a single epoch."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    adaptations_applied: int
    epoch_time: float


@dataclass
class TrainingState:
    """Current state of the training process."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_val_accuracy: float = 0.0
    
    # Adaptation tracking
    total_adaptations: int = 0
    adaptations_this_epoch: int = 0
    last_adaptation_epoch: int = -1
    
    # Early stopping
    epochs_without_improvement: int = 0
    should_stop_early: bool = False


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
        """Initialize the trainer with configuration."""
        self.config = config
        self.run_id = config.run_id or str(uuid.uuid4())[:8]
        
        # Training state
        self.state = TrainingState()
        self.device = self._setup_device()
        
        # Core components (initialized in setup)
        self.model: Optional[MorphableModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        self.criterion: Optional[nn.Module] = None
        
        # Data loaders
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        
        # Services (initialized in setup)
        self.oona_client: Optional[OonaClient] = None
        self.tamiyo_service: Optional[Any] = None  # Type will be TamiyoService when available
        
        # Training control
        self.running = False
        self._stop_requested = False
        self._last_train_loss = 0.0  # Track last training loss
        
        logger.info(f"Initialized TolariaTrainer with run_id: {self.run_id}")

    def _setup_device(self) -> torch.device:
        """Setup and return the training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        logger.info(f"Using device: {device}")
        return device

    async def initialize(self) -> None:
        """Initialize all training components and services."""
        logger.info("Initializing Tolaria trainer...")
        
        try:
            # Setup data loaders
            self._setup_data_loaders()
            
            # Setup model
            self._setup_model()
            
            # Setup optimizer and scheduler
            self._setup_optimizer()
            self._setup_scheduler()
            
            # Setup loss function
            self._setup_criterion()
            
            # Initialize services
            self._setup_services()
            
            logger.info("Tolaria trainer initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {e}")
            raise

    def _setup_data_loaders(self) -> None:
        """Setup training and validation data loaders."""
        logger.info("Setting up data loaders...")
        
        dataset_name = self.config.dataset.name.lower()
        
        if dataset_name == "cifar10":
            train_dataset, val_dataset = self._setup_cifar10()
        elif dataset_name == "cifar100":
            train_dataset, val_dataset = self._setup_cifar100()
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.dataset.batch_size,
            shuffle=True,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.dataset.val_batch_size or self.config.dataset.batch_size,
            shuffle=False,
            num_workers=self.config.dataset.num_workers,
            pin_memory=self.config.dataset.pin_memory
        )
        
        logger.info(f"Data loaders ready: {len(train_dataset)} train, {len(val_dataset)} val samples")

    def _setup_cifar10(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Setup CIFAR-10 datasets with transforms."""
        # Default transforms
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.config.dataset.data_dir,
            train=True,
            download=self.config.dataset.download,
            transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR10(
            root=self.config.dataset.data_dir,
            train=False,
            download=self.config.dataset.download,
            transform=val_transform
        )
        
        return train_dataset, val_dataset

    def _setup_cifar100(self) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """Setup CIFAR-100 datasets with transforms."""
        # Similar to CIFAR-10 but with CIFAR-100
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        train_dataset = torchvision.datasets.CIFAR100(
            root=self.config.dataset.data_dir,
            train=True,
            download=self.config.dataset.download,
            transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR100(
            root=self.config.dataset.data_dir,
            train=False,
            download=self.config.dataset.download,
            transform=val_transform
        )
        
        return train_dataset, val_dataset

    def _setup_model(self) -> None:
        """Setup the model architecture."""
        logger.info(f"Setting up model: {self.config.model.architecture}")
        
        # Create base model
        if self.config.model.architecture.lower() == "resnet18":
            base_model = torchvision.models.resnet18(pretrained=self.config.model.pretrained)
            # Adjust for CIFAR-10/100 input size
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
            base_model.fc = nn.Linear(base_model.fc.in_features, self.config.model.num_classes)
        elif self.config.model.architecture.lower() == "resnet34":
            base_model = torchvision.models.resnet34(pretrained=self.config.model.pretrained)
            base_model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base_model.maxpool = nn.Identity()
            base_model.fc = nn.Linear(base_model.fc.in_features, self.config.model.num_classes)
        else:
            raise ValueError(f"Unsupported model architecture: {self.config.model.architecture}")
        
        # Convert target layer names to module types
        target_layer_types = []
        for layer_name in self.config.model.target_layers:
            if layer_name.lower() == "linear":
                target_layer_types.append(nn.Linear)
            elif layer_name.lower() == "conv2d":
                target_layer_types.append(nn.Conv2d)
            elif layer_name.lower() == "batchnorm2d":
                target_layer_types.append(nn.BatchNorm2d)
            else:
                logger.warning(f"Unknown target layer type: {layer_name}, skipping")
        
        # Only wrap if we have target layers
        if target_layer_types:
            # Wrap with morphogenetic capabilities
            self.model = wrap(
                base_model,
                target_layers=target_layer_types,
                seeds_per_layer=self.config.model.seeds_per_layer,
                cache_size_mb=self.config.model.seed_cache_size_mb
            )
        else:
            logger.warning("No valid target layers found, using base model without wrapping")
            self.model = base_model
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Compile if requested
        if self.config.compile_model:
            try:
                self.model = torch.compile(self.model)
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        logger.info(f"Model setup complete: {sum(p.numel() for p in self.model.parameters())} parameters")

    def _setup_optimizer(self) -> None:
        """Setup the optimizer."""
        optimizer_name = self.config.optimizer.lower()
        
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        logger.info(f"Optimizer setup: {optimizer_name}")

    def _setup_scheduler(self) -> None:
        """Setup learning rate scheduler if specified."""
        if self.config.scheduler is None or self.config.scheduler.lower() == "none":
            return
        
        scheduler_name = self.config.scheduler.lower()
        
        if scheduler_name == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.max_epochs,
                **self.config.scheduler_params
            )
        elif scheduler_name == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1,
                **self.config.scheduler_params
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        logger.info(f"Scheduler setup: {scheduler_name}")

    def _setup_criterion(self) -> None:
        """Setup loss criterion."""
        self.criterion = nn.CrossEntropyLoss()

    def _setup_services(self) -> None:
        """Setup external services (Oona, Tamiyo)."""
        logger.info("Setting up services...")
        
        # Setup Oona client
        try:
            self.oona_client = OonaClient(self.config.oona)
            logger.info("Oona client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Oona client: {e}")
            self.oona_client = None
        
        # Setup Tamiyo service if policy path is provided
        if self.config.tamiyo_policy_path and Path(self.config.tamiyo_policy_path).exists():
            try:
                # Initialize Tamiyo service with the provided policy
                from esper.services.tamiyo.main import TamiyoService
                from esper.services.tamiyo.config import TamiyoConfig
                
                tamiyo_config = TamiyoConfig(
                    policy_path=self.config.tamiyo_policy_path,
                    device=str(self.device),
                    temperature=0.1,  # Conservative temperature for production
                    top_k=5,
                    batch_size=1,
                    max_context_length=512
                )
                
                self.tamiyo_service = TamiyoService(tamiyo_config)
                logger.info(f"Tamiyo service initialized with policy: {self.config.tamiyo_policy_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize Tamiyo service: {e}")
                self.tamiyo_service = None
        else:
            logger.info("No Tamiyo policy specified, running without strategic controller")
            self.tamiyo_service = None

    async def train(self) -> List[TrainingMetrics]:
        """Execute the complete training process."""
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        
        self.running = True
        metrics_history = []
        
        try:
            for epoch in range(self.config.max_epochs):
                if self._stop_requested:
                    logger.info("Training stopped by request")
                    break
                
                self.state.epoch = epoch
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self._train_epoch()
                
                # Validation phase
                val_metrics = self._validate_epoch()
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Handle end of epoch (Tamiyo integration)
                await self._handle_end_of_epoch()
                
                # Create epoch metrics
                epoch_time = time.time() - epoch_start_time
                current_lr = self.optimizer.param_groups[0]['lr']
                
                metrics = TrainingMetrics(
                    epoch=epoch,
                    train_loss=train_metrics['loss'],
                    train_accuracy=train_metrics['accuracy'],
                    val_loss=val_metrics['loss'],
                    val_accuracy=val_metrics['accuracy'],
                    learning_rate=current_lr,
                    adaptations_applied=self.state.adaptations_this_epoch,
                    epoch_time=epoch_time
                )
                
                metrics_history.append(metrics)
                
                # Log progress
                logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.4f} | "
                    f"Adaptations: {self.state.adaptations_this_epoch} | "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # Update best metrics
                if val_metrics['loss'] < self.state.best_val_loss:
                    self.state.best_val_loss = val_metrics['loss']
                    self.state.epochs_without_improvement = 0
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.state.epochs_without_improvement += 1
                
                if val_metrics['accuracy'] > self.state.best_val_accuracy:
                    self.state.best_val_accuracy = val_metrics['accuracy']
                
                # Regular checkpoint saving
                if epoch % self.config.checkpoint_frequency == 0:
                    self._save_checkpoint(epoch, is_best=False)
                
                # Reset epoch adaptation counter
                self.state.adaptations_this_epoch = 0
            
            logger.info(f"Training completed. Best val accuracy: {self.state.best_val_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            self.running = False
        
        return metrics_history

    def _train_epoch(self) -> Dict[str, float]:
        """Execute one training epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass
            if self.config.mixed_precision:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(target.view_as(pred)).sum().item()
            total_samples += data.size(0)
            
            self.state.global_step += 1
            
            # Log periodically
            if batch_idx % self.config.log_frequency == 0:
                logger.debug(
                    f"Train Batch {batch_idx:4d} | "
                    f"Loss: {loss.item():.6f} | "
                    f"Acc: {100.0 * total_correct / total_samples:.2f}%"
                )
        
        final_loss = total_loss / len(self.train_loader) if self.train_loader else 0.0
        self._last_train_loss = final_loss  # Track for Tamiyo consultation
        
        return {
            'loss': final_loss,
            'accuracy': total_correct / total_samples
        }

    def _validate_epoch(self) -> Dict[str, float]:
        """Execute one validation epoch."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(target.view_as(pred)).sum().item()
                total_samples += data.size(0)
        
        return {
            'loss': total_loss / len(self.val_loader) if self.val_loader else 0.0,
            'accuracy': total_correct / total_samples if total_samples > 0 else 0.0
        }

    async def _handle_end_of_epoch(self) -> None:
        """Handle end-of-epoch processing including Tamiyo consultation."""
        # Check if it's time for adaptation consideration
        if (self.state.epoch % self.config.adaptation_frequency != 0 or
            self.state.epoch - self.state.last_adaptation_epoch < self.config.adaptation_cooldown):
            return
        
        # Check if we've reached max adaptations for this epoch
        if self.state.adaptations_this_epoch >= self.config.max_adaptations_per_epoch:
            return
        
        # Collect health signals from model
        health_signals = self._collect_health_signals()
        
        # Consult Tamiyo if available
        if self.tamiyo_service is not None:
            try:
                decision = await self._consult_tamiyo(health_signals)
                if decision and decision.confidence > 0.5:  # Only apply high-confidence decisions
                    success = await self._apply_adaptation(decision)
                    if success:
                        self.state.adaptations_this_epoch += 1
                        self.state.total_adaptations += 1
                        self.state.last_adaptation_epoch = self.state.epoch
                        logger.info("Applied adaptation: %s", decision.adaptation_type)
            except Exception as e:
                logger.error(f"Error in Tamiyo consultation: {e}")

    def _collect_health_signals(self) -> List[HealthSignal]:
        """Collect health signals from the model's KasminaLayers."""
        health_signals = []
        
        try:
            # Collect health signals from each KasminaLayer in the model
            if self.model is not None and hasattr(self.model, 'kasmina_layers'):
                for layer_idx, (layer_name, layer) in enumerate(self.model.kasmina_layers.items()):
                    # Get performance metrics from each layer
                    if hasattr(layer, 'get_health_metrics'):
                        metrics = layer.get_health_metrics()
                        
                        # Create HealthSignal with correct fields
                        signal = HealthSignal(
                            layer_id=layer_idx,  # Use enumeration index as numeric ID
                            seed_id=0,  # Default seed ID
                            chunk_id=0,  # Default chunk ID
                            epoch=self.state.epoch,
                            activation_variance=metrics.get('activation_variance', 0.0),
                            dead_neuron_ratio=metrics.get('dead_neuron_ratio', 0.0),
                            avg_correlation=metrics.get('avg_correlation', 0.0),
                            health_score=metrics.get('health_score', 1.0),
                            execution_latency=metrics.get('execution_latency', 0.0),
                            error_count=metrics.get('error_count', 0),
                            active_seeds=metrics.get('active_seeds', 1),
                            total_seeds=metrics.get('total_seeds', 1)
                        )
                        
                        health_signals.append(signal)
            
            logger.debug("Collected %d health signals", len(health_signals))
            
        except Exception as e:
            logger.warning("Error collecting health signals: %s", e)
        
        return health_signals

    async def _consult_tamiyo(self, health_signals: List[HealthSignal]) -> Optional[AdaptationDecision]:
        """Consult Tamiyo strategic controller for adaptation decisions."""
        if self.tamiyo_service is None or not health_signals:
            return None
        
        try:
            # Format health signals for Tamiyo analysis
            context = {
                'epoch': self.state.epoch,
                'global_step': self.state.global_step,
                'learning_rate': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.0,
                'recent_loss': self._last_train_loss,
                'health_signals': [signal.model_dump() for signal in health_signals],
                'model_architecture': self.config.model.architecture,
                'adaptations_so_far': self.state.total_adaptations
            }
            
            # Request adaptation decision from Tamiyo
            # Note: This method doesn't exist yet in TamiyoService, so we'll create a placeholder
            if hasattr(self.tamiyo_service, 'make_adaptation_decision'):
                decision = await self.tamiyo_service.make_adaptation_decision(context)
            else:
                # Fallback to a simpler analysis method if available
                decision = None
            
            if decision is not None:
                logger.info("Tamiyo recommended adaptation: %s", decision.adaptation_type)
            
            return decision
            
        except Exception as e:
            logger.error("Error consulting Tamiyo: %s", e)
            return None

    async def _apply_adaptation(self, decision: AdaptationDecision) -> bool:
        """Apply an adaptation decision to the model."""
        if decision is None:
            return False
        
        try:
            logger.info("Applying adaptation: %s", decision.adaptation_type)
            
            # Validate target layer
            target_layer_name = getattr(decision, 'layer_name', None)
            if not self._validate_target_layer(target_layer_name):
                return False
            
            # Apply the adaptation (simulated for now)
            success = self._perform_adaptation_simulation()
            
            if success:
                self._update_adaptation_state()
                await self._notify_adaptation_applied(decision, target_layer_name)
                logger.info("Successfully applied adaptation to layer %s", target_layer_name)
            else:
                logger.warning("Failed to load kernel for adaptation")
            
            return success
                
        except Exception as e:
            logger.error("Error applying adaptation: %s", e)
            return False

    def _validate_target_layer(self, target_layer_name: Optional[str]) -> bool:
        """Validate that the target layer exists."""
        if target_layer_name is None:
            logger.warning("Adaptation decision has no layer_name, skipping")
            return False
        
        if (self.model is None or 
            not hasattr(self.model, 'kasmina_layers') or 
            target_layer_name not in self.model.kasmina_layers):
            available_layers = list(self.model.kasmina_layers.keys()) if hasattr(self.model, 'kasmina_layers') else []
            logger.warning("Target layer %s not found. Available layers: %s", target_layer_name, available_layers)
            return False
        
        return True

    def _perform_adaptation_simulation(self) -> bool:
        """Simulate adaptation application."""
        # For now, simulate adaptation application
        # In a full implementation, this would load actual kernels
        return True  # Placeholder for actual kernel loading

    def _update_adaptation_state(self) -> None:
        """Update adaptation tracking state."""
        self.state.adaptations_this_epoch += 1
        self.state.total_adaptations += 1
        self.state.last_adaptation_epoch = self.state.epoch

    async def _notify_adaptation_applied(self, decision: AdaptationDecision, target_layer_name: str) -> None:
        """Send adaptation event notification."""
        if self.oona_client is None:
            return
        
        try:
            message = OonaMessage(
                sender_id=f"tolaria-{self.run_id}",
                trace_id=f"adaptation-{self.state.total_adaptations}",
                topic=TopicNames.SYSTEM_EVENTS_EPOCH,
                payload={
                    'event_type': 'adaptation_applied',
                    'run_id': self.run_id,
                    'epoch': self.state.epoch,
                    'layer_name': target_layer_name,
                    'adaptation_type': decision.adaptation_type,
                    'confidence': getattr(decision, 'confidence', 0.5)
                }
            )
            await self.oona_client.publish(message)
        except Exception as e:
            logger.warning("Failed to publish adaptation event: %s", e)

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_state': self.state,
            'config': self.config,
            'run_id': self.run_id
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best checkpoint: %s", best_path)
        
        # Save latest checkpoint
        latest_path = checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        return str(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint and restore state."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.state = checkpoint['training_state']
            
            if 'scheduler' in checkpoint and self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            
            logger.info("Loaded checkpoint from epoch %s", checkpoint['epoch'])
            return True
            
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return False

    def request_stop(self) -> None:
        """Request training to stop gracefully."""
        self._stop_requested = True
        logger.info("Training stop requested")

    def get_training_state(self) -> TrainingState:
        """Get current training state."""
        return self.state

    async def shutdown(self) -> None:
        """Gracefully shutdown the trainer and services."""
        logger.info("Shutting down Tolaria trainer...")
        
        self.running = False
        self._stop_requested = True
        
        # Cleanup services
        if self.oona_client is not None:
            try:
                # Close Oona client connection
                await self.oona_client.close()
            except Exception as e:
                logger.warning("Error closing Oona client: %s", e)
        
        logger.info("Tolaria trainer shutdown complete")
