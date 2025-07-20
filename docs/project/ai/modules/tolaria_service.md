# Tolaria Service - Training Orchestrator (`src/esper/services/tolaria/`)

## Overview

Tolaria serves as the master training orchestrator for the Esper system, coordinating the complete morphogenetic training pipeline. It integrates PyTorch model training with strategic decision making from Tamiyo, manages the lifecycle of training sessions, and ensures seamless coordination between all system components.

## Architecture Summary

### Core Responsibilities
- **Training Orchestration:** Master control of training loops and optimization
- **Model Integration:** Wrapping PyTorch models with morphogenetic capabilities
- **Strategic Coordination:** Integration with Tamiyo for adaptation decisions
- **Lifecycle Management:** Complete training session state management
- **Performance Monitoring:** Comprehensive metrics collection and analysis

### Integration Points
- **Kasmina:** Model wrapping and execution coordination
- **Tamiyo:** Strategic decision making and adaptation planning
- **Oona:** System-wide event publishing and coordination
- **Urza:** Blueprint and kernel artifact management

## Files

### `__init__.py` - Tolaria Service Initialization

**Purpose:** Service module initialization for Tolaria training orchestrator.

**Contents:** Minimal initialization for Tolaria service components.

### `config.py` - Training Configuration Management

**Purpose:** Comprehensive configuration management for training parameters, model settings, and service coordination.

#### Key Components

**`ModelConfig`** - Model Architecture Configuration
```python
@dataclass
class ModelConfig:
    """Configuration for model architecture and morphogenetic settings."""
    
    # Model architecture
    architecture: str = "resnet18"  # Model architecture name
    num_classes: int = 10
    pretrained: bool = False
    
    # Morphogenetic settings
    target_layers: List[str] = field(default_factory=lambda: ["nn.Linear"])
    seeds_per_layer: int = 4
    cache_size_mb: int = 128
    telemetry_enabled: bool = True
    preserve_original: bool = True
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfig':
        """Create from dictionary."""
        return cls(**data)
```

**`DatasetConfig`** - Data Loading Configuration
```python
@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    
    # Dataset settings
    name: str = "cifar10"
    data_root: str = "./data"
    download: bool = True
    
    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    shuffle_train: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_params: Dict[str, Any] = field(default_factory=dict)
    
    # Validation split
    validation_split: float = 0.1
    test_split: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

**`OptimizationConfig`** - Training Optimization Settings
```python
@dataclass
class OptimizationConfig:
    """Configuration for training optimization parameters."""
    
    # Optimizer settings
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    
    # Learning rate scheduling
    scheduler: str = "cosine"  # "cosine", "step", "exponential", "none"
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Training parameters
    max_epochs: int = 100
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    
    # Loss function
    loss_function: str = "cross_entropy"
    loss_params: Dict[str, Any] = field(default_factory=dict)
    
    # Gradient management
    gradient_clip_norm: Optional[float] = None
    accumulation_steps: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

**`MorphogeneticConfig`** - Morphogenetic Training Settings
```python
@dataclass
class MorphogeneticConfig:
    """Configuration for morphogenetic training integration."""
    
    # Tamiyo integration
    tamiyo_enabled: bool = True
    tamiyo_url: str = "http://localhost:8001"
    decision_frequency: int = 5  # Check for decisions every N epochs
    
    # Adaptation settings
    max_adaptations_per_epoch: int = 2
    adaptation_cooldown: float = 30.0  # seconds
    confidence_threshold: float = 0.7
    urgency_threshold: float = 0.5
    
    # Performance monitoring
    health_signal_interval: int = 1  # epochs
    performance_window: int = 10  # epochs for trend analysis
    
    # Blueprint generation
    enable_blueprint_generation: bool = False  # Future feature
    blueprint_generation_frequency: int = 50  # epochs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

**`LoggingConfig`** - Logging and Monitoring Configuration
```python
@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    
    # Logging levels
    log_level: str = "INFO"
    log_format: str = "structured"
    log_file: Optional[str] = None
    
    # Metrics tracking
    track_metrics: bool = True
    metrics_interval: int = 1  # epochs
    save_metrics: bool = True
    metrics_file: str = "training_metrics.json"
    
    # Wandb integration
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_config: Dict[str, Any] = field(default_factory=dict)
    
    # TensorBoard integration
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "./runs"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

**`CheckpointConfig`** - Model Persistence Configuration
```python
@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing and persistence."""
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 10  # epochs
    keep_best_only: bool = False
    max_checkpoints: int = 5
    
    # Model saving
    save_final_model: bool = True
    model_save_path: str = "./models/final_model.pt"
    save_morphogenetic_state: bool = True
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    strict_loading: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
```

**`TolariaConfig`** - Master Configuration
```python
@dataclass
class TolariaConfig:
    """Master configuration for Tolaria training orchestrator."""
    
    # Session information
    session_name: str = "esper_training"
    session_description: str = "Morphogenetic training session"
    experiment_id: Optional[str] = None
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    morphogenetic: MorphogeneticConfig = field(default_factory=MorphogeneticConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Service URLs
    urza_url: str = "http://localhost:8000"
    tamiyo_url: str = "http://localhost:8001"
    oona_redis_url: str = "redis://localhost:6379"
    
    # Resource management
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"
    mixed_precision: bool = False
    compile_model: bool = False  # torch.compile
    
    # Debugging and development
    debug_mode: bool = False
    dry_run: bool = False
    profile_training: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TolariaConfig':
        """Create from dictionary."""
        # Handle nested configs
        config_data = data.copy()
        
        if "model" in config_data:
            config_data["model"] = ModelConfig.from_dict(config_data["model"])
        if "dataset" in config_data:
            config_data["dataset"] = DatasetConfig.from_dict(config_data["dataset"])
        if "optimization" in config_data:
            config_data["optimization"] = OptimizationConfig.from_dict(config_data["optimization"])
        if "morphogenetic" in config_data:
            config_data["morphogenetic"] = MorphogeneticConfig.from_dict(config_data["morphogenetic"])
        if "logging" in config_data:
            config_data["logging"] = LoggingConfig.from_dict(config_data["logging"])
        if "checkpoint" in config_data:
            config_data["checkpoint"] = CheckpointConfig.from_dict(config_data["checkpoint"])
        
        return cls(**config_data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'TolariaConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def validate(self) -> List[str]:
        """
        Validate configuration for common issues.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Model validation
        if self.model.seeds_per_layer < 1:
            errors.append("Model seeds_per_layer must be >= 1")
        
        if self.model.cache_size_mb < 1:
            errors.append("Model cache_size_mb must be >= 1")
        
        # Dataset validation
        if self.dataset.batch_size < 1:
            errors.append("Dataset batch_size must be >= 1")
        
        if not (0.0 <= self.dataset.validation_split <= 1.0):
            errors.append("Dataset validation_split must be between 0.0 and 1.0")
        
        # Optimization validation
        if self.optimization.learning_rate <= 0:
            errors.append("Optimization learning_rate must be > 0")
        
        if self.optimization.max_epochs < 1:
            errors.append("Optimization max_epochs must be >= 1")
        
        # Morphogenetic validation
        if not (0.0 <= self.morphogenetic.confidence_threshold <= 1.0):
            errors.append("Morphogenetic confidence_threshold must be between 0.0 and 1.0")
        
        if not (0.0 <= self.morphogenetic.urgency_threshold <= 1.0):
            errors.append("Morphogenetic urgency_threshold must be between 0.0 and 1.0")
        
        return errors
```

**Configuration Usage Patterns:**

**YAML Configuration File Example:**
```yaml
session_name: "resnet18_cifar10_morphogenetic"
session_description: "ResNet-18 training on CIFAR-10 with morphogenetic adaptations"

model:
  architecture: "resnet18"
  num_classes: 10
  pretrained: false
  target_layers: ["nn.Linear"]
  seeds_per_layer: 4
  cache_size_mb: 128
  telemetry_enabled: true

dataset:
  name: "cifar10"
  batch_size: 128
  num_workers: 4
  use_augmentation: true

optimization:
  optimizer: "adam"
  learning_rate: 0.001
  max_epochs: 200
  scheduler: "cosine"
  early_stopping_patience: 15

morphogenetic:
  tamiyo_enabled: true
  decision_frequency: 5
  max_adaptations_per_epoch: 2
  confidence_threshold: 0.75

logging:
  log_level: "INFO"
  track_metrics: true
  use_wandb: true
  wandb_project: "esper-experiments"

checkpoint:
  checkpoint_frequency: 10
  keep_best_only: false
  max_checkpoints: 3
```

**Programmatic Configuration:**
```python
# Load from YAML
config = TolariaConfig.from_yaml("configs/resnet18_cifar10.yaml")

# Validate configuration
errors = config.validate()
if errors:
    for error in errors:
        print(f"Configuration error: {error}")
    sys.exit(1)

# Use in trainer
trainer = TolariaTrainer(config)
```

### `main.py` - Service Orchestration

**Purpose:** Main service orchestration with graceful lifecycle management, health monitoring, and signal handling.

#### Key Components

**`TolariaService`** - Service Coordinator
```python
class TolariaService:
    """
    Main service orchestration for Tolaria training orchestrator.
    
    Manages service lifecycle, coordinates with other system components,
    and provides health monitoring and graceful shutdown capabilities.
    """
    
    def __init__(self, config: TolariaConfig):
        """
        Initialize Tolaria service.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.trainer: Optional[TolariaTrainer] = None
        self.running = False
        
        # Service state
        self.start_time = time.time()
        self.last_health_check = 0.0
        self.health_check_interval = 30.0  # seconds
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Logging setup
        self._setup_logging()
        
        logger.info(f"Initialized TolariaService for session '{config.session_name}'")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_config = self.config.logging
        
        # Set log level
        numeric_level = getattr(logging, log_config.log_level.upper(), logging.INFO)
        logging.getLogger().setLevel(numeric_level)
        
        # Add file handler if specified
        if log_config.log_file:
            file_handler = logging.FileHandler(log_config.log_file)
            file_handler.setLevel(numeric_level)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            
            logging.getLogger().addHandler(file_handler)
    
    async def start(self) -> None:
        """
        Start the Tolaria service.
        """
        if self.running:
            logger.warning("Service is already running")
            return
        
        logger.info("Starting Tolaria service...")
        self.running = True
        
        try:
            # Initialize trainer
            self.trainer = TolariaTrainer(self.config)
            
            # Start training
            await self.trainer.run_training()
            
        except Exception as e:
            logger.error(f"Service startup failed: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """
        Stop the Tolaria service gracefully.
        """
        if not self.running:
            return
        
        logger.info("Stopping Tolaria service...")
        self.running = False
        
        # Stop trainer
        if self.trainer:
            await self.trainer.stop()
        
        logger.info("Tolaria service stopped")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_names = {
            signal.SIGINT: "SIGINT",
            signal.SIGTERM: "SIGTERM"
        }
        
        signal_name = signal_names.get(signum, f"Signal {signum}")
        logger.info(f"Received {signal_name}, initiating graceful shutdown...")
        
        # Use asyncio to handle shutdown
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(self.stop())
        else:
            asyncio.run(self.stop())
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check and return status.
        
        Returns:
            Health status dictionary
        """
        current_time = time.time()
        
        health_status = {
            "service": "tolaria",
            "status": "healthy" if self.running else "stopped",
            "uptime": current_time - self.start_time,
            "last_check": current_time,
            "session_name": self.config.session_name
        }
        
        # Add trainer status if available
        if self.trainer:
            trainer_status = await self.trainer.get_status()
            health_status.update(trainer_status)
        
        # Check external service connectivity
        external_services = await self._check_external_services()
        health_status["external_services"] = external_services
        
        self.last_health_check = current_time
        
        return health_status
    
    async def _check_external_services(self) -> Dict[str, str]:
        """
        Check connectivity to external services.
        
        Returns:
            Dictionary of service statuses
        """
        services = {
            "tamiyo": self.config.tamiyo_url,
            "urza": self.config.urza_url,
            "oona": self.config.oona_redis_url
        }
        
        statuses = {}
        
        for service_name, service_url in services.items():
            try:
                if service_name == "oona":
                    # Check Redis connectivity
                    import redis.asyncio as redis
                    redis_client = redis.Redis.from_url(service_url)
                    await redis_client.ping()
                    statuses[service_name] = "healthy"
                    await redis_client.close()
                else:
                    # Check HTTP service
                    import aiohttp
                    timeout = aiohttp.ClientTimeout(total=5)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(f"{service_url}/health") as response:
                            if response.status == 200:
                                statuses[service_name] = "healthy"
                            else:
                                statuses[service_name] = f"unhealthy (HTTP {response.status})"
                                
            except Exception as e:
                statuses[service_name] = f"unreachable ({str(e)[:50]})"
        
        return statuses
    
    async def get_training_status(self) -> Dict[str, Any]:
        """
        Get detailed training status.
        
        Returns:
            Training status information
        """
        if not self.trainer:
            return {"status": "not_started"}
        
        return await self.trainer.get_detailed_status()
    
    async def pause_training(self) -> bool:
        """
        Pause training if possible.
        
        Returns:
            True if training was paused successfully
        """
        if self.trainer:
            return await self.trainer.pause()
        return False
    
    async def resume_training(self) -> bool:
        """
        Resume paused training.
        
        Returns:
            True if training was resumed successfully
        """
        if self.trainer:
            return await self.trainer.resume()
        return False
    
    def get_config(self) -> TolariaConfig:
        """
        Get current configuration.
        
        Returns:
            Service configuration
        """
        return self.config
```

**Service Entry Point:**
```python
async def main():
    """Main entry point for Tolaria service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tolaria Training Orchestrator")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate configuration")
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = TolariaConfig.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Configuration validation passed")
        return
    
    # Start service
    service = TolariaService(config)
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Service failed: {e}")
        sys.exit(1)
    finally:
        await service.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### `trainer.py` - Core Training Orchestrator

**Purpose:** Implements the complete training pipeline integrating PyTorch training with morphogenetic capabilities, strategic decision making, and comprehensive monitoring.

#### Key Components

**`TolariaTrainer`** - Master Training Coordinator
```python
class TolariaTrainer:
    """
    Core training orchestrator integrating all Esper components.
    
    Manages the complete morphogenetic training pipeline including model
    wrapping, strategic coordination with Tamiyo, and performance monitoring.
    """
    
    def __init__(self, config: TolariaConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Device setup
        self.device = self._setup_device()
        
        # Training state
        self.current_epoch = 0
        self.best_score = float('-inf')
        self.training_start_time = 0.0
        self.paused = False
        self.should_stop = False
        
        # Components (initialized during setup)
        self.model = None
        self.morphable_model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        
        # Service clients
        self.oona_client = None
        self.tamiyo_client = None
        
        # Metrics tracking
        self.metrics_history = defaultdict(list)
        self.last_adaptation_time = 0.0
        
        # Mixed precision
        self.scaler = None
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Initialized TolariaTrainer on device {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        device_config = self.config.device.lower()
        
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon) device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        else:
            device = torch.device(device_config)
        
        return device
    
    async def run_training(self) -> None:
        """
        Run the complete training pipeline.
        """
        logger.info(f"Starting training session: {self.config.session_name}")
        self.training_start_time = time.time()
        
        try:
            # Setup training components
            await self._setup_training()
            
            # Load checkpoint if resuming
            if self.config.checkpoint.resume_from_checkpoint:
                self._load_checkpoint(self.config.checkpoint.resume_from_checkpoint)
            
            # Main training loop
            for epoch in range(self.current_epoch, self.config.optimization.max_epochs):
                if self.should_stop:
                    break
                
                await self._train_epoch(epoch)
                
                # Handle pausing
                while self.paused and not self.should_stop:
                    await asyncio.sleep(1.0)
                
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            await self._cleanup()
    
    async def _setup_training(self) -> None:
        """Setup all training components."""
        logger.info("Setting up training components...")
        
        # Setup data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Create model
        self.model = self._create_model()
        
        # Wrap model with morphogenetic capabilities
        if self.config.morphogenetic.tamiyo_enabled:
            self.morphable_model = await self._wrap_model()
        else:
            self.morphable_model = self.model
        
        # Move to device
        self.morphable_model.to(self.device)
        
        # Compile model if requested
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                self.morphable_model = torch.compile(self.morphable_model)
                logger.info("Applied torch.compile to model")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup service clients
        await self._setup_service_clients()
        
        logger.info("Training setup complete")
    
    async def _wrap_model(self) -> MorphableModel:
        """
        Wrap model with morphogenetic capabilities.
        
        Returns:
            MorphableModel with KasminaLayers
        """
        from esper import wrap
        
        # Determine target layer types
        target_layer_types = []
        for layer_name in self.config.model.target_layers:
            if layer_name == "nn.Linear":
                target_layer_types.append(nn.Linear)
            elif layer_name == "nn.Conv2d":
                target_layer_types.append(nn.Conv2d)
            # Add more layer types as needed
        
        morphable_model = wrap(
            self.model,
            target_layers=target_layer_types,
            seeds_per_layer=self.config.model.seeds_per_layer,
            cache_size_mb=self.config.model.cache_size_mb,
            telemetry_enabled=self.config.model.telemetry_enabled,
            preserve_original=self.config.model.preserve_original
        )
        
        logger.info(f"Wrapped model with {len(morphable_model.get_layer_names())} KasminaLayers")
        return morphable_model
    
    async def _train_epoch(self, epoch: int) -> None:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.current_epoch = epoch
        epoch_start_time = time.time()
        
        logger.info(f"Starting epoch {epoch + 1}/{self.config.optimization.max_epochs}")
        
        # Training phase
        train_metrics = await self._train_phase()
        
        # Validation phase
        val_metrics = await self._validation_phase()
        
        # Combine metrics
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "epoch_time": time.time() - epoch_start_time,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
        # Add morphogenetic metrics
        if hasattr(self.morphable_model, 'get_model_stats'):
            morpho_stats = self.morphable_model.get_model_stats()
            epoch_metrics.update({
                "morphogenetic_active": morpho_stats["morphogenetic_active"],
                "active_seeds": morpho_stats["active_seeds"],
                "total_seeds": morpho_stats["total_seeds"],
                "kernel_executions": morpho_stats["total_kernel_executions"]
            })
        
        # Store metrics
        for key, value in epoch_metrics.items():
            self.metrics_history[key].append(value)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # Check for strategic decisions
        if (self.config.morphogenetic.tamiyo_enabled and 
            epoch % self.config.morphogenetic.decision_frequency == 0):
            await self._consult_tamiyo()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        # Save checkpoint
        if (self.config.checkpoint.save_checkpoints and 
            epoch % self.config.checkpoint.checkpoint_frequency == 0):
            self._save_checkpoint(epoch, val_metrics["loss"])
        
        # Early stopping check
        if self._should_early_stop(val_metrics["loss"]):
            logger.info("Early stopping triggered")
            self.should_stop = True
        
        # Publish epoch completion event
        if self.oona_client:
            await self._publish_epoch_completion(epoch_metrics)
    
    async def _train_phase(self) -> Dict[str, float]:
        """
        Execute training phase for one epoch.
        
        Returns:
            Training metrics
        """
        self.morphable_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.should_stop:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if self.scaler:
                with torch.cuda.amp.autocast():
                    output = self.morphable_model(data)
                    loss = self.criterion(output, target)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.optimization.gradient_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.morphable_model.parameters(),
                        self.config.optimization.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.morphable_model(data)
                loss = self.criterion(output, target)
                
                loss.backward()
                
                # Gradient clipping
                if self.config.optimization.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.morphable_model.parameters(),
                        self.config.optimization.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.debug(
                    f"Train Batch {batch_idx}/{len(self.train_loader)}: "
                    f"Loss: {loss.item():.6f}"
                )
        
        return {
            "loss": total_loss / len(self.train_loader),
            "accuracy": correct / total
        }
    
    async def _validation_phase(self) -> Dict[str, float]:
        """
        Execute validation phase.
        
        Returns:
            Validation metrics
        """
        self.morphable_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.morphable_model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            "loss": total_loss / len(self.val_loader),
            "accuracy": correct / total
        }
    
    async def _consult_tamiyo(self) -> None:
        """
        Consult Tamiyo for strategic adaptation decisions.
        """
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_adaptation_time < self.config.morphogenetic.adaptation_cooldown:
            logger.debug("Adaptation cooldown active, skipping Tamiyo consultation")
            return
        
        try:
            # For MVP, simulate Tamiyo decisions
            # In production, this would make HTTP request to Tamiyo service
            
            # Get model health status
            if hasattr(self.morphable_model, 'get_model_stats'):
                model_stats = self.morphable_model.get_model_stats()
                
                # Simple decision logic for MVP
                active_ratio = model_stats["active_seeds"] / max(model_stats["total_seeds"], 1)
                
                if active_ratio < 0.5 and model_stats["total_forward_calls"] > 1000:
                    # Simulate loading a kernel
                    layer_names = self.morphable_model.get_layer_names()
                    if layer_names:
                        layer_name = layer_names[0]  # Target first layer
                        success = await self.morphable_model.load_kernel(
                            layer_name, 0, "simulated-kernel-abc123"
                        )
                        
                        if success:
                            logger.info(f"Loaded simulated kernel into {layer_name}")
                            self.last_adaptation_time = current_time
                        else:
                            logger.warning("Failed to load simulated kernel")
            
        except Exception as e:
            logger.error(f"Error consulting Tamiyo: {e}")
    
    async def _publish_epoch_completion(self, metrics: Dict[str, Any]) -> None:
        """
        Publish epoch completion event via Oona.
        
        Args:
            metrics: Epoch metrics to publish
        """
        if not self.oona_client:
            return
        
        try:
            from esper.contracts.messages import OonaMessage, TopicNames
            
            message = OonaMessage(
                sender_id=f"tolaria.{self.config.session_name}",
                trace_id=f"epoch-{self.current_epoch}",
                topic=TopicNames.SYSTEM_EVENTS_EPOCH,
                payload={
                    "epoch": self.current_epoch,
                    "session_name": self.config.session_name,
                    "metrics": metrics,
                    "timestamp": time.time()
                }
            )
            
            await self.oona_client.publish(message)
            
        except Exception as e:
            logger.warning(f"Failed to publish epoch completion: {e}")
    
    # ... Additional methods for model creation, data loading, checkpointing, etc.
    # (Implementation continues with standard PyTorch training patterns)
```

**Features:**
- **Complete Training Pipeline:** Full integration of PyTorch training with morphogenetic capabilities
- **Service Integration:** Coordination with Tamiyo, Urza, and Oona services
- **Comprehensive Monitoring:** Detailed metrics collection and performance tracking
- **Graceful Lifecycle:** Proper startup, shutdown, and error handling
- **Mixed Precision Support:** Optional automatic mixed precision training
- **Checkpointing:** Complete training state persistence and recovery
- **Early Stopping:** Configurable early stopping with validation monitoring

This completes the Tolaria service documentation. The final utils module documentation will be in the next file.