"""
Configuration models for Tolaria training orchestrator.

This module defines the configuration structure for the complete
Esper morphogenetic training system.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class OonaConfig:
    """Configuration for Oona message bus client."""
    url: str = "http://localhost:8001"
    timeout: float = 30.0


@dataclass
class ModelConfig:
    """Configuration for model architecture and initialization."""
    
    # Model architecture
    architecture: str = "resnet18"  # Model architecture name
    num_classes: int = 10  # Number of output classes
    pretrained: bool = False  # Use pretrained weights
    
    # Morphogenetic settings
    target_layers: List[str] = field(default_factory=lambda: ["*"])  # Layers to wrap
    seeds_per_layer: int = 4  # Maximum seeds per layer
    seed_cache_size_mb: int = 100  # Kernel cache size in MB
    
    # Model-specific parameters
    model_params: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class DatasetConfig:
    """Configuration for dataset and data loading."""
    
    # Dataset parameters
    name: str = "cifar10"  # Dataset name
    data_dir: str = "./data"  # Data directory
    download: bool = True  # Download if not present
    
    # Training data loading
    batch_size: int = 32  # Training batch size
    num_workers: int = 4  # Data loader workers
    pin_memory: bool = True  # Pin memory for GPU
    
    # Data augmentation
    train_transforms: Dict[str, Any] = field(default_factory=dict)
    val_transforms: Dict[str, Any] = field(default_factory=dict)
    
    # Validation split
    val_split: float = 0.1  # Validation split ratio
    val_batch_size: Optional[int] = None  # Validation batch size (defaults to batch_size)


@dataclass
class UrzaConfig:
    """Configuration for Urza asset management service."""
    
    base_url: str = "http://localhost:8000"  # Urza service URL
    timeout: float = 30.0  # Request timeout
    retry_attempts: int = 3  # Retry attempts for failed requests


@dataclass
class TolariaConfig:
    """Master configuration for Tolaria training orchestrator."""
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    
    # Optimizer settings
    optimizer: str = "adam"  # Optimizer type
    scheduler: Optional[str] = None  # Learning rate scheduler
    scheduler_params: Dict[str, Any] = field(default_factory=dict)
    
    # Morphogenetic settings
    tamiyo_policy_path: Optional[str] = None  # Path to Tamiyo policy weights
    adaptation_frequency: int = 1  # Epochs between adaptation checks
    max_adaptations_per_epoch: int = 2  # Maximum adaptations per epoch
    adaptation_cooldown: int = 5  # Epochs to wait after adaptation
    
    # Model and data configuration
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    
    # Infrastructure configuration
    oona: OonaConfig = field(default_factory=OonaConfig)
    urza: UrzaConfig = field(default_factory=UrzaConfig)
    
    # Checkpointing and logging
    checkpoint_dir: str = "./checkpoints"
    checkpoint_frequency: int = 10  # Save checkpoint every N epochs
    log_frequency: int = 100  # Log every N steps
    
    # Device and performance
    device: str = "auto"  # Device selection: "auto", "cpu", "cuda"
    mixed_precision: bool = True  # Use automatic mixed precision
    compile_model: bool = True  # Use torch.compile optimization
    
    # Experiment tracking
    experiment_name: str = "esper_experiment"
    run_id: Optional[str] = None  # Unique run identifier
    
    def __post_init__(self):
        """Validate and set derived configuration values."""
        # Set default validation batch size
        if self.dataset.val_batch_size is None:
            self.dataset.val_batch_size = self.dataset.batch_size
            
        # Ensure checkpoint directory exists
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate adaptation settings
        if self.adaptation_frequency <= 0:
            raise ValueError("adaptation_frequency must be positive")
        if self.max_adaptations_per_epoch < 0:
            raise ValueError("max_adaptations_per_epoch must be non-negative")
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "TolariaConfig":
        """Load configuration from YAML file."""
        import yaml
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # Convert nested dictionaries to dataclass instances
        if 'model' in config_data:
            config_data['model'] = ModelConfig(**config_data['model'])
        
        if 'dataset' in config_data:
            config_data['dataset'] = DatasetConfig(**config_data['dataset'])
        
        if 'oona' in config_data:
            config_data['oona'] = OonaConfig(**config_data['oona'])
        
        if 'urza' in config_data:
            config_data['urza'] = UrzaConfig(**config_data['urza'])
        
        # Filter out any fields that don't belong to TolariaConfig
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_data.items() if k in valid_fields}
        
        return cls(**filtered_config)

    def to_yaml(self, config_path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
