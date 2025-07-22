# Configuration Files

This directory contains configuration files for the Esper platform.

## Structure

```
configs/
├── README.md           # This file
├── demo/              # Demo configurations
│   └── tolaria.yaml   # Demo training configuration
├── examples/          # Example configurations
│   ├── cifar10_experiment.yaml   # CIFAR-10 training config
│   └── cifar100_experiment.yaml  # CIFAR-100 training config
└── templates/         # Configuration templates
    └── default.yaml   # Default configuration template
```

## Configuration Files

### Demo Configuration

- `demo/tolaria.yaml` - Pre-configured for tech demos with reasonable defaults

### Example Configurations

- `examples/cifar10_experiment.yaml` - Standard CIFAR-10 training configuration
- `examples/cifar100_experiment.yaml` - Standard CIFAR-100 training configuration

### Templates

- `templates/default.yaml` - Base template with all available options documented

## Usage

```bash
# Use a configuration file
python -m esper.services.tolaria.main --config configs/demo/tolaria.yaml

# Or with the training script
python train.py --config configs/examples/cifar10_experiment.yaml
```

## Creating Custom Configurations

1. Copy a template or example
2. Modify values as needed
3. Validate with the config validator:
   ```python
   from esper.services.tolaria.config import TolariaConfig
   config = TolariaConfig.from_yaml("your-config.yaml")
   errors = config.validate()
   ```

## Key Configuration Sections

- **model**: Architecture and morphogenetic settings
- **dataset**: Data loading and augmentation
- **optimization**: Training parameters
- **morphogenetic**: Adaptation settings
- **logging**: Monitoring and metrics
- **checkpoint**: Model persistence

See the template file for detailed documentation of all options.