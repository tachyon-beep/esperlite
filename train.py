#!/usr/bin/env python3
"""
Esper Morphogenetic Training Platform - Main Entrypoint

This script provides the single command interface for launching complete 
morphogenetic training runs with the Esper platform.

Usage:
    python train.py --config configs/experiment.yaml
    python train.py --quick-start cifar10
    python train.py --help
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from esper.services.tolaria.config import TolariaConfig
from esper.services.tolaria.main import TolariaService
from esper.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_quick_start_config(dataset: str, output_dir: Path) -> Path:
    """Create a quick-start configuration for common scenarios."""

    quick_configs = {
        "cifar10": {
            "run_id": "quick-cifar10",
            "max_epochs": 10,
            "learning_rate": 0.001,
            "batch_size": 128,
            "device": "auto",
            "compile_model": True,
            "model": {
                "architecture": "resnet18",
                "num_classes": 10,
                "pretrained": False,
                "target_layers": ["layer1", "layer2", "layer3"],
                "seeds_per_layer": 4,
                "seed_cache_size_mb": 256
            },
            "dataset": {
                "name": "cifar10",
                "data_dir": "./data",
                "download": True,
                "val_split": 0.1
            },
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "checkpoint_frequency": 5,
            "adaptation_frequency": 2,
            "adaptation_cooldown": 1,
            "max_adaptations_per_epoch": 2,
            "oona": {
                "url": "http://localhost:8001",
                "timeout": 30.0
            }
        },
        "cifar100": {
            "run_id": "quick-cifar100",
            "max_epochs": 20,
            "learning_rate": 0.001,
            "batch_size": 128,
            "device": "auto",
            "compile_model": True,
            "model": {
                "architecture": "resnet34",
                "num_classes": 100,
                "pretrained": False,
                "target_layers": ["layer1", "layer2", "layer3", "layer4"],
                "seeds_per_layer": 6,
                "seed_cache_size_mb": 512
            },
            "dataset": {
                "name": "cifar100",
                "data_dir": "./data",
                "download": True,
                "val_split": 0.1
            },
            "optimizer": "adamw",
            "weight_decay": 0.01,
            "scheduler": "cosine",
            "checkpoint_dir": str(output_dir / "checkpoints"),
            "checkpoint_frequency": 5,
            "adaptation_frequency": 3,
            "adaptation_cooldown": 2,
            "max_adaptations_per_epoch": 3,
            "oona": {
                "url": "http://localhost:8001",
                "timeout": 30.0
            }
        }
    }

    if dataset not in quick_configs:
        raise ValueError(f"Unknown quick-start dataset: {dataset}. Available: {list(quick_configs.keys())}")

    config_data = quick_configs[dataset]
    config_path = output_dir / f"quick-{dataset}.yaml"

    # Create the config file
    output_dir.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False, indent=2)

    logger.info("Created quick-start configuration: %s", config_path)
    return config_path


def validate_environment() -> bool:
    """Validate that the environment is ready for training."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 8):
        issues.append("Python 3.8+ required")

    # Check for required packages
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, training will use CPU")
    except ImportError:
        issues.append("PyTorch not installed")

    try:
        import torchvision
    except ImportError:
        issues.append("torchvision not installed")

    if issues:
        logger.error("Environment validation failed:")
        for issue in issues:
            logger.error("  - %s", issue)
        return False

    logger.info("Environment validation passed")
    return True


async def main() -> None:
    """Main entrypoint for Esper training."""
    parser = argparse.ArgumentParser(
        description="Esper Morphogenetic Training Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with custom configuration
  python train.py --config configs/my_experiment.yaml
  
  # Quick start with CIFAR-10
  python train.py --quick-start cifar10 --output ./results
  
  # Quick start with verbose logging
  python train.py --quick-start cifar10 --verbose --output ./results
  
  # Check environment and configuration
  python train.py --config configs/test.yaml --dry-run
        """
    )

    # Configuration options
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--config",
        type=str,
        help="Path to Tolaria configuration file"
    )
    config_group.add_argument(
        "--quick-start",
        type=str,
        choices=["cifar10", "cifar100"],
        help="Quick start with predefined configuration"
    )

    # Output and logging options
    parser.add_argument(
        "--output",
        type=str,
        default="./results",
        help="Output directory for results and checkpoints (default: ./results)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and environment without training"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging("esper-train", level=log_level)

    logger.info("=== Esper Morphogenetic Training Platform ===")

    # Validate environment
    if not validate_environment():
        sys.exit(1)

    try:
        # Determine configuration path
        if args.quick_start:
            output_dir = Path(args.output)
            config_path = create_quick_start_config(args.quick_start, output_dir)
        else:
            config_path = Path(args.config)
            if not config_path.exists():
                logger.error("Configuration file not found: %s", config_path)
                sys.exit(1)

        # Load and validate configuration
        logger.info("Loading configuration from: %s", config_path)
        config = TolariaConfig.from_yaml(config_path)

        logger.info("Configuration loaded successfully:")
        logger.info("  - Model: %s", config.model.architecture)
        logger.info("  - Dataset: %s", config.dataset.name)
        logger.info("  - Epochs: %d", config.max_epochs)
        logger.info("  - Device: %s", config.device)

        if args.dry_run:
            logger.info("Dry run completed successfully - configuration is valid")
            return

        # Create and start the training service
        logger.info("Starting Tolaria training service...")
        service = TolariaService(config)

        await service.start()

        logger.info("Training completed successfully!")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        logger.error("Training failed: %s", e)
        if args.verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
