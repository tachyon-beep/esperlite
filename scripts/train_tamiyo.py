
#!/usr/bin/env python3
"""
Standalone training script for Tamiyo GNN policy.

This script trains the Tamiyo strategic controller policy using offline
reinforcement learning on collected experience data.
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from esper.services.tamiyo.policy import PolicyConfig
from esper.services.tamiyo.policy import TamiyoPolicyGNN
from esper.services.tamiyo.training import TamiyoTrainer
from esper.services.tamiyo.training import TrainingConfig
from esper.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def create_synthetic_experience_data(num_samples: int = 1000) -> list:
    """
    Create synthetic experience data for training demonstration.
    
    In a real implementation, this would load actual experience data
    collected from the Tamiyo service during operation.
    """
    import random
    import time

    experiences = []

    for i in range(num_samples):
        # Simulate varying health scenarios
        base_health = random.uniform(0.2, 0.9)

        # Simulate adaptation decisions
        should_adapt = base_health < 0.5

        # Simulate rewards based on adaptation effectiveness
        if should_adapt:
            # Positive reward if adaptation was beneficial
            reward = random.uniform(0.5, 1.0) if random.random() > 0.3 else random.uniform(-0.2, 0.2)
        else:
            # Small negative reward for unnecessary adaptation
            reward = random.uniform(-0.1, 0.1)

        experience = {
            "state": {
                "health_score": base_health,
                "latency": random.uniform(0.01, 0.05),
                "error_count": random.randint(0, 5)
            },
            "action": should_adapt,
            "reward": reward,
            "next_state": {
                "health_score": min(0.95, base_health + (0.2 if should_adapt and reward > 0 else 0)),
                "latency": random.uniform(0.01, 0.05),
                "error_count": random.randint(0, 3)
            },
            "timestamp": time.time() + i
        }

        experiences.append(experience)

    logger.info("Generated %d synthetic experiences", len(experiences))
    return experiences


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Tamiyo GNN Policy")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--experience-data",
        type=str,
        help="Path to experience data file (optional)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate if no experience data provided"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("tamiyo-trainer", level=logging.DEBUG if args.verbose else logging.INFO)

    logger.info("Starting Tamiyo policy training...")

    # Determine device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info("Using device: %s", device)

    # Create configurations
    policy_config = PolicyConfig()

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_save_path=str(Path(args.output_dir) / "tamiyo_policy.pt"),
        training_data_path=str(Path(args.output_dir) / "experience_data.pkl")
    )

    # Create model and trainer
    policy = TamiyoPolicyGNN(policy_config)
    trainer = TamiyoTrainer(policy, training_config, device)

    # Load checkpoint if specified
    if args.resume:
        logger.info("Resuming from checkpoint: %s", args.resume)
        trainer.load_checkpoint(args.resume)

    # Load or generate experience data
    if args.experience_data and Path(args.experience_data).exists():
        logger.info("Loading experience data from: %s", args.experience_data)
        experience_data = trainer.load_experience_data()
    else:
        logger.info("Generating synthetic experience data...")
        experience_data = create_synthetic_experience_data(args.synthetic_samples)

        # Save synthetic data
        trainer.save_experience_data(experience_data)

    if not experience_data:
        logger.error("No experience data available for training")
        return

    logger.info("Training on %d experiences...", len(experience_data))

    # Train the policy
    try:
        metrics = trainer.train_from_experience(experience_data)

        logger.info("Training completed successfully!")
        logger.info("Final metrics: %s", metrics)

        # Log key results
        print("\n" + "="*50)
        print("TRAINING RESULTS")
        print("="*50)
        print(f"Total epochs: {metrics.get('total_epochs', 'N/A')}")
        print(f"Final validation loss: {metrics.get('final_val_loss', 'N/A'):.4f}")
        print(f"Final validation accuracy: {metrics.get('final_val_accuracy', 'N/A'):.4f}")
        print(f"Best validation loss: {metrics.get('best_val_loss', 'N/A'):.4f}")
        print(f"Best validation accuracy: {metrics.get('best_val_accuracy', 'N/A'):.4f}")
        print(f"Model saved to: {training_config.model_save_path}")
        print("="*50)

    except Exception as e:
        logger.error("Training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
