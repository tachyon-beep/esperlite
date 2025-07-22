"""
Offline policy training infrastructure for Tamiyo.

This module implements the training procedures for improving the GNN policy
through reinforcement learning on collected experience data.
"""

import logging
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .policy import TamiyoPolicyGNN

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for policy training."""

    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0

    # PPO-specific parameters
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_bonus_coeff: float = 0.01

    # Data management
    min_buffer_size: int = 1000
    max_buffer_size: int = 50000
    checkpoint_interval: int = 10

    # Model persistence
    model_save_path: str = "models/tamiyo_policy.pt"
    training_data_path: str = "data/tamiyo_experience.pkl"


class TamiyoTrainer:
    """
    Trainer for the Tamiyo GNN policy using offline reinforcement learning.

    This class implements policy gradient methods (specifically PPO) to improve
    the strategic decision-making capabilities of the Tamiyo controller.
    """

    def __init__(
        self,
        policy: TamiyoPolicyGNN,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
    ):
        self.policy = policy
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move policy to device
        self.policy.to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(), lr=config.learning_rate, weight_decay=1e-5
        )

        # Training state
        self.training_step = 0
        self.best_validation_score = float("-inf")

        # Experience storage
        self.experience_buffer: List[Dict[str, Any]] = []

    def train_from_experience(
        self, experience_data: List[Dict[str, Any]], validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the policy from collected experience data.

        Args:
            experience_data: List of experience tuples
            validation_split: Fraction of data to use for validation

        Returns:
            Dictionary of training metrics
        """
        logger.info("Starting training with %d experiences", len(experience_data))

        # Split data
        split_idx = int(len(experience_data) * (1 - validation_split))
        train_data = experience_data[:split_idx]
        val_data = experience_data[split_idx:]

        # Training loop
        training_metrics = []

        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_data)

            # Validation phase
            val_metrics = self._validate_epoch(val_data)

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_policy_loss": train_metrics["policy_loss"],
                "train_value_loss": train_metrics["value_loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }

            training_metrics.append(epoch_metrics)

            # Logging
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Accuracy: {val_metrics['accuracy']:.4f}"
                )

            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch, val_metrics["loss"])

        # Save final checkpoint
        self._save_checkpoint(self.config.num_epochs - 1, val_metrics["loss"])

        # Final evaluation
        final_metrics = self._compute_final_metrics(training_metrics)
        logger.info("Training completed. Final metrics: %s", final_metrics)

        return final_metrics

    def _train_epoch(self, train_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train for one epoch."""
        self.policy.train()

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0

        # Shuffle data
        rng = np.random.default_rng(seed=42)
        rng.shuffle(train_data)

        # Process in batches
        for i in range(0, len(train_data), self.config.batch_size):
            batch = train_data[i : i + self.config.batch_size]

            # Convert batch to tensors
            batch_tensors = self._prepare_batch(batch)

            # Multiple PPO updates per batch
            for _ in range(self.config.ppo_epochs):
                loss_dict = self._compute_ppo_loss(batch_tensors)

                # Backward pass
                self.optimizer.zero_grad()
                loss_dict["total_loss"].backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.gradient_clip_norm
                )

                self.optimizer.step()

                # Accumulate losses
                total_loss += loss_dict["total_loss"].item()
                total_policy_loss += loss_dict["policy_loss"].item()
                total_value_loss += loss_dict["value_loss"].item()
                num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "policy_loss": total_policy_loss / max(1, num_batches),
            "value_loss": total_value_loss / max(1, num_batches),
        }

    def _validate_epoch(self, val_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Validate for one epoch."""
        self.policy.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i in range(0, len(val_data), self.config.batch_size):
                batch = val_data[i : i + self.config.batch_size]
                batch_tensors = self._prepare_batch(batch)

                # Forward pass - policy now returns a dictionary
                policy_outputs = self.policy(
                    batch_tensors["node_features"],
                    batch_tensors["edge_index"],
                    batch_tensors["batch"],
                )
                adapt_prob = policy_outputs["adaptation_prob"]
                value_est = policy_outputs["value_estimate"]

                # Compute losses
                policy_loss = self._compute_policy_loss(
                    adapt_prob, batch_tensors["actions"], batch_tensors["advantages"]
                )
                value_loss = self._compute_value_loss(
                    value_est, batch_tensors["returns"]
                )

                total_loss += (policy_loss + value_loss).item()

                # Accuracy calculation
                predictions = (adapt_prob > 0.5).float()
                targets = batch_tensors["actions"].float()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += len(predictions)

        accuracy = correct_predictions / max(1, total_predictions)
        avg_loss = total_loss / max(1, len(val_data) // self.config.batch_size + 1)

        return {"loss": avg_loss, "accuracy": accuracy}

    def _prepare_batch(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Convert a batch of experiences to tensors."""
        # For MVP, create simplified tensor representations
        batch_size = len(batch)

        # Placeholder implementation - in a real system this would
        # convert ModelGraphState to proper graph tensors
        # Use the same node feature dimension as PolicyConfig default (20 with gradient features)
        node_features = torch.randn(
            batch_size * 4, 20, device=self.device
        )  # 4 nodes per graph, 20 features (includes gradient features)

        # Create edge indices for batch of graphs
        edge_indices = []
        batch_indices = []
        for i in range(batch_size):
            base_idx = i * 4
            # Sequential connectivity
            edges = torch.tensor(
                [
                    [
                        base_idx,
                        base_idx + 1,
                        base_idx + 1,
                        base_idx + 2,
                        base_idx + 2,
                        base_idx + 3,
                    ],
                    [
                        base_idx + 1,
                        base_idx,
                        base_idx + 2,
                        base_idx + 1,
                        base_idx + 3,
                        base_idx + 2,
                    ],
                ],
                device=self.device,
            )
            edge_indices.append(edges)
            batch_indices.extend([i] * 4)

        edge_index = torch.cat(edge_indices, dim=1)
        batch_tensor = torch.tensor(batch_indices, device=self.device)

        # Extract actions and rewards (simplified)
        actions = torch.tensor(
            [1.0 if exp.get("action") else 0.0 for exp in batch], device=self.device
        )

        rewards = torch.tensor(
            [exp.get("reward", 0.0) for exp in batch], device=self.device
        )

        # Compute advantages and returns (simplified)
        advantages = rewards - rewards.mean()
        returns = rewards

        return {
            "node_features": node_features,
            "edge_index": edge_index,
            "batch": batch_tensor,
            "actions": actions,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
        }

    def _compute_ppo_loss(
        self, batch_tensors: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO loss components."""
        # Forward pass - policy now returns a dictionary
        policy_outputs = self.policy(
            batch_tensors["node_features"],
            batch_tensors["edge_index"],
            batch_tensors["batch"],
        )
        adapt_prob = policy_outputs["adaptation_prob"]
        value_est = policy_outputs["value_estimate"]

        # Policy loss
        policy_loss = self._compute_policy_loss(
            adapt_prob, batch_tensors["actions"], batch_tensors["advantages"]
        )

        # Value loss
        value_loss = self._compute_value_loss(value_est, batch_tensors["returns"])

        # Entropy bonus (for exploration)
        entropy = self._compute_entropy(adapt_prob)

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coeff * value_loss
            - self.config.entropy_bonus_coeff * entropy
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

    def _compute_policy_loss(
        self, log_probs: torch.Tensor, actions: torch.Tensor, advantages: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss."""
        # Simplified policy loss (in real implementation would use PPO clipping)
        action_log_probs = torch.log(log_probs + 1e-8) * actions + torch.log(
            1 - log_probs + 1e-8
        ) * (1 - actions)
        return -(action_log_probs * advantages).mean()

    def _compute_value_loss(
        self, value_predictions: torch.Tensor, returns: torch.Tensor
    ) -> torch.Tensor:
        """Compute value function loss."""
        return nn.MSELoss()(value_predictions.squeeze(), returns)

    def _compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy for exploration bonus."""
        return -(
            probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)
        ).mean()

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save training checkpoint."""
        # Create directory if it doesn't exist
        save_path = Path(self.config.model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "training_step": self.training_step,
        }

        # Always save to the main model path
        torch.save(checkpoint, save_path)
        logger.info("Saved checkpoint to: %s", save_path)

        # Update best validation score
        if val_loss < self.best_validation_score:
            self.best_validation_score = val_loss
            logger.info("New best model with validation loss: %.4f", val_loss)

        # Save epoch-specific checkpoint for debugging
        checkpoint_path = save_path.parent / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

    def _compute_final_metrics(
        self, training_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute final training metrics."""
        if not training_metrics:
            return {}

        final_epoch = training_metrics[-1]
        best_val_loss = min(m["val_loss"] for m in training_metrics)
        best_val_acc = max(m["val_accuracy"] for m in training_metrics)

        return {
            "final_train_loss": final_epoch["train_loss"],
            "final_val_loss": final_epoch["val_loss"],
            "final_val_accuracy": final_epoch["val_accuracy"],
            "best_val_loss": best_val_loss,
            "best_val_accuracy": best_val_acc,
            "total_epochs": len(training_metrics),
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint.get("training_step", 0)
        self.best_validation_score = checkpoint.get("val_loss", float("inf"))

        logger.info("Loaded checkpoint from epoch %d", checkpoint['epoch'])

    def save_experience_data(self, experience_data: List[Dict[str, Any]]) -> None:
        """Save experience data to disk."""
        save_path = Path(self.config.training_data_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Change extension to .json
        if save_path.suffix != '.json':
            save_path = save_path.with_suffix('.json')

        # Convert tensors to lists for JSON serialization
        serializable_data = []
        for exp in experience_data:
            serializable_exp = {}
            for key, value in exp.items():
                if isinstance(value, torch.Tensor):
                    serializable_exp[key] = value.tolist()
                else:
                    serializable_exp[key] = value
            serializable_data.append(serializable_exp)

        with open(save_path, "w") as f:
            json.dump(serializable_data, f, indent=2)

        logger.info("Saved %d experiences to %s", len(experience_data), save_path)

    def load_experience_data(self) -> List[Dict[str, Any]]:
        """Load experience data from disk."""
        load_path = Path(self.config.training_data_path)
        
        # Check for .json file
        if load_path.suffix != '.json':
            load_path = load_path.with_suffix('.json')

        if not load_path.exists():
            logger.warning("No experience data found at %s", load_path)
            return []

        with open(load_path, "r") as f:
            serializable_data = json.load(f)
        
        # Convert lists back to tensors
        experience_data = []
        for exp in serializable_data:
            reconstructed_exp = {}
            for key, value in exp.items():
                if isinstance(value, list) and key in ['state', 'action', 'reward', 'next_state']:
                    reconstructed_exp[key] = torch.tensor(value)
                else:
                    reconstructed_exp[key] = value
            experience_data.append(reconstructed_exp)

        logger.info("Loaded %d experiences from %s", len(experience_data), load_path)
        return experience_data
