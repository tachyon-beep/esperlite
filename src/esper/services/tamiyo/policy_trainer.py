"""
Production Policy Trainer for Enhanced Tamiyo GNN

This module implements advanced reinforcement learning training for the enhanced
Tamiyo policy system with uncertainty quantification, safety regularization,
and multi-metric reward integration.

Key Features:
- PPO/A2C reinforcement learning with safety constraints
- Prioritized experience replay with importance sampling
- Multi-metric loss functions with safety regularization
- Uncertainty-aware training with epistemic regularization
- Phase 1 integration for real-time feedback
- Production monitoring and convergence detection
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from esper.contracts.operational import AdaptationDecision

from .health_collector import ProductionHealthCollector
from .model_graph_builder import ModelGraphBuilder
from .model_graph_builder import ModelGraphState
from .policy import EnhancedPolicyTrainingState
from .policy import EnhancedTamiyoPolicyGNN
from .policy import PolicyConfig
from .reward_system import MultiMetricRewardSystem
from .reward_system import RewardConfig

logger = logging.getLogger(__name__)


@dataclass
class ProductionTrainingConfig:
    """Enhanced training configuration for production deployment."""

    # Core training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_training_steps: int = 100000
    gradient_clip_norm: float = 0.5
    target_update_freq: int = 100

    # PPO-specific parameters
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_bonus_coeff: float = 0.01
    gae_lambda: float = 0.95  # GAE advantage estimation

    # Safety and uncertainty regularization
    safety_loss_weight: float = 1.0
    uncertainty_regularization: float = 0.1
    epistemic_threshold: float = 0.2
    safety_penalty_weight: float = 2.0

    # Training stability
    min_buffer_size: int = 1000
    warmup_steps: int = 5000
    evaluation_interval: int = 1000
    early_stopping_patience: int = 10

    # Learning rate scheduling
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.7
    min_learning_rate: float = 1e-6

    # Model persistence
    checkpoint_interval: int = 5000
    model_save_dir: str = "models/tamiyo"
    experience_save_dir: str = "data/tamiyo_experience"

    # Training monitoring
    log_interval: int = 100
    tensorboard_logging: bool = True
    save_training_videos: bool = False


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking."""

    step: int = 0
    epoch: int = 0

    # Loss components
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    safety_loss: float = 0.0
    uncertainty_loss: float = 0.0
    entropy_bonus: float = 0.0

    # Performance metrics
    average_reward: float = 0.0
    success_rate: float = 0.0
    safety_violations: int = 0
    adaptation_accuracy: float = 0.0

    # Learning progress
    explained_variance: float = 0.0
    policy_gradient_norm: float = 0.0
    value_gradient_norm: float = 0.0
    learning_rate: float = 0.0

    # Uncertainty metrics
    average_uncertainty: float = 0.0
    epistemic_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "step": self.step,
            "epoch": self.epoch,
            "losses/total": self.total_loss,
            "losses/policy": self.policy_loss,
            "losses/value": self.value_loss,
            "losses/safety": self.safety_loss,
            "losses/uncertainty": self.uncertainty_loss,
            "performance/average_reward": self.average_reward,
            "performance/success_rate": self.success_rate,
            "performance/safety_violations": self.safety_violations,
            "performance/adaptation_accuracy": self.adaptation_accuracy,
            "learning/explained_variance": self.explained_variance,
            "learning/policy_grad_norm": self.policy_gradient_norm,
            "learning/value_grad_norm": self.value_gradient_norm,
            "learning/learning_rate": self.learning_rate,
            "uncertainty/average": self.average_uncertainty,
            "uncertainty/epistemic_confidence": self.epistemic_confidence,
        }


class AdvantageEstimator:
    """Generalized Advantage Estimation (GAE) for better policy gradients."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(
        self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.

        Args:
            rewards: Reward tensor [batch_size]
            values: Value predictions [batch_size + 1]
            dones: Done flags [batch_size]

        Returns:
            advantages: GAE advantages [batch_size]
            returns: Discounted returns [batch_size]
        """
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]

            delta = (
                rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            )
            advantages[t] = last_advantage = (
                delta + self.gamma * self.lam * next_non_terminal * last_advantage
            )

        returns = advantages + values[:-1]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


class ProductionPolicyTrainer:
    """
    Production-grade policy trainer with advanced RL and safety features.

    Features:
    - PPO with safety constraints and uncertainty regularization
    - Prioritized experience replay with importance sampling
    - Multi-metric loss functions and convergence detection
    - Real-time Phase 1 integration and feedback processing
    - Comprehensive training monitoring and checkpointing
    """

    def __init__(
        self,
        policy: EnhancedTamiyoPolicyGNN,
        policy_config: PolicyConfig,
        training_config: ProductionTrainingConfig,
        device: Optional[torch.device] = None,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.policy = policy
        self.policy_config = policy_config
        self.config = training_config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Move policy to device
        self.policy.to(self.device)

        # Create target network for stable training
        self.target_policy = EnhancedTamiyoPolicyGNN(policy_config).to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())

        # Initialize optimizer with advanced settings
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=training_config.learning_rate,
            weight_decay=1e-4,
            eps=1e-5,
        )

        # Learning rate scheduler
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=training_config.lr_scheduler_factor,
            patience=training_config.lr_scheduler_patience,
            min_lr=training_config.min_learning_rate,
        )

        # Training components
        self.replay_buffer = EnhancedPolicyTrainingState(policy_config)
        self.advantage_estimator = AdvantageEstimator(lam=training_config.gae_lambda)

        # Reward system
        self.reward_system = MultiMetricRewardSystem(reward_config)

        # Training state
        self.training_step = 0
        self.epoch = 0
        self.best_validation_score = float("inf")
        self.training_start_time = time.time()

        # Monitoring
        self.training_metrics_history = deque(maxlen=1000)
        self.validation_scores = deque(maxlen=100)
        self.convergence_window = deque(maxlen=training_config.early_stopping_patience)

        # Create save directories
        Path(training_config.model_save_dir).mkdir(parents=True, exist_ok=True)
        Path(training_config.experience_save_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"ProductionPolicyTrainer initialized with {sum(p.numel() for p in policy.parameters())} parameters"
        )

    async def train_with_experience_collection(
        self,
        health_collector: ProductionHealthCollector,
        graph_builder: ModelGraphBuilder,
        num_episodes: int = 1000,
    ) -> Dict[str, Any]:
        """
        Train policy with real-time experience collection from Phase 1 system.

        Args:
            health_collector: Production health signal collector
            graph_builder: Model graph state builder
            num_episodes: Number of training episodes

        Returns:
            Training summary metrics
        """
        logger.info("Starting production training with real-time experience collection")

        episode_metrics = []

        for episode in range(num_episodes):
            # Collect fresh health signals
            health_signals = await health_collector.get_recent_signals(count=500)

            if len(health_signals) < 10:
                logger.warning(
                    f"Episode {episode}: Insufficient health signals ({len(health_signals)})"
                )
                await asyncio.sleep(1.0)  # Wait for more signals
                continue

            # Build graph state
            graph_state = graph_builder.build_model_graph(health_signals)

            # Generate policy decision
            decision = self.policy.make_decision(graph_state)

            # Compute comprehensive reward using the reward system
            reward, _ = await self.reward_system.compute_reward(
                decision=decision,
                graph_state=graph_state,
                execution_metrics=None,  # Would be provided by Phase 1 in production
                health_signals=health_signals,
            )

            # Store experience
            if decision is not None:
                self.replay_buffer.add_experience(
                    state=graph_state,
                    action=decision,
                    reward=reward,
                    next_state=graph_state,  # Simplified for now
                    td_error=abs(reward),  # Simplified TD error
                )
            else:
                # No decision made - store neutral experience
                reward = 0.0

            # Training step
            if len(self.replay_buffer.experience_buffer) >= self.config.min_buffer_size:
                if self.training_step % 10 == 0:  # Train every 10 episodes
                    metrics = await self._training_step()
                    episode_metrics.append(metrics)

                    if self.training_step % self.config.log_interval == 0:
                        logger.info(
                            f"Episode {episode}, Step {self.training_step}: "
                            f"Loss={metrics.total_loss:.4f}, "
                            f"Reward={metrics.average_reward:.3f}, "
                            f"Safety={1.0 - metrics.safety_violations/10:.2f}"
                        )

            # Periodic target network update
            if self.training_step % self.config.target_update_freq == 0:
                self._update_target_network()

            # Checkpointing
            if self.training_step % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(episode)

            # Early stopping check
            if self._check_early_stopping():
                logger.info("Early stopping triggered at episode %d", episode)
                break

        # Final evaluation
        final_metrics = self._compute_training_summary(episode_metrics)
        logger.info("Training completed. Final metrics: %s", final_metrics)

        return final_metrics

    async def _training_step(self) -> TrainingMetrics:
        """Execute one training step with prioritized experience replay."""
        self.policy.train()

        # Sample prioritized batch
        experiences, importance_weights = self.replay_buffer.sample_batch(
            batch_size=self.config.batch_size, use_prioritization=True
        )

        if len(experiences) < self.config.batch_size // 2:
            # Not enough experiences yet
            return TrainingMetrics(step=self.training_step)

        # Convert experiences to training batch
        batch_data = self._prepare_training_batch(experiences, importance_weights)

        # Compute losses
        loss_dict = self._compute_losses(batch_data)

        # Multiple PPO epochs
        total_metrics = TrainingMetrics()

        for _ in range(self.config.ppo_epochs):
            # Backward pass
            self.optimizer.zero_grad()
            loss_dict["total_loss"].backward()

            # Gradient clipping
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.gradient_clip_norm
            )

            self.optimizer.step()

            # Accumulate metrics
            total_metrics.policy_loss += loss_dict["policy_loss"].item()
            total_metrics.value_loss += loss_dict["value_loss"].item()
            total_metrics.safety_loss += loss_dict.get(
                "safety_loss", torch.tensor(0.0)
            ).item()
            total_metrics.uncertainty_loss += loss_dict.get(
                "uncertainty_loss", torch.tensor(0.0)
            ).item()
            total_metrics.policy_gradient_norm = policy_grad_norm.item()

        # Average over PPO epochs
        total_metrics.policy_loss /= self.config.ppo_epochs
        total_metrics.value_loss /= self.config.ppo_epochs
        total_metrics.safety_loss /= self.config.ppo_epochs
        total_metrics.uncertainty_loss /= self.config.ppo_epochs
        total_metrics.total_loss = (
            total_metrics.policy_loss
            + total_metrics.value_loss
            + total_metrics.safety_loss
            + total_metrics.uncertainty_loss
        )

        # Update training state
        self.training_step += 1
        total_metrics.step = self.training_step
        total_metrics.learning_rate = self.optimizer.param_groups[0]["lr"]

        # Performance metrics from experiences
        rewards = [exp["reward"] for exp in experiences]
        total_metrics.average_reward = np.mean(rewards)
        total_metrics.success_rate = sum(1 for r in rewards if r > 0) / len(rewards)

        # Store metrics
        self.training_metrics_history.append(total_metrics)

        return total_metrics

    def _prepare_training_batch(
        self, experiences: List[Dict[str, Any]], importance_weights: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert experiences to training batch tensors."""
        # Extract graph data from experiences
        node_features_list = []
        edge_indices_list = []
        batch_indices = []

        for i, exp in enumerate(experiences):
            graph_state = exp["state"]
            graph_data = graph_state.graph_data

            node_features_list.append(graph_data.x)
            edge_indices_list.append(graph_data.edge_index)

            # Create batch indices for this graph
            num_nodes = graph_data.x.size(0)
            batch_indices.extend([i] * num_nodes)

        # Concatenate all graphs into one batch
        node_features = torch.cat(node_features_list, dim=0).to(self.device)

        # Offset edge indices for batching
        edge_index_list_offset = []
        node_offset = 0
        for edge_idx in edge_indices_list:
            edge_index_list_offset.append(edge_idx + node_offset)
            node_offset += edge_idx.max().item() + 1

        edge_indices = torch.cat(edge_index_list_offset, dim=1).to(self.device)
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long).to(self.device)

        # Extract action and reward information
        actions = []
        rewards = []

        for exp in experiences:
            action_decision = exp.get("action")
            if action_decision:
                # Convert adaptation decision to action tensor
                actions.append(1.0 if action_decision.confidence > 0.5 else 0.0)
            else:
                actions.append(0.0)

            rewards.append(exp.get("reward", 0.0))

        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)

        # Compute advantages using GAE
        with torch.no_grad():
            # Get value predictions for advantage calculation
            policy_outputs = self.policy.forward(
                node_features, edge_indices, batch_tensor, return_uncertainty=False
            )
            values = policy_outputs["value_estimate"].squeeze()

            # Pad values for GAE calculation
            values_padded = torch.cat([values, values[-1:]])  # Add final value
            dones = torch.zeros_like(rewards)  # Assume no episode termination

            advantages, returns = self.advantage_estimator.compute_advantages(
                rewards, values_padded, dones
            )

        return {
            "node_features": node_features,
            "edge_indices": edge_indices,
            "batch": batch_tensor,
            "actions": actions,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
            "importance_weights": importance_weights.to(self.device),
        }

    def _compute_losses(
        self, batch_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute all loss components for training."""
        # Forward pass
        policy_outputs = self.policy.forward(
            batch_data["node_features"],
            batch_data["edge_indices"],
            batch_data["batch"],
            return_uncertainty=self.policy_config.enable_uncertainty,
        )

        adaptation_probs = policy_outputs["adaptation_prob"]
        value_estimates = policy_outputs["value_estimate"].squeeze()
        safety_scores = policy_outputs["safety_score"]

        # PPO policy loss with clipping
        actions = batch_data["actions"]
        advantages = batch_data["advantages"]
        importance_weights = batch_data["importance_weights"]

        # Compute log probabilities
        action_log_probs = torch.log(adaptation_probs + 1e-8) * actions + torch.log(
            1 - adaptation_probs + 1e-8
        ) * (1 - actions)

        # PPO clipped objective
        ratio = torch.exp(action_log_probs)  # Simplified - should use old log probs
        clipped_ratio = torch.clamp(
            ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
        )

        policy_loss_1 = ratio * advantages
        policy_loss_2 = clipped_ratio * advantages
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        # Apply importance sampling weights
        policy_loss = policy_loss * importance_weights.mean()

        # Value function loss
        value_loss = nn.MSELoss()(value_estimates, batch_data["returns"])

        # Safety regularization loss
        safety_targets = torch.ones_like(safety_scores)  # Want high safety scores
        safety_loss = (
            nn.BCELoss()(safety_scores, safety_targets) * self.config.safety_loss_weight
        )

        # Uncertainty regularization (if enabled)
        uncertainty_loss = torch.tensor(0.0, device=self.device)
        if "uncertainty" in policy_outputs:
            # Encourage lower uncertainty for confident decisions
            uncertainty = policy_outputs["uncertainty"]
            confidence_mask = (
                adaptation_probs > self.policy_config.adaptation_confidence_threshold
            ).float()
            uncertainty_loss = (
                uncertainty * confidence_mask
            ).mean() * self.config.uncertainty_regularization

        # Entropy bonus for exploration
        entropy_bonus = -(
            adaptation_probs * torch.log(adaptation_probs + 1e-8)
            + (1 - adaptation_probs) * torch.log(1 - adaptation_probs + 1e-8)
        ).mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_loss_coeff * value_loss
            + safety_loss
            + uncertainty_loss
            - self.config.entropy_bonus_coeff * entropy_bonus
        )

        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "safety_loss": safety_loss,
            "uncertainty_loss": uncertainty_loss,
            "entropy_bonus": entropy_bonus,
        }

    def _simulate_environment_feedback(
        self, decision: Optional[AdaptationDecision], graph_state: ModelGraphState
    ) -> Tuple[float, bool]:
        """
        Simulate environment feedback for training.

        In production, this would come from actual Phase 1 execution results.
        """
        if decision is None:
            return 0.0, False  # No action taken

        # Simulate reward based on decision quality and graph state
        base_reward = 0.5

        # Reward for high confidence decisions
        confidence_bonus = (decision.confidence - 0.5) * 0.4

        # Penalty for low safety score (simulated)
        safety_penalty = 0.0
        if decision.metadata.get("safety_score", 1.0) < 0.7:
            safety_penalty = -0.3

        # Reward for addressing problematic layers
        if (
            len(graph_state.problematic_layers) > 0
            and decision.layer_name in graph_state.problematic_layers
        ):
            problem_solving_bonus = 0.3
        else:
            problem_solving_bonus = 0.0

        # Simulate some randomness for realistic training
        noise = np.random.normal(0, 0.1)

        reward = (
            base_reward
            + confidence_bonus
            + problem_solving_bonus
            - safety_penalty
            + noise
        )
        reward = np.clip(reward, -1.0, 1.0)  # Clip to reasonable range

        done = False  # Episodes don't terminate in continuous training

        return float(reward), done

    def _update_target_network(self):
        """Update target network for stable training."""
        self.target_policy.load_state_dict(self.policy.state_dict())
        logger.debug("Updated target network at step %d", self.training_step)

    async def _save_checkpoint(self, episode: int):
        """Save comprehensive training checkpoint."""
        checkpoint = {
            "episode": episode,
            "training_step": self.training_step,
            "epoch": self.epoch,
            "policy_state_dict": self.policy.state_dict(),
            "target_policy_state_dict": self.target_policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "best_validation_score": self.best_validation_score,
            "training_config": self.config,
            "policy_config": self.policy_config,
            "training_time": time.time() - self.training_start_time,
        }

        checkpoint_path = (
            Path(self.config.model_save_dir)
            / f"checkpoint_step_{self.training_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)

        # Also save as latest checkpoint
        latest_path = Path(self.config.model_save_dir) / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)

        logger.info("Saved checkpoint at step %d", self.training_step)

    def _check_early_stopping(self) -> bool:
        """Check if training should stop early."""
        if len(self.convergence_window) < self.config.early_stopping_patience:
            return False

        # Check if validation score stopped improving
        recent_scores = list(self.convergence_window)
        if all(score >= recent_scores[0] for score in recent_scores[1:]):
            logger.info("Early stopping: no improvement in validation score")
            return True

        return False

    def _compute_training_summary(
        self, episode_metrics: List[TrainingMetrics]
    ) -> Dict[str, Any]:
        """Compute comprehensive training summary."""
        if not episode_metrics:
            return {}

        recent_metrics = episode_metrics[-10:]  # Last 10 episodes

        return {
            "total_episodes": len(episode_metrics),
            "total_training_steps": self.training_step,
            "training_time_hours": (time.time() - self.training_start_time) / 3600,
            "final_average_reward": np.mean([m.average_reward for m in recent_metrics]),
            "final_success_rate": np.mean([m.success_rate for m in recent_metrics]),
            "final_policy_loss": np.mean([m.policy_loss for m in recent_metrics]),
            "final_value_loss": np.mean([m.value_loss for m in recent_metrics]),
            "final_safety_loss": np.mean([m.safety_loss for m in recent_metrics]),
            "best_validation_score": self.best_validation_score,
            "convergence_achieved": len(self.convergence_window)
            >= self.config.early_stopping_patience,
            "model_parameters": sum(p.numel() for p in self.policy.parameters()),
            "device": str(self.device),
        }

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.target_policy.load_state_dict(checkpoint["target_policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])

        self.training_step = checkpoint["training_step"]
        self.epoch = checkpoint["epoch"]
        self.best_validation_score = checkpoint["best_validation_score"]

        logger.info("Loaded checkpoint from step %d", self.training_step)

        return {
            "episode": checkpoint["episode"],
            "training_step": self.training_step,
            "training_time": checkpoint.get("training_time", 0),
            "best_validation_score": self.best_validation_score,
        }

    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        if not self.training_metrics_history:
            return {}

        recent_metrics = list(self.training_metrics_history)[-100:]  # Last 100 steps

        base_stats = {
            "training_step": self.training_step,
            "epoch": self.epoch,
            "training_time_minutes": (time.time() - self.training_start_time) / 60,
            "recent_average_reward": np.mean(
                [m.average_reward for m in recent_metrics]
            ),
            "recent_success_rate": np.mean([m.success_rate for m in recent_metrics]),
            "recent_policy_loss": np.mean([m.policy_loss for m in recent_metrics]),
            "recent_value_loss": np.mean([m.value_loss for m in recent_metrics]),
            "recent_safety_loss": np.mean([m.safety_loss for m in recent_metrics]),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "replay_buffer_size": len(self.replay_buffer.experience_buffer),
            "best_validation_score": self.best_validation_score,
            "device": str(self.device),
        }

        # Add reward system statistics
        reward_stats = self.reward_system.get_reward_statistics()
        base_stats["reward_system"] = reward_stats

        return base_stats

    def get_reward_correlations(self) -> Dict[str, Any]:
        """Get reward system correlation analysis."""
        return self.reward_system.get_correlations()
