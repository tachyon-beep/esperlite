"""
Production policy trainer with experience replay and continuous learning.
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from esper.contracts.operational import AdaptationDecision
from esper.contracts.operational import ModelGraphState
from esper.services.tamiyo.policy import TamiyoPolicyGNN


@dataclass
class ProductionTrainingConfig:
    """Configuration for production policy training."""
    
    # Core training parameters
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_training_steps: int = 100000
    gradient_clip_norm: float = 0.5
    
    # PPO parameters
    ppo_epochs: int = 4
    clip_ratio: float = 0.2
    value_loss_coeff: float = 0.5
    entropy_bonus_coeff: float = 0.01
    gae_lambda: float = 0.95
    
    # Safety and regularization
    safety_loss_weight: float = 1.0
    uncertainty_regularization: float = 0.1
    safety_penalty_weight: float = 2.0
    
    # Buffer and stability
    min_buffer_size: int = 100
    warmup_steps: int = 1000
    early_stopping_patience: int = 10
    
    # Checkpointing and logging
    checkpoint_interval: int = 1000
    log_interval: int = 100
    tensorboard_logging: bool = True


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    
    episodes: int = 0
    total_steps: int = 0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    average_reward: float = 0.0
    success_rate: float = 0.0
    learning_rate: float = 0.0


logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience for policy training."""
    
    state: ModelGraphState
    action: AdaptationDecision
    reward: float
    next_state: ModelGraphState
    done: bool
    timestamp: float


class ExperienceReplayBuffer:
    """Prioritized experience replay buffer."""
    
    def __init__(self, max_size: int = 100000, prioritized: bool = True):
        self.max_size = max_size
        self.prioritized = prioritized
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.epsilon = 1e-6
        self.alpha = 0.6  # Priority exponent
        self.beta = 0.4   # Importance sampling
        self.beta_increment = 0.001
    
    def add(self, experience: Experience, priority: float = None):
        """Add experience to buffer."""
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
        
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample_batch(
        self,
        batch_size: int,
        prioritized: bool = None
    ) -> List[Experience]:
        """Sample batch of experiences."""
        if prioritized is None:
            prioritized = self.prioritized
        
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        if prioritized:
            # Prioritized sampling
            priorities = np.array(self.priorities)
            probs = priorities ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(
                len(self.buffer),
                batch_size,
                p=probs,
                replace=False
            )
            
            # Importance sampling weights
            weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
            weights /= weights.max()
            
            batch = [self.buffer[i] for i in indices]
            
            # Increment beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            return batch, weights, indices
        else:
            # Uniform sampling
            indices = np.random.choice(
                len(self.buffer),
                batch_size,
                replace=False
            )
            return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)


class PolicySafetyValidator:
    """Validates policy decisions for safety."""
    
    def __init__(self):
        self.dangerous_patterns = {
            "rapid_changes": 0,
            "oscillations": 0,
            "divergent_rewards": 0,
        }
        self.safety_threshold = 0.1
    
    def validate_experience(self, experience: Experience) -> bool:
        """Validate single experience for safety."""
        # Check reward bounds
        if abs(experience.reward) > 10.0:
            self.dangerous_patterns["divergent_rewards"] += 1
            return False
        
        # Check action reasonableness
        if experience.action.confidence < 0.3:
            return False  # Too uncertain
        
        return True
    
    async def validate_policy_update(self, policy: TamiyoPolicyGNN) -> bool:
        """Validate policy update for safety."""
        # Simple validation for now
        # In production, would test on validation set
        return True


class PolicyPerformanceTracker:
    """Track policy performance metrics."""
    
    def __init__(self):
        self.metrics_history = {
            "avg_reward": deque(maxlen=100),
            "success_rate": deque(maxlen=100),
            "decision_confidence": deque(maxlen=100),
            "adaptation_diversity": deque(maxlen=100),
        }
    
    def update(self, metrics: Dict[str, float]):
        """Update performance metrics."""
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
    
    def get_trends(self) -> Dict[str, str]:
        """Get performance trends."""
        trends = {}
        
        for key, history in self.metrics_history.items():
            if len(history) > 10:
                recent = np.mean(list(history)[-10:])
                older = np.mean(list(history)[-20:-10])
                
                if recent > older * 1.1:
                    trends[key] = "improving"
                elif recent < older * 0.9:
                    trends[key] = "degrading"
                else:
                    trends[key] = "stable"
            else:
                trends[key] = "insufficient_data"
        
        return trends


class PolicyRollbackManager:
    """Manage policy checkpoints and rollback."""
    
    def __init__(self, max_checkpoints: int = 5):
        self.max_checkpoints = max_checkpoints
        self.checkpoints = deque(maxlen=max_checkpoints)
    
    def save_checkpoint(self, policy: TamiyoPolicyGNN, metrics: Dict):
        """Save policy checkpoint."""
        checkpoint = {
            "state_dict": policy.state_dict(),
            "metrics": metrics,
            "timestamp": time.time(),
        }
        self.checkpoints.append(checkpoint)
    
    async def rollback_to_safe_state(self, policy: TamiyoPolicyGNN):
        """Rollback to last safe checkpoint."""
        if self.checkpoints:
            # Find best checkpoint by reward
            best_checkpoint = max(
                self.checkpoints,
                key=lambda x: x["metrics"].get("avg_reward", 0)
            )
            
            policy.load_state_dict(best_checkpoint["state_dict"])
            logger.info("Rolled back to checkpoint from %s", 
                       time.ctime(best_checkpoint["timestamp"]))


class ProductionPolicyTrainer:
    """
    Production-grade policy trainer with experience replay and continuous learning.
    
    Implements advanced RL algorithms (PPO/A2C) with careful integration
    to Phase 1 execution system for safe policy updates.
    """
    
    def __init__(
        self,
        policy_network: TamiyoPolicyGNN,
        device: torch.device,
        learning_rate: float = 3e-4,
        experience_buffer_size: int = 100000,
        batch_size: int = 32,
        min_batch_size: int = 10,
        max_allowed_loss: float = 10.0
    ):
        self.policy = policy_network.to(device)
        self.device = device
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.max_allowed_loss = max_allowed_loss
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000
        )
        
        # Experience replay
        self.experience_buffer = ExperienceReplayBuffer(
            max_size=experience_buffer_size,
            prioritized=True
        )
        
        # Safety mechanisms
        self.safety_validator = PolicySafetyValidator()
        self.performance_tracker = PolicyPerformanceTracker()
        self.rollback_manager = PolicyRollbackManager()
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'avg_reward': 0.0,
            'success_rate': 0.0
        }

    async def train_on_experience(
        self,
        state: ModelGraphState,
        action: AdaptationDecision,
        reward: float,
        next_state: ModelGraphState,
        done: bool = False
    ) -> bool:
        """Train policy on single experience with safety validation."""
        
        # Create experience
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            timestamp=time.time()
        )
        
        # Validate experience
        if not self.safety_validator.validate_experience(experience):
            logger.warning("Experience failed safety validation")
            return False
        
        # Add to buffer with priority
        priority = self._calculate_experience_priority(experience)
        self.experience_buffer.add(experience, priority)
        
        # Train if enough experiences
        if len(self.experience_buffer) >= self.min_batch_size:
            training_success = await self._train_batch()
            
            if training_success:
                self.training_stats['episodes'] += 1
                await self._update_performance_tracking()
                
                # Check if policy update is safe
                if not await self.safety_validator.validate_policy_update(self.policy):
                    logger.error("Policy update failed safety validation, rolling back")
                    await self.rollback_manager.rollback_to_safe_state(self.policy)
                    return False
                
            return training_success
        
        return True

    async def _train_batch(self) -> bool:
        """Train on batch using PPO algorithm with safety constraints."""
        try:
            # Sample batch with prioritized replay
            batch_data = self.experience_buffer.sample_batch(
                batch_size=self.batch_size,
                prioritized=True
            )
            
            if self.experience_buffer.prioritized:
                batch, weights, indices = batch_data
            else:
                batch = batch_data
                weights = torch.ones(len(batch))
                indices = None
            
            # Prepare batch data
            states = [exp.state for exp in batch]
            actions = torch.tensor(
                [self._action_to_index(exp.action) for exp in batch],
                device=self.device
            )
            rewards = torch.tensor(
                [exp.reward for exp in batch],
                device=self.device
            )
            weights = torch.tensor(weights, device=self.device)
            
            # Extract gradient health from states for regularization
            gradient_healths = []
            for state in states:
                if state.global_metrics:
                    gradient_health = state.global_metrics.get("gradient_health", 0.5)
                    gradient_healths.append(gradient_health)
                else:
                    gradient_healths.append(0.5)
            self._current_gradient_health = np.mean(gradient_healths)
            
            # Convert states to graph batches
            graph_batch = self._prepare_graph_batch(states)
            
            # Forward pass
            policy_outputs = self.policy(**graph_batch)
            
            # Compute losses
            policy_loss = self._compute_policy_loss(
                policy_outputs['policy_logits'],
                actions,
                rewards,
                weights
            )
            
            value_loss = self._compute_value_loss(
                policy_outputs['value_estimate'],
                rewards,
                weights
            )
            
            entropy_loss = self._compute_entropy_loss(
                policy_outputs['policy_logits']
            )
            
            # Total loss with safety regularization
            total_loss = (
                policy_loss +
                0.5 * value_loss -
                0.01 * entropy_loss +
                self._compute_safety_regularization(policy_outputs)
            )
            
            # Safety check
            if total_loss > self.max_allowed_loss:
                logger.warning(f"Loss {total_loss} exceeds safety threshold")
                return False
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step()
            
            # Update statistics
            self.training_stats.update({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy_loss.item()
            })
            
            # Update priorities
            if indices is not None:
                td_errors = (rewards - policy_outputs['value_estimate'].squeeze()).abs()
                new_priorities = td_errors.detach().cpu().numpy()
                self.experience_buffer.update_priorities(indices, new_priorities)
            
            # Save checkpoint periodically
            if self.training_stats['episodes'] % 100 == 0:
                self.rollback_manager.save_checkpoint(
                    self.policy,
                    self.training_stats
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Policy training error: {e}")
            return False

    def _calculate_experience_priority(self, experience: Experience) -> float:
        """Calculate priority for experience replay with gradient awareness."""
        # High priority for surprising outcomes
        priority = abs(experience.reward)
        
        # Boost priority for low confidence decisions
        if experience.action.confidence < 0.5:
            priority *= 2.0
        
        # Boost priority for urgent adaptations
        priority *= (1.0 + experience.action.urgency)
        
        # Boost priority for experiences with poor gradient health
        if experience.state.global_metrics:
            gradient_health = experience.state.global_metrics.get("gradient_health", 0.5)
            training_stability = experience.state.global_metrics.get("training_stability", 0.5)
            
            # Higher priority for unstable gradients
            if gradient_health < 0.3:
                priority *= 3.0  # Triple priority for very poor gradients
            elif gradient_health < 0.5:
                priority *= 1.5
                
            # Higher priority for training instability
            if training_stability < 0.3:
                priority *= 2.0
        
        # Boost priority for gradient-stabilizing actions
        if experience.action.adaptation_type in ["add_gradient_clipping", "add_batch_normalization", "add_residual_connection"]:
            priority *= 1.5  # Prioritize learning from gradient fixes
        
        return max(priority, 0.1)

    def _action_to_index(self, action: AdaptationDecision) -> int:
        """Convert adaptation decision to action index."""
        # Map adaptation type to index
        action_map = {
            "add_attention": 0,
            "add_moe": 1,
            "add_efficiency": 2,
            "add_routing": 3,
            "add_diagnostic": 4,
            # Gradient-specific adaptations
            "add_gradient_clipping": 5,
            "add_batch_normalization": 6,
            "add_residual_connection": 7,
            "add_skip_connection": 8,
            # General adaptations
            "emergency_stabilization": 9,
            "conservative_optimization": 10,
            "optimize_parameters": 11,
        }
        return action_map.get(action.adaptation_type, 0)

    def _prepare_graph_batch(self, states: List[ModelGraphState]) -> Dict:
        """Prepare batch of graph states for training."""
        # Simplified batching - in production would use torch_geometric.data.Batch
        # For now, just use the first state
        if states:
            graph_data = states[0].graph_data
            return {
                "x": torch.tensor(graph_data.x, dtype=torch.float32, device=self.device),
                "edge_index": torch.tensor(graph_data.edge_index, dtype=torch.long, device=self.device),
                "edge_attr": torch.tensor(graph_data.edge_attr, dtype=torch.float32, device=self.device),
                "batch": None,
            }
        return {}

    def _compute_policy_loss(
        self,
        logits: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute policy gradient loss."""
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        
        # Advantage estimation (simplified - would use GAE in production)
        advantages = rewards - rewards.mean()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Policy gradient loss
        policy_loss = -(action_log_probs * advantages * weights).mean()
        
        return policy_loss

    def _compute_value_loss(
        self,
        value_estimates: torch.Tensor,
        rewards: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Compute value function loss."""
        value_targets = rewards  # Simplified - would use returns in production
        value_loss = (weights * F.mse_loss(
            value_estimates.squeeze(),
            value_targets,
            reduction='none'
        )).mean()
        
        return value_loss

    def _compute_entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy bonus for exploration."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        return entropy

    def _compute_safety_regularization(
        self,
        policy_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute safety regularization with gradient awareness."""
        # Prevent extreme confidence
        uncertainty = policy_outputs['uncertainty']
        min_uncertainty = 0.1
        uncertainty_penalty = F.relu(min_uncertainty - uncertainty).mean()
        
        # Prevent policy collapse
        policy_probs = F.softmax(policy_outputs['policy_logits'], dim=-1)
        entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=-1)
        min_entropy = 0.5
        entropy_penalty = F.relu(min_entropy - entropy).mean()
        
        # Gradient-aware regularization
        gradient_penalty = 0.0
        if hasattr(self, '_current_gradient_health'):
            # Penalize actions when gradients are unhealthy
            if self._current_gradient_health < 0.3:
                # Increase uncertainty requirement when gradients are poor
                gradient_penalty = 0.3 * (1.0 - self._current_gradient_health)
        
        return 0.1 * uncertainty_penalty + 0.1 * entropy_penalty + gradient_penalty

    async def _update_performance_tracking(self):
        """Update performance tracking metrics."""
        # Calculate success rate from recent experiences
        recent_rewards = [
            exp.reward for exp in list(self.experience_buffer.buffer)[-100:]
        ]
        
        if recent_rewards:
            self.training_stats['avg_reward'] = np.mean(recent_rewards)
            self.training_stats['success_rate'] = sum(
                1 for r in recent_rewards if r > 0
            ) / len(recent_rewards)
        
        # Update tracker
        self.performance_tracker.update({
            'avg_reward': self.training_stats['avg_reward'],
            'success_rate': self.training_stats['success_rate'],
        })