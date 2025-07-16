# **Implementation Guide: Phase 3 - Strategic Controller & Training Loop**

**Objective:** Implement the intelligent Tamiyo Strategic Controller that can analyze host models, make adaptation decisions, and improve its policy through offline reinforcement learning.

**Key Components to Implement:** Tamiyo GNN Policy, Telemetry Analysis, Control Loop, Offline Policy Training, Decision Engine.

**Timeline:** Weeks 10-13 (4 weeks)

-----

## **1. Tamiyo Strategic Controller: The Intelligent Decision Engine**

**Task:** Implement the graph neural network-based policy that analyzes model telemetry and makes strategic adaptation decisions.

### **1.1. GNN Policy Architecture (`src/esper/services/tamiyo/policy.py`)**

Implement the core Graph Neural Network that serves as Tamiyo's "brain".

```python
"""
GNN-based policy model for strategic morphogenetic control.

This module implements the core intelligence of the Tamiyo Strategic Controller,
using Graph Neural Networks to analyze model topology and performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from esper.contracts.operational import ModelGraphState, AdaptationDecision
from esper.contracts.enums import SeedLifecycleState

@dataclass
class PolicyConfig:
    """Configuration for Tamiyo policy model."""
    
    # GNN Architecture
    node_feature_dim: int = 64
    edge_feature_dim: int = 32
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    
    # Decision thresholds
    health_threshold: float = 0.3
    adaptation_confidence_threshold: float = 0.7
    max_adaptations_per_epoch: int = 2
    
    # Training parameters
    learning_rate: float = 1e-4
    replay_buffer_size: int = 10000
    batch_size: int = 32

class TamiyoPolicyGNN(nn.Module):
    """
    Graph Neural Network policy for strategic morphogenetic control.
    
    This network analyzes the topology and performance characteristics of a host
    model to make intelligent decisions about when, where, and how to adapt.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config
        
        # Node encoder - processes layer-level features
        self.node_encoder = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # GNN layers for topology-aware processing
        self.gnn_layers = nn.ModuleList([
            GCNConv(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_gnn_layers)
        ])
        
        # Decision head - outputs adaptation decisions
        self.decision_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # [should_adapt, layer_priority_logits, urgency_score]
        )
        
        # Value head for policy training
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )

    def forward(
        self, 
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            node_features: [num_nodes, node_feature_dim] tensor of layer features
            edge_index: [2, num_edges] tensor defining graph connectivity
            batch: Optional batch assignment for multiple graphs
            
        Returns:
            Tuple of (adaptation_decision, layer_priorities, value_estimate)
        """
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Apply GNN layers with residual connections
        for gnn_layer in self.gnn_layers:
            x_new = F.relu(gnn_layer(x, edge_index))
            x = x + x_new  # Residual connection
            x = F.dropout(x, training=self.training)
        
        # Global pooling to get graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_mean = global_mean_pool(x, batch)
        graph_max = global_max_pool(x, batch)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        
        # Generate decisions
        decision_logits = self.decision_head(graph_repr)
        adaptation_prob = torch.sigmoid(decision_logits[:, 0])  # Should adapt?
        layer_priorities = F.softmax(decision_logits[:, 1], dim=-1)  # Which layer?
        urgency_score = torch.sigmoid(decision_logits[:, 2])  # How urgent?
        
        # Value estimate for RL training
        value_estimate = self.value_head(graph_repr)
        
        return adaptation_prob, layer_priorities, urgency_score, value_estimate
```

### **1.2. Model State Analysis (`src/esper/services/tamiyo/analyzer.py`)**

Implement the system that converts model telemetry into graph representations.

```python
"""
Model state analysis and graph construction for Tamiyo.

This module processes telemetry from KasminaLayers and constructs graph
representations that the GNN policy can analyze.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import networkx as nx

from esper.contracts.operational import HealthSignal, ModelGraphState
from esper.execution.state_layout import SeedLifecycleState

@dataclass
class LayerNode:
    """Represents a layer in the model graph."""
    
    layer_name: str
    layer_type: str
    
    # Performance metrics
    health_score: float
    execution_latency: float
    error_count: int
    active_seeds: int
    total_seeds: int
    
    # Architectural properties
    input_size: int
    output_size: int
    parameter_count: int
    
    # Temporal dynamics
    health_trend: float  # Rate of change in health
    adaptation_history: List[str]  # Recent adaptations

class ModelGraphAnalyzer:
    """
    Analyzes model state and constructs graph representations for Tamiyo.
    
    This class processes telemetry signals from KasminaLayers and builds
    structured graph representations that capture both topology and dynamics.
    """

    def __init__(self, health_history_window: int = 10):
        self.health_history_window = health_history_window
        self.health_history: Dict[str, List[float]] = {}
        self.adaptation_history: Dict[str, List[str]] = {}

    def update_telemetry(self, health_signals: List[HealthSignal]) -> None:
        """
        Process incoming health signals and update internal state.
        
        Args:
            health_signals: List of health signals from KasminaLayers
        """
        for signal in health_signals:
            layer_name = signal.source_layer
            
            # Update health history
            if layer_name not in self.health_history:
                self.health_history[layer_name] = []
            
            self.health_history[layer_name].append(signal.health_score)
            
            # Maintain sliding window
            if len(self.health_history[layer_name]) > self.health_history_window:
                self.health_history[layer_name].pop(0)

    def construct_model_graph(
        self, 
        morphable_model,
        current_health_signals: List[HealthSignal]
    ) -> ModelGraphState:
        """
        Construct a graph representation of the current model state.
        
        Args:
            morphable_model: The MorphableModel being analyzed
            current_health_signals: Latest health signals
            
        Returns:
            ModelGraphState containing graph representation and metadata
        """
        # Build NetworkX graph for topology analysis
        G = nx.DiGraph()
        layer_nodes = {}
        
        # Add nodes for each KasminaLayer
        for layer_name, kasmina_layer in morphable_model.kasmina_layers.items():
            # Get current health signal for this layer
            layer_health = next(
                (hs for hs in current_health_signals if hs.source_layer == layer_name),
                None
            )
            
            if layer_health is None:
                continue  # Skip layers without health data
            
            # Calculate health trend
            health_trend = self._calculate_health_trend(layer_name)
            
            # Create layer node
            node = LayerNode(
                layer_name=layer_name,
                layer_type="KasminaLayer",
                health_score=layer_health.health_score,
                execution_latency=layer_health.avg_execution_time_ms,
                error_count=layer_health.error_count,
                active_seeds=kasmina_layer.state_layout.get_active_seeds().sum().item(),
                total_seeds=kasmina_layer.num_seeds,
                input_size=kasmina_layer.input_size,
                output_size=kasmina_layer.output_size,
                parameter_count=sum(p.numel() for p in kasmina_layer.parameters()),
                health_trend=health_trend,
                adaptation_history=self.adaptation_history.get(layer_name, [])
            )
            
            layer_nodes[layer_name] = node
            G.add_node(layer_name, **node.__dict__)
        
        # Add edges based on model topology
        self._add_topology_edges(G, morphable_model)
        
        # Convert to tensors for GNN processing
        node_features, edge_index = self._graph_to_tensors(G, layer_nodes)
        
        return ModelGraphState(
            node_features=node_features,
            edge_index=edge_index,
            layer_names=list(layer_nodes.keys()),
            timestamp=torch.tensor(time.time()),
            global_health=np.mean([node.health_score for node in layer_nodes.values()])
        )

    def _calculate_health_trend(self, layer_name: str) -> float:
        """Calculate the trend in health score for a layer."""
        if layer_name not in self.health_history:
            return 0.0
        
        history = self.health_history[layer_name]
        if len(history) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = np.arange(len(history))
        y = np.array(history)
        trend = np.polyfit(x, y, 1)[0]  # Slope of linear fit
        
        return float(trend)

    def _add_topology_edges(self, G: nx.DiGraph, morphable_model) -> None:
        """Add edges based on model topology."""
        # For MVP, we'll use a simple sequential topology
        # In practice, this would analyze the actual model structure
        layer_names = list(morphable_model.kasmina_layers.keys())
        
        for i in range(len(layer_names) - 1):
            G.add_edge(layer_names[i], layer_names[i + 1], 
                      edge_type="sequential", weight=1.0)

    def _graph_to_tensors(
        self, 
        G: nx.DiGraph, 
        layer_nodes: Dict[str, LayerNode]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert NetworkX graph to tensors for GNN processing."""
        # Create node feature matrix
        node_features = []
        node_names = list(G.nodes())
        
        for node_name in node_names:
            node = layer_nodes[node_name]
            features = [
                node.health_score,
                node.execution_latency,
                float(node.error_count),
                float(node.active_seeds) / max(node.total_seeds, 1),
                np.log1p(node.parameter_count),  # Log scale for parameter count
                node.health_trend,
                float(len(node.adaptation_history)),
            ]
            # Pad to fixed size (node_feature_dim from config)
            while len(features) < 64:  # Match PolicyConfig.node_feature_dim
                features.append(0.0)
            
            node_features.append(features[:64])  # Truncate if needed
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create edge index
        edge_list = []
        name_to_idx = {name: idx for idx, name in enumerate(node_names)}
        
        for edge in G.edges():
            src_idx = name_to_idx[edge[0]]
            dst_idx = name_to_idx[edge[1]]
            edge_list.append([src_idx, dst_idx])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        
        return node_features, edge_index
```

### **1.3. Tamiyo Service (`src/esper/services/tamiyo/main.py`)**

Implement the main service that orchestrates the strategic controller.

```python
"""
Tamiyo Strategic Controller Service.

This module implements the main service that analyzes model state and makes
strategic decisions about morphogenetic adaptations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch

from esper.services.oona_client import OonaClient
from esper.contracts.messages import OonaMessage, TopicNames
from esper.contracts.operational import HealthSignal, AdaptationDecision
from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from esper.services.tamiyo.analyzer import ModelGraphAnalyzer
from esper.services.tamiyo.training import ReplayBuffer, PolicyTrainer

logger = logging.getLogger(__name__)


class TamiyoService:
    """
    Strategic Controller service for morphogenetic adaptations.
    
    This service monitors model health, analyzes performance patterns,
    and makes intelligent decisions about when and how to adapt.
    """

    def __init__(
        self,
        policy_config: PolicyConfig,
        policy_checkpoint_path: Optional[str] = None,
        oona_host: str = "localhost",
        oona_port: int = 6379
    ):
        self.config = policy_config
        self.policy_checkpoint_path = policy_checkpoint_path
        
        # Initialize components
        self.oona_client = OonaClient(host=oona_host, port=oona_port)
        self.analyzer = ModelGraphAnalyzer()
        self.replay_buffer = ReplayBuffer(max_size=policy_config.replay_buffer_size)
        
        # Initialize policy model
        self.policy = TamiyoPolicyGNN(policy_config)
        if policy_checkpoint_path and Path(policy_checkpoint_path).exists():
            self.load_policy(policy_checkpoint_path)
            logger.info(f"Loaded policy from {policy_checkpoint_path}")
        else:
            logger.info("Starting with randomly initialized policy")
        
        # State tracking
        self.current_health_signals: List[HealthSignal] = []
        self.adaptation_history: List[AdaptationDecision] = []
        self.running = False

    async def start(self) -> None:
        """Start the Tamiyo service."""
        logger.info("Starting Tamiyo Strategic Controller")
        
        # Connect to Oona message bus
        await self.oona_client.connect()
        
        # Subscribe to health signals
        await self.oona_client.subscribe(
            TopicNames.HEALTH_SIGNALS,
            callback=self._on_health_signal,
            consumer_group="tamiyo-controller"
        )
        
        self.running = True
        
        # Start main control loop
        await self._control_loop()

    async def stop(self) -> None:
        """Stop the Tamiyo service."""
        logger.info("Stopping Tamiyo Strategic Controller")
        self.running = False
        await self.oona_client.disconnect()

    async def _control_loop(self) -> None:
        """Main control loop for strategic decision making."""
        while self.running:
            try:
                if len(self.current_health_signals) > 0:
                    # Analyze current state and make decisions
                    decisions = await self._analyze_and_decide()
                    
                    # Execute decisions
                    for decision in decisions:
                        await self._execute_decision(decision)
                
                # Wait before next analysis cycle
                await asyncio.sleep(5.0)  # 5-second analysis interval
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                await asyncio.sleep(1.0)

    async def _on_health_signal(self, message: OonaMessage) -> None:
        """Handle incoming health signals from KasminaLayers."""
        try:
            health_signal = HealthSignal.model_validate(message.payload)
            
            # Update current health signals (keep only latest per layer)
            self.current_health_signals = [
                hs for hs in self.current_health_signals 
                if hs.source_layer != health_signal.source_layer
            ]
            self.current_health_signals.append(health_signal)
            
            # Update analyzer with new telemetry
            self.analyzer.update_telemetry([health_signal])
            
            logger.debug(f"Received health signal from {health_signal.source_layer}: "
                        f"health={health_signal.health_score:.3f}")
        
        except Exception as e:
            logger.error(f"Error processing health signal: {e}")

    async def _analyze_and_decide(self) -> List[AdaptationDecision]:
        """Analyze current model state and make adaptation decisions."""
        try:
            # Skip if we don't have enough data
            if len(self.current_health_signals) < 2:
                return []
            
            # For MVP, we need to simulate the morphable_model
            # In Phase 4, this will come from Tolaria
            morphable_model = self._get_current_model()  # Simulated for now
            
            if morphable_model is None:
                return []
            
            # Construct graph representation
            model_graph = self.analyzer.construct_model_graph(
                morphable_model, 
                self.current_health_signals
            )
            
            # Run policy inference
            with torch.no_grad():
                adaptation_prob, layer_priorities, urgency_score, value_estimate = self.policy(
                    model_graph.node_features,
                    model_graph.edge_index
                )
            
            decisions = []
            
            # Make adaptation decisions based on policy output
            if adaptation_prob.item() > self.config.adaptation_confidence_threshold:
                # Find the layer with highest priority and lowest health
                unhealthy_layers = [
                    (hs.source_layer, hs.health_score) 
                    for hs in self.current_health_signals
                    if hs.health_score < self.config.health_threshold
                ]
                
                if unhealthy_layers:
                    # Sort by health score (lowest first)
                    unhealthy_layers.sort(key=lambda x: x[1])
                    target_layer = unhealthy_layers[0][0]
                    
                    decision = AdaptationDecision(
                        layer_name=target_layer,
                        action="load_kernel",
                        urgency_score=urgency_score.item(),
                        confidence=adaptation_prob.item(),
                        reasoning=f"Health score {unhealthy_layers[0][1]:.3f} below threshold",
                        timestamp=time.time()
                    )
                    
                    decisions.append(decision)
                    self.adaptation_history.append(decision)
                    
                    logger.info(f"Decision: Adapt {target_layer} (confidence: {adaptation_prob.item():.3f})")
            
            return decisions
        
        except Exception as e:
            logger.error(f"Error in analysis and decision making: {e}")
            return []

    async def _execute_decision(self, decision: AdaptationDecision) -> None:
        """Execute an adaptation decision by sending commands to KasminaLayers."""
        try:
            if decision.action == "load_kernel":
                # For MVP, select a random available kernel
                # In Phase 4, this will use Urza API to find optimal kernels
                kernel_id = "mock-kernel-001"  # Simulated for now
                
                command_message = OonaMessage(
                    topic=TopicNames.KASMINA_COMMANDS,
                    payload={
                        "layer_name": decision.layer_name,
                        "command": "load_kernel",
                        "kernel_id": kernel_id,
                        "seed_index": 0,  # Use first available seed
                        "decision_id": decision.decision_id
                    },
                    source="tamiyo"
                )
                
                await self.oona_client.publish(command_message)
                logger.info(f"Sent load_kernel command to {decision.layer_name}")
        
        except Exception as e:
            logger.error(f"Error executing decision: {e}")

    def _get_current_model(self):
        """Get current morphable model (simulated for MVP)."""
        # In Phase 4, this will be provided by Tolaria
        # For now, return None to skip analysis when no model available
        return None

    def load_policy(self, checkpoint_path: str) -> None:
        """Load policy weights from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        logger.info(f"Loaded policy from {checkpoint_path}")

    def save_policy(self, checkpoint_path: str) -> None:
        """Save current policy weights."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'config': self.config,
            'timestamp': time.time()
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved policy to {checkpoint_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "total_decisions": len(self.adaptation_history),
            "current_health_signals": len(self.current_health_signals),
            "replay_buffer_size": len(self.replay_buffer),
            "policy_parameters": sum(p.numel() for p in self.policy.parameters()),
            "running": self.running
        }
```

## **2. Offline Policy Training (`src/esper/services/tamiyo/training.py`)**

**Task:** Implement the reinforcement learning infrastructure for improving Tamiyo's decision-making policy.

```python
"""
Offline reinforcement learning trainer for Tamiyo policy.

This module implements the training infrastructure that allows Tamiyo to
improve its decision-making policy based on collected experience data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import random
import pickle
from pathlib import Path

from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from esper.contracts.operational import ModelGraphState, AdaptationDecision

@dataclass
class TrainingExperience:
    """Single training experience for policy learning."""
    
    state: ModelGraphState
    action: AdaptationDecision
    reward: float
    next_state: ModelGraphState
    done: bool
    timestamp: float

class ReplayBuffer:
    """Experience replay buffer for offline RL training."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: TrainingExperience) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[TrainingExperience]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def save(self, filepath: str) -> None:
        """Save buffer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(list(self.buffer), f)

    def load(self, filepath: str) -> None:
        """Load buffer from disk."""
        with open(filepath, 'rb') as f:
            experiences = pickle.load(f)
            self.buffer.extend(experiences)

    def __len__(self) -> int:
        return len(self.buffer)

class PolicyTrainer:
    """
    Offline trainer for Tamiyo policy using collected experience data.
    
    This implements a simplified actor-critic approach suitable for the
    morphogenetic control task.
    """

    def __init__(
        self,
        policy: TamiyoPolicyGNN,
        config: PolicyConfig,
        device: str = "cpu"
    ):
        self.policy = policy
        self.config = config
        self.device = device
        
        # Move policy to device
        self.policy.to(device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.learning_rate
        )
        
        # Loss functions
        self.policy_loss_fn = nn.BCELoss()
        self.value_loss_fn = nn.MSELoss()
        
        # Training metrics
        self.training_stats = {
            'total_epochs': 0,
            'total_samples': 0,
            'avg_policy_loss': 0.0,
            'avg_value_loss': 0.0,
            'avg_reward': 0.0
        }

    def train_epoch(
        self, 
        replay_buffer: ReplayBuffer,
        num_batches: int = 10
    ) -> Dict[str, float]:
        """
        Train the policy for one epoch using experience replay.
        
        Args:
            replay_buffer: Buffer containing training experiences
            num_batches: Number of batches to train on
            
        Returns:
            Dictionary of training metrics
        """
        if len(replay_buffer) < self.config.batch_size:
            return {"error": "Insufficient training data"}
        
        self.policy.train()
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_reward = 0.0
        
        for batch_idx in range(num_batches):
            # Sample batch
            experiences = replay_buffer.sample(self.config.batch_size)
            
            # Prepare batch data
            batch_data = self._prepare_batch(experiences)
            if batch_data is None:
                continue
            
            # Forward pass
            batch_loss, batch_metrics = self._compute_batch_loss(batch_data)
            
            # Backward pass
            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()
            
            # Accumulate metrics
            total_policy_loss += batch_metrics['policy_loss']
            total_value_loss += batch_metrics['value_loss']
            total_reward += batch_metrics['avg_reward']
        
        # Update training stats
        self.training_stats['total_epochs'] += 1
        self.training_stats['total_samples'] += num_batches * self.config.batch_size
        self.training_stats['avg_policy_loss'] = total_policy_loss / num_batches
        self.training_stats['avg_value_loss'] = total_value_loss / num_batches
        self.training_stats['avg_reward'] = total_reward / num_batches
        
        return {
            'policy_loss': self.training_stats['avg_policy_loss'],
            'value_loss': self.training_stats['avg_value_loss'],
            'avg_reward': self.training_stats['avg_reward'],
            'epoch': self.training_stats['total_epochs']
        }

    def _prepare_batch(
        self, 
        experiences: List[TrainingExperience]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare a batch of experiences for training."""
        try:
            # Extract components
            states = [exp.state for exp in experiences]
            actions = [exp.action for exp in experiences]
            rewards = [exp.reward for exp in experiences]
            next_states = [exp.next_state for exp in experiences]
            dones = [exp.done for exp in experiences]
            
            # Stack node features and create batch indices
            node_features_list = []
            edge_index_list = []
            batch_indices = []
            node_offset = 0
            
            for i, state in enumerate(states):
                node_features_list.append(state.node_features)
                
                # Adjust edge indices for batching
                edge_index = state.edge_index + node_offset
                edge_index_list.append(edge_index)
                
                # Create batch indices
                batch_size = state.node_features.size(0)
                batch_indices.extend([i] * batch_size)
                node_offset += batch_size
            
            # Concatenate all features
            batch_node_features = torch.cat(node_features_list, dim=0)
            batch_edge_index = torch.cat(edge_index_list, dim=1)
            batch_indices = torch.tensor(batch_indices, dtype=torch.long)
            
            # Convert other components to tensors
            batch_rewards = torch.tensor(rewards, dtype=torch.float32)
            batch_dones = torch.tensor(dones, dtype=torch.float32)
            
            # Extract action labels (simplified for MVP)
            action_labels = torch.tensor([
                1.0 if action.action == "load_kernel" else 0.0 
                for action in actions
            ], dtype=torch.float32)
            
            return {
                'node_features': batch_node_features.to(self.device),
                'edge_index': batch_edge_index.to(self.device),
                'batch': batch_indices.to(self.device),
                'action_labels': action_labels.to(self.device),
                'rewards': batch_rewards.to(self.device),
                'dones': batch_dones.to(self.device)
            }
        
        except Exception as e:
            print(f"Error preparing batch: {e}")
            return None

    def _compute_batch_loss(
        self, 
        batch_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss for a batch of training data."""
        # Forward pass through policy
        adaptation_prob, layer_priorities, urgency_score, value_estimate = self.policy(
            batch_data['node_features'],
            batch_data['edge_index'],
            batch_data['batch']
        )
        
        # Policy loss (binary classification for adaptation decision)
        policy_loss = self.policy_loss_fn(
            adaptation_prob.squeeze(), 
            batch_data['action_labels']
        )
        
        # Value loss (predict expected reward)
        value_loss = self.value_loss_fn(
            value_estimate.squeeze(), 
            batch_data['rewards']
        )
        
        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss
        
        # Metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item(),
            'avg_reward': batch_data['rewards'].mean().item()
        }
        
        return total_loss, metrics

    def evaluate(
        self, 
        replay_buffer: ReplayBuffer,
        num_eval_batches: int = 5
    ) -> Dict[str, float]:
        """Evaluate policy performance on held-out data."""
        self.policy.eval()
        
        total_accuracy = 0.0
        total_value_error = 0.0
        
        with torch.no_grad():
            for _ in range(num_eval_batches):
                experiences = replay_buffer.sample(self.config.batch_size)
                batch_data = self._prepare_batch(experiences)
                
                if batch_data is None:
                    continue
                
                # Forward pass
                adaptation_prob, _, _, value_estimate = self.policy(
                    batch_data['node_features'],
                    batch_data['edge_index'],
                    batch_data['batch']
                )
                
                # Calculate accuracy
                predictions = (adaptation_prob.squeeze() > 0.5).float()
                accuracy = (predictions == batch_data['action_labels']).float().mean()
                total_accuracy += accuracy.item()
                
                # Calculate value error
                value_error = torch.abs(
                    value_estimate.squeeze() - batch_data['rewards']
                ).mean()
                total_value_error += value_error.item()
        
        return {
            'accuracy': total_accuracy / num_eval_batches,
            'value_error': total_value_error / num_eval_batches
        }

    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
```

## **3. Training Script (`scripts/train_tamiyo.py`)**

**Task:** Create the standalone script for offline policy training.

```python
#!/usr/bin/env python3
"""
Offline training script for Tamiyo Strategic Controller.

This script trains the Tamiyo policy using collected experience data,
serving as the manual replacement for the Simic training environment.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from esper.services.tamiyo.training import ReplayBuffer, PolicyTrainer
from esper.configs import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Tamiyo Strategic Controller')
    parser.add_argument('--config', required=True, help='Configuration file path')
    parser.add_argument('--data-dir', required=True, help='Directory containing experience data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluation interval')
    parser.add_argument('--device', default='auto', help='Training device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    policy_config = PolicyConfig(**config.get('tamiyo', {}))
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Create directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model and trainer
    policy = TamiyoPolicyGNN(policy_config)
    trainer = PolicyTrainer(policy, policy_config, device=device)
    
    # Load existing checkpoint if available
    latest_checkpoint = checkpoint_dir / "latest.pth"
    if latest_checkpoint.exists():
        trainer.load_checkpoint(str(latest_checkpoint))
        logger.info(f"Resumed training from {latest_checkpoint}")
    
    # Load experience data
    replay_buffer = ReplayBuffer(max_size=policy_config.replay_buffer_size)
    
    data_dir = Path(args.data_dir)
    experience_files = list(data_dir.glob("*.pkl"))
    
    if not experience_files:
        logger.error(f"No experience files found in {data_dir}")
        return 1
    
    for exp_file in experience_files:
        try:
            replay_buffer.load(str(exp_file))
            logger.info(f"Loaded experiences from {exp_file}")
        except Exception as e:
            logger.warning(f"Failed to load {exp_file}: {e}")
    
    logger.info(f"Total experiences loaded: {len(replay_buffer)}")
    
    if len(replay_buffer) < policy_config.batch_size:
        logger.error("Insufficient training data")
        return 1
    
    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training step
        train_metrics = trainer.train_epoch(replay_buffer, num_batches=20)
        
        # Logging
        epoch_time = time.time() - start_time
        logger.info(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"(time: {epoch_time:.2f}s) - "
            f"Policy Loss: {train_metrics['policy_loss']:.4f}, "
            f"Value Loss: {train_metrics['value_loss']:.4f}, "
            f"Avg Reward: {train_metrics['avg_reward']:.4f}"
        )
        
        # Evaluation
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = trainer.evaluate(replay_buffer)
            logger.info(
                f"Evaluation - "
                f"Accuracy: {eval_metrics['accuracy']:.4f}, "
                f"Value Error: {eval_metrics['value_error']:.4f}"
            )
            
            # Save best model
            if eval_metrics['accuracy'] > best_accuracy:
                best_accuracy = eval_metrics['accuracy']
                best_path = checkpoint_dir / "best.pth"
                trainer.save_checkpoint(str(best_path))
                logger.info(f"New best model saved: accuracy {best_accuracy:.4f}")
        
        # Save latest checkpoint
        trainer.save_checkpoint(str(latest_checkpoint))
    
    logger.info("Training completed!")
    logger.info(f"Best accuracy achieved: {best_accuracy:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

## **4. Enhanced Telemetry Collection**

**Task:** Extend KasminaLayer telemetry to provide richer data for Tamiyo analysis.

### **4.1. Enhanced Health Signals (`src/esper/execution/kasmina_layer.py` - additions)**

```python
# Add to KasminaLayer class

async def _publish_enhanced_health_signal(self) -> None:
    """Publish enhanced health signal with rich telemetry data."""
    if not self.telemetry_enabled or self.oona_client is None:
        return
    
    try:
        # Calculate advanced metrics
        seed_states = self.state_layout.get_seed_states()
        active_seeds = self.state_layout.get_active_seeds()
        error_counts = self.state_layout.get_error_counts()
        
        # Performance analysis
        recent_latencies = self.state_layout.get_recent_latencies()
        avg_latency = recent_latencies.mean().item() if len(recent_latencies) > 0 else 0.0
        latency_variance = recent_latencies.var().item() if len(recent_latencies) > 0 else 0.0
        
        # Health trend calculation
        health_scores = self.state_layout.get_health_history()
        health_trend = self._calculate_health_trend(health_scores)
        
        # Adaptation readiness score
        adaptation_readiness = self._calculate_adaptation_readiness()
        
        # Create enhanced health signal
        enhanced_signal = EnhancedHealthSignal(
            source_layer=self.layer_name,
            timestamp=time.time(),
            
            # Basic health metrics
            health_score=self.health_score,
            avg_execution_time_ms=avg_latency,
            error_count=error_counts.sum().item(),
            
            # Seed-level details
            active_seeds=active_seeds.sum().item(),
            total_seeds=self.num_seeds,
            seed_states=seed_states.tolist(),
            seed_error_counts=error_counts.tolist(),
            
            # Performance characteristics
            latency_variance=latency_variance,
            throughput_trend=self._calculate_throughput_trend(),
            memory_utilization=self._get_memory_utilization(),
            
            # Adaptation metrics
            adaptation_readiness=adaptation_readiness,
            last_adaptation_time=self.last_adaptation_time,
            adaptation_count=self.total_adaptations,
            
            # Model topology context
            layer_position=self._get_layer_position(),
            upstream_health=self._get_upstream_health(),
            downstream_impact=self._get_downstream_impact()
        )
        
        # Publish to Oona
        message = OonaMessage(
            topic=TopicNames.ENHANCED_HEALTH_SIGNALS,
            payload=enhanced_signal.model_dump(),
            source=self.layer_name
        )
        
        await self.oona_client.publish(message)
        
    except Exception as e:
        logger.error(f"Error publishing enhanced health signal: {e}")

def _calculate_health_trend(self, health_scores: torch.Tensor) -> float:
    """Calculate trend in health score over time."""
    if len(health_scores) < 2:
        return 0.0
    
    # Simple linear regression for trend
    x = torch.arange(len(health_scores), dtype=torch.float32)
    y = health_scores.float()
    
    # Calculate slope
    n = len(health_scores)
    sum_x = x.sum()
    sum_y = y.sum()
    sum_xy = (x * y).sum()
    sum_x2 = (x * x).sum()
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    return slope.item()

def _calculate_adaptation_readiness(self) -> float:
    """Calculate readiness score for potential adaptations."""
    # Consider multiple factors
    factors = {
        'health_score': max(0, 1 - self.health_score),  # Lower health = higher readiness
        'error_rate': min(1, self.state_layout.get_error_counts().float().mean().item()),
        'dormant_seeds': self.state_layout.get_dormant_seeds().float().mean().item(),
        'stability': self._get_stability_score()
    }
    
    # Weighted combination
    weights = {'health_score': 0.4, 'error_rate': 0.3, 'dormant_seeds': 0.2, 'stability': 0.1}
    readiness = sum(factors[k] * weights[k] for k in factors)
    
    return min(1.0, max(0.0, readiness))
```

## **5. Phase 3 Integration Tests**

**Task:** Create comprehensive tests to validate the strategic controller.

### **5.1. Tamiyo Integration Test (`tests/integration/test_phase3_controller.py`)**

```python
"""
Integration tests for Phase 3 Strategic Controller.

This module contains tests that verify the end-to-end functionality
of the Tamiyo strategic controller and policy training system.
"""

import asyncio
import pytest
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os

from esper.services.tamiyo.main import TamiyoService
from esper.services.tamiyo.policy import TamiyoPolicyGNN, PolicyConfig
from esper.services.tamiyo.training import ReplayBuffer, PolicyTrainer, TrainingExperience
from esper.services.tamiyo.analyzer import ModelGraphAnalyzer
from esper.contracts.operational import HealthSignal, AdaptationDecision, ModelGraphState


class TestTamiyoIntegration:
    """Integration tests for Tamiyo Strategic Controller."""

    @pytest.fixture
    def policy_config(self):
        """Create test policy configuration."""
        return PolicyConfig(
            node_feature_dim=64,
            hidden_dim=64,
            num_gnn_layers=2,
            health_threshold=0.5,
            adaptation_confidence_threshold=0.6,
            batch_size=4
        )

    @pytest.fixture
    def mock_health_signals(self):
        """Create mock health signals for testing."""
        return [
            HealthSignal(
                source_layer="layer1",
                timestamp=1000.0,
                health_score=0.3,  # Unhealthy
                avg_execution_time_ms=15.5,
                error_count=3,
                active_seeds=0,
                total_seeds=4
            ),
            HealthSignal(
                source_layer="layer2", 
                timestamp=1000.0,
                health_score=0.8,  # Healthy
                avg_execution_time_ms=8.2,
                error_count=0,
                active_seeds=1,
                total_seeds=4
            )
        ]

    @pytest.mark.asyncio
    async def test_tamiyo_service_initialization(self, policy_config):
        """Test Tamiyo service initialization."""
        with patch('esper.services.tamiyo.main.OonaClient') as mock_oona:
            mock_oona.return_value = AsyncMock()
            
            service = TamiyoService(
                policy_config=policy_config,
                policy_checkpoint_path=None
            )
            
            assert service.policy is not None
            assert isinstance(service.policy, TamiyoPolicyGNN)
            assert service.analyzer is not None
            assert service.running is False

    @pytest.mark.asyncio
    async def test_health_signal_processing(self, policy_config, mock_health_signals):
        """Test processing of health signals."""
        with patch('esper.services.tamiyo.main.OonaClient') as mock_oona:
            mock_oona.return_value = AsyncMock()
            
            service = TamiyoService(policy_config=policy_config)
            
            # Process health signals
            for signal in mock_health_signals:
                from esper.contracts.messages import OonaMessage
                message = OonaMessage(
                    topic="health_signals",
                    payload=signal.model_dump(),
                    source=signal.source_layer
                )
                await service._on_health_signal(message)
            
            # Verify signals were processed
            assert len(service.current_health_signals) == 2
            assert service.current_health_signals[0].source_layer == "layer1"
            assert service.current_health_signals[1].source_layer == "layer2"

    def test_model_graph_analyzer(self, mock_health_signals):
        """Test model graph construction and analysis."""
        analyzer = ModelGraphAnalyzer()
        
        # Update with health signals
        analyzer.update_telemetry(mock_health_signals)
        
        # Verify health history tracking
        assert "layer1" in analyzer.health_history
        assert "layer2" in analyzer.health_history
        assert analyzer.health_history["layer1"][-1] == 0.3
        assert analyzer.health_history["layer2"][-1] == 0.8

    def test_policy_forward_pass(self, policy_config):
        """Test GNN policy forward pass."""
        policy = TamiyoPolicyGNN(policy_config)
        
        # Create test graph data
        num_nodes = 3
        node_features = torch.randn(num_nodes, policy_config.node_feature_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        # Forward pass
        adaptation_prob, layer_priorities, urgency_score, value_estimate = policy(
            node_features, edge_index
        )
        
        # Verify output shapes and ranges
        assert adaptation_prob.shape == torch.Size([1])
        assert 0 <= adaptation_prob.item() <= 1
        assert layer_priorities.shape == torch.Size([1])
        assert 0 <= urgency_score.item() <= 1
        assert value_estimate.shape == torch.Size([1])

    def test_replay_buffer_operations(self):
        """Test replay buffer functionality."""
        buffer = ReplayBuffer(max_size=100)
        
        # Create mock experiences
        experiences = []
        for i in range(10):
            exp = TrainingExperience(
                state=self._create_mock_graph_state(),
                action=AdaptationDecision(
                    layer_name=f"layer{i}",
                    action="load_kernel",
                    urgency_score=0.5,
                    confidence=0.7,
                    reasoning="test",
                    timestamp=1000.0 + i
                ),
                reward=np.random.random(),
                next_state=self._create_mock_graph_state(),
                done=False,
                timestamp=1000.0 + i
            )
            experiences.append(exp)
            buffer.add(exp)
        
        # Test sampling
        assert len(buffer) == 10
        sample = buffer.sample(5)
        assert len(sample) == 5
        assert all(isinstance(exp, TrainingExperience) for exp in sample)

    def test_policy_training(self, policy_config):
        """Test offline policy training."""
        policy = TamiyoPolicyGNN(policy_config)
        trainer = PolicyTrainer(policy, policy_config)
        replay_buffer = ReplayBuffer()
        
        # Add training experiences
        for i in range(20):  # Need enough for batch_size
            exp = TrainingExperience(
                state=self._create_mock_graph_state(),
                action=AdaptationDecision(
                    layer_name=f"layer{i % 3}",
                    action="load_kernel" if i % 2 == 0 else "no_action",
                    urgency_score=np.random.random(),
                    confidence=np.random.random(),
                    reasoning="test",
                    timestamp=1000.0 + i
                ),
                reward=np.random.random(),
                next_state=self._create_mock_graph_state(),
                done=False,
                timestamp=1000.0 + i
            )
            replay_buffer.add(exp)
        
        # Train for one epoch
        metrics = trainer.train_epoch(replay_buffer, num_batches=2)
        
        # Verify training metrics
        assert 'policy_loss' in metrics
        assert 'value_loss' in metrics
        assert 'avg_reward' in metrics
        assert isinstance(metrics['policy_loss'], float)

    def test_checkpoint_save_load(self, policy_config):
        """Test policy checkpoint save/load."""
        policy = TamiyoPolicyGNN(policy_config)
        trainer = PolicyTrainer(policy, policy_config)
        
        # Get initial weights
        initial_weights = policy.state_dict().copy()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            # Save checkpoint
            trainer.save_checkpoint(checkpoint_path)
            
            # Modify weights
            with torch.no_grad():
                for param in policy.parameters():
                    param.data.fill_(1.0)
            
            # Load checkpoint
            trainer.load_checkpoint(checkpoint_path)
            
            # Verify weights restored
            loaded_weights = policy.state_dict()
            for key in initial_weights:
                assert torch.allclose(initial_weights[key], loaded_weights[key])
        
        finally:
            os.unlink(checkpoint_path)

    @pytest.mark.asyncio
    async def test_decision_execution(self, policy_config, mock_health_signals):
        """Test adaptation decision execution."""
        with patch('esper.services.tamiyo.main.OonaClient') as mock_oona:
            mock_client = AsyncMock()
            mock_oona.return_value = mock_client
            
            service = TamiyoService(policy_config=policy_config)
            
            # Create decision
            decision = AdaptationDecision(
                layer_name="layer1",
                action="load_kernel",
                urgency_score=0.8,
                confidence=0.9,
                reasoning="Health below threshold",
                timestamp=1000.0
            )
            
            # Execute decision
            await service._execute_decision(decision)
            
            # Verify command was published
            mock_client.publish.assert_called_once()
            
            # Check command content
            call_args = mock_client.publish.call_args[0][0]
            assert call_args.topic == "kasmina_commands"
            assert call_args.payload["layer_name"] == "layer1"
            assert call_args.payload["command"] == "load_kernel"

    def test_performance_overhead(self, policy_config):
        """Test policy inference performance."""
        policy = TamiyoPolicyGNN(policy_config)
        policy.eval()
        
        # Create test data
        num_nodes = 10
        node_features = torch.randn(num_nodes, policy_config.node_feature_dim)
        edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)]).t()
        
        # Measure inference time
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                outputs = policy(node_features, edge_index)
        
        inference_time = time.time() - start_time
        avg_inference_time = inference_time / 100
        
        # Should be fast enough for real-time decision making
        assert avg_inference_time < 0.01  # Less than 10ms per inference
        print(f"Average inference time: {avg_inference_time*1000:.2f}ms")

    def _create_mock_graph_state(self) -> ModelGraphState:
        """Create a mock graph state for testing."""
        num_nodes = 3
        node_features = torch.randn(num_nodes, 64)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        
        return ModelGraphState(
            node_features=node_features,
            edge_index=edge_index,
            layer_names=["layer0", "layer1", "layer2"],
            timestamp=torch.tensor(1000.0),
            global_health=0.6
        )

    @pytest.mark.asyncio
    async def test_end_to_end_decision_cycle(self, policy_config, mock_health_signals):
        """Test complete decision-making cycle."""
        with patch('esper.services.tamiyo.main.OonaClient') as mock_oona:
            mock_client = AsyncMock()
            mock_oona.return_value = mock_client
            
            service = TamiyoService(policy_config=policy_config)
            
            # Process health signals
            for signal in mock_health_signals:
                from esper.contracts.messages import OonaMessage
                message = OonaMessage(
                    topic="health_signals",
                    payload=signal.model_dump(),
                    source=signal.source_layer
                )
                await service._on_health_signal(message)
            
            # Mock get_current_model to return a test model
            with patch.object(service, '_get_current_model') as mock_get_model:
                mock_model = Mock()
                mock_model.kasmina_layers = {
                    "layer1": Mock(num_seeds=4, input_size=64, output_size=32),
                    "layer2": Mock(num_seeds=4, input_size=32, output_size=16)
                }
                mock_get_model.return_value = mock_model
                
                # Run analysis and decision making
                decisions = await service._analyze_and_decide()
                
                # Should make decision for unhealthy layer1
                assert len(decisions) == 1
                assert decisions[0].layer_name == "layer1"
                assert decisions[0].action == "load_kernel"
```

## **6. Phase 3 Testing & Validation Strategy**

1. **Unit Tests:**
   * Test GNN policy architecture and forward pass
   * Test graph construction from telemetry data
   * Test decision logic and thresholds
   * Test replay buffer operations
   * Test offline training algorithm

2. **Integration Tests:**
   * Test full Tamiyo service lifecycle
   * Test telemetry processing and analysis
   * Test decision execution and command publishing
   * Test policy checkpointing and loading

3. **Performance Validation:**
   * Measure policy inference latency (target: <10ms)
   * Validate memory usage during training
   * Test scalability with larger model graphs

4. **Functional Validation:**
   * **"Strategic Decision" Test:** Verify Tamiyo makes sensible adaptation decisions
   * **"Policy Learning" Test:** Confirm offline training improves decision quality
   * **"Telemetry Integration" Test:** Validate end-to-end data flow from KasminaLayers

## **7. Definition of Done**

Phase 3 is complete when:

* ✅ **Tamiyo Service implemented** with GNN-based policy and graph analysis
* ✅ **Policy training infrastructure** with replay buffer and offline RL trainer
* ✅ **Enhanced telemetry** providing rich data for strategic analysis
* ✅ **Decision execution** with command publishing to KasminaLayers
* ✅ **All unit tests passing** with >85% code coverage
* ✅ **Integration tests passing** demonstrating end-to-end controller functionality
* ✅ **Training script** (`train_tamiyo.py`) successfully improving policy performance
* ✅ **Performance validation** showing <10ms inference latency
* ✅ **Documentation** with clear examples of policy training workflow

## **8. Phase 3 Implementation Summary**

**Phase 3 introduces the intelligent "brain" of the system:**

1. **Strategic Controller (Tamiyo)**: GNN-based policy that analyzes model topology and performance
   * Graph neural network architecture for topology-aware analysis
   * Decision engine that identifies when and where to adapt
   * Policy that learns from experience to improve decision quality

2. **Offline Training System**: Infrastructure for improving Tamiyo's decision-making
   * Experience replay buffer for collecting training data
   * Actor-critic training algorithm for policy optimization
   * Standalone training script for manual policy improvement

3. **Enhanced Telemetry**: Rich data collection for strategic analysis
   * Extended health signals with performance trends
   * Adaptation readiness scoring
   * Model topology context and impact analysis

4. **Control Loop Integration**: Connection between analysis and execution
   * Real-time telemetry processing from KasminaLayers
   * Strategic decision making based on graph analysis
   * Command execution through Oona message bus

This phase establishes the autonomous decision-making foundation required for Phase 4 (Full System Orchestration) and validates the core strategic intelligence mechanic of the morphogenetic training system.
