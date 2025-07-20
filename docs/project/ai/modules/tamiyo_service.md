# Tamiyo Service - Strategic Controller (`src/esper/services/tamiyo/`)

## Overview

Tamiyo serves as the strategic controller of the Esper system, using Graph Neural Network-based policies to analyze model health signals and make intelligent decisions about when and where to apply morphogenetic adaptations. It operates as the central intelligence that coordinates the evolution of neural network architectures during training.

## Architecture Summary

### Core Components
- **Analyzer:** Model graph construction and health analysis
- **Policy:** GNN-based strategic decision making
- **Training:** Offline reinforcement learning for policy improvement
- **Service:** Main orchestration and control loops

### Integration Points
- **Input:** Health signals from KasminaLayers via Oona
- **Output:** Adaptation decisions to execution layers
- **Learning:** Experience collection for policy improvement
- **Coordination:** Integration with Tolaria training orchestrator

## Files

### `__init__.py` - Tamiyo Service Initialization

**Purpose:** Service module initialization for Tamiyo strategic controller.

**Contents:** Minimal initialization for Tamiyo service components.

### `analyzer.py` - Model Graph Analysis

**Purpose:** Processes health signals from KasminaLayers and constructs graph representations for strategic decision making.

#### Key Components

**`LayerNode`** - Graph Node Representation
```python
@dataclass
class LayerNode:
    """Represents a layer node in the model graph."""
    
    layer_id: int
    layer_name: str
    layer_type: str
    
    # Health metrics
    current_health: float
    health_trend: float  # Rate of change
    last_updated: float
    
    # Topology information
    input_layers: List[int]
    output_layers: List[int]
    
    # Seed information
    total_seeds: int
    active_seeds: int
    error_count: int
    
    def is_problematic(self) -> bool:
        """Check if layer shows concerning health patterns."""
        return (
            self.current_health < 0.3 or
            self.health_trend < -0.1 or
            self.error_count > 5
        )
    
    def needs_attention(self) -> bool:
        """Check if layer needs immediate attention."""
        return (
            self.current_health < 0.5 and
            self.health_trend < 0.0
        )
```

**`GraphTopology`** - Model Structure Representation
```python
@dataclass
class GraphTopology:
    """Represents the topology of the neural network model."""
    
    nodes: Dict[int, LayerNode]
    edges: List[Tuple[int, int]]  # (source_layer, target_layer)
    
    # Graph properties
    num_layers: int
    max_depth: int
    has_skip_connections: bool
    
    def get_neighbors(self, layer_id: int) -> List[int]:
        """Get neighboring layers for a given layer."""
        neighbors = []
        for source, target in self.edges:
            if source == layer_id:
                neighbors.append(target)
            elif target == layer_id:
                neighbors.append(source)
        return neighbors
    
    def get_problematic_clusters(self) -> List[List[int]]:
        """Identify clusters of problematic layers."""
        problematic_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.is_problematic()
        ]
        
        # Simple clustering: connected components of problematic nodes
        clusters = []
        visited = set()
        
        for node_id in problematic_nodes:
            if node_id in visited:
                continue
                
            # BFS to find connected problematic nodes
            cluster = []
            queue = [node_id]
            visited.add(node_id)
            
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                
                for neighbor in self.get_neighbors(current):
                    if neighbor in problematic_nodes and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
            
            if cluster:
                clusters.append(cluster)
        
        return clusters
```

**`ModelGraphAnalyzer`** - Health Analysis Engine
```python
class ModelGraphAnalyzer:
    """
    Analyzes health signals and constructs model graph representations.
    
    This class processes incoming health signals from KasminaLayers and
    maintains a real-time view of model health and topology.
    """
    
    def __init__(self, health_window_size: int = 100):
        """
        Initialize the analyzer.
        
        Args:
            health_window_size: Number of health signals to keep for trend analysis
        """
        self.health_window_size = health_window_size
        
        # Health signal storage
        self.health_history: Dict[str, List[HealthSignal]] = defaultdict(list)
        self.layer_nodes: Dict[int, LayerNode] = {}
        self.topology: Optional[GraphTopology] = None
        
        # Analysis state
        self.last_analysis_time = 0.0
        self.analysis_interval = 5.0  # seconds
        
        logger.info(f"Initialized ModelGraphAnalyzer with window size {health_window_size}")
    
    def process_health_signal(self, health_signal: HealthSignal) -> None:
        """
        Process incoming health signal and update model state.
        
        Args:
            health_signal: Health signal from KasminaLayer
        """
        layer_key = f"layer_{health_signal.layer_id}"
        
        # Store health signal
        self.health_history[layer_key].append(health_signal)
        
        # Maintain window size
        if len(self.health_history[layer_key]) > self.health_window_size:
            self.health_history[layer_key].pop(0)
        
        # Update layer node
        self._update_layer_node(health_signal)
        
        logger.debug(f"Processed health signal for layer {health_signal.layer_id}")
    
    def _update_layer_node(self, health_signal: HealthSignal) -> None:
        """
        Update layer node with new health information.
        
        Args:
            health_signal: Health signal to process
        """
        layer_id = health_signal.layer_id
        layer_key = f"layer_{layer_id}"
        
        # Calculate health trend
        health_trend = self._calculate_health_trend(layer_key)
        
        # Update or create layer node
        if layer_id in self.layer_nodes:
            node = self.layer_nodes[layer_id]
            node.current_health = health_signal.health_score
            node.health_trend = health_trend
            node.last_updated = health_signal.timestamp
            node.active_seeds = health_signal.active_seeds
            node.total_seeds = health_signal.total_seeds
            node.error_count = health_signal.error_count
        else:
            # Create new node
            self.layer_nodes[layer_id] = LayerNode(
                layer_id=layer_id,
                layer_name=f"layer_{layer_id}",
                layer_type="unknown",  # Could be inferred from model inspection
                current_health=health_signal.health_score,
                health_trend=health_trend,
                last_updated=health_signal.timestamp,
                input_layers=[],  # Will be populated during topology construction
                output_layers=[],
                total_seeds=health_signal.total_seeds,
                active_seeds=health_signal.active_seeds,
                error_count=health_signal.error_count
            )
    
    def _calculate_health_trend(self, layer_key: str) -> float:
        """
        Calculate health trend for a layer.
        
        Args:
            layer_key: Layer identifier
            
        Returns:
            Health trend (positive = improving, negative = degrading)
        """
        history = self.health_history[layer_key]
        
        if len(history) < 2:
            return 0.0
        
        # Simple linear regression on recent health scores
        recent_signals = history[-min(10, len(history)):]
        health_scores = [signal.health_score for signal in recent_signals]
        
        # Calculate slope using least squares
        n = len(health_scores)
        x_sum = sum(range(n))
        y_sum = sum(health_scores)
        xy_sum = sum(i * score for i, score in enumerate(health_scores))
        x2_sum = sum(i * i for i in range(n))
        
        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def analyze_model_state(self) -> ModelGraphState:
        """
        Analyze current model state and return comprehensive analysis.
        
        Returns:
            ModelGraphState with current analysis
        """
        current_time = time.time()
        
        # Update topology if needed
        if self.topology is None or current_time - self.last_analysis_time > self.analysis_interval:
            self.topology = self._construct_topology()
            self.last_analysis_time = current_time
        
        # Calculate overall health
        overall_health = self._calculate_overall_health()
        
        # Identify problematic layers
        problematic_layers = set()
        if self.topology:
            for cluster in self.topology.get_problematic_clusters():
                problematic_layers.update(cluster)
        
        # Collect current health signals
        current_health_signals = {}
        health_trends = {}
        
        for layer_id, node in self.layer_nodes.items():
            layer_key = f"layer_{layer_id}"
            if self.health_history[layer_key]:
                current_health_signals[layer_key] = self.health_history[layer_key][-1]
                health_trends[layer_key] = node.health_trend
        
        return ModelGraphState(
            topology=self.topology,
            health_signals=current_health_signals,
            health_trends=health_trends,
            problematic_layers={f"layer_{lid}" for lid in problematic_layers},
            overall_health=overall_health,
            analysis_timestamp=current_time
        )
    
    def _construct_topology(self) -> GraphTopology:
        """
        Construct model topology from available information.
        
        Returns:
            GraphTopology representing model structure
        """
        # For MVP, create a simple sequential topology
        # In production, this would analyze actual model structure
        
        layer_ids = sorted(self.layer_nodes.keys())
        edges = []
        
        # Create sequential connections
        for i in range(len(layer_ids) - 1):
            edges.append((layer_ids[i], layer_ids[i + 1]))
        
        # Update node topology information
        for i, layer_id in enumerate(layer_ids):
            node = self.layer_nodes[layer_id]
            
            # Set input/output layers
            if i > 0:
                node.input_layers = [layer_ids[i - 1]]
            if i < len(layer_ids) - 1:
                node.output_layers = [layer_ids[i + 1]]
        
        return GraphTopology(
            nodes=self.layer_nodes.copy(),
            edges=edges,
            num_layers=len(layer_ids),
            max_depth=len(layer_ids),
            has_skip_connections=False  # MVP assumption
        )
    
    def _calculate_overall_health(self) -> float:
        """
        Calculate overall model health score.
        
        Returns:
            Overall health score (0.0 to 1.0)
        """
        if not self.layer_nodes:
            return 1.0  # No data = assume healthy
        
        # Weighted average of layer health scores
        total_weight = 0.0
        weighted_sum = 0.0
        
        for node in self.layer_nodes.values():
            # Weight by number of seeds (more important layers have more seeds)
            weight = max(node.total_seeds, 1)
            weighted_sum += node.current_health * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """
        Get summary of current analysis state.
        
        Returns:
            Analysis summary statistics
        """
        if not self.layer_nodes:
            return {"status": "no_data", "layers": 0}
        
        problematic_count = sum(
            1 for node in self.layer_nodes.values()
            if node.is_problematic()
        )
        
        needs_attention_count = sum(
            1 for node in self.layer_nodes.values()
            if node.needs_attention()
        )
        
        return {
            "status": "active",
            "layers": len(self.layer_nodes),
            "overall_health": self._calculate_overall_health(),
            "problematic_layers": problematic_count,
            "needs_attention": needs_attention_count,
            "last_analysis": self.last_analysis_time,
            "health_signals_collected": sum(
                len(history) for history in self.health_history.values()
            )
        }
```

**Features:**
- **Real-time Analysis:** Continuous processing of health signals
- **Trend Detection:** Linear regression-based health trend calculation
- **Graph Construction:** Model topology inference from layer information
- **Cluster Detection:** Identification of problematic layer groups
- **Performance Metrics:** Comprehensive analysis statistics

### `policy.py` - GNN-Based Strategic Policy

**Purpose:** Implements Graph Neural Network-based policy for strategic adaptation decisions with torch-geometric integration.

#### Key Components

**`PolicyConfig`** - Policy Configuration
```python
@dataclass
class PolicyConfig:
    """Configuration for Tamiyo policy network."""
    
    # Network architecture
    input_dim: int = 64
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    dropout: float = 0.1
    
    # Decision thresholds
    confidence_threshold: float = 0.7
    urgency_threshold: float = 0.5
    adaptation_cooldown: float = 30.0  # seconds
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    gamma: float = 0.99  # Discount factor for RL
    
    # Performance optimization
    use_torch_scatter: bool = True  # Enable if torch-scatter available
    compile_model: bool = True      # Use torch.compile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyConfig':
        """Create config from dictionary."""
        return cls(**data)
```

**`TamiyoPolicyGNN`** - Graph Neural Network Policy
```python
class TamiyoPolicyGNN(nn.Module):
    """
    Graph Neural Network-based policy for strategic adaptation decisions.
    
    This model uses Graph Convolutional Networks to analyze model topology
    and health signals to make informed decisions about morphogenetic adaptations.
    """
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the policy network.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        self.config = config
        
        # Check for torch-geometric availability
        self.has_torch_geometric = self._check_torch_geometric()
        
        if self.has_torch_geometric:
            self._init_gnn_layers()
        else:
            self._init_fallback_layers()
        
        # Decision heads
        self.adaptation_head = nn.Linear(config.hidden_dim, 3)  # [no_action, add_seed, modify_architecture]
        self.layer_priority_head = nn.Linear(config.hidden_dim, 1)  # Priority score for layer
        self.urgency_head = nn.Linear(config.hidden_dim, 1)  # Urgency of action
        self.value_head = nn.Linear(config.hidden_dim, 1)  # Value estimation for RL
        
        # Apply torch.compile if requested and available
        if config.compile_model and hasattr(torch, 'compile'):
            try:
                self.forward = torch.compile(self.forward)
                logger.info("Applied torch.compile to policy network")
            except Exception as e:
                logger.warning(f"Failed to apply torch.compile: {e}")
    
    def _check_torch_geometric(self) -> bool:
        """Check if torch-geometric is available."""
        try:
            import torch_geometric
            return True
        except ImportError:
            logger.warning("torch-geometric not available, using fallback implementation")
            return False
    
    def _init_gnn_layers(self) -> None:
        """Initialize GNN layers using torch-geometric."""
        try:
            from torch_geometric.nn import GCNConv, global_mean_pool
            
            # Node feature encoder
            self.node_encoder = nn.Sequential(
                nn.Linear(self.config.input_dim, self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout)
            )
            
            # GNN layers
            self.gnn_layers = nn.ModuleList()
            for i in range(self.config.num_gnn_layers):
                self.gnn_layers.append(
                    GCNConv(self.config.hidden_dim, self.config.hidden_dim)
                )
            
            # Global pooling
            self.global_pool = global_mean_pool
            
            # Residual connections
            self.use_residual = True
            
            logger.info(f"Initialized GNN with {self.config.num_gnn_layers} layers")
            
        except ImportError as e:
            logger.error(f"Failed to import torch-geometric components: {e}")
            self._init_fallback_layers()
    
    def _init_fallback_layers(self) -> None:
        """Initialize fallback layers for when torch-geometric is not available."""
        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.config.dropout)
        )
        
        # Simple MLP layers as fallback
        self.fallback_layers = nn.ModuleList()
        for i in range(self.config.num_gnn_layers):
            self.fallback_layers.append(
                nn.Sequential(
                    nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.config.dropout)
                )
            )
        
        self.use_residual = True
        logger.info("Initialized fallback MLP layers")
    
    def forward(
        self, 
        node_features: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            node_features: Node feature tensor [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges] (for GNN)
            batch: Batch assignment for nodes (for batched processing)
            
        Returns:
            Dictionary containing policy outputs
        """
        # Encode node features
        x = self.node_encoder(node_features)
        
        # Apply GNN or fallback layers
        if self.has_torch_geometric and edge_index is not None:
            x = self._forward_gnn(x, edge_index)
        else:
            x = self._forward_fallback(x)
        
        # Global pooling for graph-level decisions
        if batch is not None and self.has_torch_geometric:
            x = self.global_pool(x, batch)
        else:
            # Simple mean pooling as fallback
            x = x.mean(dim=0, keepdim=True)
        
        # Decision heads
        adaptation_logits = self.adaptation_head(x)
        layer_priority = torch.sigmoid(self.layer_priority_head(x))
        urgency = torch.sigmoid(self.urgency_head(x))
        value = self.value_head(x)
        
        return {
            "adaptation_logits": adaptation_logits,
            "layer_priority": layer_priority,
            "urgency": urgency,
            "value": value,
            "confidence": torch.softmax(adaptation_logits, dim=-1).max(dim=-1)[0]
        }
    
    def _forward_gnn(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through GNN layers."""
        for i, gnn_layer in enumerate(self.gnn_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = gnn_layer(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.config.dropout, training=self.training)
            
            # Residual connection
            if residual is not None:
                x = x + residual
        
        return x
    
    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fallback MLP layers."""
        for i, layer in enumerate(self.fallback_layers):
            residual = x if self.use_residual and i > 0 else None
            
            x = layer(x)
            
            # Residual connection
            if residual is not None:
                x = x + residual
        
        return x
    
    def make_decision(
        self, 
        model_state: ModelGraphState,
        layer_candidates: Optional[List[str]] = None
    ) -> List[AdaptationDecision]:
        """
        Make strategic adaptation decisions based on model state.
        
        Args:
            model_state: Current model state from analyzer
            layer_candidates: Optional list of layers to consider
            
        Returns:
            List of adaptation decisions
        """
        if not model_state.topology or not model_state.health_signals:
            return []
        
        decisions = []
        
        # Convert model state to graph format
        node_features, edge_index, layer_mapping = self._prepare_graph_input(model_state)
        
        # Run inference
        with torch.no_grad():
            outputs = self.forward(node_features, edge_index)
        
        # Process outputs for each layer
        adaptation_probs = torch.softmax(outputs["adaptation_logits"], dim=-1)
        layer_priorities = outputs["layer_priority"].squeeze()
        urgency_scores = outputs["urgency"].squeeze()
        confidence_scores = outputs["confidence"].squeeze()
        
        # Determine candidates
        if layer_candidates is None:
            layer_candidates = list(model_state.problematic_layers)
        
        # Make decisions for problematic layers
        for i, (layer_id, layer_name) in enumerate(layer_mapping.items()):
            if layer_name not in layer_candidates:
                continue
            
            # Check thresholds
            confidence = confidence_scores[i].item() if len(confidence_scores.shape) > 0 else confidence_scores.item()
            urgency = urgency_scores[i].item() if len(urgency_scores.shape) > 0 else urgency_scores.item()
            
            if confidence < self.config.confidence_threshold:
                continue
            
            # Determine adaptation type
            adaptation_idx = torch.argmax(adaptation_probs[i] if len(adaptation_probs.shape) > 1 else adaptation_probs).item()
            adaptation_types = ["no_action", "add_seed", "modify_architecture"]
            adaptation_type = adaptation_types[adaptation_idx]
            
            if adaptation_type == "no_action":
                continue
            
            # Create decision
            decision = AdaptationDecision(
                layer_name=layer_name,
                adaptation_type=adaptation_type,
                confidence=confidence,
                urgency=urgency,
                metadata={
                    "layer_priority": layer_priorities[i].item() if len(layer_priorities.shape) > 0 else layer_priorities.item(),
                    "health_score": model_state.health_signals.get(layer_name, {}).get("health_score", 0.5),
                    "health_trend": model_state.health_trends.get(layer_name, 0.0)
                }
            )
            
            decisions.append(decision)
        
        # Sort by priority and urgency
        decisions.sort(key=lambda d: d.urgency * d.confidence, reverse=True)
        
        return decisions
    
    def _prepare_graph_input(
        self, 
        model_state: ModelGraphState
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[int, str]]:
        """
        Prepare graph input tensors from model state.
        
        Args:
            model_state: Model state to convert
            
        Returns:
            Tuple of (node_features, edge_index, layer_mapping)
        """
        topology = model_state.topology
        health_signals = model_state.health_signals
        health_trends = model_state.health_trends
        
        # Create node features
        node_features = []
        layer_mapping = {}
        
        for i, (layer_id, node) in enumerate(topology.nodes.items()):
            layer_name = f"layer_{layer_id}"
            layer_mapping[i] = layer_name
            
            # Extract features
            health_signal = health_signals.get(layer_name)
            health_trend = health_trends.get(layer_name, 0.0)
            
            if health_signal:
                features = [
                    health_signal.health_score,
                    health_signal.activation_variance,
                    health_signal.dead_neuron_ratio,
                    health_signal.avg_correlation,
                    health_signal.execution_latency / 1000.0,  # Normalize to ms
                    health_signal.error_count / 10.0,  # Normalize
                    health_signal.active_seeds / max(health_signal.total_seeds, 1),  # Ratio
                    health_trend,
                    float(node.is_problematic()),
                    float(node.needs_attention())
                ]
            else:
                # Default features for missing health signals
                features = [0.5] * 10
            
            # Pad to input_dim
            while len(features) < self.config.input_dim:
                features.append(0.0)
            
            node_features.append(features[:self.config.input_dim])
        
        node_features = torch.tensor(node_features, dtype=torch.float32)
        
        # Create edge index for GNN
        edge_index = None
        if self.has_torch_geometric and topology.edges:
            edge_list = []
            layer_id_to_idx = {layer_id: i for i, layer_id in enumerate(topology.nodes.keys())}
            
            for source_layer, target_layer in topology.edges:
                if source_layer in layer_id_to_idx and target_layer in layer_id_to_idx:
                    source_idx = layer_id_to_idx[source_layer]
                    target_idx = layer_id_to_idx[target_layer]
                    edge_list.extend([[source_idx, target_idx], [target_idx, source_idx]])  # Bidirectional
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return node_features, edge_index, layer_mapping
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config.to_dict(),
            "has_torch_geometric": self.has_torch_geometric
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved policy checkpoint to {path}")
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'TamiyoPolicyGNN':
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        config = PolicyConfig.from_dict(checkpoint["config"])
        
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loaded policy checkpoint from {path}")
        return model
```

**Features:**
- **Modern GNN Architecture:** 3-layer Graph Convolutional Network with residual connections
- **Acceleration Support:** torch-scatter detection with graceful fallback
- **Multi-head Decision Making:** Adaptation type, priority, urgency, and value estimation
- **torch.compile Integration:** Automatic optimization when available
- **Flexible Input:** Handles variable graph sizes and missing features
- **Checkpointing:** Model persistence for training and deployment

**Configuration Example:**
```python
# High-performance configuration
config = PolicyConfig(
    input_dim=64,
    hidden_dim=256,
    num_gnn_layers=4,
    dropout=0.2,
    confidence_threshold=0.8,
    urgency_threshold=0.6,
    use_torch_scatter=True,
    compile_model=True
)

# Create policy
policy = TamiyoPolicyGNN(config)

# Make decisions
decisions = policy.make_decision(model_state, layer_candidates=["layer_0", "layer_1"])
```

### `training.py` - Offline Policy Training

**Purpose:** Implements comprehensive offline reinforcement learning infrastructure for policy improvement using PPO (Proximal Policy Optimization).

#### Key Components

**`TamiyoTrainer`** - Policy Training Infrastructure
```python
class TamiyoTrainer:
    """
    Offline reinforcement learning trainer for Tamiyo policy network.
    
    Uses Proximal Policy Optimization (PPO) to improve strategic decision
    making based on collected experience from live training sessions.
    """
    
    def __init__(
        self,
        policy: TamiyoPolicyGNN,
        config: PolicyConfig,
        device: str = "auto"
    ):
        """
        Initialize trainer.
        
        Args:
            policy: Policy network to train
            config: Training configuration
            device: Training device
        """
        self.policy = policy
        self.config = config
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.policy.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Experience replay buffer
        self.experience_buffer = ExperienceReplayBuffer(
            max_size=10000,
            batch_size=config.batch_size
        )
        
        # Training state
        self.epoch = 0
        self.total_steps = 0
        self.best_score = float('-inf')
        
        # Metrics tracking
        self.training_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "average_reward": [],
            "success_rate": []
        }
        
        logger.info(f"Initialized TamiyoTrainer on device {self.device}")
    
    def add_experience(
        self,
        state: ModelGraphState,
        action: AdaptationDecision,
        reward: float,
        next_state: Optional[ModelGraphState] = None,
        done: bool = False
    ) -> None:
        """
        Add experience to replay buffer.
        
        Args:
            state: Model state when decision was made
            action: Adaptation decision taken
            reward: Reward received for the action
            next_state: Resulting model state (optional)
            done: Whether episode is complete
        """
        experience = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "timestamp": time.time()
        }
        
        self.experience_buffer.add(experience)
        logger.debug(f"Added experience with reward {reward:.3f}")
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.experience_buffer) < self.config.batch_size:
            return {}
        
        # Sample batch
        batch = self.experience_buffer.sample()
        
        # Prepare batch data
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)
        
        # Forward pass
        policy_outputs = self._forward_batch(states)
        
        # Calculate losses
        policy_loss = self._calculate_policy_loss(policy_outputs, actions, rewards)
        value_loss = self._calculate_value_loss(policy_outputs, rewards, next_states, dones)
        entropy_loss = self._calculate_entropy_loss(policy_outputs)
        
        # Total loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        
        self.optimizer.step()
        
        # Update metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "average_reward": torch.mean(rewards).item(),
        }
        
        # Update training metrics
        for key, value in metrics.items():
            self.training_metrics[key].append(value)
        
        self.total_steps += 1
        
        return metrics
    
    def train_epoch(self, num_steps: int = 100) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            num_steps: Number of training steps per epoch
            
        Returns:
            Epoch training metrics
        """
        self.policy.train()
        epoch_metrics = defaultdict(list)
        
        for step in range(num_steps):
            step_metrics = self.train_step()
            
            for key, value in step_metrics.items():
                epoch_metrics[key].append(value)
            
            if step % 20 == 0 and step_metrics:
                logger.debug(f"Epoch {self.epoch}, Step {step}: Loss = {step_metrics['total_loss']:.4f}")
        
        # Calculate epoch averages
        final_metrics = {}
        for key, values in epoch_metrics.items():
            if values:
                final_metrics[f"avg_{key}"] = np.mean(values)
                final_metrics[f"std_{key}"] = np.std(values)
        
        self.epoch += 1
        
        logger.info(f"Completed epoch {self.epoch}: {final_metrics}")
        return final_metrics
    
    def evaluate(self, eval_states: List[ModelGraphState]) -> Dict[str, float]:
        """
        Evaluate policy on evaluation states.
        
        Args:
            eval_states: List of model states for evaluation
            
        Returns:
            Evaluation metrics
        """
        self.policy.eval()
        
        total_confidence = 0.0
        total_decisions = 0
        successful_decisions = 0
        
        with torch.no_grad():
            for state in eval_states:
                decisions = self.policy.make_decision(state)
                
                for decision in decisions:
                    total_decisions += 1
                    total_confidence += decision.confidence
                    
                    # Consider decision successful if high confidence and urgency
                    if decision.confidence > 0.7 and decision.urgency > 0.5:
                        successful_decisions += 1
        
        metrics = {
            "total_decisions": total_decisions,
            "average_confidence": total_confidence / max(total_decisions, 1),
            "success_rate": successful_decisions / max(total_decisions, 1),
            "decisions_per_state": total_decisions / max(len(eval_states), 1)
        }
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "epoch": self.epoch,
            "total_steps": self.total_steps,
            "best_score": self.best_score,
            "training_metrics": self.training_metrics
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved training checkpoint to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.total_steps = checkpoint["total_steps"]
        self.best_score = checkpoint["best_score"]
        self.training_metrics = checkpoint["training_metrics"]
        
        logger.info(f"Loaded training checkpoint from {path}")
    
    def _prepare_batch(self, batch: List[Dict]) -> Tuple[List, List, torch.Tensor, List, torch.Tensor]:
        """Prepare batch data for training."""
        states = [exp["state"] for exp in batch]
        actions = [exp["action"] for exp in batch]
        rewards = torch.tensor([exp["reward"] for exp in batch], dtype=torch.float32, device=self.device)
        next_states = [exp["next_state"] for exp in batch]
        dones = torch.tensor([exp["done"] for exp in batch], dtype=torch.bool, device=self.device)
        
        return states, actions, rewards, next_states, dones
    
    def _forward_batch(self, states: List[ModelGraphState]) -> List[Dict[str, torch.Tensor]]:
        """Forward pass on batch of states."""
        outputs = []
        
        for state in states:
            # Convert state to graph input
            node_features, edge_index, _ = self.policy._prepare_graph_input(state)
            node_features = node_features.to(self.device)
            
            if edge_index is not None:
                edge_index = edge_index.to(self.device)
            
            # Forward pass
            output = self.policy.forward(node_features, edge_index)
            outputs.append(output)
        
        return outputs
    
    def _calculate_policy_loss(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        actions: List[AdaptationDecision],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """Calculate policy loss (simplified PPO)."""
        policy_losses = []
        
        for i, (output, action) in enumerate(zip(outputs, actions)):
            adaptation_logits = output["adaptation_logits"]
            
            # Convert action to target
            adaptation_types = ["no_action", "add_seed", "modify_architecture"]
            if action.adaptation_type in adaptation_types:
                target = torch.tensor(
                    adaptation_types.index(action.adaptation_type),
                    device=self.device
                )
            else:
                target = torch.tensor(0, device=self.device)  # Default to no_action
            
            # Cross entropy loss
            loss = F.cross_entropy(adaptation_logits, target.unsqueeze(0))
            
            # Weight by reward
            weighted_loss = loss * rewards[i]
            policy_losses.append(weighted_loss)
        
        return torch.stack(policy_losses).mean()
    
    def _calculate_value_loss(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        rewards: torch.Tensor,
        next_states: List[Optional[ModelGraphState]],
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Calculate value function loss."""
        value_losses = []
        
        for i, output in enumerate(outputs):
            predicted_value = output["value"].squeeze()
            
            # Calculate target value (simplified)
            target_value = rewards[i]
            
            # MSE loss
            loss = F.mse_loss(predicted_value, target_value)
            value_losses.append(loss)
        
        return torch.stack(value_losses).mean()
    
    def _calculate_entropy_loss(self, outputs: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Calculate entropy loss for exploration."""
        entropy_losses = []
        
        for output in outputs:
            adaptation_logits = output["adaptation_logits"]
            probs = F.softmax(adaptation_logits, dim=-1)
            log_probs = F.log_softmax(adaptation_logits, dim=-1)
            
            entropy = -(probs * log_probs).sum()
            entropy_losses.append(entropy)
        
        return torch.stack(entropy_losses).mean()
```

**`ExperienceReplayBuffer`** - Experience Storage
```python
class ExperienceReplayBuffer:
    """Experience replay buffer for offline training."""
    
    def __init__(self, max_size: int = 10000, batch_size: int = 32):
        """
        Initialize replay buffer.
        
        Args:
            max_size: Maximum buffer size
            batch_size: Batch size for sampling
        """
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, experience: Dict[str, Any]) -> None:
        """Add experience to buffer."""
        self.buffer.append(experience)
    
    def sample(self) -> List[Dict[str, Any]]:
        """Sample batch from buffer."""
        if len(self.buffer) < self.batch_size:
            return list(self.buffer)
        
        return random.sample(self.buffer, self.batch_size)
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
```

**Features:**
- **PPO Implementation:** Proximal Policy Optimization for stable training
- **Experience Replay:** Efficient storage and sampling of training experiences
- **Multi-loss Training:** Policy, value, and entropy loss components
- **Gradient Clipping:** Stable training with gradient norm clipping
- **Comprehensive Metrics:** Detailed tracking of training progress
- **Checkpointing:** Full training state persistence

**Training Usage:**
```python
# Initialize trainer
trainer = TamiyoTrainer(policy, config)

# Add experience during live training
trainer.add_experience(
    state=current_model_state,
    action=adaptation_decision,
    reward=performance_improvement,
    next_state=resulting_model_state,
    done=False
)

# Train offline
for epoch in range(100):
    metrics = trainer.train_epoch(num_steps=50)
    
    if epoch % 10 == 0:
        eval_metrics = trainer.evaluate(eval_states)
        print(f"Epoch {epoch}: {eval_metrics}")
        
        # Save checkpoint
        trainer.save_checkpoint(f"checkpoints/tamiyo_epoch_{epoch}.pt")
```

This covers the core Tamiyo service components. The remaining service documentation (main orchestration) and other services will continue in the next section.