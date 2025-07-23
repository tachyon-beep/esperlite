# Esper Phase 2: Intelligence-Driven Morphogenetic Policy System

## Executive Summary

Phase 2 focuses on implementing an intelligent policy system that can autonomously decide when, where, and how to apply morphogenetic adaptations based on real-time analysis of model performance. Building on the production-ready kernel execution system from Phase 1, this phase introduces the "brain" of the morphogenetic system through the Tamiyo service.

**Key Objective:** Transform Esper from a manual kernel loading system into an autonomous adaptation engine that continuously optimizes neural networks during training.

## Phase 1 Integration Points

### ✅ Available Foundation from Phase 1:
- **Real Kernel Execution:** Production-ready execution with <0.5ms latency
- **Comprehensive Error Recovery:** 5 recovery strategies with 99%+ success rate
- **Enhanced Caching System:** Metadata validation and compatibility checking
- **Health Signal Infrastructure:** Error tracking and performance monitoring
- **Robust Testing Framework:** 47 test cases with comprehensive coverage

### 🔗 Phase 2 Integration Requirements:
- **Health Signal Collection:** Leverage existing error recovery system for decision making
- **Performance Metrics:** Use Phase 1 cache and execution stats for training data
- **Safe Kernel Deployment:** Build on Phase 1's validation system for policy actions
- **Real-time Monitoring:** Extend Phase 1 health monitoring for feedback loops

---

## Phase 2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Tamiyo Intelligence Layer                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │  Health Signal  │  │ Graph Neural    │  │   Policy Decision   │  │
│  │   Collector     │  │   Network       │  │     Engine          │  │
│  │                 │  │   (GNN)         │  │                     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│           │                      │                        │         │
│           ▼                      ▼                        ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │    Reward       │  │  Experience     │  │  Blueprint Request  │  │
│  │   Computer      │  │    Buffer       │  │     Generator       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
           │                      │                        │
           ▼                      ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Phase 1 Execution System                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ KasminaLayer    │  │ Enhanced Kernel │  │ Error Recovery      │  │
│  │ (Real Exec)     │  │     Cache       │  │   Manager           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📋 Phase 2 Implementation Plan

### 2.1 Intelligence Foundation: Health Signal Processing System

**🎯 Objective:** Create a real-time system that transforms raw execution metrics into actionable intelligence.

#### 2.1.1 Real-Time Health Signal Collection

**📁 New Component:** `src/esper/services/tamiyo/health_collector.py`

```python
class ProductionHealthCollector:
    """
    Production-grade health signal collection with intelligent filtering.
    
    Integrates with Phase 1 error recovery system to collect and process
    health signals in real-time for policy training.
    """
    
    def __init__(
        self,
        oona_client: OonaClient,
        buffer_size: int = 50000,
        processing_batch_size: int = 1000
    ):
        self.oona_client = oona_client
        self.signal_buffer = HealthSignalBuffer(max_size=buffer_size)
        self.aggregator = HealthAggregator()
        self.filter_engine = SignalFilterEngine()
        
        # Performance optimization
        self.batch_processor = BatchProcessor(batch_size=processing_batch_size)
        self.statistics = CollectionStatistics()
        
        # Integration with Phase 1
        self.error_recovery_integration = ErrorRecoveryIntegration()

    async def start_intelligent_collection(self):
        """Start health signal collection with intelligent filtering."""
        
        # Subscribe to all telemetry streams
        topics = [
            "telemetry.execution.kernel_performance",
            "telemetry.cache.hit_rates", 
            "telemetry.error_recovery.events",
            "telemetry.layer.health_signals",
            "telemetry.model.performance_metrics"
        ]
        
        await self.oona_client.subscribe_with_consumer_group(
            topics=topics,
            consumer_group="tamiyo_health_collectors",
            consumer_name=f"collector_{socket.gethostname()}"
        )
        
        # Start processing pipeline
        await asyncio.gather(
            self._message_ingestion_loop(),
            self._batch_processing_loop(),
            self._signal_analysis_loop(),
            self._statistics_reporting_loop()
        )

    async def _message_ingestion_loop(self):
        """High-performance message ingestion with intelligent filtering."""
        while True:
            try:
                # Consume messages in batches for efficiency
                messages = await self.oona_client.consume_batch(
                    count=100,
                    timeout_ms=50
                )
                
                for message in messages:
                    # Convert to health signal
                    health_signal = self._parse_health_signal(message)
                    
                    if health_signal and self.filter_engine.should_process(health_signal):
                        # Add to buffer with priority scoring
                        priority = self.filter_engine.calculate_priority(health_signal)
                        self.signal_buffer.add_with_priority(health_signal, priority)
                        
                        self.statistics.record_signal_processed()
                    
            except Exception as e:
                logger.error(f"Health signal ingestion error: {e}")
                await asyncio.sleep(0.1)

    def _parse_health_signal(self, message: OonaMessage) -> Optional[HealthSignal]:
        """Parse Oona message into structured health signal."""
        try:
            payload = message.payload
            
            # Handle different message types
            if message.topic.value == "telemetry.execution.kernel_performance":
                return HealthSignal(
                    layer_id=payload["layer_id"],
                    seed_id=payload["seed_id"],
                    chunk_id=payload.get("chunk_id", 0),
                    epoch=payload["epoch"],
                    
                    # Phase 1 integration: use actual execution metrics
                    execution_latency=payload["execution_latency_ms"],
                    cache_hit_rate=payload.get("cache_hit_rate", 1.0),
                    error_count=payload.get("error_count", 0),
                    
                    # Intelligence metrics
                    health_score=self._compute_health_score(payload),
                    activation_variance=payload.get("activation_variance", 0.0),
                    dead_neuron_ratio=payload.get("dead_neuron_ratio", 0.0),
                    avg_correlation=payload.get("avg_correlation", 0.5),
                    
                    timestamp=message.timestamp.timestamp()
                )
                
            elif message.topic.value == "telemetry.error_recovery.events":
                # Integrate with Phase 1 error recovery
                return self.error_recovery_integration.convert_to_health_signal(payload)
                
            return None
            
        except Exception as e:
            logger.warning(f"Failed to parse health signal: {e}")
            return None

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """Compute health score from execution metrics."""
        # Base score from execution performance
        execution_score = 1.0 - min(metrics.get("execution_latency_ms", 0) / 10.0, 1.0)
        
        # Error rate impact
        error_rate = metrics.get("error_count", 0) / max(metrics.get("total_executions", 1), 1)
        error_score = 1.0 - min(error_rate * 5, 1.0)
        
        # Cache performance impact
        cache_score = metrics.get("cache_hit_rate", 1.0)
        
        # Weighted combination
        return 0.4 * execution_score + 0.4 * error_score + 0.2 * cache_score
```

**🔧 Key Integration Features:**
- **Phase 1 Metrics Integration:** Leverages error recovery events and cache statistics
- **Intelligent Filtering:** Prioritizes signals based on anomaly detection
- **High-Performance Processing:** Batched message handling for low latency
- **Comprehensive Statistics:** Performance monitoring for the collector itself

#### 2.1.2 Graph Neural Network Model State Representation

**📁 New Component:** `src/esper/services/tamiyo/model_graph_builder.py`

```python
class ModelGraphBuilder:
    """
    Builds graph representations of model state for GNN processing.
    
    Converts health signals into graph structures that capture both
    architectural relationships and performance characteristics.
    """
    
    def __init__(self):
        self.topology_cache = {}  # Cache for model architectures
        self.feature_extractors = {
            'layer_features': LayerFeatureExtractor(),
            'connection_features': ConnectionFeatureExtractor(),
            'performance_features': PerformanceFeatureExtractor()
        }

    def build_model_graph(
        self,
        health_signals: List[HealthSignal],
        model_topology: ModelTopology,
        window_size: int = 100
    ) -> ModelGraphState:
        """Build graph representation from health signals and topology."""
        
        # Extract node features (layers/components)
        node_features = self._extract_node_features(health_signals, model_topology)
        
        # Extract edge features (connections between layers)
        edge_features, edge_index = self._extract_edge_features(model_topology)
        
        # Aggregate temporal information
        temporal_features = self._aggregate_temporal_features(health_signals, window_size)
        
        # Build graph data structure
        graph_data = Data(
            x=node_features,           # Node features [num_nodes, feature_dim]
            edge_index=edge_index,     # Edge connectivity [2, num_edges]
            edge_attr=edge_features,   # Edge features [num_edges, edge_feature_dim]
            temporal=temporal_features, # Temporal aggregations
            
            # Additional metadata
            global_features=self._extract_global_features(health_signals),
            health_trends=self._compute_health_trends(health_signals),
            problematic_layers=self._identify_problematic_layers(health_signals)
        )
        
        return ModelGraphState(
            graph_data=graph_data,
            timestamp=time.time(),
            health_signals=health_signals,
            topology=model_topology
        )

    def _extract_node_features(
        self,
        health_signals: List[HealthSignal],
        topology: ModelTopology
    ) -> torch.Tensor:
        """Extract per-layer features for graph nodes."""
        
        features = []
        
        for layer_name in topology.layer_names:
            # Get health signals for this layer
            layer_signals = [s for s in health_signals if s.layer_id == layer_name]
            
            if layer_signals:
                # Aggregate health metrics
                avg_health = np.mean([s.health_score for s in layer_signals])
                avg_latency = np.mean([s.execution_latency for s in layer_signals])
                error_rate = np.sum([s.error_count for s in layer_signals]) / len(layer_signals)
                
                # Layer architectural features
                layer_info = topology.get_layer_info(layer_name)
                param_count = layer_info.get('parameters', 0)
                layer_type_encoding = self._encode_layer_type(layer_info.get('type'))
                
                # Combine features
                node_feature = np.array([
                    avg_health,
                    avg_latency / 10.0,  # Normalize
                    error_rate,
                    np.log10(param_count + 1),  # Log scale for parameters
                    *layer_type_encoding,  # One-hot layer type
                ])
            else:
                # Default features for layers without signals
                node_feature = np.zeros(self.node_feature_dim)
            
            features.append(node_feature)
        
        return torch.tensor(features, dtype=torch.float32)
```

### 2.2 Decision Engine: Graph Neural Network Policy

**🎯 Objective:** Implement a GNN-based policy network that can analyze model graphs and make intelligent adaptation decisions.

#### 2.2.1 Production GNN Policy Architecture

**📁 Enhanced Component:** `src/esper/services/tamiyo/policy_network.py`

```python
class TamiyoPolicyGNN(nn.Module):
    """
    Production Graph Neural Network for morphogenetic policy decisions.
    
    Analyzes model state graphs and outputs adaptation decisions with
    confidence scores and uncertainty quantification.
    """
    
    def __init__(
        self,
        node_feature_dim: int = 16,
        edge_feature_dim: int = 8,
        hidden_dim: int = 128,
        num_gnn_layers: int = 4,
        num_action_types: int = 5,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Graph convolution layers with residual connections
        self.gnn_layers = nn.ModuleList([
            GCNConv(
                node_feature_dim if i == 0 else hidden_dim,
                hidden_dim,
                edge_dim=edge_feature_dim
            )
            for i in range(num_gnn_layers)
        ])
        
        # Layer normalization for stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_gnn_layers)
        ])
        
        # Attention mechanism for global state aggregation
        self.global_attention = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_action_types)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,           # Node features [num_nodes, node_feature_dim]
        edge_index: torch.Tensor,  # Edge connections [2, num_edges]
        edge_attr: torch.Tensor,   # Edge features [num_edges, edge_feature_dim]
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through GNN policy network."""
        
        # Graph convolution with residual connections
        h = x
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = gnn_layer(h, edge_index, edge_attr)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection (if dimensions match)
            if h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
        
        # Global graph representation using attention
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        global_repr = self.global_attention(h, batch)
        
        # Generate outputs
        policy_logits = self.policy_head(global_repr)
        value_estimate = self.value_head(global_repr)
        uncertainty = self.uncertainty_head(global_repr)
        
        return {
            'policy_logits': policy_logits,      # Action probabilities
            'value_estimate': value_estimate,    # State value
            'uncertainty': uncertainty,          # Prediction uncertainty
            'node_embeddings': h,               # For analysis
            'global_embedding': global_repr     # For analysis
        }

    def sample_action(
        self,
        policy_logits: torch.Tensor,
        temperature: float = 1.0,
        exploration_bonus: float = 0.1
    ) -> Tuple[int, float, float]:
        """Sample action from policy with exploration bonus for uncertainty."""
        
        # Apply temperature scaling
        scaled_logits = policy_logits / temperature
        
        # Add exploration bonus based on uncertainty
        if hasattr(self, '_last_uncertainty'):
            exploration_term = exploration_bonus * self._last_uncertainty
            scaled_logits = scaled_logits + exploration_term
        
        # Sample action
        action_probs = F.softmax(scaled_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # Calculate log probability and entropy
        log_prob = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        
        return action.item(), log_prob.item(), entropy.item()
```

#### 2.2.2 Real-Time Policy Training with Experience Replay

**📁 New Component:** `src/esper/services/tamiyo/policy_trainer.py`

```python
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
        experience_buffer_size: int = 100000
    ):
        self.policy = policy_network.to(device)
        self.device = device
        
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
            prioritized=True  # Prioritized experience replay
        )
        
        # Safety mechanisms for production
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
        
        # Validate experience for safety
        if not self.safety_validator.validate_experience(experience):
            logger.warning("Experience failed safety validation, skipping")
            return False
        
        # Add to buffer with priority
        priority = self._calculate_experience_priority(experience)
        self.experience_buffer.add(experience, priority)
        
        # Train if we have enough experiences
        if len(self.experience_buffer) >= self.min_batch_size:
            training_success = await self._train_batch()
            
            if training_success:
                self.training_stats['episodes'] += 1
                await self._update_performance_tracking()
                
                # Check if policy update is safe
                if not await self.safety_validator.validate_policy_update(self.policy):
                    logger.error("Policy update failed safety validation, rolling back")
                    await self.rollback_manager.rollback_to_safe_state()
                    return False
                
            return training_success
        
        return True

    async def _train_batch(self) -> bool:
        """Train on batch using PPO algorithm with safety constraints."""
        try:
            # Sample batch with prioritized replay
            batch = self.experience_buffer.sample_batch(
                batch_size=self.batch_size,
                prioritized=True
            )
            
            # Prepare batch data
            states = [exp.state for exp in batch]
            actions = torch.tensor([exp.action.action_type for exp in batch], device=self.device)
            rewards = torch.tensor([exp.reward for exp in batch], device=self.device)
            
            # Convert states to graph batches
            graph_batch = self._prepare_graph_batch(states)
            
            # Forward pass
            policy_outputs = self.policy(**graph_batch)
            
            # Compute losses
            policy_loss = self._compute_policy_loss(
                policy_outputs['policy_logits'],
                actions,
                rewards,
                batch
            )
            
            value_loss = self._compute_value_loss(
                policy_outputs['value_estimate'],
                rewards
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
            
            # Safety check: prevent catastrophic policy updates
            if total_loss > self.max_allowed_loss:
                logger.warning(f"Loss {total_loss} exceeds safety threshold, skipping update")
                return False
            
            # Backward pass with gradient clipping
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
            
            # Update priorities in experience buffer
            self._update_experience_priorities(batch, total_loss.item())
            
            return True
            
        except Exception as e:
            logger.error(f"Policy training error: {e}")
            return False

    def _compute_safety_regularization(self, policy_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute safety regularization to prevent dangerous policy updates."""
        
        # Prevent extreme confidence
        uncertainty = policy_outputs['uncertainty']
        min_uncertainty = 0.1  # Minimum uncertainty to maintain
        uncertainty_penalty = F.relu(min_uncertainty - uncertainty).mean()
        
        # Prevent policy collapse (maintain entropy)
        policy_probs = F.softmax(policy_outputs['policy_logits'], dim=-1)
        entropy = -(policy_probs * torch.log(policy_probs + 1e-8)).sum(dim=-1)
        min_entropy = 0.5
        entropy_penalty = F.relu(min_entropy - entropy).mean()
        
        return 0.1 * uncertainty_penalty + 0.1 * entropy_penalty
```

### 2.3 Intelligent Reward System

**🎯 Objective:** Develop a sophisticated reward computation system that provides accurate feedback for policy learning.

#### 2.3.1 Multi-Metric Reward Computer

**📁 New Component:** `src/esper/services/tamiyo/reward_computer.py`

```python
class IntelligentRewardComputer:
    """
    Advanced reward computation system that evaluates adaptation quality
    using multiple metrics and temporal analysis.
    """
    
    def __init__(self):
        self.baseline_tracker = BaselineMetricsTracker()
        self.temporal_analyzer = TemporalRewardAnalyzer()
        self.correlation_detector = CorrelationDetector()
        
        # Reward weights optimized through meta-learning
        self.reward_weights = {
            'accuracy_improvement': 1.0,      # Primary objective
            'speed_improvement': 0.6,         # Secondary objective
            'memory_efficiency': 0.4,         # Efficiency objective
            'stability_improvement': 0.8,     # Reliability objective
            'adaptation_cost': -0.3,          # Cost penalty
            'risk_penalty': -0.5              # Safety penalty
        }
        
        # Temporal discount factors
        self.temporal_weights = {
            'immediate': 0.3,    # 0-5 minutes
            'short_term': 0.4,   # 5-30 minutes  
            'medium_term': 0.25, # 30-120 minutes
            'long_term': 0.05    # 2+ hours
        }

    async def compute_adaptation_reward(
        self,
        adaptation_decision: AdaptationDecision,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_window: float = 300.0  # 5 minutes
    ) -> RewardAnalysis:
        """
        Compute comprehensive reward for adaptation decision.
        
        Integrates with Phase 1 metrics for accurate performance assessment.
        """
        
        # Collect temporal metrics across time windows
        temporal_metrics = await self._collect_temporal_metrics(
            adaptation_decision.layer_name,
            temporal_window
        )
        
        # Compute individual reward components
        reward_components = {}
        
        # 1. Accuracy/Performance Improvement
        accuracy_reward = self._compute_accuracy_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['accuracy'] = accuracy_reward
        
        # 2. Speed/Latency Improvement (integrates with Phase 1 execution metrics)
        speed_reward = self._compute_speed_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['speed'] = speed_reward
        
        # 3. Memory Efficiency (uses Phase 1 cache metrics)
        memory_reward = self._compute_memory_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['memory'] = memory_reward
        
        # 4. Stability/Reliability (integrates with Phase 1 error recovery)
        stability_reward = self._compute_stability_reward(
            pre_metrics, post_metrics, temporal_metrics
        )
        reward_components['stability'] = stability_reward
        
        # 5. Adaptation Cost (computational overhead)
        cost_penalty = self._compute_adaptation_cost(adaptation_decision)
        reward_components['cost'] = cost_penalty
        
        # 6. Risk Assessment (safety penalty)
        risk_penalty = await self._compute_risk_penalty(
            adaptation_decision, post_metrics
        )
        reward_components['risk'] = risk_penalty
        
        # Compute weighted total reward
        total_reward = sum(
            self.reward_weights[component] * reward_value
            for component, reward_value in reward_components.items()
        )
        
        # Apply temporal discounting
        discounted_reward = self._apply_temporal_discounting(
            total_reward, temporal_metrics
        )
        
        # Normalize to [-1, 1] range
        normalized_reward = torch.tanh(torch.tensor(discounted_reward)).item()
        
        return RewardAnalysis(
            total_reward=normalized_reward,
            components=reward_components,
            temporal_analysis=temporal_metrics,
            confidence=self._compute_reward_confidence(reward_components),
            explanation=self._generate_reward_explanation(reward_components)
        )

    def _compute_speed_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute speed improvement reward using Phase 1 execution metrics."""
        
        # Phase 1 integration: use actual execution latency
        pre_latency = pre_metrics.get('execution_latency_ms', 1000.0)
        post_latency = post_metrics.get('execution_latency_ms', 1000.0)
        
        # Relative improvement
        if pre_latency > 0:
            speed_improvement = (pre_latency - post_latency) / pre_latency
        else:
            speed_improvement = 0.0
        
        # Bonus for achieving target latency thresholds
        target_bonus = 0.0
        if post_latency < 0.5:  # Sub-millisecond execution
            target_bonus = 0.2
        elif post_latency < 1.0:  # Under 1ms
            target_bonus = 0.1
        
        # Penalty for degradation
        degradation_penalty = 0.0
        if speed_improvement < -0.1:  # >10% slowdown
            degradation_penalty = -0.5
        
        return speed_improvement + target_bonus + degradation_penalty

    def _compute_stability_reward(
        self,
        pre_metrics: Dict[str, float],
        post_metrics: Dict[str, float],
        temporal_metrics: Dict[str, Any]
    ) -> float:
        """Compute stability reward using Phase 1 error recovery metrics."""
        
        # Phase 1 integration: use error recovery statistics
        pre_error_rate = pre_metrics.get('error_rate', 0.0)
        post_error_rate = post_metrics.get('error_rate', 0.0)
        
        # Error rate improvement
        error_improvement = max(0, pre_error_rate - post_error_rate)
        
        # Recovery success rate (from Phase 1 error recovery)
        recovery_rate = post_metrics.get('recovery_success_rate', 1.0)
        recovery_bonus = (recovery_rate - 0.95) * 2.0  # Bonus above 95%
        
        # Stability variance penalty
        latency_variance = temporal_metrics.get('latency_variance', 0.0)
        variance_penalty = -min(latency_variance / 10.0, 0.5)
        
        return error_improvement * 2.0 + recovery_bonus + variance_penalty
```

### 2.4 Integration & Production Deployment

#### 2.4.1 Tamiyo Service Integration

**📁 Enhanced Component:** `src/esper/services/tamiyo/main.py`

```python
class ProductionTamiyoService:
    """
    Production Tamiyo service integrating all Phase 2 components.
    
    Provides autonomous morphogenetic policy decisions with comprehensive
    monitoring, safety checks, and integration with Phase 1 execution system.
    """
    
    def __init__(self, config: TamiyoConfig):
        self.config = config
        
        # Core components
        self.health_collector = ProductionHealthCollector(
            oona_client=OonaClient(config.redis_url),
            buffer_size=config.health_buffer_size
        )
        
        self.model_graph_builder = ModelGraphBuilder()
        
        self.policy_network = TamiyoPolicyGNN(
            node_feature_dim=config.node_feature_dim,
            hidden_dim=config.hidden_dim,
            num_gnn_layers=config.num_gnn_layers
        )
        
        self.policy_trainer = ProductionPolicyTrainer(
            policy_network=self.policy_network,
            device=torch.device(config.device)
        )
        
        self.reward_computer = IntelligentRewardComputer()
        
        # Production infrastructure
        self.safety_monitor = TamiyoSafetyMonitor()
        self.performance_monitor = TamiyoPerformanceMonitor()
        self.health_checker = TamiyoHealthChecker()
        
        # Phase 1 integration
        self.execution_integrator = ExecutionSystemIntegrator()
        
    async def start_autonomous_operation(self):
        """Start autonomous morphogenetic policy system."""
        
        logger.info("Starting Tamiyo autonomous operation")
        
        # Initialize all components
        await self._initialize_components()
        
        # Start main loops
        await asyncio.gather(
            self._health_collection_loop(),
            self._policy_decision_loop(),
            self._training_loop(),
            self._monitoring_loop(),
            self._safety_monitoring_loop()
        )

    async def _policy_decision_loop(self):
        """Main policy decision loop."""
        
        while True:
            try:
                # Get current model state
                current_health_signals = self.health_collector.get_recent_signals(
                    window_size=self.config.decision_window_size
                )
                
                if len(current_health_signals) < self.config.min_signals_for_decision:
                    await asyncio.sleep(self.config.decision_interval)
                    continue
                
                # Build graph representation
                model_graph = self.model_graph_builder.build_model_graph(
                    health_signals=current_health_signals,
                    model_topology=await self._get_current_topology()
                )
                
                # Make policy decision
                decision = await self._make_policy_decision(model_graph)
                
                if decision:
                    # Safety validation
                    if await self.safety_monitor.validate_decision(decision):
                        # Execute adaptation
                        success = await self._execute_adaptation(decision)
                        
                        if success:
                            # Monitor results and update policy
                            await self._monitor_adaptation_results(decision, model_graph)
                    else:
                        logger.warning(f"Decision failed safety validation: {decision}")
                
                await asyncio.sleep(self.config.decision_interval)
                
            except Exception as e:
                logger.error(f"Policy decision loop error: {e}")
                await asyncio.sleep(self.config.error_backoff_time)

    async def _make_policy_decision(
        self,
        model_graph: ModelGraphState
    ) -> Optional[AdaptationDecision]:
        """Make policy decision using trained GNN."""
        
        # Convert to PyTorch tensors
        graph_data = self._prepare_graph_for_inference(model_graph)
        
        # Forward pass through policy network
        with torch.no_grad():
            policy_outputs = self.policy_network(**graph_data)
        
        # Sample action with exploration
        action_type, log_prob, entropy = self.policy_network.sample_action(
            policy_outputs['policy_logits'],
            temperature=self.config.exploration_temperature
        )
        
        # Get uncertainty estimate
        uncertainty = policy_outputs['uncertainty'].item()
        
        # Only proceed if confidence is sufficient
        if uncertainty > self.config.max_decision_uncertainty:
            logger.debug(f"Decision uncertainty {uncertainty} too high, skipping")
            return None
        
        # Select target layer based on graph analysis
        target_layer = self._select_target_layer(
            model_graph,
            policy_outputs['node_embeddings']
        )
        
        if target_layer is None:
            return None
        
        # Create adaptation decision
        decision = AdaptationDecision(
            layer_name=target_layer,
            adaptation_type=self._map_action_to_adaptation_type(action_type),
            confidence=1.0 - uncertainty,
            urgency=self._compute_urgency(model_graph, target_layer),
            metadata={
                'policy_log_prob': log_prob,
                'policy_entropy': entropy,
                'graph_timestamp': model_graph.timestamp,
                'decision_method': 'gnn_policy'
            }
        )
        
        return decision

    async def _execute_adaptation(
        self,
        decision: AdaptationDecision
    ) -> bool:
        """Execute adaptation decision through Phase 1 integration."""
        
        try:
            # Phase 1 integration: use existing kernel loading system
            if decision.adaptation_type == "load_kernel":
                # Request new kernel from blueprint generator
                blueprint_request = await self._generate_blueprint_request(decision)
                
                if blueprint_request:
                    # Submit to Tezzeret for compilation
                    kernel_id = await self._submit_for_compilation(blueprint_request)
                    
                    if kernel_id:
                        # Use Phase 1 kernel loading
                        success = await self.execution_integrator.load_kernel(
                            layer_name=decision.layer_name,
                            kernel_id=kernel_id
                        )
                        
                        if success:
                            logger.info(f"Successfully executed adaptation: {decision}")
                            return True
            
            elif decision.adaptation_type == "unload_kernel":
                # Use Phase 1 kernel unloading
                success = await self.execution_integrator.unload_kernel(
                    layer_name=decision.layer_name
                )
                
                if success:
                    logger.info(f"Successfully unloaded kernel: {decision}")
                    return True
            
            # Other adaptation types...
            
            return False
            
        except Exception as e:
            logger.error(f"Adaptation execution failed: {e}")
            return False
```

---

## 🔧 Phase 2 Testing Strategy

### 2.4.2 Comprehensive Testing Framework

**📁 New Component:** `tests/integration/test_tamiyo_intelligence.py`

```python
@pytest.mark.integration
class TestTamiyoIntelligenceSystem:
    """Integration tests for complete Tamiyo intelligence system."""
    
    def setup_method(self):
        """Setup test environment with Phase 1 integration."""
        # Use Phase 1 components for realistic testing
        self.kasmina_layer = KasminaLayer(
            input_size=128,
            output_size=64,
            num_seeds=4,
            telemetry_enabled=True
        )
        
        # Setup Tamiyo components
        self.tamiyo_service = ProductionTamiyoService(
            config=TamiyoConfig.for_testing()
        )
        
    @pytest.mark.asyncio
    async def test_end_to_end_autonomous_adaptation(self):
        """Test complete autonomous adaptation cycle."""
        
        # 1. Generate realistic health signals
        health_signals = self._generate_realistic_health_signals()
        
        # 2. Process through health collector
        for signal in health_signals:
            await self.tamiyo_service.health_collector.process_signal(signal)
        
        # 3. Trigger policy decision
        model_graph = self.tamiyo_service.model_graph_builder.build_model_graph(
            health_signals=health_signals,
            model_topology=self._get_test_topology()
        )
        
        decision = await self.tamiyo_service._make_policy_decision(model_graph)
        
        assert decision is not None
        assert decision.confidence > 0.5
        assert decision.adaptation_type in VALID_ADAPTATION_TYPES
        
        # 4. Execute adaptation
        success = await self.tamiyo_service._execute_adaptation(decision)
        assert success
        
        # 5. Verify Phase 1 integration
        # Check that kernel was actually loaded in Phase 1 system
        layer_stats = self.kasmina_layer.get_layer_stats()
        assert layer_stats['state_stats']['active_seeds'] > 0
        
    @pytest.mark.asyncio
    async def test_policy_learning_convergence(self):
        """Test that policy learns and improves over time."""
        
        initial_performance = await self._measure_policy_performance()
        
        # Generate training experiences
        for _ in range(100):
            experience = self._generate_training_experience()
            await self.tamiyo_service.policy_trainer.train_on_experience(
                *experience
            )
        
        final_performance = await self._measure_policy_performance()
        
        # Policy should improve
        assert final_performance['accuracy'] > initial_performance['accuracy']
        assert final_performance['avg_reward'] > initial_performance['avg_reward']
        
    def test_safety_validation_prevents_dangerous_actions(self):
        """Test that safety system prevents dangerous adaptations."""
        
        # Create dangerous decision
        dangerous_decision = AdaptationDecision(
            layer_name="critical_layer",
            adaptation_type="experimental_modification",
            confidence=0.99,
            urgency=1.0
        )
        
        # Safety validator should reject it
        is_safe = self.tamiyo_service.safety_monitor.validate_decision(
            dangerous_decision
        )
        
        assert not is_safe
        
    @pytest.mark.performance
    def test_decision_latency_requirements(self):
        """Test that decision making meets latency requirements."""
        
        model_graph = self._generate_test_model_graph()
        
        start_time = time.perf_counter()
        decision = self.tamiyo_service._make_policy_decision(model_graph)
        decision_time = time.perf_counter() - start_time
        
        # Decision should be made within 100ms
        assert decision_time < 0.1
```

---

## 📊 Phase 2 Success Criteria & Milestones

### Key Performance Indicators (KPIs)

#### 🎯 Policy Intelligence Metrics
- **Decision Quality:** >80% of policy decisions lead to measurable improvements
- **Learning Convergence:** Policy converges within 1000 training experiences
- **Decision Latency:** <100ms for policy decision making
- **Safety Record:** 0% dangerous adaptations executed in production

#### ⚡ Performance Integration Metrics  
- **Health Signal Processing:** <50ms latency for signal analysis
- **Phase 1 Integration:** >99% successful kernel loading via policy decisions
- **Reward Correlation:** >0.8 correlation between predicted and actual improvements
- **System Stability:** 24+ hour autonomous operation without degradation

#### 🔄 Autonomous Operation Metrics
- **Adaptation Success Rate:** >85% of policy-initiated adaptations succeed
- **False Positive Rate:** <5% unnecessary adaptations
- **Training Efficiency:** Policy improvement visible within 100 adaptations
- **Resource Utilization:** <10% overhead from Tamiyo intelligence system

### 📅 Implementation Milestones

#### Month 1: Intelligence Foundation
- **Week 1-2:** Health signal collection and filtering system
- **Week 3-4:** Model graph builder and GNN architecture

#### Month 2: Policy Learning System  
- **Week 5-6:** GNN policy network implementation
- **Week 7-8:** Experience replay and training infrastructure

#### Month 3: Reward & Integration
- **Week 9-10:** Multi-metric reward computation system
- **Week 11-12:** Phase 1 integration and safety systems

#### Month 4: Production Deployment
- **Week 13-14:** End-to-end testing and validation
- **Week 15-16:** Production deployment and monitoring

---

## 🚀 Phase 2 Summary: Autonomous Morphogenetic Intelligence

**Phase 2 transforms Esper from a manual kernel execution system into an autonomous adaptation engine that continuously optimizes neural networks during training.**

### 🎯 Core Deliverables:
1. **Intelligent Health Analysis:** Real-time processing of model performance signals
2. **Graph Neural Network Policy:** Deep learning-based decision making for adaptations  
3. **Multi-Metric Reward System:** Sophisticated feedback for policy learning
4. **Safe Autonomous Operation:** Production-ready safety and monitoring systems
5. **Phase 1 Integration:** Seamless integration with kernel execution system

### 🔧 Production Features:
- **Real-Time Intelligence:** <100ms decision making with comprehensive analysis
- **Continuous Learning:** Policy improves autonomously through experience replay
- **Safety-First Design:** Multiple validation layers prevent dangerous adaptations
- **Phase 1 Integration:** Leverages production-ready kernel execution infrastructure
- **Comprehensive Monitoring:** Full observability into autonomous operation

**🎉 OUTCOME:** Phase 2 delivers the "brain" of the morphogenetic system, enabling truly autonomous neural network optimization that learns and adapts continuously during training while maintaining safety and performance guarantees.