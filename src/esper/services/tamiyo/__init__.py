"""
Tamiyo Strategic Controller package.

This package implements the intelligent strategic controller that analyzes
host model state and makes adaptation decisions using Graph Neural Networks.

Enhanced with REMEDIATION ACTIVITY A1 components:
- Complete blueprint library integration
- Multi-metric intelligent reward system
- Phase 1-2 seamless integration layer
- Production-grade health signal collection
"""

from .analyzer import LayerNode, ModelGraphAnalyzer
from .autonomous_service import (
    AutonomousServiceConfig,
    AutonomousTamiyoService,
    ServiceStatistics,
)
from .blueprint_integration import (
    BlueprintSelector,
    ExecutionSystemIntegrator,
    Phase2IntegrationOrchestrator,
)
from .enhanced_main import EnhancedTamiyoService
from .health_collector import ProductionHealthCollector
from .main import TamiyoService
from .model_graph_builder import ModelGraphBuilder, ModelGraphState, ModelTopology
from .policy import (
    EnhancedPolicyTrainingState,
    EnhancedTamiyoPolicyGNN,
    PolicyConfig,
    PolicyTrainingState,
    TamiyoPolicyGNN,
)
from .policy_trainer import (
    Experience,
    ExperienceReplayBuffer,
    PolicyPerformanceTracker,
    PolicyRollbackManager,
    PolicySafetyValidator,
    ProductionPolicyTrainer,
    ProductionTrainingConfig,
    TrainingMetrics,
)
from .reward_computer import (
    IntelligentRewardComputer,
    RewardAnalysis,
    TemporalRewardAnalyzer,
)
from .reward_system import (
    MultiMetricRewardSystem,
    RewardComponent,
    RewardConfig,
    RewardMetrics,
)
from .training import TamiyoTrainer, TrainingConfig

__all__ = [
    # Core Policy Components
    "TamiyoPolicyGNN",
    "EnhancedTamiyoPolicyGNN",
    "PolicyConfig",
    "PolicyTrainingState",
    "EnhancedPolicyTrainingState",
    # Training Components
    "TamiyoTrainer",
    "TrainingConfig",
    "ProductionPolicyTrainer",
    "ProductionTrainingConfig",
    "TrainingMetrics",
    "Experience",
    "ExperienceReplayBuffer",
    "PolicyPerformanceTracker",
    "PolicyRollbackManager",
    "PolicySafetyValidator",
    # Analysis Components
    "ModelGraphAnalyzer",
    "LayerNode",
    "ModelGraphBuilder",
    "ModelGraphState",
    "ModelTopology",
    # Service Components
    "TamiyoService",
    "EnhancedTamiyoService",  # Primary service with REMEDIATION A1
    "AutonomousTamiyoService",
    "AutonomousServiceConfig",
    "ServiceStatistics",
    # Health & Monitoring
    "ProductionHealthCollector",
    # Reward System
    "MultiMetricRewardSystem",
    "RewardConfig",
    "RewardMetrics",
    "RewardComponent",
    "IntelligentRewardComputer",
    "RewardAnalysis",
    "TemporalRewardAnalyzer",
    # Phase 1-2 Integration
    "BlueprintSelector",
    "ExecutionSystemIntegrator",
    "Phase2IntegrationOrchestrator",
]
