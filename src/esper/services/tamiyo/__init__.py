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

from .analyzer import LayerNode
from .analyzer import ModelGraphAnalyzer
from .autonomous_service import AutonomousServiceConfig
from .autonomous_service import AutonomousTamiyoService
from .autonomous_service import ServiceStatistics
from .blueprint_integration import BlueprintSelector
from .blueprint_integration import ExecutionSystemIntegrator
from .blueprint_integration import Phase2IntegrationOrchestrator
from .enhanced_main import EnhancedTamiyoService
from .health_collector import ProductionHealthCollector
from .main import TamiyoService
from .model_graph_builder import ModelGraphBuilder
from .model_graph_builder import ModelGraphState
from .model_graph_builder import ModelTopology
from .policy import EnhancedPolicyTrainingState
from .policy import EnhancedTamiyoPolicyGNN
from .policy import PolicyConfig
from .policy import PolicyTrainingState
from .policy import TamiyoPolicyGNN
from .policy_trainer import Experience
from .policy_trainer import ExperienceReplayBuffer
from .policy_trainer import PolicyPerformanceTracker
from .policy_trainer import PolicyRollbackManager
from .policy_trainer import PolicySafetyValidator
from .policy_trainer import ProductionPolicyTrainer
from .policy_trainer import ProductionTrainingConfig
from .policy_trainer import TrainingMetrics
from .reward_computer import IntelligentRewardComputer
from .reward_computer import RewardAnalysis
from .reward_computer import TemporalRewardAnalyzer
from .reward_system import MultiMetricRewardSystem
from .reward_system import RewardComponent
from .reward_system import RewardConfig
from .reward_system import RewardMetrics
from .training import TamiyoTrainer
from .training import TrainingConfig

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
