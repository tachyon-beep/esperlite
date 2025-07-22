"""
Production Test Scenario Factory for Realistic Test Data Generation.

This module provides factories for creating realistic production scenarios
for comprehensive testing across all Phase 2 components.
"""

import random
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
import torch.nn as nn

from esper.contracts.operational import HealthSignal, AdaptationDecision
from esper.contracts.assets import Blueprint, CompiledKernel


class ScenarioType(Enum):
    """Types of production scenarios for testing."""
    UNHEALTHY_SYSTEM = "unhealthy_system"
    STABLE_SYSTEM = "stable_system"
    HIGH_THROUGHPUT = "high_throughput"
    DEGRADING_PERFORMANCE = "degrading_performance"
    EMERGENCY_ADAPTATION = "emergency_adaptation"
    MIXED_HEALTH = "mixed_health"
    CASCADING_FAILURE = "cascading_failure"
    RECOVERY_SCENARIO = "recovery_scenario"


@dataclass
class PerformanceRequirements:
    """Performance requirements for test scenarios."""
    max_processing_latency_ms: float = 100.0
    min_throughput_hz: float = 10000.0
    max_memory_growth_mb: float = 100.0
    p95_latency_ms: float = 50.0
    p99_latency_ms: float = 150.0


@dataclass
class TestScenario:
    """Complete test scenario with expectations."""
    scenario_type: ScenarioType
    description: str
    health_signals: List[HealthSignal]
    expected_decisions: Dict[str, Any]
    performance_requirements: Optional[PerformanceRequirements] = None
    duration_seconds: float = 300.0
    metadata: Optional[Dict[str, Any]] = None


class ProductionHealthSignalFactory:
    """Factory for creating realistic health signals based on production patterns."""
    
    @staticmethod
    def create_healthy_signal(layer_id: int, seed_id: int = 0, epoch: int = 100) -> HealthSignal:
        """Create a healthy layer signal."""
        return HealthSignal(
            layer_id=layer_id,
            seed_id=seed_id,
            chunk_id=0,
            epoch=epoch,
            activation_variance=np.random.uniform(0.04, 0.08),  # Good variance
            dead_neuron_ratio=np.random.uniform(0.01, 0.03),   # Few dead neurons
            avg_correlation=np.random.uniform(0.75, 0.95),     # Good correlation
            health_score=np.random.uniform(0.8, 1.0),          # Excellent health
            error_count=np.random.poisson(0.1),                # Very few errors
            is_ready_for_transition=False
        )
    
    @staticmethod
    def create_degraded_signal(layer_id: int, seed_id: int = 0, epoch: int = 100, 
                             severity: float = 0.5) -> HealthSignal:
        """Create a degraded layer signal with specified severity (0=healthy, 1=critical)."""
        base_variance = 0.06 * (1 - severity) + 0.001 * severity
        base_dead_ratio = 0.02 * (1 - severity) + 0.5 * severity
        base_correlation = 0.85 * (1 - severity) + 0.2 * severity
        base_health = 0.9 * (1 - severity) + 0.1 * severity
        base_errors = np.random.poisson(severity * 20)
        
        return HealthSignal(
            layer_id=layer_id,
            seed_id=seed_id,
            chunk_id=0,
            epoch=epoch,
            activation_variance=max(0.001, base_variance + np.random.normal(0, 0.01)),
            dead_neuron_ratio=min(0.8, max(0.0, base_dead_ratio + np.random.normal(0, 0.05))),
            avg_correlation=min(1.0, max(0.0, base_correlation + np.random.normal(0, 0.1))),
            health_score=min(1.0, max(0.0, base_health + np.random.normal(0, 0.05))),
            error_count=max(0, base_errors + np.random.poisson(severity)),
            is_ready_for_transition=severity > 0.7
        )
    
    @staticmethod
    def create_time_series_signals(layer_id: int, num_signals: int, 
                                 trend: str = "stable") -> List[HealthSignal]:
        """Create time series of health signals showing various trends."""
        signals = []
        base_time = time.time()
        
        for i in range(num_signals):
            if trend == "degrading":
                severity = min(0.9, i / (num_signals * 0.7))  # Gradual degradation
            elif trend == "recovering":
                severity = max(0.1, 0.9 - i / (num_signals * 0.8))  # Gradual recovery
            elif trend == "oscillating":
                severity = 0.3 + 0.4 * np.sin(2 * np.pi * i / 20)  # Oscillating pattern
            else:  # stable
                severity = np.random.uniform(0.1, 0.3)  # Stable with noise
            
            signal = ProductionHealthSignalFactory.create_degraded_signal(
                layer_id=layer_id,
                seed_id=i % 4,
                epoch=100 + i,
                severity=severity
            )
            signals.append(signal)
        
        return signals


class ProductionModelFactory:
    """Factory for creating realistic neural network models for testing."""
    
    @staticmethod
    def create_vision_transformer_like() -> nn.Module:
        """Create a ViT-like model with realistic layer structure."""
        class VisionTransformerTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
                self.pos_embed = nn.Parameter(torch.randn(1, 197, 768))
                self.blocks = nn.ModuleList([
                    nn.TransformerEncoderLayer(768, 12, 3072, batch_first=True)
                    for _ in range(12)
                ])
                self.norm = nn.LayerNorm(768)
                self.head = nn.Linear(768, 1000)
            
            def forward(self, x):
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                x = x + self.pos_embed
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                return self.head(x[:, 0])
        
        return VisionTransformerTest()
    
    @staticmethod
    def create_resnet_like() -> nn.Module:
        """Create a ResNet-like model with realistic bottlenecks."""
        class ResNetBottleneck(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels//4, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels//4)
                self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, 3, 
                                     stride=stride, padding=1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels//4)
                self.conv3 = nn.Conv2d(out_channels//4, out_channels, 1, bias=False)
                self.bn3 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
                self.downsample = None
                if stride != 1 or in_channels != out_channels:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                identity = x
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.relu(self.bn2(self.conv2(out)))
                out = self.bn3(self.conv3(out))
                
                if self.downsample:
                    identity = self.downsample(x)
                
                out += identity
                return self.relu(out)
        
        class ResNetTest(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
                
                self.layer1 = nn.Sequential(*[
                    ResNetBottleneck(64 if i == 0 else 256, 256) for i in range(3)
                ])
                self.layer2 = nn.Sequential(
                    ResNetBottleneck(256, 512, stride=2),
                    *[ResNetBottleneck(512, 512) for _ in range(3)]
                )
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512, 1000)
            
            def forward(self, x):
                x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.fc(x)
        
        return ResNetTest()


class ProductionScenarioFactory:
    """Main factory for creating comprehensive production test scenarios."""
    
    @classmethod
    def create_unhealthy_system_scenario(cls) -> TestScenario:
        """Create scenario with multiple unhealthy layers requiring intervention."""
        health_signals = []
        
        # Critical layer needing immediate attention
        health_signals.append(
            ProductionHealthSignalFactory.create_degraded_signal(
                8, severity=0.85, epoch=150  # transformer_layer_8
            )
        )
        
        # Moderately degraded layer
        health_signals.append(
            ProductionHealthSignalFactory.create_degraded_signal(
                4, severity=0.45, epoch=150  # transformer_layer_4
            )
        )
        
        # Healthy layers for contrast
        for layer_idx in [0, 1, 2, 5, 6, 7, 9, 10, 11]:
            health_signals.append(
                ProductionHealthSignalFactory.create_healthy_signal(
                    layer_idx, epoch=150  # transformer_layer_{layer_idx}
                )
            )
        
        return TestScenario(
            scenario_type=ScenarioType.UNHEALTHY_SYSTEM,
            description="Multi-layer ViT model with 2 degraded layers requiring intervention",
            health_signals=health_signals,
            expected_decisions={
                'should_decide': True,
                'primary_target': 8,  # transformer_layer_8
                'secondary_target': 4,  # transformer_layer_4
                'min_confidence': 0.7,
                'min_urgency': 0.8,
                'adaptation_type': 'kernel_replacement'
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=80.0,
                min_throughput_hz=8000.0
            ),
            metadata={'model_type': 'vision_transformer', 'total_layers': 12}
        )
    
    @classmethod
    def create_stable_system_scenario(cls) -> TestScenario:
        """Create scenario with stable, healthy system that shouldn't require intervention."""
        health_signals = []
        
        # All layers healthy with realistic variance
        for layer_idx in range(12):
            health_signals.append(
                ProductionHealthSignalFactory.create_healthy_signal(
                    layer_idx, epoch=200  # transformer_layer_{layer_idx}
                )
            )
        
        return TestScenario(
            scenario_type=ScenarioType.STABLE_SYSTEM,
            description="Stable ViT model with all layers healthy",
            health_signals=health_signals,
            expected_decisions={
                'should_decide': False,
                'confidence_threshold': 0.7,
                'intervention_rate': 0.0
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=50.0,
                min_throughput_hz=12000.0
            ),
            metadata={'model_type': 'vision_transformer', 'stability_duration': '24_hours'}
        )
    
    @classmethod
    def create_high_throughput_scenario(cls, signal_count: int = 50000) -> TestScenario:
        """Create high-throughput scenario for load testing."""
        health_signals = []
        
        # Generate massive number of signals across multiple models
        for i in range(signal_count):
            model_id = i // 1000  # 50 models with 1000 signals each
            layer_id = (model_id * 10) + (i % 10)  # Unique layer IDs
            
            # Most signals healthy with occasional issues
            severity = 0.1 if random.random() > 0.05 else random.uniform(0.3, 0.8)
            
            signal = ProductionHealthSignalFactory.create_degraded_signal(
                layer_id=layer_id,
                seed_id=i % 4,
                epoch=100 + (i // 10000),
                severity=severity
            )
            health_signals.append(signal)
        
        return TestScenario(
            scenario_type=ScenarioType.HIGH_THROUGHPUT,
            description=f"High-throughput scenario with {signal_count:,} signals across 50 models",
            health_signals=health_signals,
            expected_decisions={
                'max_processing_time_ms': 30.0,
                'expected_interventions': signal_count * 0.05,  # ~5% intervention rate
                'throughput_requirement': 15000.0
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=30.0,
                min_throughput_hz=15000.0,
                max_memory_growth_mb=200.0,
                p95_latency_ms=25.0,
                p99_latency_ms=50.0
            ),
            duration_seconds=600.0,  # 10 minute load test
            metadata={'models': 50, 'signals_per_model': 1000}
        )
    
    @classmethod
    def create_cascading_failure_scenario(cls) -> TestScenario:
        """Create scenario simulating cascading failures across model layers."""
        health_signals = []
        
        # Start with one failed layer, then progressive degradation
        layer_ids = list(range(8))  # resnet_layer_0 through resnet_layer_7
        
        for i, layer_id in enumerate(layer_ids):
            if i == 0:
                # Initial failure
                severity = 0.9
            elif i <= 3:
                # Cascading degradation
                severity = 0.6 + (i * 0.1)
            else:
                # Still healthy but showing stress
                severity = 0.2 + (i * 0.05)
            
            signal = ProductionHealthSignalFactory.create_degraded_signal(
                layer_id=layer_id,
                severity=severity,
                epoch=180
            )
            health_signals.append(signal)
        
        return TestScenario(
            scenario_type=ScenarioType.CASCADING_FAILURE,
            description="Cascading failure starting from resnet_layer_0 affecting downstream layers",
            health_signals=health_signals,
            expected_decisions={
                'should_decide': True,
                'emergency_mode': True,
                'primary_target': 0,  # resnet_layer_0
                'affected_layers': [1, 2, 3],  # resnet_layer_1, 2, 3
                'min_confidence': 0.8,
                'min_urgency': 0.9
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=60.0,  # Allow more time for emergency handling
                min_throughput_hz=5000.0
            ),
            metadata={'failure_pattern': 'cascading', 'origin_layer': 0}
        )
    
    @classmethod
    def create_degrading_performance_scenario(cls) -> TestScenario:
        """Create scenario with gradually degrading performance over time."""
        health_signals = []
        
        # Create time series showing gradual degradation
        for layer_idx in range(6):
            layer_signals = ProductionHealthSignalFactory.create_time_series_signals(
                layer_idx,  # conv_layer_{layer_idx}
                num_signals=100,  # 100 time points
                trend="degrading" if layer_idx < 3 else "stable"
            )
            health_signals.extend(layer_signals)
        
        return TestScenario(
            scenario_type=ScenarioType.DEGRADING_PERFORMANCE,
            description="Gradually degrading performance in first 3 CNN layers over time",
            health_signals=health_signals,
            expected_decisions={
                'should_decide': True,
                'intervention_timing': 'progressive',
                'target_layers': [0, 1, 2],  # conv_layer_0, 1, 2
                'min_confidence': 0.6,
                'adaptation_strategy': 'gradual_replacement'
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=70.0,
                min_throughput_hz=7000.0
            ),
            duration_seconds=1800.0,  # 30 minute degradation test
            metadata={'degradation_rate': 'gradual', 'affected_layer_count': 3}
        )
    
    @classmethod
    def create_mixed_health_scenario(cls) -> TestScenario:
        """Create scenario with mixed health patterns across different model types."""
        health_signals = []
        
        # ViT model - mostly healthy (layers 0-11)
        for i in range(12):
            severity = 0.1 if i not in [3, 7] else 0.5
            health_signals.append(
                ProductionHealthSignalFactory.create_degraded_signal(
                    i, severity=severity, epoch=120  # vit_layer_{i}
                )
            )
        
        # ResNet model - some issues (layers 100-115) 
        for i in range(16):
            severity = 0.3 if i % 4 == 0 else 0.1
            health_signals.append(
                ProductionHealthSignalFactory.create_degraded_signal(
                    100 + i, severity=severity, epoch=120  # resnet_block_{i}
                )
            )
        
        # BERT model - critical issues (layers 200-223)
        for i in range(24):
            severity = 0.8 if i in [5, 12, 18] else 0.2
            health_signals.append(
                ProductionHealthSignalFactory.create_degraded_signal(
                    200 + i, severity=severity, epoch=120  # bert_layer_{i}
                )
            )
        
        return TestScenario(
            scenario_type=ScenarioType.MIXED_HEALTH,
            description="Mixed health across ViT, ResNet, and BERT models with varying severity",
            health_signals=health_signals,
            expected_decisions={
                'should_decide': True,
                'model_priority': ['bert', 'resnet', 'vit'],
                'critical_layers': [205, 212, 218],  # bert_layer_5, 12, 18
                'min_confidence': 0.7
            },
            performance_requirements=PerformanceRequirements(
                max_processing_latency_ms=90.0,
                min_throughput_hz=6000.0
            ),
            metadata={'model_types': 3, 'total_layers': 52}
        )
    
    @classmethod
    def get_scenario_by_type(cls, scenario_type: ScenarioType, **kwargs) -> TestScenario:
        """Get a specific scenario by type with optional parameters."""
        scenario_map = {
            ScenarioType.UNHEALTHY_SYSTEM: cls.create_unhealthy_system_scenario,
            ScenarioType.STABLE_SYSTEM: cls.create_stable_system_scenario,
            ScenarioType.HIGH_THROUGHPUT: lambda: cls.create_high_throughput_scenario(
                kwargs.get('signal_count', 50000)
            ),
            ScenarioType.CASCADING_FAILURE: cls.create_cascading_failure_scenario,
            ScenarioType.DEGRADING_PERFORMANCE: cls.create_degrading_performance_scenario,
            ScenarioType.MIXED_HEALTH: cls.create_mixed_health_scenario,
        }
        
        factory_func = scenario_map.get(scenario_type)
        if not factory_func:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
        
        return factory_func()
    
    @classmethod
    def create_all_scenarios(cls) -> Dict[ScenarioType, TestScenario]:
        """Create all available test scenarios for comprehensive testing."""
        scenarios = {}
        
        for scenario_type in ScenarioType:
            if scenario_type == ScenarioType.HIGH_THROUGHPUT:
                scenarios[scenario_type] = cls.create_high_throughput_scenario(10000)
            else:
                scenarios[scenario_type] = cls.get_scenario_by_type(scenario_type)
        
        return scenarios


# Utility functions for test validation
def validate_scenario_expectations(scenario: TestScenario, actual_results: Dict[str, Any]) -> bool:
    """Validate that actual test results match scenario expectations."""
    expected = scenario.expected_decisions
    
    # Check basic decision expectations
    if 'should_decide' in expected:
        if expected['should_decide'] != (actual_results.get('decision') is not None):
            return False
    
    # Check confidence and urgency thresholds
    if actual_results.get('decision'):
        decision = actual_results['decision']
        if 'min_confidence' in expected:
            if decision.confidence < expected['min_confidence']:
                return False
        if 'min_urgency' in expected:
            if decision.urgency < expected['min_urgency']:
                return False
    
    return True


def generate_performance_report(scenario: TestScenario, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Generate performance report comparing actual metrics to requirements."""
    if not scenario.performance_requirements:
        return {'status': 'no_requirements'}
    
    requirements = scenario.performance_requirements
    report = {
        'scenario_type': scenario.scenario_type.value,
        'requirements_met': True,
        'violations': [],
        'metrics': metrics
    }
    
    # Check each requirement
    if 'latency_ms' in metrics:
        if metrics['latency_ms'] > requirements.max_processing_latency_ms:
            report['requirements_met'] = False
            report['violations'].append({
                'metric': 'latency',
                'required': requirements.max_processing_latency_ms,
                'actual': metrics['latency_ms']
            })
    
    if 'throughput_hz' in metrics:
        if metrics['throughput_hz'] < requirements.min_throughput_hz:
            report['requirements_met'] = False
            report['violations'].append({
                'metric': 'throughput',
                'required': requirements.min_throughput_hz,
                'actual': metrics['throughput_hz']
            })
    
    if 'memory_growth_mb' in metrics:
        if metrics['memory_growth_mb'] > requirements.max_memory_growth_mb:
            report['requirements_met'] = False
            report['violations'].append({
                'metric': 'memory_growth',
                'required': requirements.max_memory_growth_mb,
                'actual': metrics['memory_growth_mb']
            })
    
    return report