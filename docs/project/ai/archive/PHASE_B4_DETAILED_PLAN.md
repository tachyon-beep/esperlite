# Phase B4: Dynamic Architecture Modification via Seed Orchestration - Detailed Implementation Plan

## Overview

Phase B4 enables neural networks to modify their own architecture during training through intelligent orchestration of the Kasmina seed mechanism. Rather than traditional model surgery, this achieves morphogenetic behavior by dynamically loading different kernels into seeds, adjusting blend factors, and managing seed lifecycles. This approach aligns with the HLD's vision of seeds as the fundamental unit of morphogenetic change.

## Current State Analysis

### Blocker Location
```python
# In src/esper/services/tolaria/trainer.py:~896
async def _apply_architecture_modification(self, model: nn.Module, decision: AdaptationDecision):
    """Apply architecture modification based on adaptation decision."""
    raise NotImplementedError("Architecture modification not yet implemented")
```

### Integration Points
1. **Tolaria Trainer** - Calls _apply_architecture_modification when Tamiyo decides
2. **Tamiyo Service** - Generates AdaptationDecision with architecture changes  
3. **KasminaLayers** - Already support seed-based kernel loading and blending
4. **Blueprint Integration** - Existing Phase 1-2 pipeline for kernel compilation
5. **Performance Tracker** - Phase B3's intelligent seed selection system

## Detailed Implementation Plan

### Phase 4.1: Seed Orchestration Framework (Days 1-2)

#### 1.1 Create Seed Orchestrator Infrastructure

**File: `src/esper/core/seed_orchestrator.py`**
```python
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from esper.contracts.operational import AdaptationDecision
from esper.execution.kasmina_layer import KasminaLayer
from esper.services.tamiyo.performance_tracker import PerformanceTracker
from esper.services.tamiyo.blueprint_integration import Phase2IntegrationOrchestrator

logger = logging.getLogger(__name__)

class SeedStrategy(Enum):
    """Strategy for managing seeds during architecture modification."""
    REPLACE = "replace"  # Replace underperforming kernel
    DIVERSIFY = "diversify"  # Load different kernels across seeds
    SPECIALIZE = "specialize"  # Specialize seeds for different tasks
    ENSEMBLE = "ensemble"  # Use all seeds as ensemble

@dataclass
class SeedModificationPlan:
    """Plan for modifying seeds in a layer."""
    layer_name: str
    strategy: SeedStrategy
    seed_modifications: Dict[int, Dict[str, Any]]  # seed_idx -> modification details
    expected_improvement: float
    risk_score: float
    reasoning: str

class SeedOrchestrator:
    """
    Orchestrates dynamic architecture modification through Kasmina seeds.
    
    Instead of traditional model surgery, this achieves morphogenetic behavior by:
    1. Loading different kernels into seeds
    2. Adjusting seed blend factors
    3. Managing seed lifecycle and specialization
    """
    
    def __init__(
        self,
        performance_tracker: PerformanceTracker,
        blueprint_registry: BlueprintRegistry,
        oona_client: OonaClient,
        urza_url: str
    ):
        self.performance_tracker = performance_tracker
        self.integration_orchestrator = Phase2IntegrationOrchestrator(
            blueprint_registry=blueprint_registry,
            oona_client=oona_client,
            urza_url=urza_url
        )
    
    async def apply_architecture_modification(
        self,
        model: nn.Module,
        decision: AdaptationDecision
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Apply architecture modification through seed orchestration.
        
        Args:
            model: Model containing Kasmina layers
            decision: Adaptation decision from Tamiyo
            
        Returns:
            (success, details) tuple
        """
        # Implementation details below
        
    def remove_layer(
        self,
        model: nn.Module,
        layer_name: str,
        bridge_strategy: str = "direct"
    ) -> SurgeryResult:
        """
        Remove a layer and bridge connections.
        
        Args:
            model: Target model
            layer_name: Layer to remove
            bridge_strategy: How to connect around removed layer
        """
        # Implementation details below
        
    def modify_layer_width(
        self,
        model: nn.Module,
        layer_name: str,
        new_width: int,
        initialization: str = "kaiming"
    ) -> SurgeryResult:
        """Modify the width (channels/neurons) of a layer."""
        # Implementation details below
        
    def add_skip_connection(
        self,
        model: nn.Module,
        from_layer: str,
        to_layer: str,
        connection_type: str = "add"
    ) -> SurgeryResult:
        """Add residual/skip connection between layers."""
        # Implementation details below
```

#### 1.2 Create Validation Framework

**File: `src/esper/core/surgery_validators.py`**
```python
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    check_name: str
    details: Dict[str, Any]
    errors: List[str]
    warnings: List[str]

class SurgeryValidator(ABC):
    """Base class for surgery validators."""
    
    @abstractmethod
    def validate(self, model: nn.Module, operation: str, **kwargs) -> ValidationResult:
        """Validate a surgery operation."""
        pass

class DimensionValidator(SurgeryValidator):
    """Validates tensor dimension compatibility."""
    
    def validate(self, model: nn.Module, operation: str, **kwargs) -> ValidationResult:
        """Check dimension flow through model."""
        # Trace through model with dummy input
        # Verify all dimensions match
        # Return validation result

class GradientFlowValidator(SurgeryValidator):
    """Ensures gradients can flow after surgery."""
    
    def validate(self, model: nn.Module, operation: str, **kwargs) -> ValidationResult:
        """Verify gradient flow is preserved."""
        # Create dummy loss and backward
        # Check all parameters receive gradients
        # Detect any gradient blocking

class MemoryValidator(SurgeryValidator):
    """Validates memory impact of changes."""
    
    def validate(self, model: nn.Module, operation: str, **kwargs) -> ValidationResult:
        """Check memory requirements."""
        # Calculate parameter count change
        # Estimate activation memory
        # Verify within constraints

class TopologyValidator(SurgeryValidator):
    """Validates graph topology integrity."""
    
    def validate(self, model: nn.Module, operation: str, **kwargs) -> ValidationResult:
        """Ensure valid computation graph."""
        # Check for cycles
        # Verify all paths connected
        # No orphaned modules

class SurgeryValidatorChain:
    """Runs all validators in sequence."""
    
    def __init__(self):
        self.validators = [
            DimensionValidator(),
            GradientFlowValidator(),
            MemoryValidator(),
            TopologyValidator()
        ]
    
    def validate_all(self, model: nn.Module, operation: str, **kwargs) -> List[ValidationResult]:
        """Run all validators."""
        results = []
        for validator in self.validators:
            result = validator.validate(model, operation, **kwargs)
            results.append(result)
            if not result.is_valid:
                break  # Stop on first failure
        return results
```

### Phase 4.2: Surgery Operations (Days 3-4)

#### 2.1 Layer Addition Implementation

```python
def add_layer(self, model: nn.Module, after_layer: str, new_layer: nn.Module, 
              layer_name: str, connection_type: str = "sequential") -> SurgeryResult:
    """Insert a new layer after specified layer."""
    
    # 1. Create checkpoint for rollback
    checkpoint = {
        'state_dict': copy.deepcopy(model.state_dict()),
        'architecture': self._capture_architecture(model)
    }
    
    try:
        # 2. Find insertion point
        parent_module, target_name = self._find_layer_parent(model, after_layer)
        if parent_module is None:
            raise ValueError(f"Layer {after_layer} not found")
        
        # 3. Get the module after which to insert
        after_module = getattr(parent_module, target_name)
        
        # 4. Determine dimensions
        dummy_input = self._create_dummy_input_for_layer(model, after_layer)
        with torch.no_grad():
            after_output = after_module(dummy_input)
        
        # 5. Validate new layer compatibility
        try:
            new_output = new_layer(after_output)
        except Exception as e:
            raise ValueError(f"New layer incompatible with dimensions: {e}")
        
        # 6. Insert the layer
        if isinstance(parent_module, nn.Sequential):
            # Handle Sequential container
            modules = list(parent_module.children())
            insert_idx = next(i for i, m in enumerate(modules) if m is after_module) + 1
            
            new_modules = modules[:insert_idx] + [new_layer] + modules[insert_idx:]
            new_sequential = nn.Sequential(*new_modules)
            
            # Replace in parent
            setattr(parent_module._modules, target_name, new_sequential)
            
        else:
            # Handle custom forward() - need wrapper
            wrapper = LayerWrapper(after_module, new_layer, connection_type)
            setattr(parent_module, target_name, wrapper)
        
        # 7. Update any stored layer references
        if hasattr(model, 'kasmina_layers'):
            self._update_kasmina_references(model, layer_name, new_layer)
        
        # 8. Validate the surgery
        validation_results = self.validators.validate_all(model, "add_layer", 
                                                         layer_name=layer_name)
        
        if not all(r.is_valid for r in validation_results):
            raise ValueError(f"Validation failed: {validation_results}")
        
        # 9. Record success
        result = SurgeryResult(
            success=True,
            operation="add_layer",
            details={
                'after_layer': after_layer,
                'layer_name': layer_name,
                'layer_type': type(new_layer).__name__,
                'connection_type': connection_type
            }
        )
        
    except Exception as e:
        # Rollback on any failure
        model.load_state_dict(checkpoint['state_dict'])
        result = SurgeryResult(
            success=False,
            operation="add_layer",
            details={'after_layer': after_layer, 'layer_name': layer_name},
            error=str(e),
            rollback_state=checkpoint
        )
    
    self.operation_history.append(result)
    return result
```

#### 2.2 Layer Removal Implementation

```python
def remove_layer(self, model: nn.Module, layer_name: str, 
                 bridge_strategy: str = "direct") -> SurgeryResult:
    """Remove a layer and bridge connections."""
    
    checkpoint = {
        'state_dict': copy.deepcopy(model.state_dict()),
        'architecture': self._capture_architecture(model)
    }
    
    try:
        # 1. Find layer to remove
        parent_module, target_name = self._find_layer_parent(model, layer_name)
        if parent_module is None:
            raise ValueError(f"Layer {layer_name} not found")
        
        layer_to_remove = getattr(parent_module, target_name)
        
        # 2. Find what feeds into and out of this layer
        prev_layer, next_layer = self._find_connected_layers(model, layer_name)
        
        # 3. Determine bridging strategy
        if bridge_strategy == "direct":
            # Direct connection - dimensions must match
            dummy_input = self._create_dummy_input_for_layer(model, prev_layer)
            with torch.no_grad():
                prev_output = self._get_layer_output(model, prev_layer, dummy_input)
                removed_output = layer_to_remove(prev_output)
            
            # Check dimension compatibility
            if prev_output.shape != removed_output.shape:
                # Need adapter
                adapter = self._create_dimension_adapter(prev_output.shape, 
                                                       removed_output.shape)
                setattr(parent_module, target_name, adapter)
            else:
                # Can remove directly
                if isinstance(parent_module, nn.Sequential):
                    # Remove from sequential
                    modules = [m for m in parent_module.children() 
                              if m is not layer_to_remove]
                    new_sequential = nn.Sequential(*modules)
                    # This is tricky - need parent of parent
                    self._replace_module_in_parent(model, parent_module, new_sequential)
                else:
                    # Replace with identity
                    setattr(parent_module, target_name, nn.Identity())
        
        elif bridge_strategy == "learned":
            # Replace with learned adapter
            adapter = self._create_learned_bridge(prev_output.shape, 
                                                removed_output.shape)
            setattr(parent_module, target_name, adapter)
        
        # 4. Update references
        if hasattr(model, 'kasmina_layers'):
            self._remove_kasmina_reference(model, layer_name)
        
        # 5. Validate
        validation_results = self.validators.validate_all(model, "remove_layer",
                                                         layer_name=layer_name)
        
        if not all(r.is_valid for r in validation_results):
            raise ValueError(f"Validation failed: {validation_results}")
        
        result = SurgeryResult(
            success=True,
            operation="remove_layer",
            details={
                'layer_name': layer_name,
                'bridge_strategy': bridge_strategy,
                'adapter_added': bridge_strategy != "direct" or 
                                prev_output.shape != removed_output.shape
            }
        )
        
    except Exception as e:
        model.load_state_dict(checkpoint['state_dict'])
        result = SurgeryResult(
            success=False,
            operation="remove_layer",
            details={'layer_name': layer_name},
            error=str(e),
            rollback_state=checkpoint
        )
    
    self.operation_history.append(result)
    return result
```

### Phase 4.3: Integration with Tolaria (Days 5-6)

#### 3.1 Update Trainer Integration

**Update: `src/esper/services/tolaria/trainer.py`**
```python
from esper.core.model_surgeon import ModelSurgeon
from esper.core.surgery_validators import SurgeryValidatorChain

async def _apply_architecture_modification(self, model: nn.Module, 
                                         decision: AdaptationDecision):
    """Apply architecture modification based on adaptation decision."""
    
    # Initialize surgeon
    surgeon = ModelSurgeon()
    
    # Log the decision
    logger.info(f"Applying architecture modification: {decision.adaptation_type} "
                f"to layer {decision.layer_name}")
    
    # Map decision to surgery operation
    if decision.adaptation_type == "add_neurons":
        # Add neurons/channels to existing layer
        result = surgeon.modify_layer_width(
            model=model,
            layer_name=decision.layer_name,
            new_width=decision.parameters.get('new_width', 
                                            self._get_current_width(model, decision.layer_name) * 2)
        )
        
    elif decision.adaptation_type == "add_layer":
        # Add new layer after target
        new_layer = self._create_layer_from_spec(decision.parameters['layer_spec'])
        result = surgeon.add_layer(
            model=model,
            after_layer=decision.layer_name,
            new_layer=new_layer,
            layer_name=f"{decision.layer_name}_expansion",
            connection_type=decision.parameters.get('connection_type', 'sequential')
        )
        
    elif decision.adaptation_type == "remove_layer":
        # Remove underperforming layer
        result = surgeon.remove_layer(
            model=model,
            layer_name=decision.layer_name,
            bridge_strategy=decision.parameters.get('bridge_strategy', 'direct')
        )
        
    elif decision.adaptation_type == "add_skip_connection":
        # Add residual connection
        result = surgeon.add_skip_connection(
            model=model,
            from_layer=decision.parameters['from_layer'],
            to_layer=decision.layer_name,
            connection_type=decision.parameters.get('connection_type', 'add')
        )
        
    else:
        logger.warning(f"Unknown adaptation type: {decision.adaptation_type}")
        return
    
    # Check result
    if result.success:
        logger.info(f"Architecture modification successful: {result.details}")
        
        # Update metrics
        self.metrics['architecture_modifications'] += 1
        self.metrics['last_modification_epoch'] = self.current_epoch
        
        # Notify other components
        await self._publish_architecture_change(model, result)
        
    else:
        logger.error(f"Architecture modification failed: {result.error}")
        self.metrics['failed_modifications'] += 1
        
        # Optionally trigger alternative strategy
        if self.config.get('retry_failed_surgeries', False):
            await self._handle_surgery_failure(model, decision, result)

def _create_layer_from_spec(self, spec: Dict) -> nn.Module:
    """Create a layer from specification."""
    layer_type = spec['type']
    
    if layer_type == 'linear':
        return nn.Linear(spec['in_features'], spec['out_features'])
    elif layer_type == 'conv2d':
        return nn.Conv2d(
            spec['in_channels'], 
            spec['out_channels'],
            kernel_size=spec.get('kernel_size', 3),
            padding=spec.get('padding', 1)
        )
    elif layer_type == 'attention':
        return nn.MultiheadAttention(
            spec['embed_dim'],
            spec.get('num_heads', 8)
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
```

### Phase 4.4: Testing Strategy (Day 7)

#### 4.1 Unit Tests

**File: `tests/core/test_model_surgery.py`**
```python
import pytest
import torch
import torch.nn as nn
from esper.core.model_surgeon import ModelSurgeon
from esper.core.surgery_validators import SurgeryValidatorChain

class TestModelSurgeon:
    
    def test_add_layer_to_sequential(self):
        """Test adding layer to Sequential model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        surgeon = ModelSurgeon()
        new_layer = nn.Linear(20, 20)
        
        result = surgeon.add_layer(
            model=model,
            after_layer='0',  # After first Linear
            new_layer=new_layer,
            layer_name='inserted_layer'
        )
        
        assert result.success
        assert len(list(model.children())) == 4  # Original 3 + 1
        
        # Test forward pass
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 10)
    
    def test_remove_layer_with_bridging(self):
        """Test removing layer that requires bridging."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 30),  # This changes dimensions
            nn.Linear(30, 10)
        )
        
        surgeon = ModelSurgeon()
        result = surgeon.remove_layer(
            model=model,
            layer_name='1',  # Remove middle layer
            bridge_strategy='learned'
        )
        
        assert result.success
        assert result.details['adapter_added']
        
        # Should still work
        x = torch.randn(5, 10)
        output = model(x)
        assert output.shape == (5, 10)
    
    def test_modify_layer_width(self):
        """Test changing layer width."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        surgeon = ModelSurgeon()
        result = surgeon.modify_layer_width(
            model=model,
            layer_name='0',
            new_width=40
        )
        
        assert result.success
        assert model[0].out_features == 40
        # Next layer should be adapted too
        assert model[2].in_features == 40
    
    def test_add_skip_connection(self):
        """Test adding residual connection."""
        model = CustomModel()  # Model with named layers
        
        surgeon = ModelSurgeon()
        result = surgeon.add_skip_connection(
            model=model,
            from_layer='conv1',
            to_layer='conv3'
        )
        
        assert result.success
        # Verify skip connection works
        x = torch.randn(1, 3, 32, 32)
        output = model(x)
        assert output is not None
    
    def test_validation_prevents_invalid_surgery(self):
        """Test that validation catches issues."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 5)
        )
        
        surgeon = ModelSurgeon()
        # Try to add incompatible layer
        bad_layer = nn.Linear(30, 40)  # Wrong dimensions
        
        result = surgeon.add_layer(
            model=model,
            after_layer='0',
            new_layer=bad_layer,
            layer_name='bad_layer'
        )
        
        assert not result.success
        assert 'incompatible' in result.error.lower()
    
    def test_rollback_on_failure(self):
        """Test rollback preserves original model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Linear(20, 10)
        )
        
        original_state = model.state_dict()
        
        surgeon = ModelSurgeon()
        # Attempt invalid operation
        result = surgeon.remove_layer(
            model=model,
            layer_name='nonexistent'
        )
        
        assert not result.success
        # Model should be unchanged
        for k, v in model.state_dict().items():
            assert torch.equal(v, original_state[k])
```

#### 4.2 Integration Tests

**File: `tests/integration/test_morphogenetic_surgery.py`**
```python
@pytest.mark.asyncio
async def test_full_morphogenetic_surgery_pipeline():
    """Test complete pipeline from decision to modification."""
    
    # Create morphable model
    base_model = SimpleConvNet()
    morphable = wrap(base_model)
    
    # Create adaptation decision
    decision = AdaptationDecision(
        decision_id="test_001",
        layer_name="conv1",
        adaptation_type="add_layer",
        confidence=0.85,
        parameters={
            'layer_spec': {
                'type': 'conv2d',
                'in_channels': 64,
                'out_channels': 64,
                'kernel_size': 3,
                'padding': 1
            },
            'connection_type': 'sequential'
        },
        reasoning="Layer showing high activation variance"
    )
    
    # Initialize trainer
    config = TolariaConfig(...)
    trainer = TolariaTrainer(config)
    
    # Apply modification
    await trainer._apply_architecture_modification(morphable, decision)
    
    # Verify model still works
    x = torch.randn(1, 3, 32, 32)
    output = morphable(x)
    assert output.shape == (1, 10)  # Original output shape preserved
    
    # Verify layer was added
    assert hasattr(morphable.wrapped_model, 'conv1_expansion')
```

### Phase 4.5: Safety and Constraints (Throughout)

#### 5.1 Surgery Constraints

```python
@dataclass
class SurgeryConstraints:
    """Constraints for model surgery operations."""
    
    # Size limits
    max_layers_to_add: int = 5
    max_layers_to_remove: int = 3
    max_width_multiplier: float = 2.0
    min_width_divisor: float = 2.0
    
    # Frequency limits
    min_epochs_between_surgeries: int = 10
    max_surgeries_per_epoch: int = 2
    
    # Resource limits
    max_parameter_increase: float = 1.5  # 50% increase max
    max_memory_increase_mb: float = 500
    
    # Topology limits
    max_skip_connections: int = 10
    max_graph_depth: int = 100
    
    # Recovery
    required_validation_score: float = 0.9
    rollback_on_performance_drop: float = 0.1
```

#### 5.2 Monitoring and Metrics

```python
@dataclass 
class SurgeryMetrics:
    """Track surgery operations and outcomes."""
    
    total_surgeries: int = 0
    successful_surgeries: int = 0
    failed_surgeries: int = 0
    rollback_count: int = 0
    
    surgeries_by_type: Dict[str, int] = field(default_factory=dict)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    
    average_surgery_time_ms: float = 0.0
    peak_memory_usage_mb: float = 0.0
    
    def record_surgery(self, result: SurgeryResult, performance_delta: float):
        """Record surgery outcome and impact."""
        self.total_surgeries += 1
        
        if result.success:
            self.successful_surgeries += 1
            self.surgeries_by_type[result.operation] += 1
            self.performance_impact[result.operation] = performance_delta
        else:
            self.failed_surgeries += 1
            if result.rollback_state is not None:
                self.rollback_count += 1
```

## Implementation Timeline

### Week 1 (Days 1-5)
- **Day 1-2**: Core surgery framework (ModelSurgeon, validators)
- **Day 3-4**: Surgery operations (add, remove, modify, skip)
- **Day 5**: Integration with Tolaria trainer

### Week 2 (Days 6-8) 
- **Day 6**: Advanced features (learned bridges, dimension adapters)
- **Day 7**: Comprehensive testing
- **Day 8**: Performance optimization and documentation

## Success Criteria

1. **Functionality** ✓
   - Can add layers without disrupting training
   - Can remove layers with proper bridging
   - Can modify layer widths dynamically
   - Can add skip connections
   - Rollback works 100% of the time

2. **Performance** ✓
   - Surgery operations < 500ms
   - No memory leaks
   - < 5% overhead on training speed
   - Gradient flow preserved

3. **Safety** ✓
   - All surgeries validated before application
   - Automatic rollback on failure
   - Constraints prevent destructive changes
   - Comprehensive error handling

4. **Integration** ✓
   - Seamless with existing morphogenetic system
   - Works with KasminaLayers
   - Compatible with async execution
   - Preserves seed functionality

## Risk Mitigation

1. **Start Simple**: Begin with width-only modifications
2. **Extensive Validation**: Every operation thoroughly checked
3. **Gradual Rollout**: Feature flags for production
4. **Comprehensive Monitoring**: Track all surgeries and impacts
5. **Conservative Defaults**: Tight constraints initially

## Dependencies

- PyTorch 2.0+ (for better graph manipulation)
- All Phase B1-B3 components working
- Tamiyo generating appropriate decisions
- Stable training pipeline

## Next Steps After B4

Phase B5 (Infrastructure Hardening) will:
- Add persistent caching for modified architectures
- Enable distributed surgery coordination
- Implement production monitoring
- Add surgery replay capabilities

---

This plan provides a complete implementation path for dynamic architecture modification, the key capability that enables true morphogenetic neural network evolution.