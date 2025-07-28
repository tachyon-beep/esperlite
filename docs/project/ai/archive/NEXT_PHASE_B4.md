# Phase B4: Dynamic Architecture Modification - Action Plan

## Objective
Enable runtime model surgery capabilities to allow neural networks to modify their own architecture during training.

## Current State
```python
# In src/esper/services/tolaria/trainer.py
async def _apply_architecture_modification(self, model: nn.Module, decision: AdaptationDecision):
    """Apply architecture modification based on adaptation decision."""
    raise NotImplementedError("Architecture modification not yet implemented")
```

## Implementation Plan

### 1. Model Surgery Framework (Week 1, Days 1-3)

#### Create `src/esper/core/model_surgeon.py`
```python
class ModelSurgeon:
    """Performs safe runtime modifications to PyTorch models."""
    
    def add_layer(self, model: nn.Module, position: str, layer: nn.Module) -> bool:
        """Insert a new layer at specified position."""
        
    def remove_layer(self, model: nn.Module, layer_name: str) -> bool:
        """Remove a layer and reconnect the graph."""
        
    def modify_layer(self, model: nn.Module, layer_name: str, new_config: Dict) -> bool:
        """Modify layer parameters (channels, kernel size, etc)."""
        
    def add_skip_connection(self, model: nn.Module, from_layer: str, to_layer: str) -> bool:
        """Add a residual/skip connection between layers."""
```

#### Create `src/esper/core/surgery_validators.py`
```python
class SurgeryValidator:
    """Validates model surgery operations."""
    
    def validate_graph_integrity(self, model: nn.Module) -> ValidationResult:
        """Ensure computational graph is valid after surgery."""
        
    def validate_dimension_compatibility(self, model: nn.Module) -> ValidationResult:
        """Check tensor dimensions flow correctly."""
        
    def validate_gradient_flow(self, model: nn.Module) -> ValidationResult:
        """Ensure gradients can backpropagate."""
```

### 2. Surgery Operations (Days 4-5)

#### Layer Insertion
- Identify insertion point in module hierarchy
- Initialize new layer with appropriate dimensions
- Reconnect forward() methods
- Handle state_dict updates

#### Layer Removal
- Find layer in module hierarchy
- Bridge connections around removed layer
- Clean up references
- Adjust subsequent layer dimensions if needed

#### Connection Modification
- Add skip connections using ModuleDict
- Modify forward() to handle multiple paths
- Ensure dimension matching (use 1x1 convs if needed)

### 3. Integration (Days 6-7)

#### Update `trainer.py`
```python
async def _apply_architecture_modification(self, model: nn.Module, decision: AdaptationDecision):
    """Apply architecture modification based on adaptation decision."""
    
    surgeon = ModelSurgeon()
    validator = SurgeryValidator()
    
    # Create rollback checkpoint
    checkpoint = model.state_dict()
    
    try:
        # Perform surgery based on decision type
        if decision.adaptation_type == "add_layer":
            success = surgeon.add_layer(model, decision.position, decision.layer_spec)
        elif decision.adaptation_type == "remove_layer":
            success = surgeon.remove_layer(model, decision.layer_name)
        # ... other operations
        
        # Validate the modification
        validation = validator.validate_graph_integrity(model)
        if not validation.is_valid:
            raise SurgeryError(validation.errors)
            
        # Test forward pass
        dummy_input = torch.randn(1, *self.input_shape)
        _ = model(dummy_input)
        
        logger.info(f"Architecture modification successful: {decision}")
        return True
        
    except Exception as e:
        # Rollback on failure
        model.load_state_dict(checkpoint)
        logger.error(f"Architecture modification failed, rolled back: {e}")
        return False
```

### 4. Testing Strategy

#### Unit Tests (`tests/core/test_model_surgery.py`)
- Test each surgery operation independently
- Verify dimension handling
- Check gradient flow preservation
- Test rollback mechanisms

#### Integration Tests
- Full pipeline from decision to modification
- Multi-layer surgery operations
- Performance impact measurements
- Training stability after surgery

### 5. Safety Considerations

1. **Gradual Rollout**
   - Start with simple operations (layer width changes)
   - Progress to complex operations (topology changes)

2. **Constraints**
   - Maximum layers to add/remove per epoch
   - Minimum layer sizes
   - Required cooldown between surgeries

3. **Monitoring**
   - Track success/failure rates
   - Measure performance impact
   - Log all modifications for debugging

## Success Criteria

1. **Functionality**
   - [ ] Can add layers without disrupting training
   - [ ] Can remove layers with proper connection bridging
   - [ ] Can modify layer parameters dynamically
   - [ ] Rollback works on any failure

2. **Performance**
   - [ ] Surgery operations complete in < 1 second
   - [ ] No memory leaks from modifications
   - [ ] Training convergence maintained

3. **Reliability**
   - [ ] 100% rollback success rate
   - [ ] No gradient flow disruption
   - [ ] Comprehensive error handling

## Example Use Cases

### 1. Progressive Network Growth
```python
# Start with small network
model = SimpleNet(layers=3)

# As training progresses, add complexity
if performance_plateaued:
    surgeon.add_layer(model, position="after_conv2", 
                     layer=nn.Conv2d(64, 128, 3, padding=1))
```

### 2. Adaptive Pruning
```python
# Remove underperforming layers
if layer_contribution < threshold:
    surgeon.remove_layer(model, layer_name="conv3")
```

### 3. Dynamic Skip Connections
```python
# Add residual connections to improve gradient flow
if gradient_vanishing:
    surgeon.add_skip_connection(model, from_layer="conv1", to_layer="conv4")
```

## Dependencies

- PyTorch 2.0+ (for torch.fx graph manipulation)
- Current Phase B1-B3 infrastructure
- Tamiyo decision system for surgery triggers

## Risk Mitigation

1. **Start Simple**: Begin with width-only modifications
2. **Extensive Testing**: Each operation heavily tested
3. **Gradual Deployment**: Feature flags for production
4. **Monitoring**: Comprehensive logging and metrics

---

Ready to begin Phase B4 implementation!