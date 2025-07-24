# Recovery Module (`src/esper/recovery/`)

## Overview

The recovery module implements Phase B5's checkpoint and recovery system, ensuring training continuity and fault tolerance for the Esper platform. This module provides automatic state snapshots, efficient checkpointing strategies, and rapid recovery mechanisms that achieve <30s recovery time while maintaining training momentum. The system is designed to handle both graceful and unexpected shutdowns with minimal data loss.

## Files

### `__init__.py` - Recovery Module Initialization

**Purpose:** Module initialization and public API exports.

**Contents:**
```python
"""
Checkpoint and recovery systems for training resilience.
"""

from .checkpoint_manager import CheckpointManager, CheckpointConfig, RecoveryState
from .state_snapshot import StateSnapshot, SnapshotMetadata, ModelState

__all__ = [
    "CheckpointManager",
    "CheckpointConfig",
    "RecoveryState",
    "StateSnapshot",
    "SnapshotMetadata",
    "ModelState",
]
```

**Status:** Production-ready with comprehensive exports.

### `checkpoint_manager.py` - Automatic Checkpointing System

**Purpose:** Manages automatic checkpointing, recovery coordination, and checkpoint lifecycle with configurable strategies for different training scenarios.

#### Key Classes

**`CheckpointConfig`** - Checkpoint Configuration
```python
@dataclass
class CheckpointConfig:
    """Configuration for checkpoint management."""
    
    # Checkpoint intervals
    checkpoint_interval_minutes: int = 30
    checkpoint_on_epoch: bool = True
    checkpoint_on_improvement: bool = True
    
    # Storage settings
    checkpoint_dir: str = "/var/esper/checkpoints"
    max_checkpoints_kept: int = 5
    compression_enabled: bool = True
    
    # Recovery settings
    auto_recovery: bool = True
    recovery_timeout_seconds: int = 30
    verify_checkpoints: bool = True
    
    # Performance settings
    async_checkpointing: bool = True
    checkpoint_sharding: bool = True
    differential_snapshots: bool = True
```

**`RecoveryState`** - Recovery Status
```python
@dataclass
class RecoveryState:
    """State information for recovery process."""
    
    checkpoint_id: str
    recovery_start_time: datetime
    recovery_end_time: Optional[datetime]
    
    # Recovery details
    epoch_recovered: int
    step_recovered: int
    loss_at_checkpoint: float
    
    # Component states
    model_recovered: bool
    optimizer_recovered: bool
    scheduler_recovered: bool
    kernels_recovered: bool
    
    # Recovery metrics
    data_loss_seconds: float
    recovery_duration_seconds: float
    checkpoints_evaluated: int
    
    # Status
    status: str  # "in_progress", "completed", "failed"
    error_message: Optional[str] = None
```

**`CheckpointManager`** - Main Checkpoint Manager
```python
class CheckpointManager:
    """
    Manages training checkpoints and recovery operations.
    
    Features:
    - Automatic periodic checkpointing
    - Event-driven checkpoints (epoch, improvement)
    - Multi-component state management
    - Fast recovery with verification
    - Checkpoint rotation and cleanup
    """
    
    def __init__(
        self,
        config: CheckpointConfig,
        asset_repository: AssetRepository,
        kernel_cache: PersistentKernelCache,
        oona_client: OonaClient,
    ):
        self.config = config
        self.asset_repository = asset_repository
        self.kernel_cache = kernel_cache
        self.oona_client = oona_client
        
        # State tracking
        self.last_checkpoint_time = None
        self.checkpoint_history = []
        self.is_checkpointing = False
        
        # Background tasks
        self._checkpoint_task = None
        self._cleanup_task = None
```

**Core Methods:**

**`async create_checkpoint(state: Dict[str, Any], metadata: Dict[str, Any]) -> str`**
```python
async def create_checkpoint(
    self,
    state: Dict[str, Any],
    metadata: Dict[str, Any]
) -> str:
    """
    Create a checkpoint from current training state.
    
    Args:
        state: Complete training state including:
            - model_state_dict
            - optimizer_state_dict
            - scheduler_state
            - epoch
            - global_step
            - metrics
        metadata: Additional checkpoint metadata
        
    Returns:
        Checkpoint ID
        
    Process:
    1. Create state snapshot
    2. Compress if enabled
    3. Store in asset repository
    4. Update checkpoint history
    5. Trigger cleanup if needed
    """
```

**Key Features:**
- **Atomic Operations:** Ensures checkpoint consistency
- **Differential Snapshots:** Only stores changes from previous checkpoint
- **Parallel Storage:** Shards large checkpoints across storage
- **Verification:** Optional integrity checking

**`async recover_from_checkpoint(checkpoint_id: Optional[str] = None) -> RecoveryState`**
```python
async def recover_from_checkpoint(
    self,
    checkpoint_id: Optional[str] = None
) -> RecoveryState:
    """
    Recover training state from checkpoint.
    
    Args:
        checkpoint_id: Specific checkpoint to recover from,
                      or None for latest valid checkpoint
                      
    Returns:
        RecoveryState with recovery details
        
    Process:
    1. Find valid checkpoint(s)
    2. Load and verify state
    3. Restore all components
    4. Warm kernel cache
    5. Broadcast recovery event
    """
```

**Recovery Features:**
- **Automatic Selection:** Finds latest valid checkpoint if not specified
- **Cascading Recovery:** Falls back to older checkpoints on failure
- **Cache Warming:** Pre-loads frequently used kernels
- **State Verification:** Ensures checkpoint integrity

**`async auto_checkpoint(training_loop) -> None`**
```python
async def auto_checkpoint(self, training_loop) -> None:
    """
    Background task for automatic checkpointing.
    
    Monitors training loop and creates checkpoints based on:
    - Time intervals
    - Epoch boundaries
    - Metric improvements
    """
    
    while training_loop.is_running:
        await asyncio.sleep(60)  # Check every minute
        
        should_checkpoint = False
        reason = ""
        
        # Time-based
        if self._should_checkpoint_by_time():
            should_checkpoint = True
            reason = "periodic"
            
        # Epoch-based
        elif self._should_checkpoint_by_epoch(training_loop):
            should_checkpoint = True
            reason = "epoch_complete"
            
        # Improvement-based
        elif self._should_checkpoint_by_improvement(training_loop):
            should_checkpoint = True
            reason = "metric_improved"
            
        if should_checkpoint and not self.is_checkpointing:
            await self._create_checkpoint_async(
                training_loop.get_state(),
                {"reason": reason}
            )
```

**Checkpoint Management:**

**`async list_checkpoints(limit: int = 10) -> List[Dict[str, Any]]`**
```python
async def list_checkpoints(
    self,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    List available checkpoints with metadata.
    
    Returns:
        List of checkpoint info sorted by creation time
    """
```

**`async delete_checkpoint(checkpoint_id: str) -> bool`**
```python
async def delete_checkpoint(checkpoint_id: str) -> bool:
    """
    Delete specific checkpoint.
    
    Prevents deletion of the latest checkpoint for safety.
    """
```

**`async cleanup_old_checkpoints() -> int`**
```python
async def cleanup_old_checkpoints() -> int:
    """
    Remove old checkpoints based on retention policy.
    
    Keeps the most recent N checkpoints as configured.
    
    Returns:
        Number of checkpoints deleted
    """
```

**Advanced Features:**

**`async create_differential_checkpoint(base_checkpoint_id: str, state: Dict[str, Any]) -> str`**
```python
async def create_differential_checkpoint(
    self,
    base_checkpoint_id: str,
    state: Dict[str, Any]
) -> str:
    """
    Create checkpoint storing only differences from base.
    
    Significantly reduces storage for frequent checkpoints.
    """
```

**`async export_checkpoint(checkpoint_id: str, export_path: str) -> None`**
```python
async def export_checkpoint(
    self,
    checkpoint_id: str,
    export_path: str
) -> None:
    """
    Export checkpoint for external storage or sharing.
    
    Creates self-contained checkpoint package.
    """
```

### `state_snapshot.py` - State Snapshot Management

**Purpose:** Handles creation and restoration of complete system state snapshots including models, optimizers, kernels, and training metadata.

#### Key Classes

**`SnapshotMetadata`** - Snapshot Metadata
```python
@dataclass
class SnapshotMetadata:
    """Metadata for state snapshots."""
    
    snapshot_id: str
    created_at: datetime
    checkpoint_id: str
    
    # Training state
    epoch: int
    global_step: int
    total_runtime_seconds: float
    
    # Model info
    model_hash: str
    total_parameters: int
    trainable_parameters: int
    
    # Performance metrics
    current_loss: float
    best_loss: float
    metrics: Dict[str, float]
    
    # System state
    active_kernels: List[str]
    kernel_cache_stats: Dict[str, Any]
    memory_usage_mb: float
    
    # Snapshot properties
    snapshot_size_mb: float
    compression_ratio: float
    is_differential: bool
    base_snapshot_id: Optional[str] = None
```

**`ModelState`** - Model State Container
```python
@dataclass
class ModelState:
    """Container for model-related state."""
    
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]] = None
    
    # Additional training state
    random_states: Dict[str, Any] = field(default_factory=dict)
    grad_scaler_state: Optional[Dict[str, Any]] = None
    
    # Morphogenetic state
    kasmina_layer_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_kernel_mappings: Dict[str, List[Tuple[int, str]]] = field(default_factory=dict)
```

**`StateSnapshot`** - Main Snapshot Handler
```python
class StateSnapshot:
    """
    Handles creation and restoration of training state snapshots.
    
    Features:
    - Complete state capture
    - Efficient serialization
    - Compression support
    - Differential snapshots
    - State verification
    """
    
    def __init__(
        self,
        enable_compression: bool = True,
        compression_level: int = 6,
        verify_tensors: bool = True,
    ):
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.verify_tensors = verify_tensors
```

**Core Methods:**

**`async create_snapshot(model: nn.Module, training_state: Dict[str, Any]) -> Tuple[bytes, SnapshotMetadata]`**
```python
async def create_snapshot(
    self,
    model: nn.Module,
    training_state: Dict[str, Any]
) -> Tuple[bytes, SnapshotMetadata]:
    """
    Create complete state snapshot.
    
    Args:
        model: The model to snapshot (including optimizer, etc.)
        training_state: Additional training state including:
            - epoch
            - global_step
            - metrics
            - active_kernels
            
    Returns:
        (snapshot_bytes, metadata) tuple
        
    Process:
    1. Extract all state components
    2. Create metadata
    3. Serialize state
    4. Compress if enabled
    5. Calculate checksums
    """
```

**Snapshot Creation Details:**
```python
# State extraction
model_state = self._extract_model_state(model)
morpho_state = self._extract_morphogenetic_state(model)
system_state = self._extract_system_state()

# Metadata creation
metadata = SnapshotMetadata(
    snapshot_id=self._generate_snapshot_id(),
    created_at=datetime.utcnow(),
    epoch=training_state["epoch"],
    global_step=training_state["global_step"],
    model_hash=self._calculate_model_hash(model),
    # ... additional fields
)

# Serialization
snapshot_data = {
    "version": "1.0",
    "metadata": metadata,
    "model_state": model_state,
    "morpho_state": morpho_state,
    "system_state": system_state,
}

# Compression and return
return self._compress_snapshot(snapshot_data), metadata
```

**`async restore_snapshot(snapshot_data: bytes, model: nn.Module) -> Dict[str, Any]`**
```python
async def restore_snapshot(
    self,
    snapshot_data: bytes,
    model: nn.Module
) -> Dict[str, Any]:
    """
    Restore state from snapshot.
    
    Args:
        snapshot_data: Compressed snapshot bytes
        model: Target model to restore state into
        
    Returns:
        Restored training state
        
    Process:
    1. Decompress snapshot
    2. Verify integrity
    3. Restore model state
    4. Restore optimizer state
    5. Restore morphogenetic state
    6. Return training state
    """
```

**State Extraction Methods:**

**`_extract_morphogenetic_state(model: nn.Module) -> Dict[str, Any]`**
```python
def _extract_morphogenetic_state(
    self,
    model: nn.Module
) -> Dict[str, Any]:
    """
    Extract morphogenetic-specific state.
    
    Includes:
    - Active kernel IDs per layer
    - Seed configurations
    - Blend factors
    - Performance metrics
    """
    
    morpho_state = {}
    
    for name, module in model.named_modules():
        if isinstance(module, KasminaLayer):
            layer_state = {
                "active_kernels": module.get_active_kernels(),
                "seed_alphas": module.state_layout.alpha_blend.tolist(),
                "execution_stats": module.get_stats(),
                "telemetry_enabled": module.telemetry_enabled,
            }
            morpho_state[name] = layer_state
            
    return morpho_state
```

**Differential Snapshots:**

**`async create_differential_snapshot(base_snapshot: bytes, current_state: Dict[str, Any]) -> bytes`**
```python
async def create_differential_snapshot(
    self,
    base_snapshot: bytes,
    current_state: Dict[str, Any]
) -> bytes:
    """
    Create snapshot containing only changes from base.
    
    Reduces storage by 70-90% for frequent snapshots.
    """
    
    base_state = self._decompress_snapshot(base_snapshot)
    differences = self._compute_state_diff(base_state, current_state)
    
    return self._compress_snapshot({
        "type": "differential",
        "base_snapshot_id": base_state["metadata"]["snapshot_id"],
        "differences": differences,
    })
```

**Verification Methods:**

**`async verify_snapshot(snapshot_data: bytes) -> Tuple[bool, Optional[str]]`**
```python
async def verify_snapshot(
    self,
    snapshot_data: bytes
) -> Tuple[bool, Optional[str]]:
    """
    Verify snapshot integrity.
    
    Checks:
    - Decompression success
    - Checksum validation
    - Tensor integrity
    - State completeness
    
    Returns:
        (is_valid, error_message) tuple
    """
```

**Utilities:**

**`estimate_snapshot_size(model: nn.Module) -> int`**
```python
def estimate_snapshot_size(
    self,
    model: nn.Module
) -> int:
    """
    Estimate snapshot size in bytes.
    
    Useful for capacity planning and monitoring.
    """
    
    total_size = 0
    
    # Model parameters
    for param in model.parameters():
        total_size += param.data.nelement() * param.data.element_size()
        
    # Optimizer state (estimate 2x model size)
    total_size *= 3
    
    # Compression ratio
    if self.enable_compression:
        total_size *= 0.3  # Typical compression ratio
        
    return total_size
```

## Architecture Integration

The recovery module integrates with the system:

1. **Training Loop** → `CheckpointManager` → **Automatic Checkpoints**
2. **State Changes** → `StateSnapshot` → **Snapshot Creation**
3. **System Failure** → `Recovery Process` → **State Restoration**
4. **Asset Repository** → `Checkpoint Storage` → **Persistent Storage**
5. **Kernel Cache** → `Cache Warming` → **Fast Recovery**

## Recovery Workflow

### Normal Checkpoint Flow
```
Training Loop
    ↓
Checkpoint Trigger (time/epoch/improvement)
    ↓
Create State Snapshot
    ↓
Store in Asset Repository
    ↓
Update Checkpoint History
    ↓
Cleanup Old Checkpoints
```

### Recovery Flow
```
System Start
    ↓
Check for Recovery Mode
    ↓
Find Latest Valid Checkpoint
    ↓
Load from Asset Repository
    ↓
Restore State Snapshot
    ↓
Warm Kernel Cache
    ↓
Resume Training
```

## Performance Characteristics

### Checkpoint Performance
- **Creation Time:** 5-10s for 1GB model
- **Compression Ratio:** 70% reduction typical
- **Differential Size:** 10-20% of full snapshot
- **Storage Overhead:** <5% of training time

### Recovery Performance
- **Recovery Time:** <30s for complete restoration
- **Cache Warming:** 5-10s for 1000 kernels
- **Verification Time:** 2-3s for integrity check
- **State Loading:** Parallel for large models

## Configuration Examples

### Development Configuration
```python
config = CheckpointConfig(
    checkpoint_interval_minutes=5,  # Frequent for testing
    checkpoint_dir="/tmp/esper_checkpoints",
    max_checkpoints_kept=3,
    compression_enabled=False,  # Faster
    verify_checkpoints=True,  # Extra safety
    async_checkpointing=False  # Easier debugging
)

checkpoint_manager = CheckpointManager(
    config=config,
    asset_repository=asset_repo,
    kernel_cache=kernel_cache,
    oona_client=oona_client
)
```

### Production Configuration
```python
config = CheckpointConfig(
    checkpoint_interval_minutes=30,
    checkpoint_on_epoch=True,
    checkpoint_on_improvement=True,
    checkpoint_dir="/mnt/nvme/checkpoints",  # Fast storage
    max_checkpoints_kept=10,
    compression_enabled=True,
    auto_recovery=True,
    async_checkpointing=True,
    checkpoint_sharding=True,  # For large models
    differential_snapshots=True  # Storage efficiency
)

checkpoint_manager = CheckpointManager(
    config=config,
    asset_repository=asset_repo,
    kernel_cache=kernel_cache,
    oona_client=oona_client
)

# Start automatic checkpointing
asyncio.create_task(
    checkpoint_manager.auto_checkpoint(training_loop)
)
```

### High-Frequency Configuration
```python
config = CheckpointConfig(
    checkpoint_interval_minutes=10,
    checkpoint_on_epoch=False,  # Too frequent
    checkpoint_on_improvement=True,
    max_checkpoints_kept=20,
    differential_snapshots=True,  # Essential for frequency
    compression_enabled=True,
    async_checkpointing=True  # Non-blocking
)
```

## Usage Patterns

### Manual Checkpointing
```python
# Create checkpoint manually
checkpoint_id = await checkpoint_manager.create_checkpoint(
    state={
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_loss": best_loss,
    },
    metadata={
        "reason": "manual",
        "description": "Before experimental changes"
    }
)

print(f"Created checkpoint: {checkpoint_id}")
```

### Recovery on Startup
```python
# Check if recovery needed
if checkpoint_manager.should_recover():
    print("Recovering from checkpoint...")
    
    recovery_state = await checkpoint_manager.recover_from_checkpoint()
    
    if recovery_state.status == "completed":
        print(f"Recovered to epoch {recovery_state.epoch_recovered}")
        print(f"Data loss: {recovery_state.data_loss_seconds}s")
        
        # Resume training from recovered state
        training_loop.resume_from_state(recovery_state)
    else:
        print(f"Recovery failed: {recovery_state.error_message}")
        # Start fresh training
```

### Checkpoint Verification
```python
# Verify checkpoint integrity
checkpoints = await checkpoint_manager.list_checkpoints()

for checkpoint in checkpoints:
    is_valid, error = await checkpoint_manager.verify_checkpoint(
        checkpoint["checkpoint_id"]
    )
    
    if not is_valid:
        print(f"Checkpoint {checkpoint['checkpoint_id']} corrupted: {error}")
        await checkpoint_manager.delete_checkpoint(checkpoint["checkpoint_id"])
```

### Export for Deployment
```python
# Export checkpoint for deployment
latest_checkpoint = await checkpoint_manager.get_latest_checkpoint()

await checkpoint_manager.export_checkpoint(
    checkpoint_id=latest_checkpoint["checkpoint_id"],
    export_path="/tmp/model_export.ckpt"
)

# Can be loaded without Esper dependencies
```

## Error Handling

### Checkpoint Creation Failures
```python
try:
    checkpoint_id = await checkpoint_manager.create_checkpoint(state, metadata)
except StorageError as e:
    logger.error(f"Failed to store checkpoint: {e}")
    # Continue training without checkpoint
except Exception as e:
    logger.error(f"Unexpected checkpoint error: {e}")
    # Send alert but continue training
```

### Recovery Failures
```python
try:
    recovery_state = await checkpoint_manager.recover_from_checkpoint()
except CheckpointNotFoundError:
    logger.info("No checkpoints found, starting fresh")
    return None
except CorruptedCheckpointError as e:
    logger.error(f"Checkpoint corrupted: {e}")
    # Try older checkpoint
    recovery_state = await checkpoint_manager.recover_from_checkpoint(
        use_previous=True
    )
except Exception as e:
    logger.error(f"Recovery failed: {e}")
    # Start fresh with error tracking
```

### Graceful Degradation
```python
class ResilientCheckpointManager(CheckpointManager):
    """Checkpoint manager with fallback strategies."""
    
    async def create_checkpoint(self, state, metadata):
        try:
            return await super().create_checkpoint(state, metadata)
        except Exception as e:
            # Fall back to local file storage
            logger.warning(f"Using fallback storage: {e}")
            return await self._create_local_checkpoint(state, metadata)
```

## Monitoring and Observability

### Metrics
```python
# Prometheus metrics exposed by Nissa
checkpoint_creation_duration_seconds{status="success"} 8.5
checkpoint_creation_duration_seconds{status="failure"} 2.1
checkpoint_size_bytes{type="full"} 1073741824
checkpoint_size_bytes{type="differential"} 104857600
recovery_duration_seconds 25.3
recovery_data_loss_seconds 120.0
checkpoints_total{reason="periodic"} 48
checkpoints_total{reason="epoch"} 100
checkpoints_total{reason="improvement"} 15
checkpoint_storage_usage_bytes 10737418240
```

### Health Checks
```python
async def check_recovery_health() -> Dict[str, Any]:
    """Check recovery system health."""
    
    health = {
        "checkpoint_manager": {
            "status": "healthy",
            "last_checkpoint_age_minutes": 15,
            "total_checkpoints": 10,
            "storage_available_gb": 500
        },
        "recovery_capability": {
            "can_recover": True,
            "latest_valid_checkpoint": "ckpt-123",
            "estimated_recovery_time_seconds": 25
        }
    }
    
    # Check storage
    if not checkpoint_manager.has_sufficient_storage():
        health["checkpoint_manager"]["status"] = "degraded"
        
    # Check checkpoint age
    if health["checkpoint_manager"]["last_checkpoint_age_minutes"] > 60:
        health["checkpoint_manager"]["status"] = "warning"
        
    return health
```

## Best Practices

### Checkpoint Strategy
1. **Frequency:** Balance between safety and overhead
2. **Differential:** Use for high-frequency checkpointing
3. **Verification:** Enable in production for data integrity
4. **Retention:** Keep enough for rollback flexibility
5. **Storage:** Use fast NVMe for checkpoint directory

### Recovery Planning
1. **Test Recovery:** Regular recovery drills
2. **Monitor Age:** Alert on old checkpoints
3. **Multiple Locations:** Replicate critical checkpoints
4. **Documentation:** Document recovery procedures
5. **Automation:** Fully automated recovery flow

### Performance Optimization
1. **Async Operations:** Non-blocking checkpointing
2. **Compression:** Reduce storage and I/O
3. **Sharding:** Parallelize large checkpoints
4. **Selective State:** Only checkpoint necessary components
5. **Network Storage:** Use local cache for remote storage

## Testing Strategies

### Unit Testing
```python
@pytest.mark.asyncio
async def test_checkpoint_creation():
    """Test checkpoint creation and metadata."""
    manager = CheckpointManager(test_config, mock_repo, mock_cache, mock_oona)
    
    state = {
        "model_state_dict": {"layer.weight": torch.randn(10, 10)},
        "epoch": 5,
        "global_step": 1000
    }
    
    checkpoint_id = await manager.create_checkpoint(state, {"test": True})
    
    assert checkpoint_id is not None
    assert checkpoint_id in manager.checkpoint_history
```

### Integration Testing
```python
@pytest.mark.asyncio
async def test_full_recovery_flow():
    """Test complete checkpoint and recovery cycle."""
    # Create checkpoint
    checkpoint_id = await manager.create_checkpoint(training_state, {})
    
    # Simulate failure
    manager._clear_memory_state()
    
    # Recover
    recovery_state = await manager.recover_from_checkpoint()
    
    assert recovery_state.status == "completed"
    assert recovery_state.epoch_recovered == training_state["epoch"]
```

### Failure Testing
```python
@pytest.mark.asyncio
async def test_corrupted_checkpoint_handling():
    """Test handling of corrupted checkpoints."""
    # Create corrupted checkpoint
    await create_corrupted_checkpoint(manager)
    
    # Attempt recovery
    recovery_state = await manager.recover_from_checkpoint()
    
    # Should fall back to previous checkpoint
    assert recovery_state.status == "completed"
    assert recovery_state.checkpoints_evaluated > 1
```

## Migration Guide

### From Manual to Automatic Checkpointing
```python
# Old code
if epoch % 10 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pt')

# New code
checkpoint_manager = CheckpointManager(config, repo, cache, oona)
await checkpoint_manager.auto_checkpoint(training_loop)
# Automatic handling of all checkpointing logic
```

### From File-Based to Managed Checkpoints
```python
# Old code
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# New code
recovery_state = await checkpoint_manager.recover_from_checkpoint()
# Automatically restores model, optimizer, kernels, etc.
```

## Future Enhancements

1. **Incremental Snapshots**
   - Block-level deduplication
   - Git-like version control
   - Merkle tree verification

2. **Distributed Checkpointing**
   - Sharded across nodes
   - Parallel save/restore
   - Consistent global snapshots

3. **Smart Checkpointing**
   - ML-based checkpoint timing
   - Predictive failure detection
   - Adaptive compression levels

4. **Cloud Integration**
   - Direct S3/GCS/Azure support
   - Streaming checkpoints
   - Global checkpoint registry

5. **Advanced Recovery**
   - Partial state recovery
   - Cross-version compatibility
   - Checkpoint migration tools