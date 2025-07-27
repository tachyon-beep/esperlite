"""Checkpoint management for morphogenetic seed states.

Provides persistence and recovery mechanisms for seed states,
enabling rollback, migration, and fault tolerance.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages seed state persistence and recovery.

    Handles saving and restoring seed states to disk, including
    version migration, corruption detection, and cleanup.
    """

    CURRENT_VERSION = 2
    CHECKPOINT_EXTENSION = '.pt'
    METADATA_EXTENSION = '.json'

    def __init__(self, checkpoint_dir: Path, max_checkpoints_per_seed: int = 5):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_checkpoints_per_seed: Maximum checkpoints to retain per seed
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints_per_seed = max_checkpoints_per_seed
        
        # Create subdirectories
        self.active_dir = self.checkpoint_dir / 'active'
        self.archive_dir = self.checkpoint_dir / 'archive'
        self.active_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Cache for metadata
        self.metadata_cache: Dict[str, Dict[str, Any]] = {}
    
    def save_checkpoint(
        self,
        layer_id: str,
        seed_id: int,
        state_data: Dict[str, Any],
        blueprint_state: Optional[Dict[str, torch.Tensor]] = None,
        priority: str = 'normal'
    ) -> str:
        """Save seed state to disk.
        
        Args:
            layer_id: Layer identifier
            seed_id: Seed identifier
            state_data: Core state information
            blueprint_state: Optional blueprint weights
            priority: Checkpoint priority ('normal', 'high')
            
        Returns:
            Checkpoint ID
        """
        # Generate checkpoint ID
        timestamp = int(time.time() * 1000)  # Millisecond precision
        checkpoint_id = f"{layer_id}_{seed_id}_{timestamp}"
        
        # Prepare checkpoint data
        checkpoint = {
            'version': self.CURRENT_VERSION,
            'checkpoint_id': checkpoint_id,
            'layer_id': layer_id,
            'seed_id': seed_id,
            'timestamp': timestamp,
            'priority': priority,
            'state_data': state_data,
            'blueprint_state': blueprint_state,
            'metadata': {
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'
            }
        }
        
        # Save checkpoint file
        checkpoint_path = self.active_dir / f"{checkpoint_id}{self.CHECKPOINT_EXTENSION}"
        try:
            torch.save(checkpoint, checkpoint_path)
            logger.info("Saved checkpoint %s to %s", checkpoint_id, checkpoint_path)
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_id, str(e))
            raise
        
        # Save metadata separately for fast queries
        metadata = {
            'checkpoint_id': checkpoint_id,
            'layer_id': layer_id,
            'seed_id': seed_id,
            'timestamp': timestamp,
            'priority': priority,
            'lifecycle_state': state_data.get('lifecycle_state', 'unknown'),
            'file_size': checkpoint_path.stat().st_size
        }
        
        metadata_path = self.active_dir / f"{checkpoint_id}{self.METADATA_EXTENSION}"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Update cache
        self.metadata_cache[checkpoint_id] = metadata
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints(layer_id, seed_id)
        
        return checkpoint_id
    
    def restore_checkpoint(
        self,
        checkpoint_id: str,
        target_device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Restore seed state from checkpoint.
        
        Args:
            checkpoint_id: Checkpoint identifier
            target_device: Device to load tensors to
            
        Returns:
            Checkpoint data dictionary
        """
        # Check active directory first
        checkpoint_path = self.active_dir / f"{checkpoint_id}{self.CHECKPOINT_EXTENSION}"
        if not checkpoint_path.exists():
            # Check archive
            checkpoint_path = self.archive_dir / f"{checkpoint_id}{self.CHECKPOINT_EXTENSION}"
            if not checkpoint_path.exists():
                raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Load checkpoint
        try:
            if target_device is None:
                target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load with weights_only=False to allow custom classes
            checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
            logger.info("Restored checkpoint %s from %s", checkpoint_id, checkpoint_path)
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_id, str(e))
            raise
        
        # Version migration if needed
        if checkpoint['version'] < self.CURRENT_VERSION:
            checkpoint = self._migrate_checkpoint(checkpoint)
        
        # Validate checkpoint integrity
        if not self._validate_checkpoint(checkpoint):
            raise ValueError(f"Checkpoint validation failed: {checkpoint_id}")
        
        return checkpoint
    
    def list_checkpoints(
        self,
        layer_id: Optional[str] = None,
        seed_id: Optional[int] = None,
        lifecycle_state: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List available checkpoints with filtering.
        
        Args:
            layer_id: Filter by layer ID
            seed_id: Filter by seed ID
            lifecycle_state: Filter by lifecycle state
            limit: Maximum results to return
            
        Returns:
            List of checkpoint metadata
        """
        # Load metadata if cache is empty
        if not self.metadata_cache:
            self._load_metadata_cache()
        
        # Filter checkpoints
        results = []
        for metadata in self.metadata_cache.values():
            if layer_id and metadata['layer_id'] != layer_id:
                continue
            if seed_id is not None and metadata['seed_id'] != seed_id:
                continue
            if lifecycle_state is not None and metadata.get('lifecycle_state') != lifecycle_state:
                continue
            
            results.append(metadata)
        
        # Sort by timestamp descending
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return results[:limit]
    
    def delete_checkpoint(self, checkpoint_id: str, archive: bool = True):
        """Delete or archive a checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to delete
            archive: Whether to archive instead of delete
        """
        # Find checkpoint in active or archive
        checkpoint_path = self.active_dir / f"{checkpoint_id}{self.CHECKPOINT_EXTENSION}"
        metadata_path = self.active_dir / f"{checkpoint_id}{self.METADATA_EXTENSION}"
        
        if not checkpoint_path.exists():
            # Check archive
            checkpoint_path = self.archive_dir / f"{checkpoint_id}{self.CHECKPOINT_EXTENSION}"
            metadata_path = self.archive_dir / f"{checkpoint_id}{self.METADATA_EXTENSION}"
            
            if not checkpoint_path.exists():
                logger.warning("Checkpoint not found: %s", checkpoint_id)
                return
        
        if archive:
            # Move to archive
            archive_checkpoint = self.archive_dir / checkpoint_path.name
            archive_metadata = self.archive_dir / metadata_path.name
            
            checkpoint_path.rename(archive_checkpoint)
            if metadata_path.exists():
                metadata_path.rename(archive_metadata)
            
            logger.info("Archived checkpoint %s", checkpoint_id)
        else:
            # Delete permanently
            checkpoint_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
            
            logger.info("Deleted checkpoint %s", checkpoint_id)
        
        # Remove from cache
        self.metadata_cache.pop(checkpoint_id, None)
    
    def get_latest_checkpoint(
        self,
        layer_id: str,
        seed_id: int
    ) -> Optional[str]:
        """Get the most recent checkpoint for a seed.
        
        Args:
            layer_id: Layer identifier
            seed_id: Seed identifier
            
        Returns:
            Checkpoint ID or None
        """
        checkpoints = self.list_checkpoints(
            layer_id=layer_id,
            seed_id=seed_id,
            limit=1
        )
        
        return checkpoints[0]['checkpoint_id'] if checkpoints else None
    
    def _cleanup_old_checkpoints(self, layer_id: str, seed_id: int):
        """Remove old checkpoints exceeding retention limit.
        
        Args:
            layer_id: Layer identifier  
            seed_id: Seed identifier
        """
        # Get all checkpoints for this seed
        checkpoints = self.list_checkpoints(
            layer_id=layer_id,
            seed_id=seed_id
        )
        
        # Keep high priority checkpoints
        normal_checkpoints = [
            cp for cp in checkpoints 
            if cp.get('priority', 'normal') == 'normal'
        ]
        
        # Archive excess checkpoints
        if len(normal_checkpoints) > self.max_checkpoints_per_seed:
            for cp in normal_checkpoints[self.max_checkpoints_per_seed:]:
                self.delete_checkpoint(cp['checkpoint_id'], archive=True)
    
    def _load_metadata_cache(self):
        """Load metadata cache from disk."""
        self.metadata_cache.clear()
        
        # Load from active directory
        for metadata_path in self.active_dir.glob(f"*{self.METADATA_EXTENSION}"):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    checkpoint_id = metadata['checkpoint_id']
                    self.metadata_cache[checkpoint_id] = metadata
            except Exception as e:
                logger.warning("Failed to load metadata %s: %s", metadata_path, str(e))
    
    def _migrate_checkpoint(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate checkpoint to current version.
        
        Args:
            checkpoint: Checkpoint data
            
        Returns:
            Migrated checkpoint
        """
        version = checkpoint.get('version', 1)
        
        if version == 1:
            # Migrate v1 -> v2
            logger.info("Migrating checkpoint from v1 to v2")
            
            # Add new fields
            checkpoint['version'] = 2
            checkpoint['priority'] = 'normal'
            
            # Update state data format
            if 'state_data' in checkpoint:
                state_data = checkpoint['state_data']
                if isinstance(state_data, dict) and 'lifecycle_state' not in state_data:
                    # Convert old format
                    state_data['lifecycle_state'] = state_data.get('state', 0)
                    state_data.pop('state', None)
        
        return checkpoint
    
    def _validate_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """Validate checkpoint integrity.
        
        Args:
            checkpoint: Checkpoint data
            
        Returns:
            True if valid
        """
        required_fields = [
            'version', 'checkpoint_id', 'layer_id', 
            'seed_id', 'timestamp', 'state_data'
        ]
        
        for field in required_fields:
            if field not in checkpoint:
                logger.error("Checkpoint missing required field: %s", field)
                return False
        
        # Validate version
        if checkpoint['version'] > self.CURRENT_VERSION:
            logger.error(
                "Checkpoint version %s is newer than supported version %s",
                checkpoint['version'], self.CURRENT_VERSION
            )
            return False
        
        # Validate state data
        state_data = checkpoint.get('state_data', {})
        if not isinstance(state_data, dict):
            logger.error("Invalid state_data format")
            return False
        
        return True


class CheckpointRecovery:
    """Handles checkpoint recovery and corruption repair.
    
    Provides mechanisms to recover from corrupted checkpoints
    and restore system state after failures.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        """Initialize recovery system.
        
        Args:
            checkpoint_manager: Checkpoint manager instance
        """
        self.checkpoint_manager = checkpoint_manager
    
    def recover_seed_state(
        self,
        layer_id: str,
        seed_id: int,
        target_device: Optional[torch.device] = None
    ) -> Optional[Dict[str, Any]]:
        """Attempt to recover the most recent valid state.
        
        Args:
            layer_id: Layer identifier
            seed_id: Seed identifier
            target_device: Device for loading
            
        Returns:
            Recovered checkpoint or None
        """
        # Get all checkpoints for this seed
        checkpoints = self.checkpoint_manager.list_checkpoints(
            layer_id=layer_id,
            seed_id=seed_id
        )
        
        # Try each checkpoint in order
        for checkpoint_meta in checkpoints:
            checkpoint_id = checkpoint_meta['checkpoint_id']
            
            try:
                checkpoint = self.checkpoint_manager.restore_checkpoint(
                    checkpoint_id, target_device
                )
                logger.info(
                    "Successfully recovered seed %s from checkpoint %s",
                    seed_id, checkpoint_id
                )
                return checkpoint
            except Exception as e:
                logger.warning(
                    "Failed to restore checkpoint %s: %s",
                    checkpoint_id, str(e)
                )
                continue
        
        logger.error("No valid checkpoints found for seed %s", seed_id)
        return None
    
    def repair_corrupted_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Dict[str, Any]]:
        """Attempt to repair a corrupted checkpoint.
        
        Args:
            checkpoint_id: Checkpoint to repair
            
        Returns:
            Repaired checkpoint or None
        """
        # This is a placeholder for more sophisticated repair logic
        # In practice, this might involve:
        # - Partial data recovery
        # - State reconstruction from telemetry
        # - Cross-validation with other checkpoints
        
        logger.warning(
            "Checkpoint repair not yet implemented for %s",
            checkpoint_id
        )
        return None
