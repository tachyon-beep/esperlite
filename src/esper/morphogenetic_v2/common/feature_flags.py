"""
Feature flag system for morphogenetic migration.

Enables gradual rollout and A/B testing of new features.
"""

import os
import json
import hashlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlag:
    """Configuration for a single feature flag."""
    name: str
    enabled: bool = False
    rollout_percentage: int = 0  # 0-100
    allowlist: list[str] = None  # Specific model IDs
    blocklist: list[str] = None  # Excluded model IDs
    
    def __post_init__(self):
        if self.allowlist is None:
            self.allowlist = []
        if self.blocklist is None:
            self.blocklist = []


class FeatureFlagManager:
    """Manages feature flags for the morphogenetic migration."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/morphogenetic_features.json")
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_flags()
        self._override_from_env()
        
    def _load_flags(self):
        """Load feature flags from configuration file."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    data = json.load(f)
                    for name, config in data.items():
                        self.flags[name] = FeatureFlag(name=name, **config)
                logger.info("Loaded %d feature flags", len(self.flags))
            except (json.JSONDecodeError, IOError) as e:
                logger.error("Failed to load feature flags: %s", e)
        else:
            # Default flags for Phase 0
            self._initialize_default_flags()
    
    def _initialize_default_flags(self):
        """Initialize default feature flags for migration."""
        defaults = {
            "chunked_architecture": FeatureFlag(
                name="chunked_architecture",
                enabled=False,
                rollout_percentage=0
            ),
            "triton_kernels": FeatureFlag(
                name="triton_kernels",
                enabled=False,
                rollout_percentage=0
            ),
            "extended_lifecycle": FeatureFlag(
                name="extended_lifecycle",
                enabled=False,
                rollout_percentage=0
            ),
            "message_bus": FeatureFlag(
                name="message_bus",
                enabled=False,
                rollout_percentage=0
            ),
            "neural_controller": FeatureFlag(
                name="neural_controller",
                enabled=False,
                rollout_percentage=0
            ),
            "grafting_strategies": FeatureFlag(
                name="grafting_strategies",
                enabled=False,
                rollout_percentage=0
            ),
        }
        self.flags.update(defaults)
        
    def _override_from_env(self):
        """Override flags from environment variables."""
        # Format: MORPHOGENETIC_FEATURE_<NAME>=true/false
        prefix = "MORPHOGENETIC_FEATURE_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                flag_name = key[len(prefix):].lower()
                if flag_name in self.flags:
                    self.flags[flag_name].enabled = value.lower() == "true"
                    logger.info("Overrode flag %s from env: %s", flag_name, value)
    
    def is_enabled(self, feature: str, model_id: Optional[str] = None) -> bool:
        """
        Check if a feature is enabled for a given model.
        
        Args:
            feature: Name of the feature flag
            model_id: Optional model identifier for rollout/allowlist checking
            
        Returns:
            Whether the feature is enabled
        """
        if feature not in self.flags:
            logger.warning("Unknown feature flag: %s", feature)
            return False
            
        flag = self.flags[feature]
        
        # Check blocklist first
        if model_id and model_id in flag.blocklist:
            return False
            
        # Check allowlist
        if model_id and flag.allowlist and model_id in flag.allowlist:
            return True
            
        # Check global enable
        if flag.enabled:
            return True
            
        # Check percentage rollout
        if flag.rollout_percentage > 0 and model_id:
            # Consistent hashing for stable rollout
            hash_value = int(hashlib.sha256(model_id.encode()).hexdigest(), 16)
            return (hash_value % 100) < flag.rollout_percentage
            
        return False
    
    def set_enabled(self, feature: str, enabled: bool):
        """Enable or disable a feature globally."""
        if feature in self.flags:
            self.flags[feature].enabled = enabled
            logger.info("Set feature %s enabled=%s", feature, enabled)
    
    def set_rollout_percentage(self, feature: str, percentage: int):
        """Set gradual rollout percentage for a feature."""
        if feature in self.flags:
            self.flags[feature].rollout_percentage = max(0, min(100, percentage))
            logger.info("Set feature %s rollout=%d%%", feature, percentage)
    
    def add_to_allowlist(self, feature: str, model_id: str):
        """Add a model to the feature's allowlist."""
        if feature in self.flags:
            if model_id not in self.flags[feature].allowlist:
                self.flags[feature].allowlist.append(model_id)
                logger.info("Added %s to %s allowlist", model_id, feature)
    
    def save_flags(self):
        """Persist current flag configuration."""
        data = {
            name: {
                "enabled": flag.enabled,
                "rollout_percentage": flag.rollout_percentage,
                "allowlist": flag.allowlist,
                "blocklist": flag.blocklist
            }
            for name, flag in self.flags.items()
        }
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info("Saved feature flags to %s", self.config_path)


class MigrationRouter:
    """Routes between legacy and new implementations based on feature flags."""
    
    def __init__(self, feature_manager: FeatureFlagManager):
        self.features = feature_manager
        
    def route_to_implementation(
        self,
        feature: str,
        legacy_fn: Callable,
        new_fn: Callable,
        *args,
        model_id: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Route to appropriate implementation based on feature flag.
        
        Args:
            feature: Feature flag name
            legacy_fn: Legacy implementation
            new_fn: New implementation
            model_id: Optional model identifier
            *args, **kwargs: Arguments to pass to implementation
            
        Returns:
            Result from selected implementation
        """
        if self.features.is_enabled(feature, model_id):
            logger.debug("Using new implementation for %s", feature)
            try:
                return new_fn(*args, **kwargs)
            except (RuntimeError, ValueError, TypeError) as e:
                logger.error("New implementation failed for %s: %s", feature, e)
                # Fallback to legacy on error
                return legacy_fn(*args, **kwargs)
        else:
            logger.debug("Using legacy implementation for %s", feature)
            return legacy_fn(*args, **kwargs)


# Global instance
_feature_manager = None


def get_feature_manager() -> FeatureFlagManager:
    """Get the global feature flag manager instance."""
    global _feature_manager
    if _feature_manager is None:
        _feature_manager = FeatureFlagManager()
    return _feature_manager


def is_feature_enabled(feature: str, model_id: Optional[str] = None) -> bool:
    """Convenience function to check if a feature is enabled."""
    return get_feature_manager().is_enabled(feature, model_id)