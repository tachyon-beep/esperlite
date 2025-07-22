"""
Blueprint registry for managing and loading blueprint templates.
"""

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import yaml

from esper.blueprints.metadata import BlueprintCategory
from esper.blueprints.metadata import BlueprintManifest
from esper.blueprints.metadata import BlueprintMetadata

logger = logging.getLogger(__name__)


class BlueprintRegistry:
    """
    Central registry for blueprint templates.
    
    Manages loading, validation, and access to blueprint templates
    for Tamiyo's adaptation decisions.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        """Initialize blueprint registry."""
        self.blueprints: Dict[str, BlueprintMetadata] = {}
        self.architectures: Dict[str, BlueprintManifest] = {}

        # Load from manifest if provided
        if manifest_path:
            self.load_manifest(manifest_path)
        else:
            # Load default blueprints
            self._load_default_blueprints()

    def _load_default_blueprints(self):
        """Load the default blueprint library."""
        # Import all template modules
        from esper.blueprints.templates import diagnostics
        from esper.blueprints.templates import efficiency
        from esper.blueprints.templates import moe
        from esper.blueprints.templates import routing
        from esper.blueprints.templates import transformer

        # Register blueprints from each module
        self._register_module_blueprints(transformer)
        self._register_module_blueprints(moe)
        self._register_module_blueprints(efficiency)
        self._register_module_blueprints(routing)
        self._register_module_blueprints(diagnostics)

        logger.info(f"Loaded {len(self.blueprints)} default blueprints")

    def _register_module_blueprints(self, module):
        """Register all blueprints from a template module."""
        if hasattr(module, "BLUEPRINTS"):
            for blueprint_data in module.BLUEPRINTS:
                metadata = BlueprintMetadata(**blueprint_data["metadata"])
                manifest = BlueprintManifest(**blueprint_data)
                self.register(metadata, manifest)

    def register(self, metadata: BlueprintMetadata, manifest: BlueprintManifest):
        """Register a blueprint in the registry."""
        blueprint_id = metadata.blueprint_id

        if blueprint_id in self.blueprints:
            logger.warning(f"Overwriting existing blueprint: {blueprint_id}")

        self.blueprints[blueprint_id] = metadata
        self.architectures[blueprint_id] = manifest

        logger.debug(f"Registered blueprint: {blueprint_id} v{metadata.version}")

    def get_blueprint(self, blueprint_id: str) -> Optional[BlueprintMetadata]:
        """Get blueprint metadata by ID."""
        return self.blueprints.get(blueprint_id)

    def get_architecture(self, blueprint_id: str) -> Optional[BlueprintManifest]:
        """Get blueprint architecture by ID."""
        return self.architectures.get(blueprint_id)

    def list_blueprints(
        self,
        category: Optional[BlueprintCategory] = None,
        safe_only: bool = False,
        compatible_with: Optional[str] = None,
    ) -> List[BlueprintMetadata]:
        """
        List blueprints with optional filtering.
        
        Args:
            category: Filter by category
            safe_only: Only return safe blueprints
            compatible_with: Layer type to check compatibility
            
        Returns:
            List of matching blueprint metadata
        """
        results = []

        for metadata in self.blueprints.values():
            # Category filter
            if category and metadata.category != category:
                continue

            # Safety filter
            if safe_only and not metadata.is_safe_action:
                continue

            # Compatibility filter
            if compatible_with:
                if compatible_with not in metadata.compatible_layers:
                    continue

            results.append(metadata)

        return results

    def get_cost_vector(self, blueprint_id: str) -> Optional[List[float]]:
        """Get cost vector for Tamiyo decision making."""
        metadata = self.get_blueprint(blueprint_id)
        if not metadata:
            return None

        return [
            float(metadata.param_delta),
            float(metadata.flop_delta),
            float(metadata.memory_footprint_kb),
            metadata.expected_latency_ms,
        ]

    def get_benefit_prior(self, blueprint_id: str) -> Optional[List[float]]:
        """Get benefit prior for Tamiyo decision making."""
        metadata = self.get_blueprint(blueprint_id)
        if not metadata:
            return None

        return [
            metadata.past_accuracy_gain_estimate,
            metadata.stability_improvement_estimate,
            metadata.speed_improvement_estimate,
        ]

    def load_manifest(self, manifest_path: Path):
        """Load blueprints from YAML manifest."""
        try:
            with open(manifest_path, "r") as f:
                manifest_data = yaml.safe_load(f)

            for blueprint_data in manifest_data.get("blueprints", []):
                metadata = BlueprintMetadata(**blueprint_data["metadata"])
                manifest = BlueprintManifest(**blueprint_data)
                self.register(metadata, manifest)

            logger.info(f"Loaded {len(self.blueprints)} blueprints from manifest")

        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            raise

    def save_manifest(self, manifest_path: Path):
        """Save current registry to YAML manifest."""
        blueprints = []

        for blueprint_id, metadata in self.blueprints.items():
            manifest = self.architectures[blueprint_id]
            blueprints.append({
                "metadata": metadata.to_dict(),
                "architecture": manifest.architecture.dict(),
                "validation": manifest.validation,
            })

        manifest_data = {
            "version": "1.0.0",
            "blueprints": blueprints,
        }

        with open(manifest_path, "w") as f:
            yaml.dump(manifest_data, f, default_flow_style=False)

        logger.info(f"Saved {len(blueprints)} blueprints to manifest")
