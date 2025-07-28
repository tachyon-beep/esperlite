#!/usr/bin/env python3
"""
Script to enable Phase 1 features for testing the morphogenetic migration.

This script allows gradual rollout of the chunked architecture.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.esper.morphogenetic_v2.common.feature_flags import FeatureFlagManager


def enable_phase1_features(rollout_percentage: int = 10, model_ids: list = None):
    """
    Enable Phase 1 features with specified rollout percentage.
    
    Args:
        rollout_percentage: Percentage of models to enable (0-100)
        model_ids: Specific model IDs to add to allowlist
    """
    config_path = Path("config/morphogenetic_features.json")

    # Load existing config
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Update Phase 1 features
    config["chunked_architecture"] = {
        "enabled": True,
        "rollout_percentage": rollout_percentage,
        "allowlist": model_ids or [],
        "blocklist": [],
        "description": "Phase 1: Enable chunked architecture with thousands of parallel seeds"
    }

    # Ensure Phase 0 features are enabled
    config["performance_monitoring"] = {
        "enabled": True,
        "rollout_percentage": 100,
        "allowlist": [],
        "blocklist": [],
        "description": "Phase 0: Performance baseline and monitoring"
    }

    config["ab_testing"] = {
        "enabled": True,
        "rollout_percentage": 100,
        "allowlist": [],
        "blocklist": [],
        "description": "Phase 0: A/B testing framework"
    }

    # Save updated config
    config_path.parent.mkdir(exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Phase 1 features enabled:")
    print(f"- Chunked architecture: {rollout_percentage}% rollout")
    if model_ids:
        print(f"- Allowlist: {', '.join(model_ids)}")
    print(f"- Config saved to: {config_path}")

    # Verify with feature flag manager
    manager = FeatureFlagManager(config_path)
    test_model = model_ids[0] if model_ids else "test_model_123"
    enabled = manager.is_enabled("chunked_architecture", test_model)
    print(f"\nVerification: chunked_architecture enabled for '{test_model}': {enabled}")


def disable_phase1_features():
    """Disable Phase 1 features (revert to Phase 0 only)."""
    config_path = Path("config/morphogenetic_features.json")

    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

        # Disable Phase 1 features
        if "chunked_architecture" in config:
            config["chunked_architecture"]["enabled"] = False
            config["chunked_architecture"]["rollout_percentage"] = 0
            config["chunked_architecture"]["allowlist"] = []

        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        print("Phase 1 features disabled")
    else:
        print("No config file found")


def main():
    parser = argparse.ArgumentParser(
        description="Enable/disable Phase 1 morphogenetic features"
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help="Enable Phase 1 features"
    )
    parser.add_argument(
        "--disable",
        action="store_true",
        help="Disable Phase 1 features"
    )
    parser.add_argument(
        "--rollout",
        type=int,
        default=10,
        help="Rollout percentage (0-100, default: 10)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Specific model IDs to enable"
    )

    args = parser.parse_args()

    if args.disable:
        disable_phase1_features()
    elif args.enable:
        if args.rollout < 0 or args.rollout > 100:
            print("Error: Rollout percentage must be between 0 and 100")
            sys.exit(1)
        enable_phase1_features(args.rollout, args.models)
    else:
        print("Please specify --enable or --disable")
        sys.exit(1)


if __name__ == "__main__":
    main()
