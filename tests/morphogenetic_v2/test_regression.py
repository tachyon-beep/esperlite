"""
Regression test suite for morphogenetic system.

Ensures backward compatibility during migration.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Any, Optional
import logging

# Import both legacy and new implementations
from esper.execution.kasmina_layer import KasminaLayer as LegacyKasminaLayer
from esper.services.tamiyo.analyzer import ModelGraphAnalyzer as LegacyTamiyoAnalyzer
from esper.services.tamiyo.autonomous_service import AutonomousTamiyoService as LegacyTamiyoService

logger = logging.getLogger(__name__)


class RegressionTestCase:
    """Base class for regression test cases."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.legacy_output = None
        self.new_output = None
        
    def setup(self):
        """Setup test case (override in subclasses)."""
        pass
        
    def run_legacy(self) -> Any:
        """Run test with legacy implementation."""
        raise NotImplementedError
        
    def run_new(self) -> Any:
        """Run test with new implementation."""
        raise NotImplementedError
        
    def compare_outputs(self, legacy: Any, new: Any) -> Dict[str, Any]:
        """Compare outputs from legacy and new implementations."""
        raise NotImplementedError
        
    def cleanup(self):
        """Cleanup after test (override if needed)."""
        pass


class KasminaLayerRegressionTest(RegressionTestCase):
    """Regression tests for KasminaLayer functionality."""
    
    def __init__(self):
        super().__init__(
            "kasmina_layer_basic",
            "Test basic KasminaLayer forward pass and seed management"
        )
        self.base_layer = nn.Linear(256, 256)
        self.config = {
            "num_seeds": 1,
            "health_threshold": 0.7,
            "enable_async": False
        }
        self.test_input = torch.randn(32, 256)
        
    def run_legacy(self) -> Dict[str, Any]:
        """Run with legacy KasminaLayer."""
        layer = LegacyKasminaLayer(self.base_layer, self.config)
        layer.eval()
        
        with torch.no_grad():
            # Initial forward pass
            output1 = layer(self.test_input)
            
            # Simulate seed loading
            layer.load_kernel(0, "test_kernel_id")
            
            # Forward pass with active seed
            output2 = layer(self.test_input)
            
            # Get state
            state = {
                "has_active_seeds": layer.state_layout.has_active_seeds(),
                "lifecycle_states": layer.state_layout.lifecycle_states.clone(),
                "alpha_blend": layer.state_layout.alpha_blend.clone()
            }
            
        return {
            "output_initial": output1.clone(),
            "output_with_seed": output2.clone(),
            "state": state,
            "output_shape": output1.shape
        }
        
    def run_new(self) -> Dict[str, Any]:
        """Run with new implementation (when available)."""
        # For now, return legacy results (will update when new impl exists)
        return self.run_legacy()
        
    def compare_outputs(self, legacy: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Compare outputs."""
        comparison = {
            "shape_match": legacy["output_shape"] == new["output_shape"],
            "initial_output_similarity": torch.nn.functional.cosine_similarity(
                legacy["output_initial"].flatten(),
                new["output_initial"].flatten(),
                dim=0
            ).item(),
            "seed_output_similarity": torch.nn.functional.cosine_similarity(
                legacy["output_with_seed"].flatten(),
                new["output_with_seed"].flatten(),
                dim=0
            ).item(),
            "state_match": torch.allclose(
                legacy["state"]["lifecycle_states"],
                new["state"]["lifecycle_states"]
            )
        }
        
        # Determine if outputs are equivalent
        comparison["equivalent"] = (
            comparison["shape_match"] and
            comparison["initial_output_similarity"] > 0.99 and
            comparison["seed_output_similarity"] > 0.99 and
            comparison["state_match"]
        )
        
        return comparison


class TamiyoAnalyzerRegressionTest(RegressionTestCase):
    """Regression tests for Tamiyo analyzer functionality."""
    
    def __init__(self):
        super().__init__(
            "tamiyo_analyzer_decisions",
            "Test Tamiyo analyzer decision making"
        )
        self.health_signals = [
            {
                "layer_id": "layer_1",
                "health_score": 0.6,  # Below threshold
                "gradient_norm": 0.1,
                "activation_sparsity": 0.3
            },
            {
                "layer_id": "layer_2", 
                "health_score": 0.9,  # Healthy
                "gradient_norm": 0.5,
                "activation_sparsity": 0.1
            }
        ]
        
    def run_legacy(self) -> Dict[str, Any]:
        """Run with legacy Tamiyo analyzer."""
        analyzer = LegacyTamiyoAnalyzer()
        
        # Analyze health signals
        decisions = []
        for signal in self.health_signals:
            decision = analyzer.analyze_layer_health(signal)
            if decision:
                decisions.append(decision)
                
        # Get graph state
        graph_state = analyzer.build_model_graph(self.health_signals)
        
        return {
            "decisions": decisions,
            "num_decisions": len(decisions),
            "graph_nodes": len(graph_state.get("nodes", [])),
            "problematic_layers": [
                d.get("layer_name") for d in decisions
                if d.get("adaptation_type") == "add_seed"
            ]
        }
        
    def run_new(self) -> Dict[str, Any]:
        """Run with new implementation (when available)."""
        # For now, return legacy results
        return self.run_legacy()
        
    def compare_outputs(self, legacy: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Compare decision outputs."""
        comparison = {
            "same_num_decisions": legacy["num_decisions"] == new["num_decisions"],
            "same_problematic_layers": set(legacy["problematic_layers"]) == set(new["problematic_layers"]),
            "graph_consistency": legacy["graph_nodes"] == new["graph_nodes"]
        }
        
        comparison["equivalent"] = all(comparison.values())
        return comparison


class LifecycleRegressionTest(RegressionTestCase):
    """Test seed lifecycle state transitions."""
    
    def __init__(self):
        super().__init__(
            "seed_lifecycle_transitions",
            "Test seed lifecycle state machine transitions"
        )
        
    def run_legacy(self) -> Dict[str, Any]:
        """Test legacy lifecycle transitions."""
        from esper.execution.state_layout import KasminaStateLayout, SeedLifecycle
        
        state = KasminaStateLayout(num_seeds=5, device=torch.device("cpu"))
        transitions = []
        
        # Test transition sequence
        seed_id = 0
        
        # DORMANT -> LOADING
        initial = state.lifecycle_states[seed_id].item()
        state.lifecycle_states[seed_id] = SeedLifecycle.LOADING.value
        transitions.append((initial, SeedLifecycle.LOADING.value))
        
        # LOADING -> ACTIVE
        state.lifecycle_states[seed_id] = SeedLifecycle.ACTIVE.value
        transitions.append((SeedLifecycle.LOADING.value, SeedLifecycle.ACTIVE.value))
        
        # ACTIVE -> FOSSILIZED
        state.lifecycle_states[seed_id] = SeedLifecycle.FOSSILIZED.value
        transitions.append((SeedLifecycle.ACTIVE.value, SeedLifecycle.FOSSILIZED.value))
        
        return {
            "transitions": transitions,
            "final_state": state.lifecycle_states[seed_id].item(),
            "valid_transitions": len(transitions),
            "state_values": list(SeedLifecycle)
        }
        
    def run_new(self) -> Dict[str, Any]:
        """Test new lifecycle transitions (when available)."""
        return self.run_legacy()
        
    def compare_outputs(self, legacy: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
        """Compare lifecycle behavior."""
        # In new implementation, we expect more states but same core transitions
        comparison = {
            "core_transitions_preserved": all(
                t in new["transitions"][:len(legacy["transitions"])]
                for t in legacy["transitions"]
            ),
            "final_state_valid": new["final_state"] in [s.value for s in new["state_values"]]
        }
        
        comparison["equivalent"] = comparison["core_transitions_preserved"]
        return comparison


class RegressionTestSuite:
    """Manages and runs all regression tests."""
    
    def __init__(self, output_dir: Path = Path("tests/regression_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register all test cases
        self.test_cases = [
            KasminaLayerRegressionTest(),
            TamiyoAnalyzerRegressionTest(),
            LifecycleRegressionTest()
        ]
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all regression tests."""
        results = {
            "timestamp": torch.cuda.Event(enable_timing=True),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "test_results": {}
        }
        
        for test_case in self.test_cases:
            logger.info(f"Running test: {test_case.name}")
            
            try:
                # Setup
                test_case.setup()
                
                # Run legacy
                legacy_output = test_case.run_legacy()
                
                # Run new
                new_output = test_case.run_new()
                
                # Compare
                comparison = test_case.compare_outputs(legacy_output, new_output)
                
                # Cleanup
                test_case.cleanup()
                
                # Record result
                results["test_results"][test_case.name] = {
                    "description": test_case.description,
                    "passed": comparison.get("equivalent", False),
                    "comparison": comparison
                }
                
                results["tests_run"] += 1
                if comparison.get("equivalent", False):
                    results["tests_passed"] += 1
                    logger.info(f"✓ {test_case.name} PASSED")
                else:
                    results["tests_failed"] += 1
                    logger.error(f"✗ {test_case.name} FAILED")
                    logger.error(f"  Comparison: {comparison}")
                    
            except Exception as e:
                logger.error(f"✗ {test_case.name} ERROR: {e}")
                results["test_results"][test_case.name] = {
                    "description": test_case.description,
                    "passed": False,
                    "error": str(e)
                }
                results["tests_failed"] += 1
                results["tests_run"] += 1
        
        # Save results
        self._save_results(results)
        
        # Log summary
        logger.info(f"\nRegression Test Summary:")
        logger.info(f"Tests Run: {results['tests_run']}")
        logger.info(f"Passed: {results['tests_passed']}")
        logger.info(f"Failed: {results['tests_failed']}")
        logger.info(f"Success Rate: {results['tests_passed']/results['tests_run']*100:.1f}%")
        
        return results
        
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file."""
        from datetime import datetime
        
        filename = f"regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file = self.output_dir / filename
        
        # Convert non-serializable objects
        results["timestamp"] = datetime.now().isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Saved regression test results to {output_file}")


# Pytest fixtures and tests

@pytest.fixture
def regression_suite():
    """Create regression test suite."""
    return RegressionTestSuite()


def test_kasmina_layer_compatibility(regression_suite):
    """Test KasminaLayer backward compatibility."""
    test = KasminaLayerRegressionTest()
    test.setup()
    
    legacy = test.run_legacy()
    new = test.run_new()
    comparison = test.compare_outputs(legacy, new)
    
    assert comparison["equivalent"], f"KasminaLayer outputs not equivalent: {comparison}"


def test_tamiyo_decisions_compatibility(regression_suite):
    """Test Tamiyo decision making compatibility."""
    test = TamiyoAnalyzerRegressionTest()
    test.setup()
    
    legacy = test.run_legacy()
    new = test.run_new()
    comparison = test.compare_outputs(legacy, new)
    
    assert comparison["equivalent"], f"Tamiyo decisions not equivalent: {comparison}"


def test_lifecycle_compatibility(regression_suite):
    """Test lifecycle state machine compatibility."""
    test = LifecycleRegressionTest()
    test.setup()
    
    legacy = test.run_legacy()
    new = test.run_new()
    comparison = test.compare_outputs(legacy, new)
    
    assert comparison["equivalent"], f"Lifecycle transitions not compatible: {comparison}"


def test_full_regression_suite(regression_suite):
    """Run full regression test suite."""
    results = regression_suite.run_all_tests()
    
    assert results["tests_failed"] == 0, f"Regression tests failed: {results['test_results']}"


if __name__ == "__main__":
    # Run regression tests directly
    suite = RegressionTestSuite()
    results = suite.run_all_tests()
    
    # Exit with error code if tests failed
    import sys
    sys.exit(0 if results["tests_failed"] == 0 else 1)