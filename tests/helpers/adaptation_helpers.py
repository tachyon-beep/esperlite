"""Helper functions for creating valid AdaptationDecision instances in tests."""

from typing import Any
from typing import Dict
from typing import Optional

from esper.contracts.operational import AdaptationDecision


def create_valid_adaptation_decision(
    layer_name: str,
    adaptation_type: str,
    confidence: float = 0.8,
    urgency: float = 0.5,
    parameters: Optional[Dict[str, Any]] = None,
    reasoning: Optional[str] = None,
    **kwargs
) -> AdaptationDecision:
    """
    Create a valid AdaptationDecision with proper schema.
    
    This helper maps old test patterns to the new contract-compliant schema.
    
    Args:
        layer_name: Name of the layer to adapt
        adaptation_type: Type of adaptation (will be mapped to valid types)
        confidence: Confidence level (0.0-1.0)
        urgency: Urgency level (0.0-1.0)
        parameters: Optional parameters (will be placed in metadata)
        reasoning: Optional reasoning (will be placed in metadata)
        **kwargs: Additional metadata fields
        
    Returns:
        Valid AdaptationDecision instance
    """
    # Map old adaptation types to new contract-valid ones
    type_mapping = {
        "add_neurons": "add_seed",
        "remove_neurons": "remove_seed",
        "add_layer": "modify_architecture",
        "replace_kernel": "optimize_parameters",
        "kernel_selection": "optimize_parameters",
    }

    # Use mapping if available, otherwise use as-is
    adaptation_type = type_mapping.get(adaptation_type, adaptation_type)

    # Build metadata dict
    metadata = {}
    if parameters:
        metadata["parameters"] = parameters
    if reasoning:
        metadata["reasoning"] = reasoning

    # Add any additional kwargs to metadata
    for key, value in kwargs.items():
        # Skip fields that are part of the main schema
        if key not in ["layer_name", "adaptation_type", "confidence", "urgency", "metadata", "timestamp"]:
            metadata[key] = value

    return AdaptationDecision(
        layer_name=layer_name,
        adaptation_type=adaptation_type,
        confidence=confidence,
        urgency=urgency,
        metadata=metadata if metadata else {}
    )
