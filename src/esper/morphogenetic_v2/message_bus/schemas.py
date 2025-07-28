"""Message schemas for morphogenetic system communication.

This module defines all message types used in the distributed morphogenetic system.
Messages are designed to be serializable, versioned, and efficient.
"""

import time
import uuid
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union


class MessagePriority(str, Enum):
    """Message priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Types of performance alerts."""
    DEGRADATION = "degradation"
    IMPROVEMENT = "improvement"
    ANOMALY = "anomaly"
    THRESHOLD_EXCEEDED = "threshold_exceeded"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class BaseMessage:
    """Base class for all morphogenetic messages.
    
    All messages in the system inherit from this base class, providing
    common fields for identification, routing, and correlation.
    """
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Component identifier (e.g., "kasmina_layer_12")
    version: str = "1.0"  # Message schema version
    correlation_id: Optional[str] = None  # For request/response patterns
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        result = {
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "version": self.version,
            "message_type": self.__class__.__name__
        }
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.metadata:
            result["metadata"] = self.metadata

        # Add subclass fields
        for key, value in self.__dict__.items():
            if key not in result and not key.startswith("_"):
                if hasattr(value, "to_dict"):
                    result[key] = value.to_dict()
                elif isinstance(value, list):
                    # Handle lists of objects that might have to_dict
                    result[key] = [
                        item.to_dict() if hasattr(item, "to_dict") else item
                        for item in value
                    ]
                elif isinstance(value, dict):
                    # Handle dicts that might contain objects with to_dict
                    result[key] = {
                        k: v.to_dict() if hasattr(v, "to_dict") else v
                        for k, v in value.items()
                    }
                elif isinstance(value, Enum):
                    result[key] = value.value
                else:
                    result[key] = value

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Create message from dictionary."""
        # Extract base fields
        base_fields = {
            "message_id": data.get("message_id", str(uuid.uuid4())),
            "timestamp": data.get("timestamp", time.time()),
            "source": data.get("source", ""),
            "version": data.get("version", "1.0"),
            "correlation_id": data.get("correlation_id"),
            "metadata": data.get("metadata", {})
        }

        # Extract subclass fields
        subclass_fields = {}
        for key, value in data.items():
            if key not in base_fields and key != "message_type":
                subclass_fields[key] = value

        return cls(**base_fields, **subclass_fields)


# Telemetry Messages

@dataclass
class LayerHealthReport(BaseMessage):
    """Aggregated health report for all seeds in a layer.
    
    This message provides a comprehensive health snapshot of an entire layer,
    including individual seed metrics and aggregate performance indicators.
    """
    layer_id: str = ""
    total_seeds: int = 0
    active_seeds: int = 0
    health_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)  # seed_id -> metrics
    performance_summary: Dict[str, float] = field(default_factory=dict)
    telemetry_window: Tuple[float, float] = (0.0, 0.0)  # (start_time, end_time)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def window_duration(self) -> float:
        """Calculate telemetry window duration in seconds."""
        return self.telemetry_window[1] - self.telemetry_window[0]

    @property
    def inactive_seeds(self) -> int:
        """Calculate number of inactive seeds."""
        return self.total_seeds - self.active_seeds


@dataclass
class SeedMetricsSnapshot(BaseMessage):
    """Detailed metrics for a single seed.
    
    Provides fine-grained telemetry for individual seed monitoring
    and debugging.
    """
    layer_id: str = ""
    seed_id: int = 0
    lifecycle_state: str = ""
    blueprint_id: Optional[int] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    checkpoint_id: Optional[str] = None
    compute_stats: Dict[str, float] = field(default_factory=dict)  # GPU utilization, memory, etc.
    grafting_info: Optional[Dict[str, Any]] = None


@dataclass
class TelemetryBatch(BaseMessage):
    """Batch of telemetry messages for efficient transmission.
    
    Allows grouping multiple telemetry messages to reduce overhead
    and improve throughput.
    """
    messages: List[Union[LayerHealthReport, SeedMetricsSnapshot]] = field(default_factory=list)
    compression: Optional[str] = None  # e.g., "zstd", "gzip"
    batch_size: int = 0

    def __post_init__(self):
        self.batch_size = len(self.messages)


# Control Commands

@dataclass
class CommandResult:
    """Result of command execution."""
    success: bool = False
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "success": self.success,
            "execution_time_ms": self.execution_time_ms
        }
        if self.error:
            result["error"] = self.error
        if self.details:
            result["details"] = self.details
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class LifecycleTransitionCommand(BaseMessage):
    """Command to transition a seed's lifecycle state.
    
    Supports all valid state transitions with appropriate parameters
    for each transition type.
    """
    layer_id: str = ""
    seed_id: int = 0
    target_state: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)  # State-specific parameters
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_ms: int = 5000  # Command timeout
    force: bool = False  # Force transition even if validation fails
    reason: str = ""  # Human-readable reason for transition


@dataclass
class BlueprintUpdateCommand(BaseMessage):
    """Command to update a seed's blueprint.
    
    Handles blueprint updates with various grafting strategies
    and configuration options.
    """
    layer_id: str = ""
    seed_id: int = 0
    blueprint_id: str = ""
    grafting_strategy: str = ""  # e.g., "immediate", "gradual", "conditional"
    configuration: Dict[str, Any] = field(default_factory=dict)
    rollback_on_failure: bool = True
    validation_metrics: Dict[str, float] = field(default_factory=dict)  # Required metrics for success
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_ms: int = 10000  # Longer timeout for blueprint updates


@dataclass
class BatchCommand(BaseMessage):
    """Execute multiple commands as a batch.
    
    Allows atomic execution of multiple commands with
    transactional semantics.
    """
    commands: List[Union[LifecycleTransitionCommand, BlueprintUpdateCommand]] = field(default_factory=list)
    atomic: bool = True  # All or nothing execution
    stop_on_error: bool = True  # Stop batch on first error
    priority: MessagePriority = MessagePriority.NORMAL
    timeout_ms: int = 30000  # Overall batch timeout


@dataclass
class EmergencyStopCommand(BaseMessage):
    """Emergency command to immediately stop operations.
    
    High-priority command for emergency situations.
    """
    layer_id: Optional[str] = None  # None means all layers
    seed_id: Optional[int] = None  # None means all seeds in layer
    reason: str = ""
    priority: MessagePriority = MessagePriority.CRITICAL
    timeout_ms: int = 1000  # Fast timeout for emergency


# Event Messages

@dataclass
class StateTransitionEvent(BaseMessage):
    """Event emitted when a seed transitions state.
    
    Provides detailed information about state transitions for
    monitoring and audit trails.
    """
    layer_id: str = ""
    seed_id: int = 0
    from_state: str = ""
    to_state: str = ""
    reason: str = ""
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    transition_duration_ms: float = 0.0
    triggered_by: str = ""  # "user", "system", "policy"
    success: bool = True


@dataclass
class PerformanceAlert(BaseMessage):
    """Alert for performance anomalies.
    
    Triggered when performance metrics exceed thresholds or
    anomalies are detected.
    """
    layer_id: str = ""
    seed_id: Optional[int] = None  # None means layer-wide alert
    alert_type: AlertType = AlertType.ANOMALY
    severity: AlertSeverity = AlertSeverity.INFO
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None


@dataclass
class CheckpointEvent(BaseMessage):
    """Event for checkpoint operations.
    
    Tracks checkpoint creation, deletion, and recovery operations.
    """
    layer_id: str = ""
    seed_id: int = 0
    checkpoint_id: str = ""
    operation: str = ""  # "created", "deleted", "recovered"
    checkpoint_size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class BlueprintEvent(BaseMessage):
    """Event for blueprint operations.
    
    Tracks blueprint updates, validations, and rollbacks.
    """
    layer_id: str = ""
    seed_id: int = 0
    blueprint_id: str = ""
    operation: str = ""  # "updated", "validated", "rolled_back"
    previous_blueprint_id: Optional[str] = None
    metrics_delta: Dict[str, float] = field(default_factory=dict)  # Performance change
    success: bool = True
    details: Dict[str, Any] = field(default_factory=dict)


# Coordination Messages

@dataclass
class ServiceAnnouncement(BaseMessage):
    """Service discovery announcement.
    
    Used by components to announce their presence and capabilities.
    """
    service_type: str = ""  # "kasmina_layer", "tamiyo_controller", etc.
    service_id: str = ""
    capabilities: List[str] = field(default_factory=list)
    endpoints: Dict[str, str] = field(default_factory=dict)  # protocol -> address
    health_check_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Heartbeat(BaseMessage):
    """Heartbeat message for liveness detection.
    
    Periodic message to indicate component health and availability.
    """
    service_id: str = ""
    uptime_seconds: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "healthy"  # "healthy", "degraded", "unhealthy"
    load: Dict[str, float] = field(default_factory=dict)  # CPU, memory, etc.


@dataclass
class CoordinationRequest(BaseMessage):
    """Request for distributed coordination.
    
    Used for operations requiring consensus or coordination
    across multiple components.
    """
    operation: str = ""  # "lock", "election", "barrier"
    resource_id: str = ""
    requester_id: str = ""
    timeout_ms: int = 10000
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CoordinationResponse(BaseMessage):
    """Response to coordination request."""
    request_id: str = ""  # correlation_id of request
    granted: bool = False
    holder_id: Optional[str] = None  # Current holder for locks
    wait_time_ms: Optional[int] = None  # Expected wait time
    details: Dict[str, Any] = field(default_factory=dict)


# Utility functions

def create_topic_name(message_type: str, layer_id: Optional[str] = None,
                     seed_id: Optional[int] = None) -> str:
    """Create standardized topic name from message type and identifiers.
    
    Args:
        message_type: Type of message (e.g., "telemetry", "control", "event")
        layer_id: Optional layer identifier
        seed_id: Optional seed identifier
        
    Returns:
        Standardized topic name
    """
    parts = ["morphogenetic", message_type]

    if layer_id:
        parts.extend(["layer", layer_id])

    if seed_id is not None:
        parts.extend(["seed", str(seed_id)])

    return ".".join(parts)


def parse_topic_name(topic: str) -> Dict[str, Any]:
    """Parse topic name into components.
    
    Args:
        topic: Topic name to parse
        
    Returns:
        Dictionary with parsed components
    """
    parts = topic.split(".")
    result = {
        "full_topic": topic,
        "parts": parts
    }

    if len(parts) >= 2:
        result["system"] = parts[0]
        result["message_type"] = parts[1]

    # Parse layer_id
    if "layer" in parts:
        idx = parts.index("layer")
        if idx + 1 < len(parts):
            result["layer_id"] = parts[idx + 1]

    # Parse seed_id
    if "seed" in parts:
        idx = parts.index("seed")
        if idx + 1 < len(parts):
            try:
                result["seed_id"] = int(parts[idx + 1])
            except ValueError:
                pass

    return result


# Message factory

class MessageFactory:
    """Factory for creating messages from dictionaries."""

    _message_types = {
        "LayerHealthReport": LayerHealthReport,
        "SeedMetricsSnapshot": SeedMetricsSnapshot,
        "TelemetryBatch": TelemetryBatch,
        "LifecycleTransitionCommand": LifecycleTransitionCommand,
        "BlueprintUpdateCommand": BlueprintUpdateCommand,
        "BatchCommand": BatchCommand,
        "EmergencyStopCommand": EmergencyStopCommand,
        "StateTransitionEvent": StateTransitionEvent,
        "PerformanceAlert": PerformanceAlert,
        "CheckpointEvent": CheckpointEvent,
        "BlueprintEvent": BlueprintEvent,
        "ServiceAnnouncement": ServiceAnnouncement,
        "Heartbeat": Heartbeat,
        "CoordinationRequest": CoordinationRequest,
        "CoordinationResponse": CoordinationResponse,
        "CommandResult": CommandResult,
    }

    @classmethod
    def create(cls, data: Dict[str, Any]) -> BaseMessage:
        """Create message instance from dictionary.
        
        Args:
            data: Dictionary containing message data
            
        Returns:
            Message instance
            
        Raises:
            ValueError: If message type is unknown
        """
        message_type = data.get("message_type")
        if not message_type:
            raise ValueError("Missing message_type in data")

        message_class = cls._message_types.get(message_type)
        if not message_class:
            raise ValueError(f"Unknown message type: {message_type}")

        return message_class.from_dict(data)

    @classmethod
    def register_type(cls, message_type: str, message_class: type):
        """Register a new message type.
        
        Args:
            message_type: String identifier for the message type
            message_class: Class to instantiate for this type
        """
        cls._message_types[message_type] = message_class
