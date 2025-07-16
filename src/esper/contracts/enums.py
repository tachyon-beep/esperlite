"""
Status enumerations for the Esper system.
"""

from enum import Enum


class SeedState(str, Enum):
    """Lifecycle states for morphogenetic seeds."""

    DORMANT = "dormant"
    GERMINATED = "germinated"
    TRAINING = "training"
    GRAFTING = "grafting"
    FOSSILIZED = "fossilized"
    CULLED = "culled"


class BlueprintState(str, Enum):
    """Lifecycle states for architectural blueprints."""

    PROPOSED = "proposed"
    COMPILING = "compiling"
    VALIDATING = "validating"
    CHARACTERIZED = "characterized"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class SystemHealth(str, Enum):
    """Overall system health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"


class ComponentType(str, Enum):
    """Types of system components."""

    TAMIYO = "tamiyo"  # Strategic Controller
    KARN = "karn"  # Generative Architect
    KASMINA = "kasmina"  # Execution Layer
    TEZZERET = "tezzeret"  # Compilation Forge
    URABRASK = "urabrask"  # Evaluation Engine
    TOLARIA = "tolaria"  # Training Orchestrator
    URZA = "urza"  # Central Library
    OONA = "oona"  # Message Bus
    NISSA = "nissa"  # Observability
    SIMIC = "simic"  # Policy Training Environment
    EMRAKUL = "emrakul"  # Architectural Sculptor


class BlueprintStatus(str, Enum):
    """Status of blueprint compilation and validation."""

    UNVALIDATED = "unvalidated"
    COMPILING = "compiling"
    VALIDATED = "validated"
    INVALID = "invalid"


class KernelStatus(str, Enum):
    """Status of compiled kernel artifacts."""

    VALIDATED = "validated"
    INVALID = "invalid"
    TESTING = "testing"
    DEPLOYED = "deployed"
