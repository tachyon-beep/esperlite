"""
Message bus contracts for the Oona communication system.
"""

import uuid
from datetime import UTC
from datetime import datetime
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel
from pydantic import Field


class TopicNames(str, Enum):
    """Centralized definition of all Oona message bus topics."""

    TELEMETRY_SEED_HEALTH = "telemetry.seed.health"
    CONTROL_KASMINA_COMMANDS = "control.kasmina.commands"
    COMPILATION_BLUEPRINT_SUBMITTED = "compilation.blueprint.submitted"
    COMPILATION_KERNEL_READY = "compilation.kernel.ready"
    VALIDATION_KERNEL_CHARACTERIZED = "validation.kernel.characterized"
    SYSTEM_EVENTS_EPOCH = "system.events.epoch"
    INNOVATION_FIELD_REPORTS = "innovation.field_reports"


class OonaMessage(BaseModel):
    """Base envelope for all messages published on the Oona bus."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str  # e.g., 'Tezzeret-Worker-5', 'Tamiyo-Controller'
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    trace_id: str  # To trace a request across multiple services
    topic: TopicNames
    payload: Dict[str, Any]


class BlueprintSubmitted(BaseModel):
    """Payload for blueprint submission events."""

    blueprint_id: str
    submitted_by: str
