"""Pydantic models for status.json — distinct from `status.py` which holds
the StatusManager + TRANSITIONS."""
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from .errors import ErrorClass


class State(StrEnum):
    CREATED = "created"
    PREPARING = "preparing"
    PREPARED = "prepared"
    UPLOADING = "uploading"
    QUEUED = "queued"
    LAUNCHING = "launching"
    PHASE_RUNNING = "phase_running"
    COMPLETED = "completed"
    FAILED_RECOVERABLE = "failed_recoverable"
    FAILED_TERMINAL = "failed_terminal"
    CANCELLED = "cancelled"


class PhaseHistoryEntry(BaseModel):
    phase: str
    status: str  # "ok" | "error"
    wall_s: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    stats: dict[str, Any] = Field(default_factory=dict)


class FailureInfoModel(BaseModel):
    """Pydantic version of ErrorInfo for embedding in status.json."""

    phase: str | None = None
    error_class: ErrorClass
    message: str
    retryable: bool
    stderr_tail: str | None = None
    attempt: int = 1
    occurred_at: datetime


class InstanceInfo(BaseModel):
    instance_id: str
    instance_type: str
    az: str
    spot: bool
    launch_at: datetime


class ResourceTelemetry(BaseModel):
    timestamp: datetime
    gpu_util_pct: float | None = None
    gpu_mem_gb: float | None = None
    cpu_util_pct: float | None = None
    ram_gb: float | None = None


class Status(BaseModel):
    schema_version: int = 1
    job_id: str
    state: State = State.CREATED
    created_at: datetime
    updated_at: datetime
    last_heartbeat_at: datetime | None = None
    current_phase: str | None = None
    current_phase_started_at: datetime | None = None
    current_phase_progress: str | None = None
    phase_history: list[PhaseHistoryEntry] = Field(default_factory=list)
    attempts: dict[str, int] = Field(default_factory=dict)
    failure: FailureInfoModel | None = None
    instance: InstanceInfo | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    resource_telemetry: list[ResourceTelemetry] = Field(default_factory=list, max_length=1000)
