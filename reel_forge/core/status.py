"""StatusManager + declared TRANSITIONS.

Atomic writes: every transition writes <local_path>.tmp, replaces, and
calls storage.upload_atomic. Never write directly to status.json — go
through the manager.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import IllegalTransition
from .status_models import (
    FailureInfoModel,
    InstanceInfo,
    PhaseHistoryEntry,
    ResourceTelemetry,
    State,
    Status,
)
from .storage import ObjectStore

# Single source of truth for legal state transitions.
TRANSITIONS: dict[State, frozenset[State]] = {
    State.CREATED: frozenset({State.PREPARING, State.FAILED_TERMINAL}),
    State.PREPARING: frozenset(
        {State.PREPARED, State.FAILED_RECOVERABLE, State.FAILED_TERMINAL}
    ),
    State.PREPARED: frozenset({State.UPLOADING, State.CANCELLED}),
    State.UPLOADING: frozenset({State.QUEUED, State.FAILED_RECOVERABLE, State.CANCELLED}),
    State.QUEUED: frozenset({State.LAUNCHING, State.CANCELLED}),
    State.LAUNCHING: frozenset({State.PHASE_RUNNING, State.FAILED_RECOVERABLE}),
    State.PHASE_RUNNING: frozenset(
        {
            State.PHASE_RUNNING,
            State.COMPLETED,
            State.FAILED_RECOVERABLE,
            State.FAILED_TERMINAL,
            State.CANCELLED,
        }
    ),
    State.FAILED_RECOVERABLE: frozenset({State.QUEUED, State.CANCELLED}),
    State.COMPLETED: frozenset(),
    State.FAILED_TERMINAL: frozenset(),
    State.CANCELLED: frozenset(),
}


def _now() -> datetime:
    return datetime.now(UTC)


class StatusManager:
    """Owns the status.json for one job. Thread-safe-enough for our use:
    the heartbeat thread and the main thread both call into it; mutations
    are serial because each call writes the full file."""

    def __init__(
        self,
        job_id: str,
        local_path: Path,
        storage: ObjectStore,
        s3_key: str,
    ) -> None:
        self.job_id = job_id
        self.local_path = local_path
        self.storage = storage
        self.s3_key = s3_key
        self.status: Status = self._load_or_create()

    def _load_or_create(self) -> Status:
        if self.local_path.exists():
            return Status.model_validate_json(self.local_path.read_text())
        # Try S3 (resume scenario)
        if self.storage.exists(self.s3_key):
            self.storage.download(self.s3_key, self.local_path)
            return Status.model_validate_json(self.local_path.read_text())
        now = _now()
        return Status(job_id=self.job_id, state=State.CREATED, created_at=now, updated_at=now)

    def transition(self, to: State, **fields: Any) -> None:
        if to not in TRANSITIONS[self.status.state]:
            raise IllegalTransition(
                f"illegal transition: {self.status.state.value} → {to.value}"
            )
        update = {**fields, "state": to, "updated_at": _now()}
        self.status = self.status.model_copy(update=update)
        self._flush()

    def heartbeat(self, **fields: Any) -> None:
        update = {**fields, "last_heartbeat_at": _now()}
        self.status = self.status.model_copy(update=update)
        self._flush()

    def append_phase_history(
        self,
        phase: str,
        status: str,
        *,
        wall_s: float | None = None,
        stats: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        ended_at: datetime | None = None,
    ) -> None:
        entry = PhaseHistoryEntry(
            phase=phase,
            status=status,
            wall_s=wall_s,
            started_at=started_at,
            ended_at=ended_at or _now(),
            stats=stats or {},
        )
        history = [*self.status.phase_history, entry]
        self.status = self.status.model_copy(
            update={"phase_history": history, "updated_at": _now()}
        )
        self._flush()

    def bump_attempt(self, phase: str) -> int:
        attempts = dict(self.status.attempts)
        attempts[phase] = attempts.get(phase, 0) + 1
        self.status = self.status.model_copy(
            update={"attempts": attempts, "updated_at": _now()}
        )
        self._flush()
        return attempts[phase]

    def append_telemetry(self, sample: ResourceTelemetry) -> None:
        # Cap at 1000 samples (model_max_length); drop oldest.
        tele = [*self.status.resource_telemetry, sample][-1000:]
        self.status = self.status.model_copy(
            update={"resource_telemetry": tele, "last_heartbeat_at": _now()}
        )
        self._flush()

    def set_instance(self, info: InstanceInfo) -> None:
        self.status = self.status.model_copy(update={"instance": info, "updated_at": _now()})
        self._flush()

    def fail(
        self,
        *,
        phase: str | None,
        error: Any,  # ErrorInfo from core.errors — kept Any to avoid circular import
        terminal: bool | None = None,
    ) -> None:
        retryable = bool(getattr(error, "retryable", False))
        target = State.FAILED_RECOVERABLE if retryable and not terminal else State.FAILED_TERMINAL
        failure = FailureInfoModel(
            phase=phase,
            error_class=error.error_class,
            message=error.message,
            retryable=retryable,
            stderr_tail=getattr(error, "stderr_tail", None),
            attempt=getattr(error, "attempt", 1),
            occurred_at=_now(),
        )
        attempts = dict(self.status.attempts)
        if phase:
            attempts[phase] = attempts.get(phase, 0) + 1
        self.status = self.status.model_copy(
            update={
                "state": target,
                "failure": failure,
                "attempts": attempts,
                "updated_at": _now(),
            }
        )
        self._flush()

    def _flush(self) -> None:
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.local_path.with_suffix(".json.tmp")
        tmp.write_text(self.status.model_dump_json(indent=2, exclude_none=False))
        tmp.replace(self.local_path)
        # Best-effort upload — storage failures must not crash a phase, but
        # they MUST be logged so transient S3 issues don't disappear silently.
        # The next heartbeat or phase boundary will retry.
        try:
            self.storage.upload_atomic(self.local_path, self.s3_key)
        except Exception as exc:
            import structlog as _structlog

            _log = _structlog.get_logger(__name__)
            _log.warning(
                "status.s3_mirror_failed",
                job_id=self.job_id,
                key=self.s3_key,
                err=str(exc),
            )


__all__ = ["TRANSITIONS", "State", "StatusManager"]


# Surface a small helper for json dumping outside the manager (used by tools).
def status_to_json(s: Status) -> str:
    return s.model_dump_json(indent=2)


_ = json  # keep import; useful for ad-hoc debugging
