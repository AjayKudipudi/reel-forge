"""PhaseResult / PhaseContext / ErrorInfo — the data shapes a phase exchanges
with the orchestrator. PhaseResult also carries a strict JSON codec used by
the subprocess runner; never pickle."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from .errors import ErrorClass, ErrorInfo

if TYPE_CHECKING:
    import structlog

    from .manifest import Manifest
    from .storage import ObjectStore


@dataclass
class PhaseContext:
    """Everything a phase needs to run.

    INTRA-PROCESS ONLY — never serialized. The subprocess runner
    reconstructs this in the child from primitive args + on-disk manifest.
    """

    job_id: str
    work_dir: Path
    storage: ObjectStore
    s3_prefix: str
    manifest: Manifest
    seed: int
    logger: structlog.stdlib.BoundLogger
    on_progress: Callable[[str], None]


@dataclass
class PhaseResult:
    status: Literal["ok", "error"]
    stats: dict[str, Any] = field(default_factory=dict)
    error: ErrorInfo | None = None
    artifacts: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        *,
        stats: dict[str, Any] | None = None,
        artifacts: dict[str, Path] | None = None,
    ) -> PhaseResult:
        return cls(status="ok", stats=stats or {}, artifacts=artifacts or {})

    @classmethod
    def fail(
        cls,
        *,
        error_class: ErrorClass,
        message: str,
        retryable: bool,
        stats: dict[str, Any] | None = None,
        stderr_tail: str | None = None,
    ) -> PhaseResult:
        return cls(
            status="error",
            stats=stats or {},
            error=ErrorInfo(
                error_class=error_class,
                message=message,
                retryable=retryable,
                stderr_tail=stderr_tail,
            ),
        )

    def to_json(self) -> str:
        d: dict[str, Any] = {
            "status": self.status,
            "stats": self.stats,
            "artifacts": {k: str(v) for k, v in self.artifacts.items()},
        }
        if self.error is not None:
            d["error"] = {
                "class": self.error.error_class.value,
                "message": self.error.message,
                "retryable": self.error.retryable,
                "stderr_tail": self.error.stderr_tail,
                "attempt": self.error.attempt,
            }
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> PhaseResult:
        d = json.loads(s)
        err: ErrorInfo | None = None
        if d.get("error"):
            e = d["error"]
            err = ErrorInfo(
                error_class=ErrorClass(e["class"]),
                message=e["message"],
                retryable=e["retryable"],
                stderr_tail=e.get("stderr_tail"),
                attempt=e.get("attempt", 1),
            )
        return cls(
            status=d["status"],
            stats=d.get("stats", {}),
            error=err,
            artifacts={k: Path(v) for k, v in d.get("artifacts", {}).items()},
        )
