"""PhaseResult.to_json / from_json round-trip."""
from __future__ import annotations

from pathlib import Path

from reel_forge.core.errors import ErrorClass
from reel_forge.core.result import PhaseResult


def test_ok_roundtrip() -> None:
    r = PhaseResult.ok(
        stats={"wall_s": 1.5, "frames": 81},
        artifacts={"animated": Path("/tmp/a.mp4")},
    )
    s = r.to_json()
    r2 = PhaseResult.from_json(s)
    assert r2.status == "ok"
    assert r2.stats == r.stats
    assert r2.artifacts["animated"] == Path("/tmp/a.mp4")
    assert r2.error is None


def test_fail_roundtrip() -> None:
    r = PhaseResult.fail(
        error_class=ErrorClass.MODEL_OOM,
        message="oom",
        retryable=True,
        stderr_tail="CUDA out of memory",
    )
    s = r.to_json()
    r2 = PhaseResult.from_json(s)
    assert r2.status == "error"
    assert r2.error is not None
    assert r2.error.error_class == ErrorClass.MODEL_OOM
    assert r2.error.retryable is True
    assert r2.error.stderr_tail == "CUDA out of memory"
