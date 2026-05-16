"""Phase Protocol — every ec2/phases/*.py implements this."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from .result import PhaseContext, PhaseResult


@runtime_checkable
class Phase(Protocol):
    """Single unit of pipeline work.

    Phases are plain classes. The orchestrator iterates over Phase
    instances and calls `phase.run(ctx)`. Subprocess isolation is an
    implementation detail of the runner, not the Phase contract.
    """

    name: str
    timeout_s: int

    def run(self, ctx: PhaseContext) -> PhaseResult: ...
