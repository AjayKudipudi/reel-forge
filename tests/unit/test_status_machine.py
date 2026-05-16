"""Every illegal transition raises IllegalTransition."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from reel_forge.core.errors import IllegalTransition
from reel_forge.core.status import TRANSITIONS, State, StatusManager
from reel_forge.core.storage import InMemoryObjectStore


def _build(tmp_path: Path) -> StatusManager:
    return StatusManager(
        job_id="abc123abc123",
        local_path=tmp_path / "status.json",
        storage=InMemoryObjectStore(),
        s3_key="abc123abc123/status.json",
    )


def test_transitions_table_covers_all_states() -> None:
    for s in State:
        assert s in TRANSITIONS, f"missing {s} from TRANSITIONS"


@pytest.mark.parametrize(
    "from_, to_",
    [(f, t) for f in State for t in State if t not in TRANSITIONS[f] and f != t],
)
def test_illegal_transitions_raise(tmp_path: Path, from_: State, to_: State) -> None:
    s = _build(tmp_path)
    s.status = s.status.model_copy(update={"state": from_, "updated_at": datetime.now(UTC)})
    with pytest.raises(IllegalTransition):
        s.transition(to_)


def test_legal_transitions_pass(tmp_path: Path) -> None:
    s = _build(tmp_path)
    # A representative legal walk
    s.transition(State.PREPARING)
    s.transition(State.PREPARED)
    s.transition(State.UPLOADING)
    s.transition(State.QUEUED)
    s.transition(State.LAUNCHING)
    s.transition(State.PHASE_RUNNING)
    s.transition(State.COMPLETED)
    assert s.status.state == State.COMPLETED


def test_status_persists_atomically(tmp_path: Path) -> None:
    s = _build(tmp_path)
    s.transition(State.PREPARING)
    # Local file exists and parses cleanly.
    assert (tmp_path / "status.json").exists()
    # In-memory store mirrors it.
    assert s.storage.exists("abc123abc123/status.json")
