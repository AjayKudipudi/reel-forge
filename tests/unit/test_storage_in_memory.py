"""InMemoryObjectStore parity."""
from __future__ import annotations

from pathlib import Path

import pytest

from insta_influencer.core.storage import InMemoryObjectStore


def test_roundtrip(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    src = tmp_path / "x"
    src.write_bytes(b"data")
    store.upload(src, "k")
    assert store.exists("k")
    out = tmp_path / "y"
    store.download("k", out)
    assert out.read_bytes() == b"data"


def test_missing_key_raises(tmp_path: Path) -> None:
    store = InMemoryObjectStore()
    with pytest.raises(FileNotFoundError):
        store.download("nope", tmp_path / "y")
