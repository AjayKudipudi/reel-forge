"""LocalObjectStore happy + edge."""
from __future__ import annotations

from pathlib import Path

import pytest

from reel_forge.core.storage import LocalObjectStore


def test_upload_download_roundtrip(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path / "store")
    src = tmp_path / "src.txt"
    src.write_text("hello")
    store.upload(src, "a/b/c.txt")
    assert store.exists("a/b/c.txt")
    out = tmp_path / "out.txt"
    store.download("a/b/c.txt", out)
    assert out.read_text() == "hello"


def test_upload_atomic_replaces(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path / "store")
    a = tmp_path / "a.txt"
    a.write_text("v1")
    store.upload_atomic(a, "k.txt")
    a.write_text("v2")
    store.upload_atomic(a, "k.txt")
    out = tmp_path / "out.txt"
    store.download("k.txt", out)
    assert out.read_text() == "v2"


def test_list_walks_files(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path / "store")
    for i in range(3):
        p = tmp_path / f"{i}.txt"
        p.write_text(str(i))
        store.upload(p, f"jobs/abc/file{i}.txt")
    keys = list(store.list("jobs/abc"))
    assert len(keys) == 3


def test_path_escape_blocked(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path / "store")
    with pytest.raises(ValueError):
        store.exists("../escape.txt")


def test_delete(tmp_path: Path) -> None:
    store = LocalObjectStore(root=tmp_path / "store")
    p = tmp_path / "x.txt"
    p.write_text("x")
    store.upload(p, "k.txt")
    store.delete("k.txt")
    assert not store.exists("k.txt")
