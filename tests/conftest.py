"""Shared pytest fixtures."""
from __future__ import annotations

import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def _set_required_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterable[None]:
    """Provide minimum required Config env so import doesn't crash."""
    monkeypatch.setenv("HF_TOKEN", "dummy-test-token")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "dummy")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "dummy")
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("ANIMATE_FAKE", "1")
    monkeypatch.setenv("INSTA_SPOT_WATCH", "0")  # don't poll EC2 metadata in tests
    monkeypatch.setenv("LOCAL_STORE_ROOT", str(tmp_path / "store"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "batch"))
    monkeypatch.setenv("ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("EC2_WORK_DIR", str(tmp_path / "ec2_work"))
    # Reset the module-level cached config so each test gets a fresh one.
    import reel_forge.config as cfg_mod
    cfg_mod._cached = None
    yield
    cfg_mod._cached = None


@pytest.fixture
def in_memory_store() -> InMemoryObjectStore:  # noqa: F821
    from reel_forge.core.storage import InMemoryObjectStore
    return InMemoryObjectStore()


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_photo(tmp_path: Path) -> Path:
    """Synthetic 720x1280 photo (9:16 portrait — required by
    photo_prep's aspect-ratio guard, matches Instagram Reels native)."""
    p = tmp_path / "jane.png"
    arr = np.full((1280, 720, 3), 200, dtype=np.uint8)
    Image.fromarray(arr).save(p)
    return p


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """Synthetic 1-second 24-fps mp4."""
    p = tmp_path / "sample.mp4"
    frames = np.tile(np.full((480, 640, 3), 100, dtype=np.uint8)[None], (24, 1, 1, 1))
    iio.imwrite(p, frames, fps=24, codec="libx264")
    return p


@pytest.fixture
def now_utc() -> datetime:
    return datetime.now(UTC)


# Convenience: path adjustments so `python -m reel_forge` resolves in tests.
@pytest.fixture(autouse=True)
def _ensure_pkg_importable() -> Iterable[None]:
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in os.sys.path:
        os.sys.path.insert(0, str(repo))
    yield
