"""PoseExtractor Protocol — fake satisfies it."""
from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest
from PIL import Image

from reel_forge.ec2.models._base import PoseExtractor
from reel_forge.ec2.models.factory import get_pose_extractor
from reel_forge.ec2.models.fake import FakePoseExtractor


def test_fake_extractor_satisfies_protocol() -> None:
    e = FakePoseExtractor()
    assert isinstance(e, PoseExtractor)


def test_factory_returns_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMATE_FAKE", "1")
    e = get_pose_extractor()
    assert e.name == "fake-pose"


def test_fake_extract_writes_video(tmp_path: Path) -> None:
    e = FakePoseExtractor()
    e.load()
    ref = Image.new("RGB", (1024, 576))
    frames = np.zeros((10, 100, 100, 3), dtype=np.uint8)
    driving = tmp_path / "driving.mp4"
    iio.imwrite(driving, frames, fps=24, codec="libx264")
    out = e.extract_aligned(
        reference_image=ref,
        reference_image_path=tmp_path / "ref.png",
        driving_video=driving,
        work_dir=tmp_path,
    )
    assert out["pose_video"].exists()
    assert out["pose_dir"].exists()
    assert out["pose_dir"] == tmp_path / "pose"
    assert out["confidence"] > 0
