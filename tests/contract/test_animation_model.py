"""AnimationModel Protocol — Fake satisfies it; real Steady is GPU-only smoke."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from reel_forge.ec2.models._base import AnimationModel
from reel_forge.ec2.models.factory import get_animation_model
from reel_forge.ec2.models.fake import FakeAnimationModel


def test_fake_model_satisfies_protocol() -> None:
    m = FakeAnimationModel()
    assert isinstance(m, AnimationModel)


def test_factory_returns_fake_when_env_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMATE_FAKE", "1")
    m = get_animation_model()
    assert m.name == "fake-animation"


def test_fake_animation_writes_mp4(tmp_path: Path) -> None:
    m = FakeAnimationModel()
    m.load(quant="gguf-q5-m")
    ref_path = tmp_path / "ref.png"
    Image.new("RGB", (256, 144), color=(50, 100, 150)).save(ref_path)
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    pose_neg_dir = tmp_path / "pose_neg"
    pose_neg_dir.mkdir()
    out = tmp_path / "out.mp4"
    result = m.animate(
        reference_image_path=ref_path,
        pose_dir=pose_dir,
        pose_neg_dir=pose_neg_dir,
        prompt="a person dancing",
        negative_prompt="cartoon",
        num_frames=5,
        fps=24,
        seed=42,
        output_path=out,
        progress_cb=lambda _: None,
    )
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.smoke
def test_real_steadydancer_protocol() -> None:
    if os.getenv("ANIMATE_FAKE") == "1":
        pytest.skip("ANIMATE_FAKE=1 — fake mode")
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch not installed")
    from typing import cast

    from reel_forge.ec2.models.steadydancer import SteadyDancerModel

    m = cast(AnimationModel, SteadyDancerModel())
    assert isinstance(m, AnimationModel)


_ = np  # keep
