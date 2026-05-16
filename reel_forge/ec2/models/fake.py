"""Fake model + pose extractor for GPU-free local development & tests.

Both impls are deterministic given a seed.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from PIL import Image


class FakeAnimationModel:
    name: str = "fake-animation"

    def load(self, *, quant: str) -> None:
        self._quant = quant

    def animate(
        self,
        *,
        reference_image_path: Path,
        pose_dir: Path,
        pose_neg_dir: Path,
        prompt: str,
        negative_prompt: str,
        num_frames: int,
        fps: int,
        seed: int,
        output_path: Path,
        progress_cb: Callable[[str], None],
    ) -> Path:
        _ = pose_neg_dir  # accepted for protocol parity; fake doesn't use it
        rng = np.random.default_rng(seed)
        ref_arr = np.array(Image.open(reference_image_path).convert("RGB"), dtype=np.uint8)
        h, w, _ = ref_arr.shape
        frames = rng.integers(0, 255, (num_frames, h, w, 3), dtype=np.uint8)
        frames[0] = ref_arr  # first-frame preservation
        for i in range(num_frames):
            progress_cb(f"frame {i + 1}/{num_frames}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(output_path, frames, fps=fps, codec="libx264")
        return output_path


class FakePoseExtractor:
    name: str = "fake-pose"

    def load(self) -> None:
        pass

    def extract_aligned(
        self,
        *,
        reference_image: Image.Image,
        reference_image_path: Path,
        driving_video: Path,
        work_dir: Path,
    ) -> dict[str, Any]:
        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_video = pose_dir / "aligned_pose.mp4"
        gray = np.full((1, 256, 256, 3), 64, dtype=np.uint8)
        frames = np.tile(gray, (24, 1, 1, 1))
        iio.imwrite(pose_video, frames, fps=24, codec="libx264")
        return {
            "pose_dir": pose_dir,
            "pose_video": pose_video,
            "reference_size": reference_image.size,
            "driving_size": (1920, 1080),
            "confidence": 0.95,
            "overlay_path": None,
        }
