"""Protocols for swappable model implementations.

Real impls live in steadydancer.py + dwpose.py and require the [ec2] extras.
Fake impls (fake.py) enable GPU-free local development.

Note: the upstream SteadyDancer toolchain is CLI-driven, not Python-API
driven. `pose_align.py` outputs a video file (aligned pose frames), and
`generate_dancer.py` consumes a directory containing that pose video.
So our Protocol passes Paths, not tensors.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@runtime_checkable
class AnimationModel(Protocol):
    name: str

    def load(self, *, quant: str) -> None: ...

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
        """Generate the animation. Returns the path to the produced mp4.

        `pose_dir` and `pose_neg_dir` are EXPLICIT (passed by the phase)
        rather than derived inside the wrapper — this lets the animate
        phase build per-chunk pose directories for chunked generation
        without the wrapper having to know about chunking.
        """
        ...


@runtime_checkable
class PoseExtractor(Protocol):
    name: str

    def load(self) -> None: ...

    def extract_aligned(
        self,
        *,
        reference_image: Image.Image,
        reference_image_path: Path,
        driving_video: Path,
        work_dir: Path,
    ) -> dict[str, Any]:
        """Returns a dict with keys:
        - pose_dir: Path — directory the animate phase will pass to generate_dancer
        - pose_video: Path — aligned pose video produced by upstream
        - reference_size: (w, h)
        - driving_size: (w, h)
        - confidence: float
        - overlay_path: Path | None
        """
        ...
