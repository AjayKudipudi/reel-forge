"""Real DWPose-based pose extractor — subprocess wrapper around upstream.

Upstream `preprocess/pose_align.py` is CLI-driven (no clean Python API).
Args we use (verified against MCG-NJU/SteadyDancer main):
  --imgfn_refer <photo>
  --vidfn <driving video>
  --outfn_align_pose_video <output aligned pose video>
  --outfn <visualization overlay>
  --max_frame 300         # cap frames considered
  --align_frame 0         # use frame 0 of driving video to align scale

mmcv/mmpose/mmdet are required (built into the AMI).

FPS normalization: SteadyDancer outputs at a fixed 16 fps. If we feed 30
or 60 fps source at native fps, 81 pose frames cover less real time than
the model plays them back — producing slow-motion output. We resample
the driving video to 16 fps before invoking pose_align.py so the pose
track has the temporal density the model expects.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import structlog
from PIL import Image

from ...core.errors import ErrorClass, PoseNoPerson, ToolFailed
from ...core.external_tool import ToolSpec, run_tool
from ...core.tools.ffmpeg import FFMPEG

UPSTREAM_REPO_DIR = Path("/opt/insta-influencer/third_party/SteadyDancer")
log = structlog.get_logger(__name__)


def _classify_pose_align(_rc: int, _out: str, err: str) -> ErrorClass:
    e = err.lower()
    if any(s in e for s in ("no person", "no detection", "empty", "0 keypoints")):
        return ErrorClass.POSE_EXTRACTION_NO_PERSON
    return ErrorClass.UNKNOWN


# Upstream pose_align.py is run from the SteadyDancer repo root (it uses
# os.path.dirname(__file__) for default config/ckpt paths).
POSE_ALIGN_TOOL = ToolSpec(
    name="pose_align.py",
    binary=sys.executable,  # use whichever python invoked us
    timeout_s=900,
    classifier=_classify_pose_align,
)


class DwPoseExtractor:
    name: str = "dwpose-aligned"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        # Heavy mmcv/mmpose imports happen inside the subprocess; nothing
        # to load in the parent process.
        self._loaded = True

    def extract_aligned(
        self,
        *,
        reference_image: Image.Image,
        reference_image_path: Path,
        driving_video: Path,
        work_dir: Path,
    ) -> dict[str, Any]:
        self.load()
        pose_dir = work_dir / "pose"
        pose_dir.mkdir(parents=True, exist_ok=True)
        pose_video = pose_dir / "aligned_pose.mp4"
        overlay_path = work_dir / "pose_overlay.mp4"

        # Resample driving video to 16 fps. The model outputs at fixed 16 fps;
        # feeding 30/60 fps source at native fps makes 81 pose frames cover
        # less real time than the model plays them back, producing slow-motion
        # output. -vsync cfr forces constant frame rate via duplication/drop
        # (vs the default vfr which would preserve irregular timestamps and
        # break frame-index lookup downstream). -an strips audio; the
        # audio_attach phase reads from the ORIGINAL reference.mp4, not this
        # resampled copy. Use -vsync cfr instead of -fps_mode cfr for ffmpeg
        # 4.x compatibility (the AMI ships ffmpeg 4.x; -fps_mode is 5.0+).
        normalized = work_dir / "driving_16fps.mp4"
        run_tool(
            FFMPEG,
            [
                "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(driving_video),
                "-r", "16",
                "-vsync", "cfr",
                "-an",
                str(normalized),
            ],
        )
        driving_video = normalized

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{UPSTREAM_REPO_DIR}:{UPSTREAM_REPO_DIR / 'preprocess'}"

        t0 = time.time()
        try:
            run_tool(
                POSE_ALIGN_TOOL,
                [
                    str(UPSTREAM_REPO_DIR / "preprocess" / "pose_align.py"),
                    "--imgfn_refer", str(reference_image_path),
                    "--vidfn", str(driving_video),
                    "--outfn_align_pose_video", str(pose_video),
                    "--outfn", str(overlay_path),
                    # 500 frames @ 16 fps = ~31 sec of source pose, enough for
                    # up to 6 chunks of 81 frames (chunked animate for long
                    # outputs). Capped here; if the source video is shorter,
                    # pose_align.py uses the source length naturally.
                    "--max_frame", "500",
                    "--align_frame", "0",
                ],
                cwd=UPSTREAM_REPO_DIR,
                env=env,
            )
        except ToolFailed as exc:
            if exc.error_class == ErrorClass.POSE_EXTRACTION_NO_PERSON:
                raise PoseNoPerson(exc.result.stderr[-400:]) from exc
            raise

        if not pose_video.exists():
            raise PoseNoPerson(
                f"pose_align.py exited 0 but produced no output at {pose_video}"
            )

        # generate_dancer.py's --cond_pos_folder expects per-frame JPGs named
        # 0000.jpg, 0001.jpg, ... — not the .mp4. Extract here so the consumer
        # doesn't have to.
        run_tool(
            FFMPEG,
            [
                "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(pose_video),
                "-start_number", "0",
                "-q:v", "2",
                str(pose_dir / "%04d.jpg"),
            ],
        )

        # --- Negative (augmented) pose for classifier-free guidance ---
        # Upstream README's inference pipeline runs pose_align_withdiffaug.py
        # to produce a differentially-augmented pose track (random offset
        # +/-0.2, scale 0.7-1.3, aspect-ratio 0.6-1.4). That track is the
        # NEGATIVE pose condition consumed by generate_dancer.py's
        # --cond_neg_folder. The CFG between exact-pose (positive) and
        # jittered-pose (negative) is what lets condition_guide_scale enforce
        # pose precision AND suppress pose-derived artifacts like multiple
        # people leaking into the frame. Without it (the prior code shipped
        # blank black JPGs as the negative), CFG had no informative signal
        # to subtract, weakening pose adherence and identity preservation.
        pose_neg_dir = work_dir / "pose_neg"
        pose_neg_dir.mkdir(parents=True, exist_ok=True)
        pose_neg_base = pose_neg_dir / "aligned_pose_neg.mp4"
        run_tool(
            POSE_ALIGN_TOOL,
            [
                str(UPSTREAM_REPO_DIR / "preprocess" / "pose_align_withdiffaug.py"),
                "--imgfn_refer", str(reference_image_path),
                "--vidfn", str(driving_video),
                "--outfn_align_pose_video", str(pose_neg_base),
                "--outfn", str(work_dir / "pose_overlay_neg.mp4"),
                "--max_frame", "300",
                "--align_frame", "0",
            ],
            cwd=UPSTREAM_REPO_DIR,
            env=env,
        )
        # pose_align_withdiffaug.py writes <outfn_align_pose_video>_aug.mp4
        # alongside the non-aug version (see upstream
        # preprocess/pose_align_withdiffaug.py:622). The _aug.mp4 is what
        # generate_dancer.py wants as the negative condition.
        pose_neg_aug_video = pose_neg_dir / "aligned_pose_neg_aug.mp4"
        if not pose_neg_aug_video.exists():
            raise PoseNoPerson(
                f"pose_align_withdiffaug.py exited 0 but did not produce "
                f"{pose_neg_aug_video}"
            )
        run_tool(
            FFMPEG,
            [
                "-hide_banner", "-loglevel", "error",
                "-y", "-i", str(pose_neg_aug_video),
                "-start_number", "0",
                "-q:v", "2",
                str(pose_neg_dir / "%04d.jpg"),
            ],
        )

        log.info(
            "pose.extracted",
            pose_video=str(pose_video),
            pose_neg_video=str(pose_neg_aug_video),
            wall_s=round(time.time() - t0, 2),
            ref_size=reference_image.size,
        )
        return {
            "pose_dir": pose_dir,
            "pose_neg_dir": pose_neg_dir,
            "pose_video": pose_video,
            "pose_neg_video": pose_neg_aug_video,
            "reference_size": reference_image.size,
            "driving_size": (0, 0),  # could probe via ffprobe if needed
            "confidence": 1.0,        # upstream doesn't expose a confidence score
            "overlay_path": overlay_path if overlay_path.exists() else None,
        }


# Avoid unused-import warnings for stdlib modules kept for future use.
_ = subprocess
