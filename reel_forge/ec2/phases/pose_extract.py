"""Phase 1: pose extraction.

Reads photo.png + reference.mp4 from work_dir.
Writes a `pose/` directory containing the aligned pose video that the
animate phase will pass to upstream generate_dancer.py via
--cond_pos_folder.
"""
from __future__ import annotations

import time

from PIL import Image

from ...core import keys as K
from ...core.errors import classify
from ...core.result import PhaseContext, PhaseResult
from ...core.seed import seed_everything
from ..models.factory import get_pose_extractor


class PoseExtractPhase:
    name: str = "pose_extract"
    timeout_s: int = 900

    def run(self, ctx: PhaseContext) -> PhaseResult:
        seed_everything(ctx.seed)
        t0 = time.time()
        try:
            extractor = get_pose_extractor()
            extractor.load()
            ref_path = ctx.work_dir / K.PHOTO
            ref = Image.open(ref_path).convert("RGB")
            out = extractor.extract_aligned(
                reference_image=ref,
                reference_image_path=ref_path,
                driving_video=ctx.work_dir / K.REFERENCE_VIDEO,
                work_dir=ctx.work_dir,
            )
            return PhaseResult.ok(
                stats={
                    "wall_s": round(time.time() - t0, 2),
                    "pose_video": str(out["pose_video"]),
                    "confidence": float(out["confidence"]),
                },
                artifacts={"pose_dir": out["pose_dir"], "pose_video": out["pose_video"]},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
                stats={"wall_s": round(time.time() - t0, 2)},
                stderr_tail=info.stderr_tail,
            )
