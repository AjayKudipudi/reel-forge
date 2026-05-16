"""Phase 4 (conditional): mux reference Reel audio onto the animated video."""
from __future__ import annotations

import time

from ...core import keys as K
from ...core.errors import classify
from ...core.external_tool import run_tool
from ...core.result import PhaseContext, PhaseResult
from ...core.tools.ffmpeg import FFMPEG


class AudioAttachPhase:
    name: str = "audio_attach"
    timeout_s: int = 120

    def run(self, ctx: PhaseContext) -> PhaseResult:
        t0 = time.time()
        try:
            # Source preference (most-finished first):
            #   1. animated_60fps_face.mp4 — interp + GFPGAN face restoration
            #   2. animated_60fps.mp4      — interp only
            #   3. animated.mp4            — raw 16fps generation
            src_face = ctx.work_dir / K.ANIMATED_60FPS_FACE
            src_60 = ctx.work_dir / K.ANIMATED_60FPS
            if src_face.exists():
                src = src_face
            elif src_60.exists():
                src = src_60
            else:
                src = ctx.work_dir / K.ANIMATED
            ref = ctx.work_dir / K.REFERENCE_VIDEO
            dst = ctx.work_dir / K.ANIMATED_W_AUDIO
            run_tool(
                FFMPEG,
                [
                    "-y",
                    "-i",
                    str(src),
                    "-i",
                    str(ref),
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-shortest",
                    "-movflags",
                    "+faststart",
                    str(dst),
                ],
            )
            return PhaseResult.ok(
                stats={"wall_s": round(time.time() - t0, 2)},
                artifacts={"animated_with_audio": dst},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
                stderr_tail=info.stderr_tail,
            )
