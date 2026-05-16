"""Phase 5: format to 1080x1920 vertical Instagram Reels mp4."""
from __future__ import annotations

import time

from ...core import keys as K
from ...core.errors import classify
from ...core.external_tool import run_tool
from ...core.result import PhaseContext, PhaseResult
from ...core.tools.ffmpeg import FFMPEG


def _pick_source_filename(ctx: PhaseContext) -> str:
    """Most-finished video available, in priority order."""
    work = ctx.work_dir
    for fname in (K.ANIMATED_W_AUDIO, K.ANIMATED_60FPS_FACE, K.ANIMATED_60FPS, K.ANIMATED):
        if (work / fname).exists():
            return fname
    return K.ANIMATED  # default — animate phase always produces this


def _build_filter(strategy: str, w: int, h: int) -> str:
    if strategy == "letterbox":
        # Scale longest edge to W, pad to WxH, center vertically.
        return (
            f"scale=w={w}:h=-2:flags=lanczos,"
            f"pad={w}:{h}:0:(oh-ih)/2:black"
        )
    # pillarbox: fill height, crop sides
    return (
        f"scale=w=-2:h={h}:flags=lanczos,"
        f"crop={w}:{h}:(iw-{w})/2:0"
    )


class ReelsFormatPhase:
    name: str = "reels_format"
    timeout_s: int = 120

    def run(self, ctx: PhaseContext) -> PhaseResult:
        t0 = time.time()
        try:
            src_name = _pick_source_filename(ctx)
            src = ctx.work_dir / src_name
            dst = ctx.work_dir / K.FINAL
            w = ctx.manifest.output.width
            h = ctx.manifest.output.height
            vf = _build_filter(ctx.manifest.output.format_strategy, w, h)
            args = [
                "-y",
                "-i",
                str(src),
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(dst),
            ]
            # If the source has no audio, ffmpeg's `-c:a copy` errors with
            # "Output file does not contain any stream". ANIMATED_60FPS and
            # ANIMATED_60FPS_FACE are also audio-less (interp + face_restore
            # only re-encode video). audio_attach is the only phase that
            # produces an audio-bearing mp4.
            if src_name in (K.ANIMATED, K.ANIMATED_60FPS, K.ANIMATED_60FPS_FACE):
                args = [a for a in args if a not in ("-c:a", "copy")]
                args.append("-an")
            run_tool(FFMPEG, args)
            return PhaseResult.ok(
                stats={"wall_s": round(time.time() - t0, 2), "source": src_name},
                artifacts={"final": dst},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
                stderr_tail=info.stderr_tail,
            )
