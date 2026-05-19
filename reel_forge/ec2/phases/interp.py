"""Phase 3: ML-based frame interpolation from the model's native 16 fps to
the source reel's fps (clamped to [16, 60]).

Default backend: **Practical-RIFE v4.25** (MIT licensed, MCG Megvii).
RIFE replaced `ffmpeg minterpolate=mci` after the 2026-05-14 §5.16 review:
mci hallucinated colored blobs where it couldn't resolve fast hand motion,
and the `mi_mode=blend` fallback produced visible motion blur. RIFE
synthesizes intermediate frames via a learned optical-flow + image-synthesis
network — temporally coherent and artifact-free.

Target fps: tracks source reel's native fps so output feels native to the
user's input (30 fps Reel in → 30 fps out; 60 fps source → 60 fps out).
Clamped to [16, 60]: <16 raises to 16 (Instagram judders below); >60 caps
at 60 (RIFE multipliers + file size stay sane). Lower multi vs the prior
hardcoded 60 means fewer fast-hand-motion artifacts when source is 30 fps.

If RIFE is not installed at the expected path
(`/opt/insta-influencer/third_party/Practical-RIFE`), the phase falls back
to `ffmpeg minterpolate=mi_mode=blend` so the pipeline never hard-fails on
a missing optional dep.

When the animate phase produced per-chunk mp4s (num_clips>=2), each chunk
is interpolated independently and the 60fps chunks are concatenated. This
keeps the interpolator from seeing the chunk boundary at all — the boundary
becomes a single hard cut at 60fps (1/60s, sub-perceptual) instead of a
smeared transition between chunk N's last frame and chunk N+1's re-anchored
first frame.
"""
from __future__ import annotations

import math
import shutil
import subprocess
import time
from pathlib import Path

from ...core import keys as K
from ...core.errors import classify
from ...core.external_tool import run_tool
from ...core.result import PhaseContext, PhaseResult
from ...core.tools.ffmpeg import FFMPEG

# AMI / cloud-init expected layout.
RIFE_REPO_DIR = Path("/opt/insta-influencer/third_party/Practical-RIFE")
RIFE_WEIGHTS_DIR = Path("/opt/insta-influencer/rife-weights")
# Practical-RIFE's inference_video.py reads weights from a `train_log/`
# subdir under its repo. cloud-init symlinks RIFE_WEIGHTS_DIR -> repo/train_log.


def _detect_target_fps(ctx: PhaseContext) -> int:
    """Read the source reel's native fps via ffprobe; return it clamped to [16, 60].

    The source reel is at <work_dir>/reference.mp4 (synced from S3 by the
    orchestrator before phases run). If ffprobe fails or returns nonsense,
    fall back to 30 — Instagram Reels native rate, the most common case.

    Clamp rationale: the model outputs at fixed 16 fps. RIFE upsamples to
    multi*16 where multi = round(target/16). Target <16 doesn't make sense
    (Instagram judders); target >60 wastes interpolation work because the
    platform downsamples to 30 fps on most playback anyway, and high multi
    values amplify RIFE artifacts on fast hand motion.
    """
    reel = ctx.work_dir / K.REFERENCE_VIDEO
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
             str(reel)],
            check=True, capture_output=True, text=True, timeout=10,
        )
        num, _, den = res.stdout.strip().partition("/")
        src_fps = round(int(num) / int(den or "1"))
    except Exception:
        src_fps = 30
    return min(max(src_fps, 16), 60)


def _rife_available() -> bool:
    """RIFE is usable when both the repo and a train_log/ weights dir exist."""
    return (RIFE_REPO_DIR / "inference_video.py").exists() and (
        (RIFE_REPO_DIR / "train_log").exists()
        or (RIFE_REPO_DIR / "train_log").is_symlink()
    )


def _rife_one(src: Path, dst: Path, target_fps: int) -> None:
    """Run Practical-RIFE's inference_video.py on a single mp4.

    Practical-RIFE writes its output next to the input with a fixed suffix
    pattern like `<basename>_4X_60fps.mp4`. We invoke it then move the
    produced file to `dst`. Uses --fp16 to halve VRAM + ~25% faster.

    Must pass `--multi=N` explicitly. Without it Practical-RIFE defaults to
    `multi=2` regardless of `--fps`, producing only 2x the source frames
    stamped at the target fps → output duration becomes (source_dur * 2 /
    target_fps_ratio), i.e. half as long. With source=16fps, target=60fps,
    we need multi >= ceil(60/16) = 4 so RIFE generates ~64fps of content,
    then --fps=60 resamples down to 60fps with the correct duration.
    """
    multi = max(2, math.ceil(target_fps / 16))  # 30fps -> 2, 60fps -> 4
    cmd = [
        "python",
        str(RIFE_REPO_DIR / "inference_video.py"),
        "--model", str(RIFE_REPO_DIR / "train_log"),
        "--video", str(src),
        "--multi", str(multi),
        "--fps", str(target_fps),
        "--fp16",
    ]
    subprocess.run(cmd, check=True, cwd=str(RIFE_REPO_DIR), timeout=900)
    # Practical-RIFE names output `<stem>_<multi>X_<fps>fps.mp4` and drops
    # it next to the source. Find the newest mp4 in src's dir matching the
    # stem and move it to dst.
    src_stem = src.stem
    candidates = sorted(
        src.parent.glob(f"{src_stem}_*X_{target_fps}fps.mp4"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if not candidates:
        # Fallback: any mp4 in src's parent newer than src itself.
        candidates = sorted(
            (p for p in src.parent.glob("*.mp4")
             if p != src and p.stat().st_mtime > src.stat().st_mtime),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(
            f"RIFE produced no output mp4 for {src} (expected "
            f"{src_stem}_*X_{target_fps}fps.mp4 in {src.parent})",
        )
    shutil.move(str(candidates[0]), str(dst))


def _ffmpeg_blend_one(src: Path, dst: Path, target_fps: int) -> None:
    """Fallback: ffmpeg minterpolate=mi_mode=blend. Produces motion blur on
    fast movement (worse than RIFE) but never hallucinates content.
    """
    run_tool(
        FFMPEG,
        [
            "-y", "-i", str(src),
            "-filter:v", f"minterpolate=fps={target_fps}:mi_mode=blend",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(dst),
        ],
    )


def _interp_one(src: Path, dst: Path, target_fps: int, use_rife: bool) -> str:
    """Return the backend name actually used."""
    # Model output is already 16 fps. If the source reel was also ~16 fps
    # (clamp floor), we have nothing to interpolate — just copy.
    if target_fps <= 16:
        shutil.copy2(src, dst)
        return "passthrough-16fps"
    if use_rife:
        try:
            _rife_one(src, dst, target_fps)
            return "rife-v4.25-fp16"
        except Exception:
            # Fall back to blend so the pipeline never hard-fails.
            _ffmpeg_blend_one(src, dst, target_fps)
            return "rife-failed-fallback-blend"
    _ffmpeg_blend_one(src, dst, target_fps)
    return "ffmpeg-minterpolate-blend"


def _concat_lossless(parts: list[Path], dst: Path, work_dir: Path) -> None:
    """ffmpeg concat-demuxer with -c copy. Lossless when codecs match."""
    listfile = work_dir / "_interp_concat.txt"
    listfile.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    run_tool(
        FFMPEG,
        [
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(listfile),
            "-c", "copy",
            str(dst),
        ],
    )


class InterpPhase:
    name: str = "interp"
    timeout_s: int = 1200  # RIFE on 162 frames at 1080p ≈ 30-90s; padded.

    def run(self, ctx: PhaseContext) -> PhaseResult:
        t0 = time.time()
        try:
            dst = ctx.work_dir / K.ANIMATED_60FPS
            target_fps = _detect_target_fps(ctx)
            use_rife = _rife_available()

            chunk_files = sorted(ctx.work_dir.glob("animated_chunk_*.mp4"))
            backend_used: str
            if len(chunk_files) >= 2:
                interp_outs: list[Path] = []
                backends: list[str] = []
                for i, ch in enumerate(chunk_files):
                    out = ctx.work_dir / f"animated_chunk_{i}_60fps.mp4"
                    backends.append(_interp_one(ch, out, target_fps, use_rife))
                    interp_outs.append(out)
                _concat_lossless(interp_outs, dst, ctx.work_dir)
                # Report the predominant backend.
                backend_used = backends[0] if backends else "unknown"
            else:
                src = ctx.work_dir / K.ANIMATED
                backend_used = _interp_one(src, dst, target_fps, use_rife)

            return PhaseResult.ok(
                stats={
                    "wall_s": round(time.time() - t0, 2),
                    "interp": backend_used,
                    "fps_in": 16, "fps_out": target_fps,
                    "chunks_interpolated": len(chunk_files) if chunk_files else 1,
                    "rife_available": use_rife,
                },
                artifacts={"animated_60fps": dst},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
            )
