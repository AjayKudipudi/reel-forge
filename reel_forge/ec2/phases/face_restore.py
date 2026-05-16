"""Phase 3.5: GFPGAN per-frame face restoration with NATURAL-LOOK blend.

Generated faces are small (~50px wide in source 576×1024) and look soft
after Lanczos upscale to 1080×1920. GFPGAN applies a StyleGAN2 face prior
to sharpen facial features, while leaving the rest of the image alone.

CRITICAL FIX (2026-05-15 audit): GFPGAN's `weight` kwarg is INERT in
`arch='clean'` — the parameter falls into `**kwargs` in
`gfpganv1_clean_arch.py::forward` and is never used. Specifying
`weight=0.5` was a no-op; we were running GFPGAN at FULL strength on
every frame. That's the cause of "too sharpened / plasticky / doesn't
look like real video" complaints.

The real strength control is a manual alpha-blend on the OUTPUT:
    out = alpha * restored + (1 - alpha) * original
Lower alpha = more natural / closer to the model's raw generation.
We default to ALPHA=0.30 (community sweet spot for video; AUTOMATIC1111's
"GFPGAN visibility" slider default range is 0.3-0.5).

Plus two more naturalness fixes:
1. Monkey-patch `randomize_noise=False` on the StyleGAN2 forward so the
   per-frame noise injection that causes temporal flicker is disabled.
2. Sharpness gate: skip restoration entirely on frames where the face
   crop is already sharp enough (Laplacian variance > threshold). For
   already-clear frames the lossy upsample→restore→downsample round-trip
   only adds artifacts.

References:
- gfpgan/archs/gfpganv1_clean_arch.py:277 (weight swallowed by **kwargs)
- A1111 webui GFPGAN visibility = alpha blend
  (github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/5257)
- Temporal stability discussion: arxiv.org/html/2410.11828v1
- GFPGAN issues #533 (open, no built-in temporal stable mode), #639
  (Laplacian sharpness gating)

GFPGAN does NOT re-open closed eyes. That's an upstream generation issue
(see prompt + sample_guide_scale).

Apache 2.0 license, ships on PyPI as `gfpgan==1.3.8`. Weights at
/opt/insta-influencer/gfpgan-weights/GFPGANv1.4.pth.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ...core import keys as K
from ...core.errors import classify
from ...core.external_tool import run_tool
from ...core.result import PhaseContext, PhaseResult
from ...core.tools.ffmpeg import FFMPEG

GFPGAN_WEIGHTS = Path("/opt/insta-influencer/gfpgan-weights/GFPGANv1.4.pth")

# Naturalness tunables.
# ALPHA: lower = more natural / closer to model's raw frame. 0.30 is the
# A1111-default-range midpoint. If still too plasticky, drop to 0.20 or
# 0.15. If face is too soft, raise to 0.40-0.50.
ALPHA = 0.30
# LAPLACIAN_SKIP_THRESHOLD: if the source face crop's Laplacian variance
# exceeds this, skip restoration. ~120 is the community-suggested floor
# (face is "clear enough"); raise to make restoration more aggressive.
LAPLACIAN_SKIP_THRESHOLD = 120.0


def _gfpgan_available() -> tuple[bool, str]:
    """Return (available, reason). Reason is empty when available."""
    if not GFPGAN_WEIGHTS.exists():
        return False, f"weights missing at {GFPGAN_WEIGHTS}"
    try:
        from gfpgan import GFPGANer  # noqa: F401
    except ImportError as exc:
        return False, f"gfpgan import failed: {exc}"
    return True, ""


def _build_restorer() -> Any:
    """Construct GFPGANer + monkey-patch StyleGAN2 forward to disable
    `randomize_noise`. The default `randomize_noise=True` injects fresh
    Gaussian noise on every call, producing per-frame variation in skin
    micro-details — which reads as flicker across video frames. There's
    no public API to disable it (the GFPGANer wrapper doesn't expose it
    and arch='clean' forward defaults to True), so we replace the forward
    method on the loaded module.
    """
    from gfpgan import GFPGANer

    restorer = GFPGANer(
        model_path=str(GFPGAN_WEIGHTS),
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    _orig_forward = restorer.gfpgan.forward

    def _stable_forward(
        x: Any,
        return_latents: bool = False,
        return_rgb: bool = True,
        randomize_noise: bool = False,  # forced off — see docstring above
        **kwargs: Any,
    ) -> Any:
        return _orig_forward(
            x,
            return_latents=return_latents,
            return_rgb=return_rgb,
            randomize_noise=False,
        )

    restorer.gfpgan.forward = _stable_forward
    return restorer


def _restore_video(src: Path, dst: Path, tmp_dir: Path) -> dict[str, Any]:
    """Decode src → PNG frames, GFPGAN-restore each frame with natural
    alpha-blend + sharpness gate + temporal-stable noise, re-encode to
    dst. Returns stats dict.
    """
    import cv2

    frames_dir = tmp_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # 1. decode to PNG
    run_tool(
        FFMPEG,
        [
            "-hide_banner", "-loglevel", "error",
            "-y", "-i", str(src),
            "-pix_fmt", "rgb24",
            str(frames_dir / "%05d.png"),
        ],
    )

    restorer = _build_restorer()

    pngs = sorted(frames_dir.glob("*.png"))
    restored_count = 0
    skipped_no_face = 0
    skipped_sharp = 0
    for p in pngs:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        cropped_faces, _restored_faces, restored_img = restorer.enhance(
            bgr,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
            # weight=ALPHA  <-- OMITTED. weight is inert in arch='clean'
            # (gfpganv1_clean_arch.py:277 swallows it via **kwargs). We
            # do the blend ourselves below.
        )
        if restored_img is None or not cropped_faces:
            # No face detected this frame — keep the original.
            skipped_no_face += 1
            continue

        # Sharpness gate: skip restoration when face is already clear.
        # The crop is at 512×512 face resolution from facexlib.
        src_crop = cropped_faces[0]
        sharpness = cv2.Laplacian(src_crop, cv2.CV_64F).var()
        if sharpness > LAPLACIAN_SKIP_THRESHOLD:
            skipped_sharp += 1
            continue

        # Manual alpha blend. This is the REAL strength control — the
        # GFPGANer `weight` parameter is a no-op (see module docstring).
        blended = cv2.addWeighted(restored_img, ALPHA, bgr, 1.0 - ALPHA, 0)
        cv2.imwrite(str(p), blended)
        restored_count += 1

    # 3. re-encode at same fps as src
    import subprocess as _sp
    res = _sp.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(src)],
        check=True, capture_output=True, text=True, timeout=10,
    )
    fps_str = res.stdout.strip() or "60/1"
    run_tool(
        FFMPEG,
        [
            "-hide_banner", "-loglevel", "error",
            "-y",
            "-framerate", fps_str,
            "-i", str(frames_dir / "%05d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(dst),
        ],
    )
    return {
        "frames_total": len(pngs),
        "frames_restored": restored_count,
        "frames_skipped_no_face": skipped_no_face,
        "frames_skipped_already_sharp": skipped_sharp,
        "alpha_blend": ALPHA,
        "laplacian_skip_threshold": LAPLACIAN_SKIP_THRESHOLD,
        "randomize_noise": False,
    }


class FaceRestorePhase:
    name: str = "face_restore"
    timeout_s: int = 1800  # 30 min

    def run(self, ctx: PhaseContext) -> PhaseResult:
        t0 = time.time()
        try:
            src = ctx.work_dir / K.ANIMATED_60FPS
            dst = ctx.work_dir / K.ANIMATED_60FPS_FACE
            if not src.exists():
                return PhaseResult.ok(
                    stats={
                        "wall_s": round(time.time() - t0, 2),
                        "skipped": True,
                        "reason": "no animated_60fps source",
                    },
                    artifacts={},
                )

            available, reason = _gfpgan_available()
            if not available:
                import shutil
                shutil.copy2(src, dst)
                return PhaseResult.ok(
                    stats={
                        "wall_s": round(time.time() - t0, 2),
                        "skipped": True,
                        "reason": reason,
                    },
                    artifacts={"animated_60fps_face": dst},
                )

            tmp_dir = ctx.work_dir / "_gfpgan_tmp"
            stats = _restore_video(src, dst, tmp_dir)
            stats["wall_s"] = round(time.time() - t0, 2)
            stats["skipped"] = False
            stats["model"] = "GFPGANv1.4"
            return PhaseResult.ok(
                stats=stats,
                artifacts={"animated_60fps_face": dst},
            )
        except Exception as exc:
            info = classify(exc)
            return PhaseResult.fail(
                error_class=info.error_class,
                message=info.message,
                retryable=info.retryable,
            )
