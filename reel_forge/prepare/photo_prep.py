"""Photo validation, optional background composite, resize/crop to
SteadyDancer's expected reference dimensions (1024x576)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog
from PIL import Image

from ..core.errors import PhotoInvalid

log = structlog.get_logger(__name__)

# Reels-native portrait 9:16. SteadyDancer's i2v-14B supports 576x1024
# (per SUPPORTED_SIZES in wan/configs/__init__.py). This eliminates the
# black bars that landscape 1024x576 produced after letterboxing into
# 1080x1920 Reels canvas — same aspect ratio means no bars, subject
# fills the full vertical frame.
TARGET_W = 576
TARGET_H = 1024
TARGET_ASPECT = TARGET_W / TARGET_H  # 0.5625 (9:16)

# Reject photos whose aspect ratio is too far from 9:16 portrait.
# Center-crop on an off-aspect input removes important subject content —
# e.g. a landscape input (~16:9) center-cropped to portrait strips the
# sides; a square input loses head AND feet. Failing fast with a clear
# message is better than silently producing a poorly-framed animation.
#
# Tolerance: src_aspect / TARGET_ASPECT must be within (1 - X, 1 + X).
# 0.15 accepts 9:16 (0.5625), 2:3 (0.667), 3:5 (0.6), 10:16 (0.625);
# it rejects 1:1 (1.0), 4:3 (1.33), 16:9 (1.78), and all landscape.
ASPECT_TOLERANCE = 0.15


@dataclass(frozen=True)
class PreparedPhoto:
    staged_path: Path
    sha256: str
    original_path: Path


def prepare_photo(
    src: Path,
    *,
    out_path: Path,
    background_path: Path | None = None,
    matte_model: str = "birefnet",
) -> PreparedPhoto:
    """Resize/crop the photo to TARGET_W x TARGET_H. If background_path is
    given, run matting via the named model and composite the subject onto
    the new background before resizing.

    NOTE: matte models (BiRefNet/SAM2/RMBG) are heavy ML deps deferred to
    the [ec2] extras; if BACKGROUND_REPLACE is True we shell out to the
    appropriate binary. For v1 we default to the no-replace path."""
    from .job_id import file_sha256

    if not src.exists():
        raise PhotoInvalid(f"photo not found: {src}")
    try:
        img = Image.open(src).convert("RGB")
    except Exception as exc:
        raise PhotoInvalid(f"cannot open photo {src}: {exc}") from exc

    _check_aspect(img, src)

    if background_path is not None:
        img = _composite_on_background(img, background_path, matte_model)

    img = _resize_and_center_crop(img, TARGET_W, TARGET_H)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")

    log.info(
        "photo.prepared",
        src=str(src),
        out=str(out_path),
        size=(img.size[0], img.size[1]),
        bg_replace=background_path is not None,
    )
    return PreparedPhoto(
        staged_path=out_path,
        sha256=file_sha256(out_path),
        original_path=src,
    )


def _check_aspect(img: Image.Image, src: Path) -> None:
    """Reject photos whose aspect ratio is too far from 16:9 landscape.

    Symptom this guards against: a portrait input (e.g. 901x1746, aspect
    ~0.52) silently center-cropped to 16:9 strips the top 620px (head)
    and the bottom 620px (feet) of a 1746px-tall photo, leaving only the
    mid-body slice. The animation then shows a waist-only dancer.
    """
    src_w, src_h = img.size
    if src_w <= 0 or src_h <= 0:
        raise PhotoInvalid(f"photo has invalid dimensions {src_w}x{src_h}: {src}")
    src_aspect = src_w / src_h
    ratio = src_aspect / TARGET_ASPECT
    if abs(ratio - 1.0) > ASPECT_TOLERANCE:
        if src_aspect > 1.4:
            orientation = "landscape"
        elif src_aspect > 0.9:
            orientation = "near-square"
        else:
            orientation = "narrower-portrait-than-9:16"
        raise PhotoInvalid(
            f"photo {src.name} is {src_w}x{src_h} ({orientation}, aspect "
            f"{src_aspect:.2f}:1) but the pipeline needs 9:16 portrait "
            f"(aspect ~0.56:1, tolerance +/-{int(ASPECT_TOLERANCE * 100)}%) "
            f"to match Instagram Reels native 1080x1920 format. "
            f"Center-cropping a {orientation} photo would remove subject content. "
            f"Please re-crop the photo to portrait and resubmit. "
            f"Recommended dimensions (any of these): "
            f"1080x1920 (Reels native, BEST), 720x1280, "
            f"{TARGET_W}x{TARGET_H} (model native, smallest acceptable) "
            f"-- anything 9:16 with the subject's full body visible (head to feet).",
        )


def _resize_and_center_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    """Center-crop to aspect, then resize. Preserves the subject if they
    are roughly centered."""
    src_w, src_h = img.size
    target_aspect = w / h
    src_aspect = src_w / src_h
    if src_aspect > target_aspect:
        # too wide — crop horizontally
        new_w = int(src_h * target_aspect)
        x0 = (src_w - new_w) // 2
        img = img.crop((x0, 0, x0 + new_w, src_h))
    else:
        new_h = int(src_w / target_aspect)
        y0 = (src_h - new_h) // 2
        img = img.crop((0, y0, src_w, y0 + new_h))
    return img.resize((w, h), Image.Resampling.LANCZOS)


def _composite_on_background(
    img: Image.Image,
    background_path: Path,
    matte_model: str,
) -> Image.Image:
    """Stub: real BiRefNet/SAM2 invocation lives in ec2 land. For now,
    paste the photo over the background at full alpha — operator can
    refine on EC2 where the matting deps live."""
    bg = Image.open(background_path).convert("RGB")
    bg = _resize_and_center_crop(bg, img.width, img.height)
    # Without a real matte, this is just identity — we still respect the
    # user's stated intent and log a warning so they know to run on EC2
    # for actual replacement.
    log.warning(
        "photo.composite.naive",
        matte_model=matte_model,
        note="background_replace requires ec2 matting deps; using identity composite",
    )
    return bg.convert("RGB") if bg.size == img.size else img
