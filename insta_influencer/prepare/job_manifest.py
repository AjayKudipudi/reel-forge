"""Build a Pydantic Manifest from CLI args + Config.DEFAULT_* seeds."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from ..core.manifest import (
    Background,
    BackgroundFromPhoto,
    BackgroundReplace,
    Manifest,
    ModelConfig,
    OutputSpec,
    ReferenceLocal,
    ReferenceSource,
    ReferenceURL,
)
from ..data.prompts.animate import NEGATIVE as DEFAULT_NEGATIVE
from ..data.prompts.animate import POSITIVE as DEFAULT_POSITIVE
from .job_id import derive_job_id
from .photo_prep import PreparedPhoto
from .reel_fetcher import ReferenceVideo


def build_reference_source(ref: ReferenceVideo) -> ReferenceSource:
    if ref.source_kind == "url":
        assert ref.original_url is not None
        return ReferenceURL(
            type="url",
            url=ref.original_url,
            staged_path=ref.staged_path,
            sha256=ref.sha256,
        )
    assert ref.original_local_path is not None
    return ReferenceLocal(
        type="local",
        original_path=ref.original_local_path,
        staged_path=ref.staged_path,
        sha256=ref.sha256,
    )


def build_background(
    *,
    replacement_path: Path | None,
    matte_model: str,
) -> Background:
    if replacement_path is None:
        return BackgroundFromPhoto()
    return BackgroundReplace(
        replacement_path=replacement_path,
        matte_model=matte_model,
    )


def build_manifest(
    *,
    cfg: Config,
    reference: ReferenceVideo,
    photo: PreparedPhoto,
    prompt: str | None,
    seed: int | None,
    background: Background,
    tags: list[str] | None = None,
    num_clips: int = 1,
) -> Manifest:
    output = OutputSpec(
        width=cfg.REELS_OUTPUT_W,
        height=cfg.REELS_OUTPUT_H,
        fps=cfg.DEFAULT_OUTPUT_FPS,
        num_frames=cfg.DEFAULT_OUTPUT_FRAMES,
        num_clips=num_clips,
        keep_reference_audio=cfg.DEFAULT_KEEP_REFERENCE_AUDIO,
        format_strategy=cfg.DEFAULT_REELS_FORMAT_STRATEGY,
        frame_interp=cfg.DEFAULT_FRAME_INTERP,
    )
    model = ModelConfig(
        quant=cfg.DEFAULT_MODEL_QUANT,
        seed=seed if seed is not None else cfg.DEFAULT_SEED,
    )
    final_prompt = prompt or DEFAULT_POSITIVE
    bg_mode = background.mode
    job_id = derive_job_id(
        photo_sha256=photo.sha256,
        reference_sha256=reference.sha256,
        model_quant=model.quant,
        seed=model.seed,
        output_w=output.width,
        output_h=output.height,
        prompt=final_prompt,
        background_mode=bg_mode,
    )
    return Manifest(
        schema_version=1,
        job_id=job_id,
        created_at=datetime.now(UTC),
        reference_source=build_reference_source(reference),
        photo_path=photo.staged_path,
        photo_sha256=photo.sha256,
        background=background,
        prompt=final_prompt,
        negative_prompt=DEFAULT_NEGATIVE,
        model=model,
        output=output,
        tags=tags or [],
    )
