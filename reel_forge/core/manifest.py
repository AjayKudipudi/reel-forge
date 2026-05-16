"""Pydantic schema for a job manifest.

Discriminated unions on `reference_source.type` (url|local) and
`background.mode` (from_photo|replace). Per-job overrides for everything
that's tunable; defaults seeded by `prepare/job_manifest.build_manifest`.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, HttpUrl

# ── reference_source ────────────────────────────────────────────────────


class ReferenceURL(BaseModel):
    type: Literal["url"] = "url"
    url: HttpUrl
    staged_path: Path
    sha256: str


class ReferenceLocal(BaseModel):
    type: Literal["local"] = "local"
    original_path: Path
    staged_path: Path
    sha256: str


ReferenceSource = Annotated[
    ReferenceURL | ReferenceLocal,
    Field(discriminator="type"),
]


# ── background ──────────────────────────────────────────────────────────


class BackgroundFromPhoto(BaseModel):
    mode: Literal["from_photo"] = "from_photo"


class BackgroundReplace(BaseModel):
    mode: Literal["replace"] = "replace"
    replacement_path: Path
    matte_model: Literal["birefnet", "sam2", "rmbg"] = "birefnet"


Background = Annotated[
    BackgroundFromPhoto | BackgroundReplace,
    Field(discriminator="mode"),
]


# ── nested specs ────────────────────────────────────────────────────────


class ModelConfig(BaseModel):
    quant: Literal["fp16", "gguf-q4-s", "gguf-q4-m", "gguf-q5-m", "gguf-q6"] = "gguf-q5-m"
    seed: int = 42


class OutputSpec(BaseModel):
    """Per-job output knobs. Mutable by the orchestrator (e.g. OOM-retry
    shrinks num_frames). Never read from CONFIG at phase runtime — always
    from the manifest, which is the per-job source of truth."""

    width: int = 1080
    height: int = 1920
    fps: int = 24
    # Per-clip frame count. SteadyDancer-14B is trained on 81 frames (4n+1
    # constraint from VAE temporal stride). To produce videos longer than
    # ~5 sec we run animate `num_clips` times within the same spot
    # instance and ffmpeg-concat — each clip uses the previous clip's last
    # frame as its first-frame condition (the paper author's recommended
    # chaining approach from MCG-NJU/SteadyDancer issue #17).
    num_frames: int = 81
    num_clips: int = Field(default=1, ge=1, le=10)
    keep_reference_audio: bool = True
    format_strategy: Literal["letterbox", "pillarbox"] = "letterbox"
    # Enable ffmpeg minterpolate (mci) 16fps -> 30fps in interp phase. Model
    # outputs at sample_fps=16 which feels stuttery / slow-motion on Reels;
    # 30fps is Reels native and reads as smooth natural motion.
    frame_interp: bool = True


# ── top-level Manifest ──────────────────────────────────────────────────


class Manifest(BaseModel):
    schema_version: int = 1
    job_id: str = Field(min_length=12, max_length=12)
    created_at: datetime
    reference_source: ReferenceSource
    photo_path: Path
    photo_sha256: str
    background: Background = Field(default_factory=BackgroundFromPhoto)
    prompt: str
    negative_prompt: str | None = None
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputSpec = Field(default_factory=OutputSpec)
    tags: list[str] = Field(default_factory=list)


# ── _pending.json queue ─────────────────────────────────────────────────


class PendingQueue(BaseModel):
    schema_version: int = 1
    job_ids: list[str] = Field(default_factory=list)
    enqueued_at: datetime
