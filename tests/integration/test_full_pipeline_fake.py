"""End-to-end pipeline with fakes. STORAGE_BACKEND=local + ANIMATE_FAKE=1.

Runs in <30s on a laptop — exercises the full Phase chain.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from reel_forge.config import get_config
from reel_forge.core import keys as K
from reel_forge.core.manifest import (
    BackgroundFromPhoto,
    Manifest,
    ModelConfig,
    OutputSpec,
    ReferenceLocal,
)
from reel_forge.core.result import PhaseContext
from reel_forge.core.seed import seed_everything
from reel_forge.core.status import State, StatusManager
from reel_forge.core.storage import get_object_store
from reel_forge.ec2.phases.animate import AnimatePhase
from reel_forge.ec2.phases.audio_attach import AudioAttachPhase
from reel_forge.ec2.phases.pose_extract import PoseExtractPhase
from reel_forge.ec2.phases.reels_format import ReelsFormatPhase

log = structlog.get_logger(__name__)


def _make_manifest(work: Path, photo: Path, ref: Path) -> Manifest:
    return Manifest(
        schema_version=1,
        job_id="abc123abc123",
        created_at=datetime.now(UTC),
        reference_source=ReferenceLocal(
            type="local",
            original_path=ref,
            staged_path=ref,
            sha256="a" * 64,
        ),
        photo_path=photo,
        photo_sha256="b" * 64,
        background=BackgroundFromPhoto(),
        prompt="a person dancing",
        model=ModelConfig(seed=42),
        output=OutputSpec(num_frames=8, fps=24, keep_reference_audio=True),
    )


def test_full_pipeline_with_fakes(
    tmp_path: Path,
    sample_photo: Path,
    sample_video: Path,
) -> None:
    cfg = get_config()
    work = tmp_path / "work"
    work.mkdir()
    # Resize photo to SteadyDancer's reference dims so AnimatePhase keeps it.
    from reel_forge.prepare.photo_prep import prepare_photo
    prepared = prepare_photo(sample_photo, out_path=work / K.PHOTO)
    # Stage reference video into the work dir under canonical name.
    (work / K.REFERENCE_VIDEO).write_bytes(sample_video.read_bytes())

    manifest = _make_manifest(work, prepared.staged_path, work / K.REFERENCE_VIDEO)
    (work / K.MANIFEST).write_text(manifest.model_dump_json(indent=2))

    storage = get_object_store(cfg)
    status = StatusManager(
        job_id=manifest.job_id,
        local_path=work / K.STATUS,
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, manifest.job_id),
    )
    # Walk through the legal sequence
    for s in (
        State.PREPARING,
        State.PREPARED,
        State.UPLOADING,
        State.QUEUED,
        State.LAUNCHING,
        State.PHASE_RUNNING,
    ):
        status.transition(s)
    seed_everything(manifest.model.seed)

    ctx = PhaseContext(
        job_id=manifest.job_id,
        work_dir=work,
        storage=storage,
        s3_prefix=K.s3_job_prefix(cfg.S3_PREFIX, manifest.job_id),
        manifest=manifest,
        seed=manifest.model.seed,
        logger=log,
        on_progress=lambda p: status.heartbeat(current_phase_progress=p),
    )

    for phase in (PoseExtractPhase(), AnimatePhase(), AudioAttachPhase(), ReelsFormatPhase()):
        result = phase.run(ctx)
        assert result.status == "ok", f"phase {phase.name} failed: {result.error}"

    final = work / K.FINAL
    assert final.exists()
    assert final.stat().st_size > 0


def test_seed_determinism(
    tmp_path: Path,
    sample_photo: Path,
    sample_video: Path,
) -> None:
    """Two runs with same seed produce identical animated.mp4."""
    from reel_forge.prepare.photo_prep import prepare_photo

    cfg = get_config()
    sizes: list[int] = []
    for run in range(2):
        work = tmp_path / f"work{run}"
        work.mkdir()
        prepared = prepare_photo(sample_photo, out_path=work / K.PHOTO)
        (work / K.REFERENCE_VIDEO).write_bytes(sample_video.read_bytes())
        manifest = _make_manifest(work, prepared.staged_path, work / K.REFERENCE_VIDEO)
        (work / K.MANIFEST).write_text(manifest.model_dump_json(indent=2))
        storage = get_object_store(cfg)
        StatusManager(
            job_id=manifest.job_id,
            local_path=work / K.STATUS,
            storage=storage,
            s3_key=K.s3_status_key(cfg.S3_PREFIX, manifest.job_id) + f"-{run}",
        )
        seed_everything(manifest.model.seed)
        ctx = PhaseContext(
            job_id=manifest.job_id,
            work_dir=work,
            storage=storage,
            s3_prefix=f"{cfg.S3_PREFIX}/run{run}",
            manifest=manifest,
            seed=manifest.model.seed,
            logger=log,
            on_progress=lambda _: None,
        )
        for phase in (PoseExtractPhase(), AnimatePhase()):
            r = phase.run(ctx)
            assert r.status == "ok"
        sizes.append((work / K.ANIMATED).stat().st_size)
    assert sizes[0] == sizes[1], f"non-deterministic output sizes: {sizes}"
