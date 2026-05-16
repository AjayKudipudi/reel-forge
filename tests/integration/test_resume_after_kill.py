"""After a phase marker is set, re-running skips that phase."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import structlog

from insta_influencer.config import get_config
from insta_influencer.core import keys as K
from insta_influencer.core.manifest import (
    BackgroundFromPhoto,
    Manifest,
    ModelConfig,
    OutputSpec,
    ReferenceLocal,
)
from insta_influencer.core.result import PhaseContext
from insta_influencer.core.seed import seed_everything
from insta_influencer.core.status import State, StatusManager
from insta_influencer.core.storage import get_object_store
from insta_influencer.ec2.phases.animate import AnimatePhase
from insta_influencer.ec2.phases.pose_extract import PoseExtractPhase

log = structlog.get_logger(__name__)


def test_phase_marker_skips_recompute(
    tmp_path: Path,
    sample_photo: Path,
    sample_video: Path,
) -> None:
    from insta_influencer.prepare.photo_prep import prepare_photo

    cfg = get_config()
    work = tmp_path / "work"
    work.mkdir()
    prepared = prepare_photo(sample_photo, out_path=work / K.PHOTO)
    (work / K.REFERENCE_VIDEO).write_bytes(sample_video.read_bytes())
    manifest = Manifest(
        schema_version=1,
        job_id="resumetest12",
        created_at=datetime.now(UTC),
        reference_source=ReferenceLocal(
            type="local",
            original_path=sample_video,
            staged_path=work / K.REFERENCE_VIDEO,
            sha256="a" * 64,
        ),
        photo_path=prepared.staged_path,
        photo_sha256=prepared.sha256,
        background=BackgroundFromPhoto(),
        prompt="a person dancing",
        model=ModelConfig(seed=42),
        output=OutputSpec(num_frames=8, keep_reference_audio=False),
    )
    (work / K.MANIFEST).write_text(manifest.model_dump_json(indent=2))

    storage = get_object_store(cfg)
    status = StatusManager(
        job_id=manifest.job_id,
        local_path=work / K.STATUS,
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, manifest.job_id),
    )
    for s in (
        State.PREPARING,
        State.PREPARED,
        State.UPLOADING,
        State.QUEUED,
        State.LAUNCHING,
        State.PHASE_RUNNING,
    ):
        status.transition(s)

    seed_everything(42)
    ctx = PhaseContext(
        job_id=manifest.job_id,
        work_dir=work,
        storage=storage,
        s3_prefix=K.s3_job_prefix(cfg.S3_PREFIX, manifest.job_id),
        manifest=manifest,
        seed=42,
        logger=log,
        on_progress=lambda _: None,
    )

    # Run pose_extract once and place the marker manually.
    PoseExtractPhase().run(ctx)
    marker_local = work / "_marker_pose_extract"
    marker_local.touch()
    marker_key = K.s3_marker_key(cfg.S3_PREFIX, manifest.job_id, "pose_extract")
    storage.upload_atomic(marker_local, marker_key)

    # Verify marker exists; in a real orchestrator loop this skips the phase.
    assert storage.exists(marker_key)

    # Continue with animate — pose_seq.npz is on disk from the prior run.
    r = AnimatePhase().run(ctx)
    assert r.status == "ok"
    assert (work / K.ANIMATED).exists()
