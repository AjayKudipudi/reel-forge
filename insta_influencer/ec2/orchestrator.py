"""Per-job phase driver. CLI entrypoint for EC2 cloud-init.

Concurrency: jobs are processed SERIALLY. One EC2 instance = one job at a
time = predictable VRAM, predictable status. Concurrent jobs on one
instance is out of scope for v1.
"""
from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import click
import structlog

from ..config import get_config
from ..core import keys as K
from ..core.errors import ErrorClass, ErrorInfo, max_attempts
from ..core.log_setup import configure_logging
from ..core.manifest import Manifest, PendingQueue
from ..core.phase import Phase
from ..core.result import PhaseContext, PhaseResult
from ..core.seed import seed_everything
from ..core.status import State, StatusManager
from ..core.storage import ObjectStore, get_object_store
from ._subprocess_runner import run_phase_in_subprocess
from .heartbeat import HeartbeatThread
from .phases.animate import AnimatePhase
from .phases.audio_attach import AudioAttachPhase
from .phases.face_restore import FaceRestorePhase
from .phases.interp import InterpPhase
from .phases.pose_extract import PoseExtractPhase
from .phases.reels_format import ReelsFormatPhase
from .spot_watch import SpotWatchThread

log = structlog.get_logger(__name__)

ISOLATED_PHASES = {"pose_extract", "animate", "interp", "face_restore"}


# Each phase's primary on-disk output. Used by the marker-skip guard
# (`_phase_outputs_present_locally`) so an S3 marker from a prior spot
# whose EBS is gone doesn't cause us to skip a phase whose outputs aren't
# actually here. Pose_extract's outputs (per-frame JPGs under pose/ and
# pose_neg/) are NOT uploaded to S3 — so they MUST be regenerated on every
# fresh spot regardless of marker. Animate's outputs ARE uploaded to S3
# (post-§5.17) and pulled in by the artifact preseed step earlier in
# process_job, so its marker remains honorable.
_PHASE_OUTPUT_CHECKS: dict[str, str] = {
    "pose_extract": "pose/0000.jpg",
    "animate": K.ANIMATED,
    "interp": K.ANIMATED_60FPS,
    "face_restore": K.ANIMATED_60FPS_FACE,
    "audio_attach": K.ANIMATED_W_AUDIO,
    "reels_format": K.FINAL,
}


def _phase_outputs_present_locally(phase_name: str, work_dir: Path) -> bool:
    """Return True iff the phase's primary output is on local disk in
    `work_dir`. Used to prevent the marker-skip from honoring a marker
    when the phase's outputs aren't actually here (e.g. after spot
    termination + relaunch on a fresh EBS).
    """
    rel = _PHASE_OUTPUT_CHECKS.get(phase_name)
    if rel is None:
        return True  # unknown phase — fall back to legacy "honor marker" behavior
    return (work_dir / rel).exists()


def build_phase_pipeline(manifest: Manifest) -> list[Phase]:
    pipeline: list[Phase] = [PoseExtractPhase(), AnimatePhase()]
    # Always include InterpPhase. Old code gated this on
    # manifest.output.frame_interp, but stale manifests from prepare runs
    # before v8.8 (when DEFAULT_FRAME_INTERP was False) shipped with the
    # flag False and silently skipped interp — leaving Reels output at
    # the model's stuttery 16fps. The phase itself does the right thing
    # internally now (auto-detects source fps, targets max(src,60)).
    pipeline.append(InterpPhase())
    # Per-frame GFPGAN face restoration. Phase short-circuits to a copy if
    # GFPGAN deps aren't present on the box (defensive — pipeline never
    # hard-fails on a missing optional dep). See ec2/phases/face_restore.py.
    pipeline.append(FaceRestorePhase())
    if manifest.output.keep_reference_audio:
        pipeline.append(AudioAttachPhase())
    pipeline.append(ReelsFormatPhase())
    return pipeline


def _save_manifest_local_and_s3(
    manifest: Manifest,
    work: Path,
    storage: ObjectStore,
    s3_prefix: str,
) -> None:
    p = work / K.MANIFEST
    p.write_text(manifest.model_dump_json(indent=2))
    storage.upload_atomic(p, f"{s3_prefix}/{K.MANIFEST}")


def _on_oom_retry(
    manifest: Manifest,
    work: Path,
    storage: ObjectStore,
    s3_prefix: str,
) -> Manifest:
    """Shrink num_frames by 25% (floor 33), persist."""
    new = manifest.model_copy(deep=True)
    new.output.num_frames = max(33, int(new.output.num_frames * 0.75))
    _save_manifest_local_and_s3(new, work, storage, s3_prefix)
    log.warning("animate.oom.retry", new_num_frames=new.output.num_frames)
    return new


def process_job(job_id: str) -> int:
    """Run all phases for one job_id. Returns 0 on completed, non-zero on failure."""
    cfg = get_config()
    storage = get_object_store(cfg)
    work = cfg.EC2_WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    s3_prefix = K.s3_job_prefix(cfg.S3_PREFIX, job_id)
    storage.download(K.s3_manifest_key(cfg.S3_PREFIX, job_id), work / K.MANIFEST)
    manifest = Manifest.model_validate_json((work / K.MANIFEST).read_text())
    status = StatusManager(
        job_id=job_id,
        local_path=work / K.STATUS,
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, job_id),
    )

    seed_everything(manifest.model.seed)

    heartbeat = HeartbeatThread(status, interval_s=cfg.HEARTBEAT_INTERVAL_S)
    spot_watch = SpotWatchThread(status)
    heartbeat.start()
    spot_watch.start()

    try:
        # Pull the photo + reference video.
        for fname in (K.PHOTO, K.REFERENCE_VIDEO):
            storage.download(f"{s3_prefix}/{fname}", work / fname)

        # Optional pre-seed: if a prior animate phase uploaded its outputs to
        # S3 (and the marker for that phase is present), pull them here so
        # the existing per-phase-marker skip logic below will short-circuit
        # pose_extract + animate. Enables a "postprocess-only" iteration loop
        # for tuning RIFE / GFPGAN / audio / reels without paying the
        # ~1h 34m animate cost. See animate.py for the upload side.
        for fname in (K.ANIMATED,):
            s3_key = f"{s3_prefix}/{fname}"
            if storage.exists(s3_key) and not (work / fname).exists():
                storage.download(s3_key, work / fname)
                log.info("orchestrator.preseeded", file=fname)
        # Per-chunk animated mp4s (for num_clips>=2; interp prefers these).
        i = 0
        while True:
            fname = f"animated_chunk_{i}.mp4"
            s3_key = f"{s3_prefix}/{fname}"
            if not storage.exists(s3_key):
                break
            if not (work / fname).exists():
                storage.download(s3_key, work / fname)
                log.info("orchestrator.preseeded", file=fname)
            i += 1
            if i > 16:  # safety bound
                break

        if status.status.state in (State.QUEUED, State.LAUNCHING):
            status.transition(State.LAUNCHING)
        if status.status.state == State.LAUNCHING:
            status.transition(State.PHASE_RUNNING, current_phase=None)

        for phase in build_phase_pipeline(manifest):
            marker = K.s3_marker_key(cfg.S3_PREFIX, job_id, phase.name)
            if storage.exists(marker):
                # Guard against the "marker exists but outputs missing"
                # case that bit run #5 (Bug 46): a prior spot completed
                # the phase, uploaded the marker, then got reclaimed; the
                # marker survives on S3 but the outputs lived on the
                # prior spot's EBS and are gone. Only honor the marker if
                # the phase's primary output is actually on this spot's
                # local disk.
                if _phase_outputs_present_locally(phase.name, work):
                    log.info("phase.skip", phase=phase.name)
                    continue
                log.info(
                    "phase.marker_present_but_outputs_missing.rerun",
                    phase=phase.name,
                )

            if storage.exists(K.s3_cancel_key(cfg.S3_PREFIX, job_id)):
                status.transition(State.CANCELLED)
                return 2

            status.transition(
                State.PHASE_RUNNING,
                current_phase=phase.name,
                current_phase_started_at=datetime.now(UTC),
            )

            ctx = PhaseContext(
                job_id=job_id,
                work_dir=work,
                storage=storage,
                s3_prefix=s3_prefix,
                manifest=manifest,
                seed=manifest.model.seed,
                logger=log.bind(phase=phase.name),
                on_progress=lambda p: status.heartbeat(current_phase_progress=p),
            )
            isolated = phase.name in ISOLATED_PHASES
            t0 = datetime.now(UTC)
            result: PhaseResult = (
                run_phase_in_subprocess(phase, ctx, stderr_tail_bytes=cfg.STDERR_TAIL_BYTES)
                if isolated
                else phase.run(ctx)
            )
            t1 = datetime.now(UTC)

            if result.status == "ok":
                # Marker = zero-byte file.
                marker_local = work / f"_marker_{phase.name}"
                marker_local.touch()
                storage.upload_atomic(marker_local, marker)
                status.append_phase_history(
                    phase.name,
                    "ok",
                    wall_s=result.stats.get("wall_s"),
                    started_at=t0,
                    ended_at=t1,
                    stats=result.stats,
                )
                continue

            # Failure path
            assert result.error is not None
            err = result.error
            attempts = status.bump_attempt(phase.name)

            if (
                err.error_class == ErrorClass.MODEL_OOM
                and attempts <= max_attempts(ErrorClass.MODEL_OOM)
            ):
                manifest = _on_oom_retry(manifest, work, storage, s3_prefix)
                continue

            if err.retryable and attempts < max_attempts(err.error_class):
                log.warning(
                    "phase.retry",
                    phase=phase.name,
                    attempt=attempts,
                    error_class=err.error_class.value,
                )
                continue

            status.fail(phase=phase.name, error=err)
            return 1

        final = work / K.FINAL
        if not final.exists():
            status.fail(
                phase=None,
                error=ErrorInfo(
                    error_class=ErrorClass.UNKNOWN,
                    message="pipeline completed but no final.mp4 produced",
                    retryable=False,
                ),
            )
            return 1
        storage.upload(final, K.s3_final_key(cfg.S3_PREFIX, job_id))
        status.transition(State.COMPLETED)
        return 0
    finally:
        heartbeat.stop()
        spot_watch.stop()


# ── CLI ───────────────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """`python -m insta_influencer.ec2.orchestrator ...`"""


@cli.command("process-job")
@click.argument("job_id")
def process_job_cmd(job_id: str) -> None:
    cfg = get_config()
    configure_logging(job_id=job_id, log_dir=cfg.LOG_DIR, level=cfg.LOG_LEVEL, fmt=cfg.LOG_FORMAT)
    sys.exit(process_job(job_id))


@cli.command("process-pending")
def process_pending_cmd() -> None:
    """Drain s3://.../jobs/_pending.json (oldest first). EC2 cloud-init entry."""
    cfg = get_config()
    configure_logging(job_id=None, log_dir=cfg.LOG_DIR, level=cfg.LOG_LEVEL, fmt=cfg.LOG_FORMAT)
    storage = get_object_store(cfg)
    pending_key = K.s3_pending_key(cfg.S3_PREFIX)
    if not storage.exists(pending_key):
        log.info("pending.empty")
        return
    tmp = cfg.EC2_WORK_DIR / "_pending.json"
    cfg.EC2_WORK_DIR.mkdir(parents=True, exist_ok=True)
    storage.download(pending_key, tmp)
    queue = PendingQueue.model_validate_json(tmp.read_text())
    failures = 0
    for jid in list(queue.job_ids):
        try:
            rc = process_job(jid)
        except BaseException as exc:
            log.exception("process_job.crash", job_id=jid, err=str(exc))
            rc = 1
        if rc != 0:
            failures += 1
        # After each job: re-read queue (new jobs may have been enqueued)
        # then drop just-processed id and write back.
        try:
            storage.download(pending_key, tmp)
            queue2 = PendingQueue.model_validate_json(tmp.read_text())
            queue2.job_ids = [j for j in queue2.job_ids if j != jid]
            queue2.enqueued_at = datetime.now(UTC)
            (cfg.EC2_WORK_DIR / "_pending.write.json").write_text(
                queue2.model_dump_json(indent=2)
            )
            storage.upload_atomic(cfg.EC2_WORK_DIR / "_pending.write.json", pending_key)
        except Exception as exc:
            log.warning("pending.update_failed", err=str(exc))
    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    cli()


_ = json  # keep import; useful for ad-hoc CLI integration
