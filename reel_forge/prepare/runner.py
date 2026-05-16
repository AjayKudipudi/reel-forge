"""Local-side launcher: prepare → upload → enqueue → launch → poll."""
from __future__ import annotations

import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import structlog

from ..config import Config
from ..core import keys as K
from ..core.manifest import Manifest, PendingQueue
from ..core.status import State, StatusManager
from ..core.status_models import Status
from ..core.storage import ObjectStore, get_object_store

log = structlog.get_logger(__name__)


def job_dir(cfg: Config, job_id: str) -> Path:
    return cfg.OUTPUT_DIR / job_id


def manifest_path(cfg: Config, job_id: str) -> Path:
    return job_dir(cfg, job_id) / K.MANIFEST


def status_path(cfg: Config, job_id: str) -> Path:
    return job_dir(cfg, job_id) / K.STATUS


def write_manifest_locally(cfg: Config, manifest: Manifest) -> Path:
    p = manifest_path(cfg, manifest.job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.model_dump_json(indent=2))
    return p


def stage_inputs(cfg: Config, manifest: Manifest) -> dict[str, Path]:
    """Copy photo and reference video into the job dir using canonical names."""
    out: dict[str, Path] = {}
    target_photo = job_dir(cfg, manifest.job_id) / K.PHOTO
    target_ref = job_dir(cfg, manifest.job_id) / K.REFERENCE_VIDEO
    if not target_photo.exists():
        shutil.copy2(manifest.photo_path, target_photo)
    if not target_ref.exists():
        shutil.copy2(manifest.reference_source.staged_path, target_ref)
    out["photo"] = target_photo
    out["reference"] = target_ref
    return out


def existing_status(cfg: Config, job_id: str) -> Status | None:
    p = status_path(cfg, job_id)
    if p.exists():
        return Status.model_validate_json(p.read_text())
    return None


def upload_inputs(
    *,
    cfg: Config,
    storage: ObjectStore,
    manifest: Manifest,
    status: StatusManager,
) -> None:
    status.transition(State.UPLOADING)
    jdir = job_dir(cfg, manifest.job_id)
    jprefix = K.s3_job_prefix(cfg.S3_PREFIX, manifest.job_id)
    storage.upload(jdir / K.MANIFEST, f"{jprefix}/{K.MANIFEST}")
    storage.upload(jdir / K.PHOTO, f"{jprefix}/{K.PHOTO}")
    storage.upload(jdir / K.REFERENCE_VIDEO, f"{jprefix}/{K.REFERENCE_VIDEO}")


def enqueue(*, cfg: Config, storage: ObjectStore, job_ids: list[str]) -> None:
    """Append job_ids to s3://.../<prefix>/_pending.json (atomic write)."""
    pending_key = K.s3_pending_key(cfg.S3_PREFIX)
    existing: list[str] = []
    if storage.exists(pending_key):
        tmp = cfg.OUTPUT_DIR / "_pending.json.read"
        try:
            storage.download(pending_key, tmp)
            existing = PendingQueue.model_validate_json(tmp.read_text()).job_ids
        finally:
            tmp.unlink(missing_ok=True)
    seen = set(existing)
    merged = [*existing, *(j for j in job_ids if j not in seen)]
    queue = PendingQueue(
        schema_version=1,
        job_ids=merged,
        enqueued_at=datetime.now(UTC),
    )
    out = cfg.OUTPUT_DIR / "_pending.json.write"
    out.write_text(queue.model_dump_json(indent=2))
    try:
        storage.upload_atomic(out, pending_key)
    finally:
        out.unlink(missing_ok=True)


def poll_until_done(
    *,
    storage: ObjectStore,
    s3_status_key: str,
    interval_s: float = 5.0,
    timeout_s: float = 7200.0,
) -> Status:
    """Block until the remote status reaches a terminal state. Used by
    `generate --watch`."""
    started = time.time()
    last_state: str | None = None
    terminal = {
        State.COMPLETED.value,
        State.FAILED_TERMINAL.value,
        State.CANCELLED.value,
    }
    import tempfile
    while True:
        if not storage.exists(s3_status_key):
            time.sleep(interval_s)
            if time.time() - started > timeout_s:
                raise TimeoutError("status never appeared in S3")
            continue
        # Download to a temp and parse.
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            storage.download(s3_status_key, tmp_path)
            s = Status.model_validate_json(tmp_path.read_text())
        if s.state.value != last_state:
            log.info(
                "status.tick",
                state=s.state.value,
                phase=s.current_phase,
                progress=s.current_phase_progress,
            )
            last_state = s.state.value
        if s.state.value in terminal:
            return s
        if time.time() - started > timeout_s:
            raise TimeoutError("polling exceeded timeout")
        time.sleep(interval_s)


def prepare_and_register(*, cfg: Config, manifest: Manifest) -> Path:
    """Idempotent: writes manifest + status (state=prepared) locally.
    Returns the manifest path."""
    jid = manifest.job_id
    write_manifest_locally(cfg, manifest)
    storage = get_object_store(cfg)
    status = StatusManager(
        job_id=jid,
        local_path=status_path(cfg, jid),
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, jid),
    )
    if status.status.state == State.CREATED:
        status.transition(State.PREPARING)
        stage_inputs(cfg, manifest)
        status.transition(State.PREPARED)
    return manifest_path(cfg, jid)


def parse_existing_manifest(cfg: Config, job_id: str) -> Manifest | None:
    p = manifest_path(cfg, job_id)
    if not p.exists():
        return None
    return Manifest.model_validate_json(p.read_text())


def list_local_jobs(cfg: Config) -> list[str]:
    if not cfg.OUTPUT_DIR.exists():
        return []
    return sorted(p.name for p in cfg.OUTPUT_DIR.iterdir() if p.is_dir())


def write_pending_local(cfg: Config, queue: PendingQueue) -> Path:
    p = cfg.OUTPUT_DIR / "_pending.json"
    p.write_text(queue.model_dump_json(indent=2))
    return p


_ = json  # keep
