"""`forge retry` — re-queue a failed_recoverable job."""
from __future__ import annotations

import sys
from datetime import UTC, datetime

import click

from ..core import keys as K
from ..core.manifest import PendingQueue
from ..core.status import State, StatusManager
from ..core.storage import get_object_store
from ._common import cfg_and_log


@click.command("retry")
@click.option("--job", "job_id", type=str, required=True)
@click.option("--force", is_flag=True, default=False, help="Allow retry of failed_terminal")
def retry_cmd(job_id: str, force: bool) -> None:
    cfg = cfg_and_log(job_id)
    storage = get_object_store(cfg)
    status = StatusManager(
        job_id=job_id,
        local_path=cfg.OUTPUT_DIR / job_id / K.STATUS,
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, job_id),
    )
    if status.status.state == State.FAILED_RECOVERABLE:
        status.transition(State.QUEUED)
    elif status.status.state == State.FAILED_TERMINAL and force:
        # Allowed only with --force; operator should have changed inputs.
        status.status = status.status.model_copy(update={"failure": None})
        status._flush()
        status.transition(State.FAILED_RECOVERABLE)
        status.transition(State.QUEUED)
    else:
        click.echo(
            f"refuse: state={status.status.state.value}; only failed_recoverable is retryable "
            f"(use --force for failed_terminal after fixing inputs)",
            err=True,
        )
        sys.exit(1)

    # Re-enqueue
    pending_key = K.s3_pending_key(cfg.S3_PREFIX)
    queue = PendingQueue(job_ids=[job_id], enqueued_at=datetime.now(UTC))
    if storage.exists(pending_key):
        import tempfile
        from pathlib import Path
        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
            tmp_path = Path(tmp.name)
            storage.download(pending_key, tmp_path)
            existing = PendingQueue.model_validate_json(tmp_path.read_text())
        if job_id not in existing.job_ids:
            existing.job_ids.append(job_id)
            existing.enqueued_at = datetime.now(UTC)
        queue = existing
    out = cfg.OUTPUT_DIR / "_pending.write.json"
    out.write_text(queue.model_dump_json(indent=2))
    try:
        storage.upload_atomic(out, pending_key)
    finally:
        out.unlink(missing_ok=True)
    click.echo(f"requeued: {job_id}")
