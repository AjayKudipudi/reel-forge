"""`forge cancel` — write cancel marker to S3 and update local status."""
from __future__ import annotations

import click

from ..core import keys as K
from ..core.status import State, StatusManager
from ..core.storage import get_object_store
from ._common import cfg_and_log


@click.command("cancel")
@click.option("--job", "job_id", type=str, required=True)
def cancel_cmd(job_id: str) -> None:
    cfg = cfg_and_log(job_id)
    storage = get_object_store(cfg)
    # Cancel marker (orchestrator checks this between phases).
    marker_path = cfg.OUTPUT_DIR / job_id / "_cancel"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.touch()
    storage.upload(marker_path, K.s3_cancel_key(cfg.S3_PREFIX, job_id))

    status = StatusManager(
        job_id=job_id,
        local_path=cfg.OUTPUT_DIR / job_id / K.STATUS,
        storage=storage,
        s3_key=K.s3_status_key(cfg.S3_PREFIX, job_id),
    )
    try:
        status.transition(State.CANCELLED)
    except Exception as exc:
        click.echo(f"warning: could not transition to cancelled ({exc}); marker placed anyway",
                   err=True)
    click.echo(f"cancelled: {job_id}")
