"""`insta cleanup` — retention policy."""
from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta

import click

from ..core import keys as K
from ..core.status_models import State
from ..core.storage import get_object_store
from ._common import cfg_and_log, list_jobs, read_local_status

_KEEP_STATES: set[State] = {State.PHASE_RUNNING, State.FAILED_RECOVERABLE,
                            State.UPLOADING, State.QUEUED, State.LAUNCHING}


@click.command("cleanup")
@click.option("--older-than", "days", type=int, default=None,
              help="Override RETENTION_DAYS")
@click.option("--apply/--dry-run", default=False)
@click.option("--scope", type=click.Choice(["local", "s3", "both"]), default="local")
def cleanup_cmd(days: int | None, apply: bool, scope: str) -> None:
    cfg = cfg_and_log(None)
    storage = get_object_store(cfg)
    cutoff = datetime.now(UTC) - timedelta(days=days or cfg.RETENTION_DAYS)
    deleted_local: list[str] = []
    deleted_s3: list[str] = []
    skipped: list[tuple[str, str]] = []
    for jid in list_jobs(cfg):
        s = read_local_status(cfg, jid)
        if s is None:
            continue
        if s.state in _KEEP_STATES:
            skipped.append((jid, f"state={s.state.value}"))
            continue
        if s.updated_at >= cutoff:
            skipped.append((jid, "too-recent"))
            continue
        if scope in ("local", "both"):
            if apply:
                shutil.rmtree(cfg.OUTPUT_DIR / jid, ignore_errors=True)
            deleted_local.append(jid)
        if scope in ("s3", "both"):
            prefix = K.s3_job_prefix(cfg.S3_PREFIX, jid)
            keys = list(storage.list(prefix))
            if apply:
                for k in keys:
                    storage.delete(k)
            deleted_s3.append(jid)
    click.echo(f"would delete (local): {len(deleted_local)} jobs")
    click.echo(f"would delete (s3): {len(deleted_s3)} jobs")
    click.echo(f"skipped: {len(skipped)} jobs")
    if not apply:
        click.echo("(dry-run; pass --apply to actually delete)")
