"""`forge logs` — fetch log file from S3 (or local) and print."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import click

from ..core import keys as K
from ..core.storage import get_object_store
from ._common import cfg_and_log


@click.command("logs")
@click.option("--job", "job_id", type=str, required=True)
@click.option("--phase", type=str, default=None)
@click.option("--full", is_flag=True, default=False)
def logs_cmd(job_id: str, phase: str | None, full: bool) -> None:
    cfg = cfg_and_log(job_id)
    storage = get_object_store(cfg)
    name = f"{phase}.stderr" if phase else "run.log"
    key = K.s3_log_key(cfg.S3_PREFIX, job_id, name)
    if not storage.exists(key):
        click.echo(f"no log at {key}", err=True)
        sys.exit(1)
    with tempfile.NamedTemporaryFile(suffix=f".{name}", delete=True) as tmp:
        tmp_path = Path(tmp.name)
        storage.download(key, tmp_path)
        text = tmp_path.read_text(errors="replace")
    if full:
        click.echo(text)
    else:
        tail = "\n".join(text.splitlines()[-200:])
        click.echo(tail)
