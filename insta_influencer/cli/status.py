"""`insta status` — table view of jobs and their state."""
from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import click

from ..core import keys as K
from ..core.status_models import Status
from ..core.storage import get_object_store
from ._common import cfg_and_log, list_jobs, read_local_status


def _fmt_age(t: datetime | None) -> str:
    if t is None:
        return "—"
    delta = datetime.now(UTC) - t
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        return f"{s // 3600}h"
    return f"{s // 86400}d"


def _row(s: Status) -> dict[str, str]:
    return {
        "job": s.job_id,
        "state": s.state.value,
        "phase": s.current_phase or "—",
        "age": _fmt_age(s.created_at),
        "heartbeat": _fmt_age(s.last_heartbeat_at),
        "progress": s.current_phase_progress or "—",
    }


@click.command("status")
@click.option("--job", "job_filter", type=str, default=None)
@click.option("--watch", is_flag=True, default=False)
@click.option("--json", "as_json", is_flag=True, default=False)
def status_cmd(job_filter: str | None, watch: bool, as_json: bool) -> None:
    cfg = cfg_and_log(None)
    storage = get_object_store(cfg)

    def render() -> None:
        ids = [job_filter] if job_filter else list_jobs(cfg)
        rows: list[dict[str, str]] = []
        for jid in ids:
            s = read_local_status(cfg, jid)
            if s is None and storage.exists(K.s3_status_key(cfg.S3_PREFIX, jid)):
                # Pull from S3 on demand.
                import tempfile
                from pathlib import Path
                with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
                    tmp_path = Path(tmp.name)
                    storage.download(K.s3_status_key(cfg.S3_PREFIX, jid), tmp_path)
                    s = Status.model_validate_json(tmp_path.read_text())
            if s is None:
                continue
            rows.append(_row(s))
        if as_json:
            click.echo(json.dumps(rows, indent=2))
            return
        if not rows:
            click.echo("no jobs.")
            return
        # Simple table
        cols = ["job", "state", "phase", "age", "heartbeat", "progress"]
        widths = {c: max(len(c), max(len(r[c]) for r in rows)) for c in cols}
        header = "  ".join(c.upper().ljust(widths[c]) for c in cols)
        click.echo(header)
        click.echo("-" * len(header))
        for r in rows:
            click.echo("  ".join(r[c].ljust(widths[c]) for c in cols))

    if not watch:
        render()
        return
    try:
        while True:
            click.clear()
            render()
            time.sleep(5)
    except KeyboardInterrupt:
        pass
