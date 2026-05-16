"""`forge stats` — cost aggregation by month."""
from __future__ import annotations

from datetime import UTC, datetime

import click

from ..core.status_models import State
from ._aws_prices import hourly_price
from ._common import cfg_and_log, list_jobs, read_local_status


def _month_key(d: datetime) -> str:
    return d.strftime("%Y-%m")


@click.command("stats")
@click.option("--month", type=str, default=None, help="YYYY-MM (default: current month)")
@click.option("--scope", type=click.Choice(["jobs", "gpu", "both"]), default="both")
def stats_cmd(month: str | None, scope: str) -> None:
    cfg = cfg_and_log(None)
    target_month = month or _month_key(datetime.now(UTC))

    rows: list[dict[str, str | float | int]] = []
    total_wall_s = 0.0
    total_cost_usd = 0.0
    for jid in list_jobs(cfg):
        s = read_local_status(cfg, jid)
        if s is None or s.state != State.COMPLETED:
            continue
        if _month_key(s.updated_at) != target_month:
            continue
        wall_s = sum(p.wall_s or 0.0 for p in s.phase_history)
        instance_type = (s.instance.instance_type if s.instance else cfg.EC2_INSTANCE_TYPE)
        spot = s.instance.spot if s.instance else cfg.USE_SPOT
        per_hr = hourly_price(instance_type, spot=spot)
        cost = (wall_s / 3600.0) * per_hr
        rows.append(
            {
                "job": jid,
                "wall_s": round(wall_s, 1),
                "instance": instance_type,
                "spot": "Y" if spot else "N",
                "$/hr": per_hr,
                "cost_usd": round(cost, 4),
            }
        )
        total_wall_s += wall_s
        total_cost_usd += cost

    if scope in ("jobs", "both"):
        if not rows:
            click.echo(f"no completed jobs in {target_month}")
        else:
            cols = ["job", "wall_s", "instance", "spot", "$/hr", "cost_usd"]
            widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
            click.echo("  ".join(c.ljust(widths[c]) for c in cols))
            click.echo("-" * (sum(widths.values()) + 2 * len(cols)))
            for r in rows:
                click.echo("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    if scope in ("gpu", "both"):
        click.echo("")
        click.echo(f"month: {target_month}")
        click.echo(f"  jobs:       {len(rows)}")
        click.echo(f"  total wall: {round(total_wall_s, 1)}s ({round(total_wall_s/60, 1)} min)")
        click.echo(f"  total cost: ${round(total_cost_usd, 2)}")
