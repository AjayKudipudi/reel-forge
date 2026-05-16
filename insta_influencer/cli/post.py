"""`insta post` — copy / reveal / (future) Instagram Graph API publish."""
from __future__ import annotations

import shutil
import subprocess
import sys

import click

from ..core import keys as K
from ..core.storage import get_object_store
from ..data.prompts.caption import render_caption
from ._common import cfg_and_log, read_local_manifest


@click.command("post")
@click.option("--job", "job_id", type=str, required=True)
@click.option(
    "--mode",
    type=click.Choice(["copy", "reveal", "graph"]),
    default="copy",
)
def post_cmd(job_id: str, mode: str) -> None:
    cfg = cfg_and_log(job_id)
    manifest = read_local_manifest(cfg, job_id)
    if manifest is None:
        click.echo(f"no manifest for {job_id}", err=True)
        sys.exit(1)

    storage = get_object_store(cfg)
    final_local = cfg.OUTPUT_DIR / job_id / K.FINAL
    if not final_local.exists():
        # Try pulling from S3.
        s3_key = f"{cfg.S3_PREFIX}/{job_id}/{K.FINAL}"
        if not storage.exists(s3_key):
            click.echo(f"no final.mp4 yet for {job_id}", err=True)
            sys.exit(1)
        storage.download(s3_key, final_local)

    if mode == "graph":
        click.echo(
            "graph mode is not implemented in v1; "
            "use --mode copy and upload manually for now.",
            err=True,
        )
        sys.exit(5)

    if mode == "reveal":
        # macOS / Linux fallbacks
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.run([opener, str(final_local)], check=False)
        except FileNotFoundError:
            click.echo(str(final_local))
        return

    # mode == "copy"
    ready_dir = cfg.OUTPUT_DIR / "_ready" / job_id
    ready_dir.mkdir(parents=True, exist_ok=True)
    target = ready_dir / K.FINAL
    shutil.copy2(final_local, target)
    caption = render_caption(prompt=manifest.prompt, hashtags=cfg.DEFAULT_HASHTAGS)
    (ready_dir / "caption.txt").write_text(caption)
    click.echo(f"ready: {ready_dir}")
