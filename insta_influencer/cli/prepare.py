"""`insta prepare` — build a job manifest from a Reel URL or local mp4."""
from __future__ import annotations

import sys
from pathlib import Path

import click

from ..core.errors import PipelineError
from ..core.manifest import Background, BackgroundFromPhoto, BackgroundReplace
from ..prepare.content_moderation import moderate
from ..prepare.job_manifest import build_manifest
from ..prepare.photo_prep import prepare_photo
from ..prepare.reel_fetcher import acquire_reference
from ..prepare.runner import (
    existing_status,
    job_dir,
    prepare_and_register,
)
from ._common import cfg_and_log


@click.command("prepare")
@click.option("--reel", type=str, default=None, help="Instagram Reel URL")
@click.option(
    "--video",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Local mp4 path",
)
@click.option(
    "--photo",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--background",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional new background image; opt into BackgroundReplace mode",
)
@click.option("--prompt", type=str, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--matte-model", type=click.Choice(["birefnet", "sam2", "rmbg"]), default="birefnet")
@click.option(
    "--clips",
    type=click.IntRange(1, 10),
    default=1,
    help="Number of 81-frame clips to generate and concat. 1 = ~5s, 4 = ~20s. "
    "Each additional clip adds ~30 min to animate phase on g6e.2xlarge.",
)
def prepare_cmd(
    reel: str | None,
    video: Path | None,
    photo: Path,
    background: Path | None,
    prompt: str | None,
    seed: int | None,
    matte_model: str,
    clips: int,
) -> None:
    """Build a job manifest. Mutually-exclusive --reel | --video."""
    if (reel is None) == (video is None):
        raise click.UsageError("provide exactly one of --reel or --video")
    cfg = cfg_and_log(None)

    try:
        ref = acquire_reference(
            url=reel,
            local_path=video,
            cache_dir=cfg.ASSETS_DIR / "references",
        )
        # Photo prep target: temporary location until we know the job_id;
        # we'll move it under the job dir after job_id derivation.
        tmp_photo = cfg.ASSETS_DIR / "photos" / f"{photo.stem}.staged.png"
        prepared = prepare_photo(
            photo,
            out_path=tmp_photo,
            background_path=background,
            matte_model=matte_model,
        )
        moderate(
            photo=prepared.staged_path,
            reference=ref.staged_path,
            binary=cfg.CONTENT_MODERATION_BINARY,
            enabled=cfg.CONTENT_MODERATION_ENABLED,
        )
        bg: Background = (
            BackgroundReplace(
                replacement_path=background,
                matte_model=matte_model,
            )
            if background is not None
            else BackgroundFromPhoto()
        )
        manifest = build_manifest(
            cfg=cfg,
            reference=ref,
            photo=prepared,
            prompt=prompt,
            seed=seed,
            background=bg,
            num_clips=clips,
        )
    except PipelineError as exc:
        click.echo(f"prepare failed: {type(exc).__name__}: {exc}", err=True)
        sys.exit(1)

    # Idempotent: if the job already exists in prepared/completed, no-op.
    existing = existing_status(cfg, manifest.job_id)
    if existing and existing.state.value in {"prepared", "completed"}:
        click.echo(f"already prepared: {manifest.job_id}")
        sys.exit(0)

    prepare_and_register(cfg=cfg, manifest=manifest)
    click.echo(f"prepared: {manifest.job_id}")
    click.echo(f"  job dir: {job_dir(cfg, manifest.job_id)}")
