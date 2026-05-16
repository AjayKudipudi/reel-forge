"""`forge generate` — upload prepared jobs and launch a spot batch."""
from __future__ import annotations

import sys
from datetime import datetime, timezone

import click

from ..core import keys as K
from ..core.manifest import PendingQueue
from ..core.status import State, StatusManager
from ..core.storage import get_object_store
from ..prepare.runner import (
    enqueue,
    poll_until_done,
    upload_inputs,
)
from ._common import cfg_and_log, list_jobs, read_local_manifest


@click.command("generate")
@click.option("--job", "job_filter", type=str, default=None,
              help="Process only this job (default: all prepared jobs)")
@click.option("--watch/--no-watch", default=True, help="Stream remote status")
@click.option("--instance-type", type=str, default=None)
@click.option("--no-spot/--spot", default=False, help="Force on-demand")
def generate_cmd(
    job_filter: str | None,
    watch: bool,
    instance_type: str | None,
    no_spot: bool,
) -> None:
    cfg = cfg_and_log(None)
    if instance_type:
        cfg.EC2_INSTANCE_TYPE = instance_type
    if no_spot:
        cfg.USE_SPOT = False

    storage = get_object_store(cfg)
    job_ids: list[str] = []
    candidates = [job_filter] if job_filter else list_jobs(cfg)
    for jid in candidates:
        manifest = read_local_manifest(cfg, jid)
        if manifest is None:
            click.echo(f"skip {jid}: no manifest", err=True)
            continue
        status = StatusManager(
            job_id=jid,
            local_path=cfg.OUTPUT_DIR / jid / K.STATUS,
            storage=storage,
            s3_key=K.s3_status_key(cfg.S3_PREFIX, jid),
        )
        if status.status.state not in (State.PREPARED, State.FAILED_RECOVERABLE):
            click.echo(f"skip {jid}: state={status.status.state.value}", err=True)
            continue
        try:
            upload_inputs(cfg=cfg, storage=storage, manifest=manifest, status=status)
            status.transition(State.QUEUED)
            job_ids.append(jid)
        except Exception as exc:
            click.echo(f"upload failed for {jid}: {exc}", err=True)
            status.fail(phase=None, error=type("E", (), {"error_class": "s3_upload_failed",
                                                          "message": str(exc),
                                                          "retryable": True})())

    if not job_ids:
        click.echo("nothing to generate.")
        sys.exit(0)

    enqueue(cfg=cfg, storage=storage, job_ids=job_ids)
    click.echo(f"enqueued: {len(job_ids)} jobs")

    # FSR setup (if enabled). Without FSR, fresh spot volumes lazy-load from
    # S3 at ~7 MB/s on first read regardless of gp3 throughput. With FSR on
    # the AMI's snapshot in the launch AZ, full gp3 speed from the start.
    # We enable per-AZ before launch and disable after watch completes to
    # avoid the standing $0.75/AZ/hour charge between batches.
    fsr_snapshot: str | None = None
    fsr_az: str | None = None
    if cfg.USE_FSR:
        try:
            import boto3

            from ..ec2.launch import _root_snapshot_id, enable_fsr
            ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
            fsr_snapshot = _root_snapshot_id(ec2, cfg.EC2_AMI_ID)
            # Pick the first AZ from rotation as the FSR target.
            fsr_az = cfg.SPOT_AZ_ROTATION[0]
            click.echo(f"enabling FSR on {fsr_snapshot} in {fsr_az} "
                       f"(timeout {cfg.FSR_ENABLE_TIMEOUT_S}s)...")
            enable_fsr(ec2, fsr_snapshot, fsr_az, timeout_s=cfg.FSR_ENABLE_TIMEOUT_S)
            click.echo(f"FSR enabled in {fsr_az}")
        except Exception as exc:
            click.echo(f"FSR enable failed: {exc}; continuing without FSR", err=True)
            fsr_snapshot = None
            fsr_az = None

    # Launch — best-effort import; fall back to "manually launch" message
    # if boto3 / AWS creds are not configured in this environment.
    launched = False
    try:
        from ..ec2.launch import launch_for_pending
        result = launch_for_pending(cfg, preferred_az=fsr_az)
        click.echo(f"launched: {result.instance_id} az={result.az} spot={result.spot}")
        launched = True
    except Exception as exc:
        click.echo(f"could not launch automatically: {exc}", err=True)
    finally:
        # FSR's effect is consumed at volume creation. Once the spot has
        # launched, the volume is fully initialized; disabling FSR now does
        # NOT undo that initialization, but it does stop the ~$0.75/AZ/hr
        # standing charge while the job continues running.
        if fsr_snapshot and fsr_az:
            import boto3

            from ..ec2.launch import disable_fsr
            disable_fsr(boto3.client("ec2", region_name=cfg.AWS_REGION), fsr_snapshot, fsr_az)
            click.echo(f"FSR disabled in {fsr_az}")
    if not launched:
        click.echo("Pending queue is in S3. Launch the EC2 instance manually "
                   "(it will pick up the queue via cloud-init).")
        sys.exit(4)

    if watch:
        for jid in job_ids:
            click.echo(f"watching {jid} ...")
            try:
                final = poll_until_done(
                    storage=storage,
                    s3_status_key=K.s3_status_key(cfg.S3_PREFIX, jid),
                )
                click.echo(f"  {jid}: {final.state.value}")
            except TimeoutError as exc:
                click.echo(f"  {jid}: timeout ({exc})", err=True)
                sys.exit(3)


_ = datetime, timezone, PendingQueue  # keep imports
