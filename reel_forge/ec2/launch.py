"""Spot lifecycle: request spot, wait for ready, attach cloud-init
user-data that runs the orchestrator and self-terminates.

The local-side `cli/generate.py` calls into `launch_for_pending()`, which
returns once the spot request is fulfilled. Status is then polled from
S3 by the local CLI; no SSH needed.
"""
from __future__ import annotations

import base64
import contextlib
import time
from dataclasses import dataclass
from typing import Any

import structlog

from ..config import Config, get_config
from ..core.errors import ErrorClass, classify

log = structlog.get_logger(__name__)


CLOUD_INIT_TEMPLATE = """\
#!/bin/bash
set -euo pipefail
exec >> /var/log/insta-influencer.log 2>&1
echo "[$(date -uIs)] boot: insta-influencer entrypoint"

cd /opt/insta-influencer
# Wipe any stale work dirs baked into the AMI (Bug 21). The orchestrator's
# StatusManager prefers local status.json over S3, so a stale failed_recoverable
# baked in causes IllegalTransition crashes on fresh spot boots.
rm -rf /opt/insta-influencer/work/* 2>/dev/null || true

# Pull the latest source.tar.gz from S3 and overlay it on the AMI's frozen
# source. This is the "source-refresh-on-boot" pattern: any code-only fix
# can ship by re-tarring + `aws s3 cp` — no AMI re-bake. The bake script
# also uploads source.tar.gz to the same key during bake.
echo "[$(date -uIs)] refreshing source from S3..."
aws s3 cp "s3://{s3_bucket}/setup/insta-influencer-source.tar.gz" /tmp/src.tar.gz \
    --region "{aws_region}" || true
if [ -f /tmp/src.tar.gz ]; then
    tar -xzf /tmp/src.tar.gz -C /opt/insta-influencer --overwrite
    echo "[$(date -uIs)] source refreshed"
fi

source /opt/insta-influencer/.venv/bin/activate
export AWS_REGION="{aws_region}"
export S3_BUCKET="{s3_bucket}"
export S3_PREFIX="{s3_prefix}"
export HF_TOKEN="{hf_token}"
export HUGGING_FACE_HUB_TOKEN="{hf_token}"
export HF_HOME=/opt/insta-influencer/hf-cache

# Download SteadyDancer-14B fine-tuned weights if not already cached.
# This is the actual fine-tuned model — `Wan-AI/Wan2.1-I2V-14B-480P` is just
# the BASE Wan2.1 weights and is missing SteadyDancer's pose-conditioning
# adapter layers (condition_embedding_*, patch_embedding_fuse, etc.). Without
# these the model is silently randomly-initialized and produces garbage
# (Bug 27 — see handoff). ~28 GB safetensors over HF (~5-10 min).
if [ ! -d /opt/insta-influencer/hf-cache/hub/models--MCG-NJU--SteadyDancer-14B ]; then
    echo "[$(date -uIs)] downloading MCG-NJU/SteadyDancer-14B (fine-tuned weights)..."
    python - <<PYEOF
from huggingface_hub import snapshot_download
snapshot_download("MCG-NJU/SteadyDancer-14B")
print("steadydancer-14b cached")
PYEOF
    echo "[$(date -uIs)] SteadyDancer-14B cached"
fi
# torch.distributed.optim.functional_* uses @torch.jit.script which SEGVs on
# torch 2.5.1+cu121. Disable JIT — these functional optimizers are deprecated
# and we don't use them for inference.
export PYTORCH_JIT=0

# === Optional post-process deps: RIFE (frame interp) + GFPGAN (face restore) ===
# Best-effort. If any step fails, the corresponding phase falls back gracefully:
#   - interp.py: minterpolate=blend if RIFE missing
#   - face_restore.py: cp-through if GFPGAN missing
# All steps idempotent so successive boots (or AMI re-bake later) are fast.

# 1. Clone Practical-RIFE (MIT). ~3 MB shallow clone.
if [ ! -f /opt/insta-influencer/third_party/Practical-RIFE/inference_video.py ]; then
    echo "[$(date -uIs)] cloning Practical-RIFE..."
    mkdir -p /opt/insta-influencer/third_party
    git clone --depth 1 https://github.com/hzwer/Practical-RIFE.git \
        /opt/insta-influencer/third_party/Practical-RIFE \
        2>&1 | tail -5 || echo "[warn] RIFE clone failed; interp will use ffmpeg blend"
fi

# 2. Download RIFE v4.26 weights from HF mirror hzwer/RIFE.
# The repo serves zipped weight bundles (.zip per release), not loose
# .pkl/.py files. Latest is RIFEv4.26_0921.zip; unzip into train_log/ so
# Practical-RIFE's inference_video.py --model=train_log finds the weights
# and the architecture .py files it imports at runtime.
RIFE_TRAIN_LOG=/opt/insta-influencer/third_party/Practical-RIFE/train_log
if [ ! -f "$RIFE_TRAIN_LOG/flownet.pkl" ] && [ -f /opt/insta-influencer/third_party/Practical-RIFE/inference_video.py ]; then
    echo "[$(date -uIs)] downloading RIFE v4.26 weights bundle..."
    mkdir -p "$RIFE_TRAIN_LOG"
    curl -fL -o /tmp/rife.zip \
        https://huggingface.co/hzwer/RIFE/resolve/main/RIFEv4.26_0921.zip \
        2>&1 | tail -3 || echo "[warn] RIFE zip download failed"
    if [ -f /tmp/rife.zip ]; then
        unzip -o -d /tmp/rife-unzipped /tmp/rife.zip 2>&1 | tail -5 || true
        # The zip may unpack into a subdir like train_log/ or train_log_v4.26/.
        # Find the dir containing flownet.pkl and move its contents into train_log/.
        SRC_DIR=$(find /tmp/rife-unzipped -name "flownet.pkl" -exec dirname {{}} \\; | head -1)
        if [ -n "$SRC_DIR" ]; then
            cp -r "$SRC_DIR"/* "$RIFE_TRAIN_LOG"/ || true
            echo "RIFE weights installed:"
            ls -la "$RIFE_TRAIN_LOG" | head -20
        else
            echo "[warn] flownet.pkl not found in RIFE zip"
        fi
        rm -rf /tmp/rife-unzipped /tmp/rife.zip
    fi
fi

# 3. pip install gfpgan (Apache 2.0) + scikit-video (for Practical-RIFE's
# inference_video.py which `import skvideo.io`). Pin numpy<2 —
# gfpgan/basicsr/facexlib may transitively upgrade numpy, which breaks the
# pre-built xtcocotools wheel that mmpose (pose_extract) depends on
# (Bug 15 / numpy.dtype ABI error). Pin basicsr/facexlib to known-good
# versions.
if ! python -c "import gfpgan" 2>/dev/null; then
    echo "[$(date -uIs)] installing GFPGAN + scikit-video..."
    pip install \
        "numpy<2" \
        basicsr==1.4.2 facexlib==0.3.0 gfpgan==1.3.8 \
        scikit-video \
        2>&1 | tail -10 || echo "[warn] gfpgan pip install failed; face_restore will cp-through"
    # Defensive: ensure numpy stayed <2 even if pip's resolver wobbled.
    pip install --force-reinstall --no-deps "numpy<2" 2>&1 | tail -3 || true

    # Patch basicsr's torchvision.transforms.functional_tensor import (the
    # symbol was removed in torchvision 0.17+). Locate the file via `find`
    # NOT via `python -c "import basicsr..."` — the very import we're trying
    # to fix is the one that breaks, so importing to find the path would
    # silently return nothing and skip the patch.
    BASICSR_DEG=$(find /opt/insta-influencer/.venv -path '*basicsr/data/degradations.py' 2>/dev/null | head -1)
    if [ -n "$BASICSR_DEG" ] && [ -f "$BASICSR_DEG" ]; then
        echo "[$(date -uIs)] patching basicsr degradations.py at $BASICSR_DEG"
        sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' "$BASICSR_DEG"
        # Verify patch took.
        if grep -q 'functional_tensor' "$BASICSR_DEG"; then
            echo "[ERROR] basicsr patch did NOT apply — gfpgan will still fail to import"
        else
            echo "[$(date -uIs)] basicsr patch applied OK"
        fi
    else
        echo "[ERROR] basicsr degradations.py not found under .venv — gfpgan likely not installed"
    fi

    # Diagnostic: confirm gfpgan + skvideo + xtcocotools all import. Log
    # explicit success/failure so the next boot log makes the state obvious.
    python - <<'PYEOF' || true
import sys
def check(mod, what):
    try:
        __import__(mod)
        print(f"  [ok ] {{what}} import: {{mod}}")
    except Exception as e:
        print(f"  [FAIL] {{what}} import: {{mod}} -> {{type(e).__name__}}: {{e}}", file=sys.stderr)

check("gfpgan", "GFPGAN")
check("skvideo.io", "RIFE skvideo dep")
check("xtcocotools.coco", "pose_extract xtcocotools")
import numpy
print(f"  numpy version: {{numpy.__version__}}")
PYEOF
fi

# 4. Download GFPGAN v1.4 weights (~349 MB) from the TencentARC GitHub release.
GFPGAN_WEIGHTS=/opt/insta-influencer/gfpgan-weights/GFPGANv1.4.pth
if [ ! -f "$GFPGAN_WEIGHTS" ]; then
    echo "[$(date -uIs)] downloading GFPGANv1.4.pth..."
    mkdir -p /opt/insta-influencer/gfpgan-weights
    curl -fL -o "$GFPGAN_WEIGHTS" \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
        2>&1 | tail -3 || echo "[warn] GFPGAN weights download failed"
fi

# 5. Patch scikit-video's deprecated np.float / np.int / np.bool usages.
# skvideo is unmaintained and uses removed numpy 1.20+ aliases. Without
# this patch, Practical-RIFE's inference_video.py crashes immediately on
# `lastframe = next(videogen)` because skvideo.io.vreader calls np.float.
SKVIDEO_DIR=$(find /opt/insta-influencer/.venv -path '*site-packages/skvideo' -type d 2>/dev/null | head -1)
if [ -n "$SKVIDEO_DIR" ] && [ -d "$SKVIDEO_DIR" ]; then
    echo "[$(date -uIs)] patching skvideo np.float/np.int/np.bool in $SKVIDEO_DIR"
    grep -rl 'np\\.float\\b\\|np\\.int\\b\\|np\\.bool\\b' "$SKVIDEO_DIR" 2>/dev/null | while read -r f; do
        sed -i -e 's|np\\.float(|float(|g' \
               -e 's|np\\.int(|int(|g' \
               -e 's|np\\.bool(|bool(|g' \
               -e 's|= *np\\.float\\b|= float|g' \
               -e 's|= *np\\.int\\b|= int|g' \
               -e 's|= *np\\.bool\\b|= bool|g' \
               "$f"
    done
    if grep -rq 'np\\.float(\\|np\\.int(\\|np\\.bool(' "$SKVIDEO_DIR" 2>/dev/null; then
        echo "[ERROR] skvideo np.* patches incomplete; RIFE will still fail"
    else
        echo "[$(date -uIs)] skvideo patches applied OK"
    fi
fi

# 6. Pre-warm facexlib detection weights to ~/.cache so first-frame
# face_restore doesn't pay the download cost mid-run.
if [ -f "$GFPGAN_WEIGHTS" ] && python -c "import gfpgan" 2>/dev/null; then
    python - <<'PYEOF' || true
from gfpgan import GFPGANer
try:
    GFPGANer(
        model_path="/opt/insta-influencer/gfpgan-weights/GFPGANv1.4.pth",
        upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None,
    )
    print("facexlib detection weights warmed")
except Exception as e:
    print(f"facexlib warm failed: {{e}}")
PYEOF
fi

# NOTE: page-cache prewarm was REMOVED here. With FSR enabled on the AMI's
# snapshot, the volume is already fully initialized at creation — mmap reads
# hit full gp3 speed without prewarm. Adding prewarm on top of FSR caused
# double-counting of memory (page cache + model RSS) and triggered swap
# thrashing on g6e.2xlarge (Bug 26 — see handoff for the timeline).

# Capture orchestrator exit code without letting `set -e` abort the script —
# we MUST reach the terminate-instances call below even on failure, or the
# spot lingers and burns money.
set +e
python -m reel_forge.ec2.orchestrator process-pending
EXIT=$?
set -e

# Always upload cloud-init log to S3 so failures are diagnosable from local.
aws s3 cp /var/log/insta-influencer.log \
    "s3://{s3_bucket}/{s3_prefix}/_runtime-logs/last-boot.log" \
    --region "{aws_region}" || true

TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
        http://169.254.169.254/latest/meta-data/instance-id)

# Debug-hold mechanism. When {keep_alive_on_failure} is "1" AND the
# pipeline exited non-zero, sleep here for {keep_alive_seconds}s instead
# of terminating. Operator can SSH in (SG must allow port 22), fix the
# issue interactively, and run `python -m reel_forge.ec2.orchestrator
# process-job <job_id>` to retry. After the hold window expires the
# instance terminates as usual.
if [ "{keep_alive_on_failure}" = "1" ] && [ "$EXIT" != "0" ]; then
    echo "[$(date -uIs)] pipeline failed (exit=$EXIT); KEEP_ALIVE_ON_FAILURE=1 so sleeping {keep_alive_seconds}s before terminate."
    echo "[$(date -uIs)] SSH: ssh -i <your-key.pem> ubuntu@<public-ip>"
    # Best-effort upload of partial log so operator can grep on local box too.
    aws s3 cp /var/log/insta-influencer.log \
        "s3://{s3_bucket}/{s3_prefix}/_runtime-logs/keep-alive.log" \
        --region "{aws_region}" || true
    sleep {keep_alive_seconds}
fi

aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "{aws_region}" || true
exit $EXIT
"""


@dataclass(frozen=True)
class LaunchResult:
    instance_id: str
    az: str
    spot: bool


def render_user_data(cfg: Config, *, keep_alive_on_failure: bool = False, keep_alive_seconds: int = 3600) -> str:
    """Render cloud-init user-data.

    When `keep_alive_on_failure=True`, a failed pipeline run will sleep for
    `keep_alive_seconds` before terminating instead of self-terminating
    immediately — letting the operator SSH in (port 22 is open on the SG)
    and debug interactively before the spot dies.
    """
    import os
    # Allow env-var override (cleaner than threading through every CLI call).
    if not keep_alive_on_failure:
        keep_alive_on_failure = os.environ.get("KEEP_ALIVE_ON_FAILURE", "0") == "1"
    env_secs = os.environ.get("KEEP_ALIVE_SECONDS")
    if env_secs and env_secs.isdigit():
        keep_alive_seconds = int(env_secs)
    return CLOUD_INIT_TEMPLATE.format(
        aws_region=cfg.AWS_REGION,
        s3_bucket=cfg.S3_BUCKET,
        s3_prefix=cfg.S3_PREFIX,
        hf_token=cfg.HF_TOKEN,
        keep_alive_on_failure="1" if keep_alive_on_failure else "0",
        keep_alive_seconds=keep_alive_seconds,
    )


def _ec2_client(cfg: Config) -> Any:
    import boto3

    return boto3.client("ec2", region_name=cfg.AWS_REGION)


def request_spot_in_az(*, cfg: Config, az: str, user_data_b64: str) -> dict[str, Any]:
    """Launch a spot instance via the modern `run_instances` API with
    `InstanceMarketOptions`. The legacy `request_spot_instances` API was
    observed to create volumes without honoring FSR (FastRestored=False)
    even when the snapshot is FSR-enabled; the modern API picks it up
    correctly. Returns a synthesized response shape compatible with the
    spot-request-id flow in `launch_for_pending`."""
    ec2 = _ec2_client(cfg)
    kwargs: dict[str, Any] = {
        "ImageId": cfg.EC2_AMI_ID,
        "InstanceType": cfg.EC2_INSTANCE_TYPE,
        "KeyName": cfg.EC2_KEY_NAME,
        "MinCount": 1,
        "MaxCount": 1,
        "Placement": {"AvailabilityZone": az},
        "UserData": user_data_b64,
        "InstanceMarketOptions": {
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": str(cfg.EC2_SPOT_MAX_PRICE),
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        },
    }
    if cfg.EC2_SECURITY_GROUP_ID:
        kwargs["SecurityGroupIds"] = [cfg.EC2_SECURITY_GROUP_ID]
    if cfg.EC2_SUBNET_ID:
        kwargs["SubnetId"] = cfg.EC2_SUBNET_ID
    if cfg.EC2_IAM_INSTANCE_PROFILE:
        kwargs["IamInstanceProfile"] = {"Name": cfg.EC2_IAM_INSTANCE_PROFILE}
    resp = ec2.run_instances(**kwargs)
    inst = resp["Instances"][0]
    # Match the legacy `request_spot_instances` shape so the polling loop
    # in `launch_for_pending` doesn't need to know which API was used.
    return {
        "SpotInstanceRequests": [
            {
                "SpotInstanceRequestId": inst["SpotInstanceRequestId"],
                "InstanceId": inst["InstanceId"],
                "State": "active",
                "Status": {"Code": "fulfilled"},
            }
        ]
    }


def request_ondemand(*, cfg: Config, user_data_b64: str) -> dict[str, Any]:
    ec2 = _ec2_client(cfg)
    kwargs: dict[str, Any] = {
        "ImageId": cfg.EC2_AMI_ID,
        "InstanceType": cfg.EC2_INSTANCE_TYPE,
        "KeyName": cfg.EC2_KEY_NAME,
        "MinCount": 1,
        "MaxCount": 1,
        "UserData": user_data_b64,
    }
    if cfg.EC2_SECURITY_GROUP_ID:
        kwargs["SecurityGroupIds"] = [cfg.EC2_SECURITY_GROUP_ID]
    if cfg.EC2_SUBNET_ID:
        kwargs["SubnetId"] = cfg.EC2_SUBNET_ID
    if cfg.EC2_IAM_INSTANCE_PROFILE:
        kwargs["IamInstanceProfile"] = {"Name": cfg.EC2_IAM_INSTANCE_PROFILE}
    resp: dict[str, Any] = ec2.run_instances(**kwargs)
    return resp


def wait_for_running(ec2: Any, instance_id: str, *, timeout_s: int = 600) -> str:
    """Block until instance is running. Returns the AZ."""
    started = time.time()
    while True:
        d = ec2.describe_instances(InstanceIds=[instance_id])
        inst = d["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        if state == "running":
            return str(inst["Placement"]["AvailabilityZone"])
        if state in ("terminated", "shutting-down"):
            raise RuntimeError(f"instance {instance_id} ended in {state}")
        if time.time() - started > timeout_s:
            raise TimeoutError(f"instance {instance_id} not running after {timeout_s}s")
        time.sleep(10)


def _root_snapshot_id(ec2: Any, ami_id: str) -> str:
    """Look up the EBS snapshot backing /dev/sda1 for an AMI."""
    img = ec2.describe_images(ImageIds=[ami_id])["Images"][0]
    for b in img.get("BlockDeviceMappings", []):
        if "Ebs" in b and b["Ebs"].get("SnapshotId"):
            return str(b["Ebs"]["SnapshotId"])
    raise RuntimeError(f"AMI {ami_id} has no EBS block device with a snapshot")


def enable_fsr(
    ec2: Any,
    snapshot_id: str,
    az: str,
    *,
    timeout_s: int = 1800,
    credit_wait_s: int = 360,
) -> None:
    """Enable Fast Snapshot Restore for `snapshot_id` in `az`. Blocks until
    state=`enabled` AND at least `credit_wait_s` seconds have passed since
    EnabledTime — the latter is the actually-load-bearing part of the wait.

    Confirmed empirically (2026-05-11): AWS gives FSR credits at roughly
    `60 / size_gib` credits per minute (~1 credit per 5 min for a 300 GB
    snapshot). Volume creation requires a credit at that exact moment, or
    AWS silently falls back to lazy-load (FastRestored=False on the
    resulting volume — no error, no warning). State=enabled with zero
    credits behaves identically to state=disabled from the caller's POV.

    Idempotent: if FSR is already enabled and has been so for longer than
    `credit_wait_s`, returns immediately.
    """
    # Idempotent: check current state first.
    cur = ec2.describe_fast_snapshot_restores(
        Filters=[
            {"Name": "snapshot-id", "Values": [snapshot_id]},
            {"Name": "availability-zone", "Values": [az]},
        ]
    ).get("FastSnapshotRestores", [])
    state = cur[0]["State"] if cur else "disabled"
    if state in ("disabling", "disabled"):
        log.info("fsr.enable", snapshot=snapshot_id, az=az)
        ec2.enable_fast_snapshot_restores(
            AvailabilityZones=[az],
            SourceSnapshotIds=[snapshot_id],
        )
    # Wait for state=enabled. "enabling" → "optimizing" → "enabled".
    started = time.time()
    last_state = ""
    enabled_time: float | None = None
    while True:
        cur = ec2.describe_fast_snapshot_restores(
            Filters=[
                {"Name": "snapshot-id", "Values": [snapshot_id]},
                {"Name": "availability-zone", "Values": [az]},
            ]
        )["FastSnapshotRestores"]
        rec = cur[0] if cur else None
        state = rec["State"] if rec else "disabled"
        if state != last_state:
            log.info("fsr.state", snapshot=snapshot_id, az=az, state=state)
            last_state = state
        if state == "enabled" and rec and rec.get("EnabledTime"):
            enabled_time = rec["EnabledTime"].timestamp()
            break
        if state in ("disabling", "disabled"):
            raise RuntimeError(f"FSR for {snapshot_id} in {az} unexpectedly disabled mid-wait")
        if time.time() - started > timeout_s:
            raise TimeoutError(
                f"FSR for {snapshot_id} in {az} not enabled after {timeout_s}s "
                f"(last state: {state})"
            )
        time.sleep(15)

    # Credit-accumulation wait. If FSR has been enabled for longer than
    # `credit_wait_s` already (e.g. across batches), no further wait needed.
    elapsed_since_enabled = time.time() - enabled_time
    remaining = credit_wait_s - elapsed_since_enabled
    if remaining > 0:
        log.info(
            "fsr.credit_wait",
            snapshot=snapshot_id,
            az=az,
            elapsed_since_enabled=int(elapsed_since_enabled),
            remaining=int(remaining),
        )
        time.sleep(remaining)
    log.info("fsr.ready", snapshot=snapshot_id, az=az)


def disable_fsr(ec2: Any, snapshot_id: str, az: str) -> None:
    """Disable Fast Snapshot Restore. Tolerant of already-disabled state."""
    log.info("fsr.disable", snapshot=snapshot_id, az=az)
    with contextlib.suppress(Exception):
        ec2.disable_fast_snapshot_restores(
            AvailabilityZones=[az],
            SourceSnapshotIds=[snapshot_id],
        )


def _wait_for_quota_release(ec2: Any, *, timeout_s: int = 300) -> None:
    """Block until any of our G-family instances stuck in `shutting-down`
    have fully terminated. Otherwise the next `run_instances` /
    `request_spot_instances` hits VcpuLimitExceeded — this was v3's failure
    mode and recurs whenever a failed instance is rapid-relaunched."""
    started = time.time()
    while True:
        d = ec2.describe_instances(
            Filters=[
                {"Name": "instance-state-name", "Values": ["shutting-down"]},
                {"Name": "instance-type", "Values": ["g6.xlarge", "g6e.xlarge"]},
            ]
        )
        stuck = [i["InstanceId"] for r in d["Reservations"] for i in r["Instances"]]
        if not stuck:
            return
        if time.time() - started > timeout_s:
            raise TimeoutError(f"instances stuck in shutting-down past {timeout_s}s: {stuck}")
        log.info("launch.waiting_for_quota_release", stuck=stuck)
        time.sleep(10)


def launch_for_pending(
    cfg: Config | None = None,
    *,
    preferred_az: str | None = None,
) -> LaunchResult:
    """Launch a spot (or OD if FALLBACK_TO_OD=true after exhaustion).

    Rotates AZs from cfg.SPOT_AZ_ROTATION on capacity errors. The instance
    runs cloud-init user-data on boot which drains the pending queue and
    self-terminates.

    If `preferred_az` is set, the rotation is constrained to that single AZ
    — used by the FSR flow to ensure the spot lands in the AZ where FSR is
    enabled on the AMI's snapshot.
    """
    cfg = cfg or get_config()
    if not cfg.EC2_AMI_ID:
        raise RuntimeError("EC2_AMI_ID is empty — run setup_ami first")
    user_data = render_user_data(cfg)
    user_data_b64 = base64.b64encode(user_data.encode()).decode()
    ec2 = _ec2_client(cfg)
    _wait_for_quota_release(ec2)

    az_rotation = (preferred_az,) if preferred_az else cfg.SPOT_AZ_ROTATION

    if cfg.USE_SPOT:
        last_err: Exception | None = None
        for az in az_rotation:
            try:
                resp = request_spot_in_az(cfg=cfg, az=az, user_data_b64=user_data_b64)
                req_id = resp["SpotInstanceRequests"][0]["SpotInstanceRequestId"]
                # Wait briefly for a spot instance id to be assigned.
                inst_id: str | None = None
                for _ in range(60):
                    desc = ec2.describe_spot_instance_requests(SpotInstanceRequestIds=[req_id])
                    sir = desc["SpotInstanceRequests"][0]
                    if sir.get("InstanceId"):
                        inst_id = sir["InstanceId"]
                        break
                    code = sir.get("Status", {}).get("Code", "")
                    if code in ("capacity-not-available", "constraint-not-fulfillable"):
                        raise RuntimeError(f"spot capacity-not-available in {az}")
                    if code == "bad-parameters" or sir.get("State") == "failed":
                        # Most common: instance type not offered in this AZ.
                        # Cancel the failed request and try the next AZ.
                        with contextlib.suppress(Exception):
                            ec2.cancel_spot_instance_requests(SpotInstanceRequestIds=[req_id])
                        raise RuntimeError(
                            f"spot request {req_id} failed in {az}: "
                            f"{sir.get('Status', {}).get('Message', code)}"
                        )
                    time.sleep(5)
                if not inst_id:
                    raise RuntimeError(f"spot request {req_id} never produced an instance")
                az_real = wait_for_running(ec2, inst_id)
                log.info("launch.spot", instance_id=inst_id, az=az_real)
                return LaunchResult(instance_id=inst_id, az=az_real, spot=True)
            except Exception as exc:
                info = classify(exc)
                msg = str(exc)
                if (
                    info.error_class == ErrorClass.SPOT_CAPACITY_UNAVAILABLE
                    or "capacity-not-available" in msg
                    or "failed in" in msg  # bad-parameters / invalid AZ
                ):
                    log.warning("spot.az_unavailable", az=az, reason=msg[:200])
                    last_err = exc
                    continue
                raise
        if not cfg.FALLBACK_TO_OD:
            raise RuntimeError(f"all AZs exhausted: {last_err!r}")
        log.warning("launch.fallback_to_od")

    resp = request_ondemand(cfg=cfg, user_data_b64=user_data_b64)
    inst = resp["Instances"][0]
    inst_id = inst["InstanceId"]
    az_real = wait_for_running(ec2, inst_id)
    log.info("launch.ondemand", instance_id=inst_id, az=az_real)
    return LaunchResult(instance_id=inst_id, az=az_real, spot=False)
