"""One-command AMI bake for the SteadyDancer-14B pipeline.

Spins a fresh g6e.xlarge from the AWS Deep Learning Base AMI
(Ubuntu 22.04 + CUDA 12.x), installs the package + ML deps,
snapshot-downloads the SteadyDancer-GGUF + DWPose + Wan2.1-base weights,
runs a smoke inference, captures an AMI, writes `EC2_AMI_ID` back to
`.env`.

Expects the following AWS resources to already exist (set in `.env`):
- `EC2_KEY_NAME`            EC2 keypair name for SSH access
- `EC2_SECURITY_GROUP_ID`   Security group with port 22 ingress
- `EC2_IAM_INSTANCE_PROFILE` IAM instance profile with S3 + EC2 permissions

Usage:
    python -m insta_influencer.ec2.setup_ami              # full bake
    python -m insta_influencer.ec2.setup_ami --smoke-only  # stop after smoke
    python -m insta_influencer.ec2.setup_ami --bake-ami i-...   # capture from smoke-passed instance
"""
from __future__ import annotations

import datetime as _dt
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import click

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deep Learning Base AMI: ships with CUDA 12.x + nvidia drivers + Python 3.11.
# Cuts ~30 min of bake time vs starting from plain Ubuntu.
BASE_AMI_PATTERN = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*"

ROOT_VOLUME_GB = 300  # DLAMI base ~140 GB + torch/CUDA ~15 GB + GGUF/Wan2.1/DWPose ~30 GB
# + HF download tmp ~10 GB. v4 hit 100% mid-download on 200 GB; never go below 250.

# S3 keys for setup artifacts (shared bucket; under a setup/ prefix so it
# doesn't collide with job artifacts).
S3_SOURCE_TARBALL = "setup/insta-influencer-source.tar.gz"


# ─────────────────────────────────────────────────────────────────────────
# Source packaging
# ─────────────────────────────────────────────────────────────────────────


def package_source(out_path: Path) -> Path:
    """Tar the package source (no tests, no volumes, no venv) for upload."""
    excludes = {
        ".venv",
        "volumes",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".git",
        "build",
        "dist",
        ".env",
        ".env.bak",
    }

    def _filter(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        parts = set(Path(tar_info.name).parts)
        if parts & excludes:
            return None
        if tar_info.name.endswith(".pyc"):
            return None
        return tar_info

    with tarfile.open(out_path, "w:gz") as tar:
        for name in (
            "insta_influencer",
            "tests",
            "pyproject.toml",
            "README.md",
            "Makefile",
        ):
            src = REPO_ROOT / name
            if src.exists():
                tar.add(src, arcname=name, filter=_filter)
    return out_path


def upload_source_to_s3(cfg: Any, source_tarball: Path) -> str:
    import boto3

    s3 = boto3.client("s3", region_name=cfg.AWS_REGION)
    s3.upload_file(str(source_tarball), cfg.S3_BUCKET, S3_SOURCE_TARBALL)
    return f"s3://{cfg.S3_BUCKET}/{S3_SOURCE_TARBALL}"


def ensure_s3_bucket(cfg: Any) -> None:
    """Create the bucket if it doesn't exist (region-scoped)."""
    import boto3

    s3 = boto3.client("s3", region_name=cfg.AWS_REGION)
    try:
        s3.head_bucket(Bucket=cfg.S3_BUCKET)
        click.echo(f"  [ok]  bucket exists: s3://{cfg.S3_BUCKET}")
        return
    except Exception as exc:
        click.echo(f"  [info] head_bucket: {type(exc).__name__} — will create")
    kwargs: dict[str, Any] = {"Bucket": cfg.S3_BUCKET}
    if cfg.AWS_REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": cfg.AWS_REGION}
    s3.create_bucket(**kwargs)
    click.echo(f"  [+]   bucket created: s3://{cfg.S3_BUCKET}")


# ─────────────────────────────────────────────────────────────────────────
# AWS resource verification — verify required resources exist, don't create
# ─────────────────────────────────────────────────────────────────────────


def verify_aws_resources(cfg: Any) -> tuple[str, str, str]:
    """Returns (key_name, sg_id, profile_name). Raises if anything missing."""
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    iam = boto3.client("iam", region_name=cfg.AWS_REGION)

    key_name = cfg.EC2_KEY_NAME
    sg_id = cfg.EC2_SECURITY_GROUP_ID
    profile = cfg.EC2_IAM_INSTANCE_PROFILE
    if not (key_name and sg_id and profile):
        raise click.UsageError(
            "EC2_KEY_NAME, EC2_SECURITY_GROUP_ID, and EC2_IAM_INSTANCE_PROFILE "
            "must all be set in .env before running setup_ami."
        )

    ec2.describe_key_pairs(KeyNames=[key_name])
    click.echo(f"  [ok]  keypair: {key_name}")
    ec2.describe_security_groups(GroupIds=[sg_id])
    click.echo(f"  [ok]  security group: {sg_id}")
    iam.get_instance_profile(InstanceProfileName=profile)
    click.echo(f"  [ok]  iam profile: {profile}")
    return key_name, sg_id, profile


def find_base_ami(cfg: Any) -> str:
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    resp = ec2.describe_images(
        Owners=["amazon"],
        Filters=[
            {"Name": "name", "Values": [BASE_AMI_PATTERN]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": ["x86_64"]},
        ],
    )
    images = sorted(resp["Images"], key=lambda x: x["CreationDate"], reverse=True)
    if not images:
        raise RuntimeError(f"no Deep Learning Base AMI in {cfg.AWS_REGION}")
    img = images[0]
    click.echo(f"  [ok]  base AMI: {img['ImageId']}  ({img['Name'][:70]})")
    return str(img["ImageId"])


# ─────────────────────────────────────────────────────────────────────────
# User-data builder
# ─────────────────────────────────────────────────────────────────────────


USER_DATA_TEMPLATE = r"""#!/bin/bash
set -euxo pipefail
exec >> /var/log/insta-influencer-setup.log 2>&1
echo "=== insta-influencer AMI setup starting at $(date -uIs) ==="

# Tag the instance as in-progress; the local poller watches for a terminal value.
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60")
INSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 create-tags --region "{aws_region}" --resources "$INSTANCE_ID" \
    --tags "Key=SetupStatus,Value=installing" || true

# Expand root volume to full size.
growpart /dev/nvme0n1 1 || true
resize2fs /dev/nvme0n1p1 || resize2fs /dev/root || true
df -h /

# 32 GB swap (g6e.xlarge has 32 GB RAM but model loads can spike).
if [ ! -f /swapfile ]; then
    fallocate -l 32G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    grep -q '^/swapfile ' /etc/fstab || echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
fi

# Wait for any in-flight apt/dpkg locks.
echo "waiting for apt locks..."
for i in $(seq 1 60); do
    if ! fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
       && ! fuser /var/lib/apt/lists/lock >/dev/null 2>&1 \
       && ! fuser /var/lib/dpkg/lock >/dev/null 2>&1; then
        break
    fi
    sleep 10
done

systemctl stop unattended-upgrades || true
systemctl disable unattended-upgrades || true

apt-get update -y
apt-get install -y ffmpeg python3.11-venv python3.11-dev awscli git build-essential

# Project directory
mkdir -p /opt/insta-influencer
cd /opt/insta-influencer

# Pull source tarball from S3.
aws s3 cp "s3://{s3_bucket}/{s3_source}" /tmp/source.tar.gz
tar -xzf /tmp/source.tar.gz -C /opt/insta-influencer
ls -la /opt/insta-influencer

# Build venv
python3.11 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# Pin torch + CUDA build matching the DLAMI's CUDA 12.1.
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Project deps including [ec2] extras (diffusers, transformers, gguf, onnxruntime-gpu,
# opencv-python-headless, einops, decord, moviepy, av, easydict, ftfy, dashscope, numpy<2).
pip install -e '.[ec2]'

# ── Upstream SteadyDancer install ladder (per its README) ───────────────
# flash_attn, xformers, xfuser are required by Wan2.1's attention path.
# mmcv MUST be built from source with CUDA ops; mmpose / mmdet then attach.
# This block adds ~25-45 min to the bake (mmcv compile dominates).
FLASH_ATTN_WHL='https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1'
FLASH_ATTN_WHL="$FLASH_ATTN_WHL/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
pip install "$FLASH_ATTN_WHL" \
    || pip install flash_attn==2.7.4.post1 --no-build-isolation
pip install xformers==0.0.29.post1
pip install 'xfuser[diffusers,flash-attn]' || true   # optional; SteadyDancer can run without

# Pose extraction stack (mmpose + mmdet + mmcv built with CUDA ops).
pip install -U pip setuptools wheel
pip install openmim
pip uninstall -y mmcv mmcv-full mmcv-lite mmpose mmdet mmengine 2>/dev/null || true
mim install mmengine
git clone --depth=1 -b v2.1.0 https://github.com/open-mmlab/mmcv.git \
    /opt/insta-influencer/third_party/mmcv
cd /opt/insta-influencer/third_party/mmcv
MMCV_WITH_OPS=1 MAX_JOBS=$(nproc) python setup.py build_ext
MMCV_WITH_OPS=1 MAX_JOBS=$(nproc) python setup.py develop
cd /opt/insta-influencer
mim install 'mmdet>=3.1.0'
# chumpy is a transitive mmpose dep whose setup.py does `import pip` from a
# subprocess that has no pip under PEP 517 build isolation. Pre-install with
# --no-build-isolation against the venv's existing setuptools+numpy.
pip install --no-build-isolation chumpy==0.70 || true
pip install mmpose==1.3.2

# torch 2.5.1 has C ABI breaks against numpy 2.x; mmcv was compiled here
# against numpy 1.x earlier in this script and would SEGV on numpy 2.x at
# runtime. Some transitive in the install ladder above can bump numpy to 2.x,
# so re-pin it last. opencv-python <4.13 also keeps numpy 1.x-friendly (4.13
# requires numpy>=2). Headless variant avoids libGL on the DLAMI base.
#
# IMPORTANT (Bug 24): opencv-python and opencv-python-headless share the same
# `cv2/` directory in site-packages. Uninstalling opencv-python AFTER
# installing the headless variant wipes cv2's files even though the headless
# pip metadata is preserved. Always uninstall the full variant FIRST, then
# install headless.
pip uninstall -y opencv-python || true
pip install --upgrade --force-reinstall --no-deps 'opencv-python-headless<4.13'
pip install --upgrade 'numpy>=1.26,<2'

# Free disk before the heavy weight downloads. The DLAMI base + our pip
# installs leave only ~55 GB on a 200 GB volume; v4 ran out mid-download.
# 300 GB volume + cache trim gives us comfortable headroom.
echo "=== disk before cleanup ==="
df -h /
apt-get clean -y
pip cache purge || true
rm -rf /root/.cache/pip /home/ubuntu/.cache/pip 2>/dev/null || true
# Drop unused pre-baked DLAMI conda artifacts (we ship our own venv).
rm -rf /opt/conda/pkgs /home/ubuntu/anaconda3/pkgs 2>/dev/null || true
echo "=== disk after cleanup ==="
df -h /

# Persist HF token so huggingface_hub auths automatically during the snapshot
# downloads below (the GGUF + base Wan2.1 repos require accepting model card terms).
export HF_TOKEN="{hf_token}"
export HUGGING_FACE_HUB_TOKEN="{hf_token}"
# Predictable cache location: cloud-init runs as root by default (HOME=/root),
# but at runtime the orchestrator may run under a different user. Pin HF_HOME
# so the same cache is found in both phases.
export HF_HOME=/opt/insta-influencer/hf-cache
mkdir -p "$HF_HOME"

# Clone SteadyDancer upstream repo at the pinned commit (or main).
mkdir -p /opt/insta-influencer/third_party
git clone --depth=1 https://github.com/MCG-NJU/SteadyDancer \
    /opt/insta-influencer/third_party/SteadyDancer
SDANCER_SHA="{steadydancer_git_sha}"
# Only attempt checkout if the value looks like a valid git SHA (hex,
# 7-40 chars). Defensive against accidental inline comments leaking
# into the env value via .env parsing.
if [[ "$SDANCER_SHA" =~ ^[0-9a-f]{{7,40}}$ ]]; then
    cd /opt/insta-influencer/third_party/SteadyDancer
    git fetch --depth=1 origin "$SDANCER_SHA"
    git checkout "$SDANCER_SHA"
    cd /opt/insta-influencer
else
    echo "(no valid STEADYDANCER_GIT_SHA, using main: '$SDANCER_SHA')"
fi

# Pre-download model weights into the HuggingFace cache so the AMI is
# warm-start at runtime. This is the single biggest chunk of bake time
# (~25-40 min for the GGUF + Wan2.1 base).
echo "=== disk before SteadyDancer download ==="
df -h /
echo "=== snapshot_download SteadyDancer GGUF (Q5_K_M only) ==="
python - <<'PYEOF'
from huggingface_hub import snapshot_download
# Pull the Q5_K_M variant only — saves ~15 GB vs the full multi-quant repo.
snapshot_download(
    "{hf_steadydancer_gguf}",
    # Tightened post-v5: previous over-broad pattern pulled the full bf16
    # safetensors (~65 GB unnecessary). Q5_K_M GGUF + configs only.
    allow_patterns=["*Q5_K_M*.gguf", "*.json", "*.txt"],
)
print("steadydancer-gguf cached")
PYEOF

echo "=== disk after SteadyDancer; before Wan2.1 ==="
df -h /
echo "=== snapshot_download Wan2.1-I2V base (encoders + VAE + DiT shards) ==="
python - <<'PYEOF'
from huggingface_hub import snapshot_download
# Originally we only pulled encoders/VAE on the assumption the GGUF DiT
# would be the inference path. But upstream generate_dancer.py uses
# diffusers' WanModel.from_pretrained() which only knows safetensors —
# it raises FileNotFoundError on the missing diffusion_pytorch_model
# shards even when the GGUF is present (Bug 19 in 07_session_log_handoff).
# Pull the bf16 shards too (~28 GB). Total cache after this is ~38 GB,
# which fits in the 300 GB root volume.
snapshot_download(
    "Wan-AI/Wan2.1-I2V-14B-480P",
    allow_patterns=[
        "*config.json",
        "*tokenizer*",
        "*.json",
        "models_t5*",
        "models_clip*",
        "Wan2.1_VAE*",
        "google*",
        "diffusion_pytorch_model*",
    ],
)
print("wan2.1-base cached")
PYEOF

echo "=== disk after Wan2.1; before DWPose ==="
df -h /
echo "=== snapshot_download DWPose ==="
python - <<'PYEOF'
from huggingface_hub import snapshot_download
snapshot_download("{hf_dwpose}")
print("dwpose cached")
PYEOF

# pose_align.py defaults look for ckpts at <repo>/preprocess/pretrained_weights/dwpose/.
# DWPose HF repo only ships dw-ll_ucoco_384.pth (and yolox_l.onnx, not .pth).
# Stage both into upstream's expected location so default args work without
# having to thread --yolox_ckpt/--dwpose_ckpt through our wrapper.
PRETRAINED_DIR=/opt/insta-influencer/third_party/SteadyDancer/preprocess/pretrained_weights/dwpose
mkdir -p "$PRETRAINED_DIR"
DW_LL=$(find /opt/insta-influencer/hf-cache -name "dw-ll_ucoco_384.pth" | head -1)
ln -sf "$DW_LL" "$PRETRAINED_DIR/dw-ll_ucoco_384.pth"
wget -q -O "$PRETRAINED_DIR/yolox_l_8x8_300e_coco.pth" \
    "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth"

# torch 2.5.1+cu121 ships torch.distributed.optim.functional_* with
# @torch.jit.script class wrappers that SEGV at import time on this stack
# (libstdc++/libtorch combo on the DLAMI). The functional optimizers are
# deprecated upstream and we don't use them. Pin the env var globally.
echo 'export PYTORCH_JIT=0' >> /etc/profile.d/insta-influencer.sh
chmod 0755 /etc/profile.d/insta-influencer.sh

# Pre-warm the page cache for the heaviest model files. Without this,
# `WanModel.from_pretrained()` does mmap-style random 4KB page faults on
# cold EBS gp3 (~7 MB/s effective) and a 14B model load takes 60-90 min.
# Sequential `cat ... > /dev/null` reads at gp3's 125 MB/s sequential
# bandwidth, finishing the same 28 GB in ~4 min. (See Bugs 20+23 and the
# critical follow-up about EBS pre-warm.)
echo "=== prewarm page cache for model files ==="
for f in $(find /opt/insta-influencer/hf-cache -name "diffusion_pytorch_model*.safetensors" \
                -o -name "models_t5*.safetensors" -o -name "*.gguf" -o -name "Wan2.1_VAE*"); do
    echo "  prewarm: $f"
    cat "$f" > /dev/null || true
done

# Smoke test: render a 33-frame clip end-to-end. Validates that the model
# weights load + inference path runs. Visual quality is NOT asserted —
# real validation comes from the operator's first prepare+generate run.
echo "=== disk after all downloads ==="
df -h /
echo "=== running smoke ==="
export STORAGE_BACKEND=local ANIMATE_FAKE=0 INSTA_SPOT_WATCH=0 PYTORCH_JIT=0
mkdir -p /opt/insta-influencer/volumes/output/batch /opt/insta-influencer/volumes/store \
         /opt/insta-influencer/volumes/logs /opt/insta-influencer/volumes/assets
if python -m insta_influencer.ec2.smoke_test --num-frames 33; then
    SMOKE=passed
else
    SMOKE=failed
fi

# Upload the smoke log + final.mp4 (if any) for operator review.
aws s3 cp /var/log/insta-influencer-setup.log \
    "s3://{s3_bucket}/setup/last-bake.log" --region "{aws_region}" || true
if [ -f /opt/insta-influencer/volumes/output/batch/smk000000000/final.mp4 ]; then
    aws s3 cp /opt/insta-influencer/volumes/output/batch/smk000000000/final.mp4 \
        "s3://{s3_bucket}/setup/last-smoke-final.mp4" --region "{aws_region}" || true
fi

aws ec2 create-tags --region "{aws_region}" --resources "$INSTANCE_ID" \
    --tags "Key=SetupStatus,Value=smoke-$SMOKE"

echo "=== insta-influencer AMI setup finished ($SMOKE) at $(date -uIs) ==="
"""


def build_user_data(cfg: Any) -> str:
    if not cfg.HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN missing. SteadyDancer-GGUF + Wan2.1-base require an HF token "
            "with model card terms accepted."
        )
    return USER_DATA_TEMPLATE.format(
        aws_region=cfg.AWS_REGION,
        s3_bucket=cfg.S3_BUCKET,
        s3_source=S3_SOURCE_TARBALL,
        hf_token=cfg.HF_TOKEN,
        hf_steadydancer_gguf=cfg.HF_STEADYDANCER_GGUF,
        hf_dwpose=cfg.HF_DWPOSE,
        steadydancer_git_sha=cfg.STEADYDANCER_GIT_SHA,
    )


# ─────────────────────────────────────────────────────────────────────────
# EC2 lifecycle
# ─────────────────────────────────────────────────────────────────────────


def launch_setup_instance(
    cfg: Any,
    *,
    base_ami: str,
    sg_id: str,
    key_name: str,
    profile_name: str,
    user_data: str,
    instance_type: str,
) -> str:
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    kwargs: dict[str, Any] = dict(
        ImageId=base_ami,
        InstanceType=instance_type,
        MinCount=1,
        MaxCount=1,
        KeyName=key_name,
        SecurityGroupIds=[sg_id],
        UserData=user_data,
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {"VolumeSize": ROOT_VOLUME_GB, "VolumeType": "gp3"},
            }
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {"Key": "Name", "Value": "insta-influencer-ami-setup"},
                    {"Key": "Project", "Value": "insta-influencer"},
                    {"Key": "SetupStatus", "Value": "boot"},
                ],
            }
        ],
    )
    if profile_name:
        kwargs["IamInstanceProfile"] = {"Name": profile_name}
    if getattr(cfg, "USE_SPOT", False):
        # Spot quota in this account is 8+ vCPU; OD is 4 vCPU. g6e.2xlarge
        # bakes only fit on spot. SpotOptions.SpotInstanceType=one-time +
        # InstanceInterruptionBehavior=terminate matches the production
        # launch path's behavior. MaxPrice from .env (default $1.00/hr).
        kwargs["InstanceMarketOptions"] = {
            "MarketType": "spot",
            "SpotOptions": {
                "MaxPrice": str(cfg.EC2_SPOT_MAX_PRICE),
                "SpotInstanceType": "one-time",
                "InstanceInterruptionBehavior": "terminate",
            },
        }
    resp = ec2.run_instances(**kwargs)
    instance_id: str = resp["Instances"][0]["InstanceId"]
    market = "spot" if getattr(cfg, "USE_SPOT", False) else "on-demand"
    click.echo(f"  [+]   launched: {instance_id}  ({market})")
    return instance_id


def poll_setup_status(cfg: Any, instance_id: str, *, timeout_s: int = 5400) -> str:
    """Poll the SetupStatus tag until smoke-passed | smoke-failed | timeout."""
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    started = time.time()
    last_status = ""
    while time.time() - started < timeout_s:
        try:
            resp = ec2.describe_tags(
                Filters=[
                    {"Name": "resource-id", "Values": [instance_id]},
                    {"Name": "key", "Values": ["SetupStatus"]},
                ]
            )
            for t in resp.get("Tags", []):
                v = t.get("Value", "")
                if v != last_status:
                    elapsed = int(time.time() - started)
                    click.echo(f"    [{elapsed:>4}s] SetupStatus={v}")
                    last_status = v
                if v in ("smoke-passed", "smoke-failed"):
                    return str(v)
        except Exception as exc:
            click.echo(f"    [warn] describe_tags: {exc}")
        time.sleep(20)
    return "timeout"


def stream_setup_log_tail(cfg: Any, n: int = 60) -> None:
    """Best-effort: pull the last bake log from S3 and print the tail."""
    import boto3

    s3 = boto3.client("s3", region_name=cfg.AWS_REGION)
    try:
        body = s3.get_object(Bucket=cfg.S3_BUCKET, Key="setup/last-bake.log")["Body"].read()
        text = body.decode("utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-n:])
        click.echo("─── last-bake.log tail ───")
        click.echo(tail)
        click.echo("─── end ───")
    except Exception as exc:
        click.echo(f"(could not fetch bake log: {exc})")


def create_ami(cfg: Any, instance_id: str) -> str:
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    # Spot one-time instances can't be stopped (UnsupportedOperation: Bug 25).
    # Detect lifecycle: 'spot' means SpotInstanceType=one-time → must skip
    # the stop. NoReboot=True below means the snapshot is taken live; for our
    # bake-then-snapshot flow there are no active writes, so this is safe.
    desc = ec2.describe_instances(InstanceIds=[instance_id])
    inst = desc["Reservations"][0]["Instances"][0]
    is_spot = inst.get("InstanceLifecycle") == "spot"
    if is_spot:
        click.echo("  spot instance detected; skipping stop (NoReboot snapshot is safe)")
    else:
        click.echo("  stopping instance...")
        ec2.stop_instances(InstanceIds=[instance_id])
        ec2.get_waiter("instance_stopped").wait(InstanceIds=[instance_id])

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%d-%H%M")
    name = f"insta-influencer-steadydancer-{ts}"
    resp = ec2.create_image(
        InstanceId=instance_id,
        Name=name,
        Description=(
            "insta-influencer pipeline: SteadyDancer-14B GGUF + DWPose + Wan2.1 base, "
            "smoke-tested"
        ),
        NoReboot=True,
        TagSpecifications=[
            {
                "ResourceType": "image",
                "Tags": [
                    {"Key": "Project", "Value": "insta-influencer"},
                    {"Key": "BakedAt", "Value": ts},
                    {"Key": "Pipeline", "Value": "steadydancer"},
                ],
            }
        ],
    )
    ami_id: str = resp["ImageId"]
    click.echo(f"  [+]   AMI creating: {ami_id} ({name})")
    click.echo("  waiting for AMI available (200 GB snapshot — ~30 min)...")
    ec2.get_waiter("image_available").wait(
        ImageIds=[ami_id], WaiterConfig={"Delay": 30, "MaxAttempts": 240}
    )
    click.echo(f"  [ok]  AMI ready: {ami_id}")
    return ami_id


def save_ami_to_env(ami_id: str) -> None:
    env_path = REPO_ROOT / ".env"
    text = env_path.read_text() if env_path.exists() else ""
    lines = text.splitlines()
    found = False
    for i, ln in enumerate(lines):
        if ln.startswith("EC2_AMI_ID="):
            lines[i] = f"EC2_AMI_ID={ami_id}"
            found = True
            break
    if not found:
        lines.append(f"EC2_AMI_ID={ami_id}")
    env_path.write_text("\n".join(lines) + "\n")
    click.echo(f"  [+]   .env updated: EC2_AMI_ID={ami_id}")


def terminate_instance(cfg: Any, instance_id: str) -> None:
    import boto3

    ec2 = boto3.client("ec2", region_name=cfg.AWS_REGION)
    ec2.terminate_instances(InstanceIds=[instance_id])
    click.echo(f"  [-]   instance terminated: {instance_id}")


# ─────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────


@click.command("setup-ami")
@click.option(
    "--instance-type",
    default="g6e.2xlarge",
    help=(
        "Bake instance. g6e.2xlarge (L40S 48GB GPU + 64GB RAM) is required: "
        "Wan2.1-I2V-14B in bf16 (~28GB) + T5-XXL on CPU (~9GB) overflows the "
        "30GB RAM on g6e.xlarge and causes swap thrashing (Bug 20)."
    ),
)
@click.option(
    "--smoke-only",
    is_flag=True,
    default=False,
    help="Stop after smoke; don't capture AMI. For manual review.",
)
@click.option(
    "--bake-ami",
    "bake_from_instance",
    type=str,
    default=None,
    help="Skip install/smoke and capture an AMI from an existing smoke-passed instance.",
)
@click.option(
    "--keep-instance/--no-keep-instance",
    default=False,
    help="Don't terminate the bake instance after success (for SSH debugging).",
)
def setup_ami(
    instance_type: str,
    smoke_only: bool,
    bake_from_instance: str | None,
    keep_instance: bool,
) -> None:
    """Bake a SteadyDancer-ready AMI. ~75-90 min, ~$3."""
    from ..config import load_config

    cfg = load_config()

    # Short-circuit: just bake an AMI from a known good instance.
    if bake_from_instance:
        click.echo(f"=== capturing AMI from {bake_from_instance} ===")
        ami_id = create_ami(cfg, bake_from_instance)
        save_ami_to_env(ami_id)
        terminate_instance(cfg, bake_from_instance)
        click.echo(f"\n[done] EC2_AMI_ID={ami_id}")
        return

    click.echo("=== insta-influencer AMI bake ===")
    click.echo(f"  region:        {cfg.AWS_REGION}")
    click.echo(f"  instance:      {instance_type}")
    click.echo(f"  bucket:        s3://{cfg.S3_BUCKET}")
    click.echo(f"  hf_steadydancer: {cfg.HF_STEADYDANCER_GGUF}")
    click.echo(f"  hf_dwpose:       {cfg.HF_DWPOSE}")
    click.echo(f"  pinned commit:   {cfg.STEADYDANCER_GIT_SHA or '(main)'}")
    click.echo("")

    # Step 1: AWS prereqs
    click.echo("[1/8] verifying AWS resources...")
    ensure_s3_bucket(cfg)
    key_name, sg_id, profile = verify_aws_resources(cfg)
    base_ami = find_base_ami(cfg)

    # Step 2: package source
    click.echo("[2/8] packaging source...")
    tarball = REPO_ROOT / "build" / "insta-influencer-source.tar.gz"
    tarball.parent.mkdir(parents=True, exist_ok=True)
    package_source(tarball)
    size_mb = tarball.stat().st_size / 1024 / 1024
    click.echo(f"  [ok]  tarball: {tarball.name} ({size_mb:.1f} MB)")

    # Step 3: upload to S3
    click.echo("[3/8] uploading source to S3...")
    s3_uri = upload_source_to_s3(cfg, tarball)
    click.echo(f"  [ok]  uploaded: {s3_uri}")

    # Step 4: build user-data
    click.echo("[4/8] building user-data script...")
    user_data = build_user_data(cfg)
    click.echo(f"  [ok]  user-data length: {len(user_data)} bytes")
    if len(user_data) > 16 * 1024:
        click.echo(
            f"  [WARN] user-data is {len(user_data)} bytes (max 16 KB). "
            "Trim or move logic into the source tarball."
        )

    # Step 5: launch
    click.echo("[5/8] launching bake instance...")
    instance_id = launch_setup_instance(
        cfg,
        base_ami=base_ami,
        sg_id=sg_id,
        key_name=key_name,
        profile_name=profile,
        user_data=user_data,
        instance_type=instance_type,
    )

    # Step 6: poll status
    click.echo(f"[6/8] polling setup status (timeout 90 min)... instance: {instance_id}")
    status = poll_setup_status(cfg, instance_id, timeout_s=5400)
    click.echo(f"  status: {status}")

    if status != "smoke-passed":
        click.echo(f"\n[fail] bake did not pass smoke ({status}).")
        stream_setup_log_tail(cfg)
        click.echo(f"\nInstance preserved for inspection: {instance_id}")
        click.echo(
            f"SSH: ssh -i <your-keypair.pem> ubuntu@<public-ip>"
        )
        sys.exit(1)

    if smoke_only:
        click.echo("\n[smoke-only] stopping here. To capture the AMI later:")
        click.echo(f"  python -m insta_influencer.ec2.setup_ami --bake-ami {instance_id}")
        return

    # Step 7: AMI capture
    click.echo("[7/8] capturing AMI...")
    ami_id = create_ami(cfg, instance_id)
    save_ami_to_env(ami_id)

    # Step 8: cleanup
    click.echo("[8/8] cleanup...")
    if keep_instance:
        click.echo(f"  [keep] instance {instance_id} preserved")
    else:
        terminate_instance(cfg, instance_id)

    click.echo(f"\n[done] EC2_AMI_ID={ami_id}")


if __name__ == "__main__":
    setup_ami()


