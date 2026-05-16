# 05 — AWS Setup

Reproducible AWS configuration for the pipeline. Covers IAM, AMI
baking, Fast Snapshot Restore (FSR) lifecycle, cost expectations, and
spot-interruption recovery.

---

## 1. Required AWS resources

| Resource | Purpose |
|---|---|
| IAM user (programmatic access) | Local-machine credentials for boto3 |
| IAM role for EC2 instances | Allows the spot instance to pull/push S3 |
| EC2 keypair | SSH access for debugging |
| Security group | Allow SSH (port 22) from operator IP only |
| S3 bucket | Job inputs, outputs, logs, status, phase markers |
| (Optional) Default VPC + subnet | Uses default if subnet is not set |

---

## 2. IAM policies

### 2.1 Local user (boto3 client)

Minimum policy for the local driver. Permissions needed:

| Service | Action(s) | Resource |
|---|---|---|
| S3 | `GetObject`, `PutObject`, `ListBucket`, `DeleteObject` | `<bucket>`, `<bucket>/*` |
| EC2 | `RunInstances`, `RequestSpotInstances`, `DescribeInstances`, `DescribeImages`, `DescribeSpotInstanceRequests`, `TerminateInstances`, `CreateImage`, `DescribeSnapshots` | `*` |
| EC2 FSR | `EnableFastSnapshotRestores`, `DisableFastSnapshotRestores`, `DescribeFastSnapshotRestores` | snapshot ARN |
| IAM | `PassRole` | the EC2 role ARN |

### 2.2 EC2 instance role

The role attached to launched spots needs:

| Service | Action(s) | Resource |
|---|---|---|
| S3 | `GetObject`, `PutObject`, `ListBucket` | `<bucket>`, `<bucket>/*` |

Boto3 on the instance gets credentials from IMDS; no need to set
`AWS_ACCESS_KEY_ID` in the instance environment.

---

## 3. AMI bake (one-time)

```
local                                EC2 bake instance (g6e.xlarge)
-----                                ------------------------------
setup_ami.py                  -->    launch DLAMI base
   |  upload bake user-data
   |                                 install OS packages
   |                                 create Python venv
   |                                 pip install project + [ec2] extras
   |                                 snapshot_download HF weights:
   |                                   - SteadyDancer-14B (~28 GB)
   |                                   - Wan-AI/Wan2.1-I2V-14B-480P (~28 GB DiT + 12 GB T5 + 5 GB CLIP + 0.5 GB VAE)
   |                                   - DWPose ckpts
   |                                 wget yolox_l_8x8_300e_coco.pth from mmdetection release
   |                                 symlink DWPose ckpts to upstream path
   |                                 prewarm facexlib detection weights
   |                                 (optional) clone Practical-RIFE + download v4.26 weights
   |                                 (optional) download GFPGANv1.4.pth
   |                                 (optional) sed-patch basicsr and skvideo
   |                                 1-frame 1-step smoke
   |                                 stop services / clean work dirs
   |                                 create-image
   |
   |<-- AMI id
```

The bake instance is g6e.xlarge to keep cost contained. Total wall
time for a clean bake from DLAMI base is ~75-90 minutes; cost
~$3 at standard pricing or proportionally lower under credits.

After the bake, the AMI id is written into `.env` as `EC2_AMI_ID` and
subsequent `generate` runs launch from it.

### 3.1 What to include in the AMI

For reduced cold-launch time, the AMI should include everything that
cloud-init currently installs at runtime:

- SteadyDancer-14B weights at the cache path
- Wan2.1-I2V-14B-480P DiT shards, pre-converted to bf16 if loader
  permits (~10 min/launch savings)
- DWPose ckpts at the upstream-expected path
- Practical-RIFE clone + v4.26 weights
- GFPGAN v1.4 weights
- basicsr and skvideo with sed-patches applied
- Pinned `numpy<2` and verified `xtcocotools` import

This eliminates the dep-race conditions in cloud-init (Bugs 38-41)
and saves ~7-10 min per spot launch.

---

## 4. Fast Snapshot Restore (FSR) lifecycle

### 4.1 Why FSR is required

A gp3 volume created from an AMI snapshot reads at ~7-11 MB/s
effective on first-touch blocks unless FSR is enabled. With ~38 GB
of HF weight cache to read, this throttle alone adds ~80 minutes to
the first run.

The drivers enable FSR just-in-time before launching the spot and
disable it immediately after. FSR's effect is consumed at
`create_volume`; the volume keeps its initialised state after FSR is
disabled.

### 4.2 FSR credit-wait

AWS issues FSR credits at approximately `60 / snapshot_gib` credits
per minute (~1 credit per 5 minutes for a 300 GB snapshot). Volume
creation consumes one credit. If the bucket is empty at
`create_volume`, AWS silently falls back to lazy-load: `FastRestored:
False` on the volume, no error, no warning.

The launch driver reads `EnabledTime` from the API after FSR reaches
`state=enabled` and sleeps until at least 360 seconds have elapsed.
This is idempotent.

### 4.3 FSR cost

| Item | Cost |
|---|---|
| FSR standing charge | ~$0.75 / AZ / hour while enabled |
| Per-batch FSR window (enable + optimise + credit wait + launch) | ~12-20 min, ~$0.15-$0.25 |

vs:

| Alternative | Penalty |
|---|---|
| Leave FSR enabled 24/7 | ~$18/day per AZ |
| Run without FSR | ~80 min wall penalty per first-cold run |

Just-in-time FSR is the cost-minimising choice for low-frequency use.

### 4.4 Manual FSR override

If the operator needs to enable/disable FSR outside the driver
flow:

```bash
aws ec2 enable-fast-snapshot-restores \
    --region <region> \
    --availability-zones <az> \
    --source-snapshot-ids <ami-root-snapshot-id>
# wait at least 360 s after state=enabled
aws ec2 describe-fast-snapshot-restores \
    --region <region> \
    --filters Name=availability-zone,Values=<az> \
              Name=snapshot-id,Values=<ami-root-snapshot-id>
# launch spot in <az>
aws ec2 disable-fast-snapshot-restores \
    --region <region> \
    --availability-zones <az> \
    --source-snapshot-ids <ami-root-snapshot-id>
```

### 4.5 Code references

| Function | File | Behaviour |
|---|---|---|
| `enable_fsr` | `ec2/launch.py` | Idempotent; blocks until `state=enabled`; sleeps for credit-wait if `EnabledTime` is recent |
| `disable_fsr` | `ec2/launch.py` | Idempotent; suppresses errors |
| `launch_for_pending(preferred_az=...)` | `ec2/launch.py` | Single-AZ override so the spot lands where FSR was enabled |
| `cli/generate.py` | top level | Wraps launch with try/except/finally: enable FSR -> launch -> disable FSR (even on launch failure) |

---

## 5. Cost expectations

### 5.1 Per-run cost (2-clip / 10 sec output)

| Item | Cost |
|---|---|
| g6e.2xlarge spot effective rate | ~$0.04/hr (with credits applied to ~$2.24/hr on-demand) |
| Wall time | ~2h 20m |
| Spot cost | ~$0.10 effective / ~$5 on-demand |
| FSR window cost | ~$0.20 |
| **Net per run with credits** | **~$3** |

Components of the wall time:

| Phase | Wall |
|---|---|
| FSR enable + optimise + credit wait | ~22 min (local-side) |
| Spot boot + cloud-init + downloads | ~11 min |
| pose_extract (dual pass) | ~2 min |
| animate (DiT load + 2 x 33 min sampling, daemonised) | ~1h 33m |
| interp (RIFE) | ~30 s |
| face_restore (GFPGAN) | ~4 min |
| audio_attach + reels_format | <10 s |

### 5.2 Per-run cost scaling

The marginal cost of an additional chunk is the GPU sampling time
plus a small interp delta — no additional FSR / boot / model-load
overhead.

| Clips | Output duration | Wall (approx) | Net cost with credits |
|---|---|---|---|
| 1 | 5 s | ~1h 17m | ~$2 |
| 2 | 10 s | ~2h 20m | ~$3 |
| 3 | 15 s | ~2h 50m | ~$3.50 |
| 4 | 20 s | ~3h 25m | ~$4 |

---

## 6. Spot interruption recovery

### 6.1 What happens on reclamation

AWS sends a two-minute interruption notice to instance metadata. The
running orchestrator does not currently react to this notice; the
spot is simply terminated.

On the next launch, the orchestrator's startup sequence:

1. Pulls `status.json` and any cached artefacts from S3 into the
   fresh work directory.
2. Iterates phases. For each phase: if both the S3 marker and the
   on-disk output are present, the phase is skipped; otherwise, it
   re-runs.

Phases whose outputs are uploaded to S3 (`animate` -> `animated.mp4`
plus per-chunk mp4s; final phases -> `final.mp4`) recover their
outputs from S3 at startup and skip cleanly. Phases whose outputs
are local-only (`pose_extract` -> hundreds of MB of per-frame JPGs)
re-run on the fresh spot. See
[`04_bugs_encountered.md`](./04_bugs_encountered.md) Bug 46.

### 6.2 KEEP_ALIVE_ON_FAILURE

For debugging post-animate phase failures without re-paying FSR +
boot, set `KEEP_ALIVE_ON_FAILURE=1` (default 0) and
`KEEP_ALIVE_SECONDS=3600` (default 3600). On non-zero pipeline exit,
the spot sleeps before termination. The operator can:

```bash
ssh -i <key> ubuntu@<public-ip>
# fix the issue, e.g. patch a phase
python -m reel_forge.ec2.orchestrator process-job <job_id>
```

This is the right default for any infrastructure-change run.

---

## 7. Where AWS configuration lives in code

| Knob | File | Default |
|---|---|---|
| Region | `app/config.py` (Config.AWS_REGION) | `eu-south-2` |
| Spot instance type | `app/config.py` | `g6e.2xlarge` |
| Spot max price | `app/config.py` | per-`.env` |
| FSR enable/disable | `ec2/launch.py` | code |
| AMI id | `.env` (EC2_AMI_ID) | written by `setup_ami.py` |
| Spot AZ rotation | `app/config.py` | comma-separated; FSR overrides to single AZ |
| Root volume size | `setup_ami.py` (ROOT_VOLUME_GB) | 300 |
| S3 bucket / prefix | `app/config.py` | `.env`-driven |

---

## 8. Pre-flight checklist for a fresh AWS account

1. Create an IAM user with programmatic access; attach the policy
   from §2.1. Record the access key id + secret in `.env`.
2. Create an EC2 instance role with the policy from §2.2. Record the
   instance profile name in `.env` as `EC2_IAM_INSTANCE_PROFILE`.
3. Verify the G-family vCPU quota in the target region is at least
   8 (a g6e.2xlarge takes 8 vCPU). Request a quota increase if not.
4. Create an EC2 keypair; download the `.pem`; record the name in
   `.env` as `EC2_KEY_NAME`.
5. Create a security group allowing inbound TCP 22 from the
   operator's IP; record the group id in `.env` as
   `EC2_SECURITY_GROUP_ID`.
6. Create an S3 bucket; record the name in `.env` as `S3_BUCKET`.
7. Set `HF_TOKEN` in `.env` (read-only token is sufficient).
   Accept the SteadyDancer-14B and Wan-AI/Wan2.1-I2V-14B-480P model
   card terms on HuggingFace.
8. Run `python -m reel_forge.ec2.setup_ami` to bake the AMI.
   Wait ~75-90 min.
9. `setup_ami.py` writes the AMI id to `.env` on success.
10. Run a smoke `generate` job. See §5.1 for cost expectations.
