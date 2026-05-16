#!/usr/bin/env python
"""Seed a postprocess-only iteration from an existing 60fps final.mp4.

Downsamples a 60fps mp4 to 16fps (the model's native rate), uploads it as
animated.mp4 to S3 under the target job_id, and uploads pose_extract +
animate phase markers so the orchestrator skips both phases. Result: a
spot launch will run only interp + face_restore + audio_attach +
reels_format (~25 min, ~$0.40 vs ~2h 40m / ~$1.80 for a full run).

Usage:
    python scripts/seed_postprocess.py --job <job_id> --source <60fps.mp4>

The job_id must already have a manifest.json + photo.png + reference.mp4
on S3 (any completed prior run works). The script overwrites animated.mp4
+ status.json + pose_extract/animate markers, then prints the generate
command to run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import boto3

from reel_forge.config import get_config
from reel_forge.core import keys as K


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", required=True, help="Existing job_id (manifest must be on S3)")
    p.add_argument("--source", required=True, type=Path, help="Local 60fps mp4 (will be downsampled to 16fps proxy)")
    args = p.parse_args()

    if not args.source.exists():
        print(f"source not found: {args.source}", file=sys.stderr)
        sys.exit(1)

    cfg = get_config()
    s3 = boto3.client("s3", region_name=cfg.AWS_REGION)

    # Sanity-check the job's manifest exists on S3.
    manifest_key = K.s3_manifest_key(cfg.S3_PREFIX, args.job)
    try:
        s3.head_object(Bucket=cfg.S3_BUCKET, Key=manifest_key)
    except Exception as e:
        print(f"manifest not found on S3 at {manifest_key}: {e}", file=sys.stderr)
        print("(run `forge prepare ...` first, then `forge generate` once to upload the manifest)", file=sys.stderr)
        sys.exit(1)

    # 1. Downsample 60fps -> 16fps proxy.
    work = cfg.OUTPUT_DIR / args.job
    work.mkdir(parents=True, exist_ok=True)
    proxy = work / "animated_16fps_proxy.mp4"
    print(f"[1/4] downsampling {args.source} -> {proxy} (16fps)...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(args.source),
            "-filter:v", "fps=16",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            str(proxy),
        ],
        check=True,
    )
    print(f"      proxy size: {proxy.stat().st_size} bytes")

    # 2. Upload proxy as animated.mp4.
    animated_key = f"{cfg.S3_PREFIX}/{args.job}/{K.ANIMATED}"
    s3.upload_file(str(proxy), cfg.S3_BUCKET, animated_key)
    print(f"[2/4] uploaded -> s3://{cfg.S3_BUCKET}/{animated_key}")

    # 3a. Upload pose_extract + animate phase markers so the orchestrator
    # skips them (the existing per-phase-marker skip logic in process_job
    # short-circuits on storage.exists(marker)).
    for phase in ("pose_extract", "animate"):
        marker_key = K.s3_marker_key(cfg.S3_PREFIX, args.job, phase)
        s3.put_object(Bucket=cfg.S3_BUCKET, Key=marker_key, Body=b"")
        print(f"[3a] marker uploaded -> {marker_key}")

    # 3b. DELETE any leftover downstream markers from a prior completion. If
    # those markers exist, the orchestrator would skip ALL phases (including
    # interp/face_restore/audio/reels) and exit with "no final.mp4 produced".
    for phase in ("interp", "face_restore", "audio_attach", "reels_format"):
        marker_key = K.s3_marker_key(cfg.S3_PREFIX, args.job, phase)
        try:
            s3.delete_object(Bucket=cfg.S3_BUCKET, Key=marker_key)
            print(f"[3b] marker deleted -> {marker_key}")
        except Exception as e:
            print(f"[3b] marker delete (best-effort): {marker_key} -> {e}")

    # 4. Reset status.json to PREPARED so `forge generate` accepts the job.
    status_local = work / K.STATUS
    if status_local.exists():
        d = json.loads(status_local.read_text())
    else:
        d = {
            "schema_version": 1,
            "job_id": args.job,
            "state": "prepared",
            "phase_history": [],
            "attempts": {},
            "failure": None,
            "resource_telemetry": [],
        }
    d["state"] = "prepared"
    d["current_phase"] = None
    d["current_phase_started_at"] = None
    d["phase_history"] = []
    d["failure"] = None
    d["attempts"] = {}
    status_local.write_text(json.dumps(d, indent=2))
    s3.put_object(
        Bucket=cfg.S3_BUCKET,
        Key=K.s3_status_key(cfg.S3_PREFIX, args.job),
        Body=json.dumps(d, indent=2).encode(),
    )
    print("[4/4] state reset to PREPARED (local + S3)")

    print()
    print("Ready. Launch the postprocess-only spot with:")
    print(f"  forge generate --job {args.job}")
    print()
    print("Expected wall time: ~25 min (FSR + cloud-init + interp + face_restore + audio + reels).")
    print("Expected cost: ~$0.40-0.60.")


if __name__ == "__main__":
    main()
