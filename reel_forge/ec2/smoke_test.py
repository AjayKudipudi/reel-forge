"""Pre-bake AMI smoke: minimal validation that imports + binaries respond.

Strategy: don't run the full pipeline on synthetic input — DWPose would
fail to find a person in a flat-gray image. Instead, verify the install
ladder imports cleanly and that upstream CLIs respond to `--help`. If
all checks pass, the AMI is valid; the operator's first real
prepare+generate is the end-to-end validation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from ..config import get_config
from ..core.log_setup import configure_logging

UPSTREAM_REPO_DIR = Path("/opt/insta-influencer/third_party/SteadyDancer")


@click.command("smoke")
@click.option("--num-frames", default=33, help="kept for CLI compatibility; ignored")
def smoke(num_frames: int) -> None:
    cfg = get_config()
    log = configure_logging(
        job_id="smoke",
        log_dir=cfg.LOG_DIR,
        level=cfg.LOG_LEVEL,
        fmt=cfg.LOG_FORMAT,
    )
    log.info("smoke.start", strategy="import-check")

    failures: list[str] = []

    # Core ML stack
    for mod in ("torch", "diffusers", "transformers", "accelerate", "huggingface_hub"):
        try:
            __import__(mod)
            log.info("smoke.import.ok", module=mod)
        except Exception as exc:
            log.error("smoke.import.fail", module=mod, err=str(exc))
            failures.append(f"import {mod}: {exc}")

    # Pose stack
    for mod in ("cv2", "mmcv", "mmpose", "mmdet", "decord", "moviepy"):
        try:
            __import__(mod)
            log.info("smoke.import.ok", module=mod)
        except Exception as exc:
            log.error("smoke.import.fail", module=mod, err=str(exc))
            failures.append(f"import {mod}: {exc}")

    # Attention libs (optional but expected on this AMI)
    for mod in ("flash_attn", "xformers"):
        try:
            __import__(mod)
            log.info("smoke.import.ok", module=mod)
        except Exception as exc:
            log.warning("smoke.import.optional_missing", module=mod, err=str(exc))

    # Upstream CLIs respond to --help
    for script in ("preprocess/pose_align.py", "generate_dancer.py"):
        path = UPSTREAM_REPO_DIR / script
        if not path.exists():
            failures.append(f"missing upstream: {path}")
            log.error("smoke.upstream.missing", path=str(path))
            continue
        try:
            proc = subprocess.run(
                [sys.executable, str(path), "--help"],
                cwd=str(UPSTREAM_REPO_DIR),
                capture_output=True,
                timeout=120,
                check=False,
            )
            out = proc.stdout.decode("utf-8", errors="replace")
            err = proc.stderr.decode("utf-8", errors="replace")
            if proc.returncode == 0 or "usage:" in out.lower() or "usage:" in err.lower():
                log.info("smoke.upstream.help_ok", script=script, exitcode=proc.returncode)
            else:
                log.error(
                    "smoke.upstream.help_fail",
                    script=script,
                    exitcode=proc.returncode,
                    stderr_tail=err[-400:],
                )
                failures.append(f"{script} --help failed: {err[-200:]}")
        except Exception as exc:
            log.error("smoke.upstream.help_exception", script=script, err=str(exc))
            failures.append(f"{script} --help raised: {exc}")

    if failures:
        log.error("smoke.failed", failures=failures)
        sys.exit(1)
    log.info("smoke.passed")


if __name__ == "__main__":
    smoke()
