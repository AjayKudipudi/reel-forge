"""Run a Phase in a fresh subprocess.

IPC contract — parent → child:
  - argv: [python, -u, -m, insta_influencer.ec2._subprocess_runner, <phase_qualname>]
  - env:  parent's env + Config.to_subprocess_dict() + INSTA_JOB_ID + PYTHONUNBUFFERED=1
  - stdin: a small JSON payload — { job_id, work_dir, s3_prefix } only.

IPC contract — child → parent:
  - stdout: PhaseResult JSON on the LAST line.
  - stderr: streamed line-by-line to <work_dir>/logs/<phase>.stderr (file open
    in unbuffered mode); parent tails that file every N seconds and re-emits
    each line as a structlog event (so cloud-init's /var/log/insta-influencer.log
    captures it live, even if subprocess is later SIGKILLed by timeout).
  - exit code: 0 on PhaseResult.ok or PhaseResult.fail; non-zero ONLY for
               unrecoverable child crashes (segfault, OOM-kill, subprocess timeout).

This module is BOTH the parent-side helper (`run_phase_in_subprocess`)
and the child-side entrypoint (`__main__` → `child_main`).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import threading
import time
from importlib import import_module
from pathlib import Path
from typing import cast

import structlog

from ..config import get_config
from ..core.errors import ErrorClass, classify
from ..core.log_setup import configure_logging
from ..core.manifest import Manifest
from ..core.phase import Phase
from ..core.result import PhaseContext, PhaseResult
from ..core.status import StatusManager
from ..core.storage import get_object_store

_log = structlog.get_logger(__name__)


def _phase_qualname(phase: Phase) -> str:
    return f"{phase.__class__.__module__}:{phase.__class__.__name__}"


def _import_phase(qualname: str) -> Phase:
    mod_name, cls_name = qualname.split(":")
    module = import_module(mod_name)
    cls = getattr(module, cls_name)
    return cast(Phase, cls())


def _tail_into_log(stop: threading.Event, path: Path, phase_name: str) -> None:
    """Tail a file growing in real time and emit each new line as a structlog
    event. Lets cloud-init's /var/log/insta-influencer.log show live phase
    output instead of nothing-then-empty-stderr on timeout."""
    while not path.exists() and not stop.is_set():
        time.sleep(0.2)
    if stop.is_set():
        return
    with open(path, errors="replace") as f:
        f.seek(0, io.SEEK_END)
        while not stop.is_set():
            line = f.readline()
            if line:
                _log.info("phase.stderr", phase=phase_name, line=line.rstrip())
            else:
                time.sleep(0.5)


def run_phase_in_subprocess(
    phase: Phase,
    ctx: PhaseContext,
    *,
    stderr_tail_bytes: int,
) -> PhaseResult:
    payload = json.dumps(
        {
            "job_id": ctx.job_id,
            "work_dir": str(ctx.work_dir),
            "s3_prefix": ctx.s3_prefix,
        }
    )
    cfg = get_config()
    env = {
        **os.environ,
        **cfg.to_subprocess_dict(),
        "INSTA_JOB_ID": ctx.job_id,
        "PYTHONUNBUFFERED": "1",
    }
    logs_dir = ctx.work_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = logs_dir / f"{phase.name}.stderr"
    stdout_path = logs_dir / f"{phase.name}.stdout"
    # Start the tail thread before the subprocess so we don't miss lines.
    stop = threading.Event()
    tail = threading.Thread(
        target=_tail_into_log,
        args=(stop, stderr_path, phase.name),
        daemon=True,
    )
    tail.start()
    timed_out = False
    with open(stderr_path, "wb") as ferr, open(stdout_path, "wb") as fout:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "insta_influencer.ec2._subprocess_runner",
                _phase_qualname(phase),
            ],
            stdin=subprocess.PIPE,
            stdout=fout,
            stderr=ferr,
            env=env,
        )
        try:
            proc.communicate(input=payload.encode(), timeout=phase.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.communicate()
    stop.set()
    tail.join(timeout=2)

    stderr_bytes = stderr_path.read_bytes() if stderr_path.exists() else b""
    stdout_bytes = stdout_path.read_bytes() if stdout_path.exists() else b""
    tail_str = stderr_bytes[-stderr_tail_bytes:].decode("utf-8", errors="replace")
    if timed_out:
        # Upload logs to S3 so we can post-mortem after the spot terminates.
        try:
            ctx.storage.upload(
                stderr_path,
                f"{ctx.s3_prefix}/_runtime-logs/{phase.name}.stderr",
            )
            ctx.storage.upload(
                stdout_path,
                f"{ctx.s3_prefix}/_runtime-logs/{phase.name}.stdout",
            )
        except Exception as up:
            _log.warning("phase.log_upload_failed", phase=phase.name, err=str(up))
        return PhaseResult.fail(
            error_class=ErrorClass.UNKNOWN,
            message=f"timeout: {phase.name} exceeded {phase.timeout_s}s; "
            f"stdout {len(stdout_bytes)} bytes, stderr {len(stderr_bytes)} bytes",
            retryable=False,
            stats={"timeout_s": phase.timeout_s},
            stderr_tail=tail_str,
        )
    if proc.returncode != 0:
        ec = (
            ErrorClass.MODEL_OOM
            if "OutOfMemory" in tail_str or "CUDA out of memory" in tail_str
            else ErrorClass.UNKNOWN
        )
        with contextlib.suppress(Exception):
            ctx.storage.upload(
                stderr_path,
                f"{ctx.s3_prefix}/_runtime-logs/{phase.name}.stderr",
            )
        return PhaseResult.fail(
            error_class=ec,
            message=f"child exited {proc.returncode}: {tail_str[-400:]}",
            retryable=ec == ErrorClass.MODEL_OOM,
            stats={"child_exit": proc.returncode},
            stderr_tail=tail_str,
        )
    lines = [ln for ln in stdout_bytes.decode("utf-8", "replace").splitlines() if ln.strip()]
    if not lines:
        return PhaseResult.fail(
            error_class=ErrorClass.UNKNOWN,
            message="child produced no PhaseResult on stdout",
            retryable=False,
            stderr_tail=tail_str,
        )
    try:
        return PhaseResult.from_json(lines[-1])
    except Exception as exc:
        return PhaseResult.fail(
            error_class=ErrorClass.UNKNOWN,
            message=f"PhaseResult.from_json failed: {exc!r}; last line={lines[-1]!r}",
            retryable=False,
            stderr_tail=tail_str,
        )


def child_main() -> int:
    """Entry: `python -m insta_influencer.ec2._subprocess_runner <phase_qualname>`."""
    if len(sys.argv) < 2:
        sys.stderr.write("usage: _subprocess_runner <phase_qualname>\n")
        return 2
    phase_qualname = sys.argv[1]
    args = json.loads(sys.stdin.read())
    work_dir = Path(args["work_dir"])
    job_id = args["job_id"]
    s3_prefix = args["s3_prefix"]

    cfg = get_config()
    log = configure_logging(
        job_id=job_id,
        log_dir=cfg.LOG_DIR,
        level=cfg.LOG_LEVEL,
        fmt=cfg.LOG_FORMAT,
    )

    storage = get_object_store(cfg)
    manifest = Manifest.model_validate_json((work_dir / "manifest.json").read_text())
    status = StatusManager(
        job_id=job_id,
        local_path=work_dir / "status.json",
        storage=storage,
        s3_key=f"{s3_prefix}/status.json",
    )
    phase = _import_phase(phase_qualname)
    ctx = PhaseContext(
        job_id=job_id,
        work_dir=work_dir,
        storage=storage,
        s3_prefix=s3_prefix,
        manifest=manifest,
        seed=manifest.model.seed,
        logger=log.bind(phase=phase.name),
        on_progress=lambda p: status.heartbeat(current_phase_progress=p),
    )
    try:
        result = phase.run(ctx)
    except BaseException as exc:
        info = classify(exc)
        result = PhaseResult.fail(
            error_class=info.error_class,
            message=info.message,
            retryable=info.retryable,
        )
    sys.stdout.write("\n" + result.to_json() + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(child_main())
