"""Unified subprocess runner with classified failures.

One runner, not five copies of the same try/except. Per-tool ToolSpecs in
`core/tools/*.py` plug in classifiers that turn returncode + stderr into
typed ErrorClass values.
"""
from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ErrorClass, ToolFailed


@dataclass(frozen=True)
class ToolSpec:
    name: str
    binary: str
    timeout_s: int = 600
    classifier: Callable[[int, str, str], ErrorClass] = field(
        default=lambda rc, out, err: ErrorClass.UNKNOWN
    )


@dataclass
class ToolResult:
    returncode: int
    stdout: str
    stderr: str
    wall_s: float


def run_tool(
    spec: ToolSpec,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_data: bytes | None = None,
    expected_artifacts: tuple[Path, ...] = (),
    stream_to_log: Path | None = None,
) -> ToolResult:
    """Run an external tool. Raises ToolFailed on non-zero exit OR a
    missing expected artifact, with the spec's classifier mapping
    stderr→ErrorClass.

    If `stream_to_log` is set, the tool's stdout AND stderr are merged and
    written to that file in real time (unbuffered). This survives
    `TimeoutExpired` — the buffer-in-memory of `capture_output=True` is lost
    on timeout, so for long-running tools (generate_dancer.py) the file is
    the only post-mortem. Streams are merged because upstream Python tools
    typically log via `logging.StreamHandler(stream=sys.stdout)` while
    progress bars (tqdm) and torch warnings go to stderr — capturing only
    stderr loses all the "Generating video..." style status messages we need
    to diagnose where it stalled.
    """
    started = time.time()
    if stream_to_log is not None:
        stream_to_log.parent.mkdir(parents=True, exist_ok=True)
        with open(stream_to_log, "wb", buffering=0) as log_f:
            proc_p = subprocess.Popen(
                [spec.binary, *args],
                cwd=str(cwd) if cwd else None,
                env=env,
                stdin=subprocess.PIPE if input_data else None,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            try:
                proc_p.communicate(input=input_data, timeout=spec.timeout_s)
                rc = proc_p.returncode
            except subprocess.TimeoutExpired:
                proc_p.kill()
                proc_p.communicate()
                log_f.flush()
                merged_partial = stream_to_log.read_bytes() if stream_to_log.exists() else b""
                result = ToolResult(
                    returncode=-9,
                    stdout="",
                    stderr=merged_partial.decode("utf-8", errors="replace"),
                    wall_s=round(time.time() - started, 2),
                )
                raise ToolFailed(
                    spec=spec, result=result, error_class=ErrorClass.UNKNOWN,
                ) from None
        merged_b = stream_to_log.read_bytes() if stream_to_log.exists() else b""
        result = ToolResult(
            returncode=rc,
            stdout="",
            stderr=merged_b.decode("utf-8", errors="replace"),
            wall_s=round(time.time() - started, 2),
        )
    else:
        proc = subprocess.run(
            [spec.binary, *args],
            cwd=str(cwd) if cwd else None,
            env=env,
            input=input_data,
            capture_output=True,
            timeout=spec.timeout_s,
            check=False,
        )
        result = ToolResult(
            returncode=proc.returncode,
            stdout=proc.stdout.decode("utf-8", errors="replace"),
            stderr=proc.stderr.decode("utf-8", errors="replace"),
            wall_s=round(time.time() - started, 2),
        )
    if result.returncode != 0:
        ec = spec.classifier(result.returncode, result.stdout, result.stderr)
        raise ToolFailed(spec=spec, result=result, error_class=ec)
    for artifact in expected_artifacts:
        if not artifact.exists():
            raise ToolFailed(
                spec=spec,
                result=result,
                error_class=ErrorClass.UNKNOWN,
                missing_artifact=artifact,
            )
    return result
