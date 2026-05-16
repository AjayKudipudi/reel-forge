"""structlog configuration. One entry point per process; safe to call again
in a subprocess (rebinds contextvars cleanly)."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import structlog


def configure_logging(
    *,
    job_id: str | None,
    log_dir: Path,
    level: str = "INFO",
    fmt: Literal["text", "json"] = "text",
) -> structlog.stdlib.BoundLogger:
    """Configure structlog + stdlib logging. Idempotent."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id or 'no-job'}.log"

    # Reset stdlib root handlers in case of re-config (subprocess re-entry).
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(getattr(logging, level))
    root.addHandler(logging.StreamHandler(sys.stdout))
    root.addHandler(logging.FileHandler(log_path, encoding="utf-8"))

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        timestamper,
    ]
    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.reset_defaults()
    structlog.configure(
        processors=[*shared_processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    log: structlog.stdlib.BoundLogger = structlog.get_logger("insta")
    structlog.contextvars.clear_contextvars()
    if job_id:
        structlog.contextvars.bind_contextvars(job_id=job_id)
    return log
