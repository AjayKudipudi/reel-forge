"""Shared CLI helpers."""
from __future__ import annotations

from pathlib import Path

from ..config import Config, get_config
from ..core import keys as K
from ..core.log_setup import configure_logging
from ..core.manifest import Manifest
from ..core.status_models import Status


def cfg_and_log(job_id: str | None = None) -> Config:
    cfg = get_config()
    configure_logging(job_id=job_id, log_dir=cfg.LOG_DIR, level=cfg.LOG_LEVEL, fmt=cfg.LOG_FORMAT)
    return cfg


def job_dir(cfg: Config, job_id: str) -> Path:
    return cfg.OUTPUT_DIR / job_id


def read_local_manifest(cfg: Config, job_id: str) -> Manifest | None:
    p = job_dir(cfg, job_id) / K.MANIFEST
    if not p.exists():
        return None
    return Manifest.model_validate_json(p.read_text())


def read_local_status(cfg: Config, job_id: str) -> Status | None:
    p = job_dir(cfg, job_id) / K.STATUS
    if not p.exists():
        return None
    return Status.model_validate_json(p.read_text())


def list_jobs(cfg: Config) -> list[str]:
    if not cfg.OUTPUT_DIR.exists():
        return []
    return sorted(p.name for p in cfg.OUTPUT_DIR.iterdir() if p.is_dir() and len(p.name) == 12)
