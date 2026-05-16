"""Dual-mode reference acquisition.

acquire_reference(url=...) → downloads via yt-dlp.
acquire_reference(local_path=...) → stages an existing mp4.

Both paths produce a sha256-keyed file in the cache_dir + ffprobe-validated
metadata.
"""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

from ..core.errors import LocalVideoInvalid, ReelDownloadFailed
from ..core.external_tool import run_tool
from ..core.tools.ffprobe import FFPROBE
from ..core.tools.ytdlp import YTDLP
from .job_id import file_sha256

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ReferenceVideo:
    staged_path: Path
    sha256: str
    duration_s: float
    width: int
    height: int
    source_kind: str  # "url" | "local"
    original_url: str | None = None
    original_local_path: Path | None = None


def acquire_reference(
    *,
    url: str | None = None,
    local_path: Path | None = None,
    cache_dir: Path,
) -> ReferenceVideo:
    """EXACTLY ONE of url / local_path required."""
    if (url is None) == (local_path is None):
        raise ValueError("provide exactly one of url= or local_path=")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if url is not None:
        staged = _download_url(url, cache_dir)
        ref = _validate_and_describe(staged, "url", original_url=url)
    else:
        assert local_path is not None
        staged = _stage_local(local_path, cache_dir)
        ref = _validate_and_describe(staged, "local", original_local_path=local_path)
    log.info(
        "reference.acquired",
        staged=str(ref.staged_path),
        duration_s=ref.duration_s,
        sha256=ref.sha256,
        kind=ref.source_kind,
    )
    return ref


def _download_url(url: str, cache_dir: Path) -> Path:
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
    # Cache hit: a previous successful download is on disk. yt-dlp always hits
    # Instagram's metadata endpoint before deciding to skip, so under rate
    # limits even cached URLs would fail. Short-circuit before that round trip.
    cached = list(cache_dir.glob(f"{url_hash}.*"))
    if cached:
        log.info("reel.download.cache_hit", url=url, staged=str(cached[0]))
        return cached[0]
    out_template = cache_dir / f"{url_hash}.%(ext)s"
    log.info("reel.download.start", url=url, target=str(out_template))
    run_tool(
        YTDLP,
        [
            "-f",
            "mp4/best[ext=mp4]/best",
            "--no-playlist",
            "--no-warnings",
            "-o",
            str(out_template),
            url,
        ],
    )
    matches = list(cache_dir.glob(f"{url_hash}.*"))
    if not matches:
        raise ReelDownloadFailed(f"yt-dlp produced no file for {url}")
    return matches[0]


def _stage_local(src: Path, cache_dir: Path) -> Path:
    if not src.exists():
        raise LocalVideoInvalid(f"file not found: {src}")
    if not src.is_file():
        raise LocalVideoInvalid(f"not a regular file: {src}")
    sha = file_sha256(src)
    target = cache_dir / f"{sha[:12]}.mp4"
    if not target.exists():
        shutil.copy2(src, target)
    return target


def _validate_and_describe(
    path: Path,
    kind: str,
    *,
    original_url: str | None = None,
    original_local_path: Path | None = None,
) -> ReferenceVideo:
    res = run_tool(
        FFPROBE,
        [
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
    )
    parts = [p for p in res.stdout.strip().splitlines() if p.strip()]
    if len(parts) < 3:
        raise LocalVideoInvalid(f"ffprobe could not describe {path}: {res.stdout!r}")
    try:
        w, h = int(parts[0]), int(parts[1])
        dur = float(parts[2])
    except ValueError as exc:
        raise LocalVideoInvalid(f"ffprobe gave non-numeric output: {res.stdout!r}") from exc
    return ReferenceVideo(
        staged_path=path,
        sha256=file_sha256(path),
        duration_s=dur,
        width=w,
        height=h,
        source_kind=kind,
        original_url=original_url,
        original_local_path=original_local_path,
    )
