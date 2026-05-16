"""yt-dlp ToolSpec — Reel downloader."""
from __future__ import annotations

from ..errors import ErrorClass
from ..external_tool import ToolSpec


def _classify(_rc: int, _out: str, err: str) -> ErrorClass:
    e = err.lower()
    if any(s in e for s in ("unavailable", "private", "404", "removed", "not found")):
        return ErrorClass.REEL_DOWNLOAD_UNAVAILABLE
    return ErrorClass.REEL_DOWNLOAD_FAILED


YTDLP = ToolSpec(name="yt-dlp", binary="yt-dlp", timeout_s=600, classifier=_classify)
