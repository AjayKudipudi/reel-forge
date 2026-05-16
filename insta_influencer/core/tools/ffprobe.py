"""ffprobe ToolSpec — used to validate downloaded/local mp4s."""
from __future__ import annotations

from ..errors import ErrorClass
from ..external_tool import ToolSpec


def _classify(_rc: int, _out: str, err: str) -> ErrorClass:
    e = err.lower()
    if "no such file" in e or "invalid data" in e or "moov atom not found" in e:
        return ErrorClass.LOCAL_VIDEO_INVALID
    return ErrorClass.LOCAL_VIDEO_INVALID


FFPROBE = ToolSpec(name="ffprobe", binary="ffprobe", timeout_s=30, classifier=_classify)
