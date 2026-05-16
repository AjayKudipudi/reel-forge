"""FFmpeg ToolSpec."""
from __future__ import annotations

from ..errors import ErrorClass
from ..external_tool import ToolSpec


def _classify(_rc: int, _out: str, err: str) -> ErrorClass:
    e = err.lower()
    if "no such file" in e or "no such file or directory" in e:
        return ErrorClass.LOCAL_VIDEO_INVALID
    if "no space left on device" in e:
        return ErrorClass.DISK_FULL
    if "permission denied" in e:
        return ErrorClass.DISK_FULL
    return ErrorClass.FFMPEG_FAILED


FFMPEG = ToolSpec(name="ffmpeg", binary="ffmpeg", timeout_s=600, classifier=_classify)
