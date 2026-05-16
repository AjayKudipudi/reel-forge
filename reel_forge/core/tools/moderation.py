"""Content-moderation ToolSpec. Binary path comes from CONFIG.CONTENT_MODERATION_BINARY."""
from __future__ import annotations

from ..errors import ErrorClass
from ..external_tool import ToolSpec


def _classify(_rc: int, _out: str, _err: str) -> ErrorClass:
    return ErrorClass.CONTENT_MODERATION_REJECTED


def make_spec(binary: str) -> ToolSpec:
    return ToolSpec(name="content-moderation", binary=binary, timeout_s=120, classifier=_classify)
