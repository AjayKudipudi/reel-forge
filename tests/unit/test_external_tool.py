"""run_tool happy + classified failure."""
from __future__ import annotations

import pytest

from insta_influencer.core.errors import ErrorClass, ToolFailed
from insta_influencer.core.external_tool import ToolSpec, run_tool


def test_run_tool_happy_path() -> None:
    spec = ToolSpec(name="echo", binary="/bin/echo", timeout_s=5)
    res = run_tool(spec, ["hello"])
    assert res.returncode == 0
    assert "hello" in res.stdout


def test_run_tool_classifies_failure() -> None:
    spec = ToolSpec(
        name="false",
        binary="/usr/bin/false",
        timeout_s=5,
        classifier=lambda rc, out, err: ErrorClass.FFMPEG_FAILED,
    )
    with pytest.raises(ToolFailed) as exc_info:
        run_tool(spec, [])
    assert exc_info.value.error_class == ErrorClass.FFMPEG_FAILED
