"""Optional content-moderation pre-flight. Calls an operator-supplied
binary; exit 0 = pass, anything else = ContentModerationRejected."""
from __future__ import annotations

from pathlib import Path

import structlog

from ..core.errors import ContentModerationRejected, ToolFailed
from ..core.external_tool import run_tool
from ..core.tools.moderation import make_spec

log = structlog.get_logger(__name__)


def moderate(
    *,
    photo: Path,
    reference: Path,
    binary: str,
    enabled: bool,
) -> None:
    if not enabled:
        log.debug("moderation.disabled")
        return
    if not binary:
        log.warning("moderation.enabled_but_no_binary")
        return
    spec = make_spec(binary)
    try:
        run_tool(spec, [str(photo), str(reference)])
    except ToolFailed as exc:
        raise ContentModerationRejected(
            f"moderator rejected: rc={exc.result.returncode}, stderr={exc.result.stderr[:400]}"
        ) from exc
    log.info("moderation.passed", binary=binary)
