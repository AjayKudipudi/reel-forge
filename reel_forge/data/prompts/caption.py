"""Caption template (Phase F). Rendered with manifest fields + branding."""
from __future__ import annotations

CAPTION_TEMPLATE: str = (
    "{prompt} 🎬\n"
    "\n"
    "{hashtags}"
)


def render_caption(*, prompt: str, hashtags: str) -> str:
    return CAPTION_TEMPLATE.format(prompt=prompt, hashtags=hashtags)
