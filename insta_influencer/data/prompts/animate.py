"""Animate-phase prompt defaults. Edit here, not in .env.

Per-job overrides arrive via Manifest.prompt and Manifest.negative_prompt.

Minimal-prompt philosophy (2026-05-16): let the model generate. The
upstream Wan I2V-14B was trained against a rich Chinese sample_neg_prompt
(`色调艳丽, 过曝, 多余的手指, 手指融合, 形态畸形的肢体, ...`) that
already targets the exact artifacts we care about. UMT5-XXL is
multilingual; the official English translation (Wan README) is
equivalent. We keep that as the BASE and add ONLY the few defects the
model improvises that aren't in the upstream — minimal additions, no
anatomical enumeration.

POSITIVE is one natural-language sentence — Wan2.1#496 documents that
long attribute lists hurt quality.
"""
from __future__ import annotations

POSITIVE: str = (
    "A young woman dancing gracefully on a rooftop, "
    "full body in frame, photorealistic."
)

# Upstream Wan I2V-14B sample_neg_prompt (official English from Wan README;
# UMT5 is multilingual so this is equivalent to the trained Chinese form).
_UPSTREAM_NEGATIVE_EN: str = (
    "Camera shake, bright tones, overexposed, static, blurred details, "
    "subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, "
    "walking backwards"
)

# Only what the upstream doesn't cover, targeting specific defects we
# observed across runs #1-#5: identity drift, closed eyes mid-dance,
# hallucinated hair accessory, extra hands.
_OUR_ADDITIONS_NEGATIVE: str = (
    "face change, identity change, closed eyes, "
    "flower in hair, hair clip, hair accessory, "
    "extra hands, extra arms, third arm"
)

NEGATIVE: str = _UPSTREAM_NEGATIVE_EN + ", " + _OUR_ADDITIONS_NEGATIVE
