"""Real-GPU smoke. Requires ANIMATE_FAKE unset + CUDA + AWS creds."""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.smoke
def test_real_steadydancer_one_clip() -> None:
    if os.getenv("ANIMATE_FAKE") == "1":
        pytest.skip("ANIMATE_FAKE=1 — fake mode")
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("torch not installed")
    if not torch.cuda.is_available():
        pytest.skip("no GPU")

    # Click invocation: simulate `insta-smoke --num-frames 33`
    from click.testing import CliRunner

    from insta_influencer.ec2.smoke_test import smoke

    runner = CliRunner()
    result = runner.invoke(smoke, ["--num-frames", "33"])
    assert result.exit_code == 0, result.output
