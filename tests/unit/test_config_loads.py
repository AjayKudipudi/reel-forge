"""Config loads from env; missing required raises."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from reel_forge.config import Config, load_config


def test_load_with_min_env() -> None:
    cfg = load_config()
    assert cfg.HF_TOKEN
    assert cfg.AWS_REGION
    assert cfg.STORAGE_BACKEND in ("s3", "local")


def test_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)

    # We have to bypass the autouse fixture's env to test this, so build
    # Config directly with no init args.
    class _NoEnvSettings(Config):
        model_config = Config.model_config | {"env_file": None}

    with pytest.raises(ValidationError):
        _NoEnvSettings(  # type: ignore[call-arg]
            AWS_ACCESS_KEY_ID="x",
            AWS_SECRET_ACCESS_KEY="y",
        )


def test_to_subprocess_dict_roundtrip() -> None:
    cfg = load_config()
    d = cfg.to_subprocess_dict()
    assert d["AWS_REGION"] == cfg.AWS_REGION
    assert d["STORAGE_BACKEND"] == cfg.STORAGE_BACKEND
    assert "," in d["SPOT_AZ_ROTATION"]


def test_spot_az_rotation_parses_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: pydantic-settings would otherwise try to JSON-decode this
    field. NoDecode annotation defers parsing to the field_validator, which
    accepts a comma-separated string."""
    monkeypatch.setenv("SPOT_AZ_ROTATION", "us-east-1a,us-east-1b,us-east-1c")
    cfg = load_config()
    assert cfg.SPOT_AZ_ROTATION == ("us-east-1a", "us-east-1b", "us-east-1c")


def test_spot_az_rotation_single_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPOT_AZ_ROTATION", "us-east-1a")
    cfg = load_config()
    assert cfg.SPOT_AZ_ROTATION == ("us-east-1a",)
