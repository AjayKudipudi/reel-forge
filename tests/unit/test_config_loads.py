     1|"""Config loads from env; missing required raises."""
     2|from __future__ import annotations
     3|
     4|import pytest
     5|from pydantic import ValidationError
     6|
     7|from reel_forge.config import Config, load_config
     8|
     9|
    10|def test_load_with_min_env() -> None:
    11|    cfg = load_config()
    12|    assert cfg.HF_TOKEN
    13|    assert cfg.AWS_REGION
    14|    assert cfg.STORAGE_BACKEND in ("s3", "local")
    15|
    16|
    17|def test_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    18|    monkeypatch.delenv("HF_TOKEN", raising=False)
    19|
    20|    # HF_TOKEN is now optional (empty default). Verify that Config()
    21|    # no longer raises ValidationError when HF_TOKEN is unset.
    22|    class _NoEnvSettings(Config):
    23|        model_config = Config.model_config | {"env_file": None}
    24|
    25|    cfg = _NoEnvSettings(
    26|        AWS_ACCESS_KEY_ID="x",
    27|        AWS_SECRET_ACCESS_KEY="***",
    28|    )
    29|    assert cfg.HF_TOKEN == ""
    30|
    31|
    32|def test_to_subprocess_dict_roundtrip() -> None:
    33|    cfg = load_config()
    34|    d = cfg.to_subprocess_dict()
    35|    assert d["AWS_REGION"] == cfg.AWS_REGION
    36|    assert d["STORAGE_BACKEND"] == cfg.STORAGE_BACKEND
    37|    assert "," in d["SPOT_AZ_ROTATION"]
    38|
    39|
    40|def test_spot_az_rotation_parses_comma_separated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    41|    """Regression: pydantic-settings would otherwise try to JSON-decode this
    42|    field. NoDecode annotation defers parsing to the field_validator, which
    43|    accepts a comma-separated string."""
    44|    monkeypatch.setenv("SPOT_AZ_ROTATION", "us-east-1a,us-east-1b,us-east-1c")
    45|    cfg = load_config()
    46|    assert cfg.SPOT_AZ_ROTATION == ("us-east-1a", "us-east-1b", "us-east-1c")
    47|
    48|
    49|def test_spot_az_rotation_single_value(monkeypatch: pytest.MonkeyPatch) -> None:
    50|    monkeypatch.setenv("SPOT_AZ_ROTATION", "us-east-1a")
    51|    cfg = load_config()
    52|    assert cfg.SPOT_AZ_ROTATION == ("us-east-1a",)
    53|