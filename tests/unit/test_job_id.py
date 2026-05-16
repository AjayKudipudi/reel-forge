"""Content-addressed job id contract."""
from __future__ import annotations

from reel_forge.prepare.job_id import JOB_ID_SCHEMA_VERSION, derive_job_id


def _kw(**overrides: object) -> dict[str, object]:
    base = dict(
        photo_sha256="a" * 64,
        reference_sha256="b" * 64,
        model_quant="gguf-q5-m",
        seed=42,
        output_w=1080,
        output_h=1920,
        prompt="a person dancing",
        background_mode="from_photo",
    )
    base.update(overrides)
    return base


def test_same_inputs_produce_same_id() -> None:
    a = derive_job_id(**_kw())  # type: ignore[arg-type]
    b = derive_job_id(**_kw())  # type: ignore[arg-type]
    assert a == b
    assert len(a) == 12


def test_changing_input_changes_id() -> None:
    base = derive_job_id(**_kw())  # type: ignore[arg-type]
    diffs = [
        derive_job_id(**_kw(seed=43)),  # type: ignore[arg-type]
        derive_job_id(**_kw(prompt="something else")),  # type: ignore[arg-type]
        derive_job_id(**_kw(model_quant="fp16")),  # type: ignore[arg-type]
        derive_job_id(**_kw(background_mode="replace")),  # type: ignore[arg-type]
        derive_job_id(**_kw(photo_sha256="c" * 64)),  # type: ignore[arg-type]
    ]
    for d in diffs:
        assert d != base


def test_schema_version_invariant() -> None:
    assert JOB_ID_SCHEMA_VERSION >= 1
