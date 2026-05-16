"""Pydantic round-trip + discriminator behavior."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from insta_influencer.core.manifest import (
    BackgroundFromPhoto,
    BackgroundReplace,
    Manifest,
    ModelConfig,
    OutputSpec,
    ReferenceLocal,
    ReferenceURL,
)


def _mk(**overrides: object) -> Manifest:
    base = Manifest(
        schema_version=1,
        job_id="0" * 12,
        created_at=datetime.now(UTC),
        reference_source=ReferenceLocal(
            type="local",
            original_path=Path("/x/y.mp4"),
            staged_path=Path("/cache/y.mp4"),
            sha256="a" * 64,
        ),
        photo_path=Path("/cache/photo.png"),
        photo_sha256="b" * 64,
        background=BackgroundFromPhoto(),
        prompt="a person dancing",
        model=ModelConfig(),
        output=OutputSpec(),
    )
    return base.model_copy(update=overrides)


def test_round_trip_local() -> None:
    m = _mk()
    s = m.model_dump_json()
    m2 = Manifest.model_validate_json(s)
    assert m2 == m


def test_round_trip_url() -> None:
    m = _mk(
        reference_source=ReferenceURL(
            type="url",
            url="https://www.instagram.com/reel/x/",  # type: ignore[arg-type]
            staged_path=Path("/cache/y.mp4"),
            sha256="a" * 64,
        )
    )
    m2 = Manifest.model_validate_json(m.model_dump_json())
    assert m2.reference_source.type == "url"


def test_background_replace_round_trip() -> None:
    m = _mk(background=BackgroundReplace(replacement_path=Path("/bg/cafe.jpg")))
    m2 = Manifest.model_validate_json(m.model_dump_json())
    assert m2.background.mode == "replace"


def test_invalid_job_id_length() -> None:
    # model_copy doesn't validate; round-trip through model_validate to enforce constraints.
    bad = _mk().model_copy(update={"job_id": "too-short"})
    with pytest.raises(ValidationError):
        Manifest.model_validate(bad.model_dump())
