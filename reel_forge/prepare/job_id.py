"""Content-addressed job IDs. Same inputs → same ID → idempotent prepare."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

JOB_ID_SCHEMA_VERSION: int = 1


def derive_job_id(
    *,
    photo_sha256: str,
    reference_sha256: str,
    model_quant: str,
    seed: int,
    output_w: int,
    output_h: int,
    prompt: str,
    background_mode: str,
) -> str:
    payload = json.dumps(
        {
            "v": JOB_ID_SCHEMA_VERSION,
            "photo": photo_sha256,
            "ref": reference_sha256,
            "quant": model_quant,
            "seed": seed,
            "out": [output_w, output_h],
            "prompt": prompt,
            "bg": background_mode,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
