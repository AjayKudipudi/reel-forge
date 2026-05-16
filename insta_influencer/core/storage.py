"""ObjectStore Protocol and three implementations.

S3ObjectStore — production (boto3, retries via botocore adaptive mode).
LocalObjectStore — filesystem-backed for `STORAGE_BACKEND=local` dev.
InMemoryObjectStore — dict-backed for unit tests.
"""
from __future__ import annotations

import io
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


@runtime_checkable
class ObjectStore(Protocol):
    def upload(self, local: Path, key: str) -> None: ...
    def download(self, key: str, local: Path) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> Iterable[str]: ...
    def upload_atomic(self, local: Path, key: str) -> None: ...
    def url(self, key: str) -> str: ...
    def delete(self, key: str) -> None: ...


# ── S3 ───────────────────────────────────────────────────────────────────


class S3ObjectStore:
    """boto3-backed implementation. Retries are configured via botocore."""

    def __init__(self, bucket: str, region: str) -> None:
        import boto3
        from botocore.config import Config as BotoConfig

        self.bucket = bucket
        self.region = region
        self._s3 = boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(retries={"max_attempts": 5, "mode": "adaptive"}),
        )

    def upload(self, local: Path, key: str) -> None:
        self._s3.upload_file(str(local), self.bucket, key)

    def upload_atomic(self, local: Path, key: str) -> None:
        tmp_key = f"{key}.tmp"
        self._s3.upload_file(str(local), self.bucket, tmp_key)
        self._s3.copy_object(
            Bucket=self.bucket,
            CopySource={"Bucket": self.bucket, "Key": tmp_key},
            Key=key,
        )
        self._s3.delete_object(Bucket=self.bucket, Key=tmp_key)

    def download(self, key: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        self._s3.download_file(self.bucket, key, str(local))

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:  # botocore.exceptions.ClientError 404
            return False

    def list(self, prefix: str) -> Iterable[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=key)


# ── Local FS ─────────────────────────────────────────────────────────────


class LocalObjectStore:
    """Filesystem-backed. Used for `STORAGE_BACKEND=local` dev runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Disallow absolute paths or `..` segments — keys must stay under root.
        p = (self.root / key).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"key {key!r} escapes store root")
        return p

    def upload(self, local: Path, key: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)

    def upload_atomic(self, local: Path, key: str) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(local, tmp)
        tmp.replace(target)

    def download(self, key: str, local: Path) -> None:
        src = self._path(key)
        if not src.exists():
            raise FileNotFoundError(f"local store has no key: {key}")
        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str) -> Iterable[str]:
        base = self._path(prefix) if prefix else self.root
        if not base.exists():
            return
        for p in base.rglob("*"):
            if p.is_file():
                yield str(p.relative_to(self.root))

    def url(self, key: str) -> str:
        return f"file://{self._path(key)}"

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


# ── In-memory ────────────────────────────────────────────────────────────


class InMemoryObjectStore:
    """dict[str, bytes]. Used in unit tests."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}

    def upload(self, local: Path, key: str) -> None:
        self._d[key] = local.read_bytes()

    def upload_atomic(self, local: Path, key: str) -> None:
        # Atomicity is trivial in-memory.
        self.upload(local, key)

    def download(self, key: str, local: Path) -> None:
        if key not in self._d:
            raise FileNotFoundError(f"in-memory store has no key: {key}")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(self._d[key])

    def exists(self, key: str) -> bool:
        return key in self._d

    def list(self, prefix: str) -> Iterable[str]:
        return [k for k in self._d if k.startswith(prefix)]

    def url(self, key: str) -> str:
        return f"mem://{key}"

    def delete(self, key: str) -> None:
        self._d.pop(key, None)

    # Test helpers
    def write_bytes(self, key: str, data: bytes) -> None:
        self._d[key] = data

    def read_bytes(self, key: str) -> bytes:
        return self._d[key]


def get_object_store(cfg: object) -> ObjectStore:
    """Pick a backend based on config. Imported lazily by phase entrypoints."""
    backend = getattr(cfg, "STORAGE_BACKEND", "s3")
    if backend == "s3":
        return S3ObjectStore(bucket=cfg.S3_BUCKET, region=cfg.AWS_REGION)  # type: ignore[attr-defined]
    if backend == "local":
        return LocalObjectStore(root=cfg.LOCAL_STORE_ROOT)  # type: ignore[attr-defined]
    raise ValueError(f"Unknown STORAGE_BACKEND={backend!r}")


# Re-export for convenience.
__all__ = [
    "InMemoryObjectStore",
    "LocalObjectStore",
    "ObjectStore",
    "S3ObjectStore",
    "get_object_store",
]


# unused import kept intentionally minimal
_ = io
