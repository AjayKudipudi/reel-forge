"""Single source of truth for configuration.

Built on `pydantic_settings.BaseSettings`. Eager validation at import time:
required env vars missing → ValidationError immediately, never at runtime.
Prompts are NOT here — they live in `data/prompts/animate.py` as code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Secrets ──────────────────────────────────────────────────────────
    HF_TOKEN: str
    # AWS keys are optional: on EC2 with an IAM instance profile attached,
    # boto3 fetches credentials from IMDS automatically. Local dev MUST
    # provide them via .env or the shell.
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    ANTHROPIC_API_KEY: str | None = None
    INSTAGRAM_GRAPH_TOKEN: str | None = None

    # ── AWS infra ────────────────────────────────────────────────────────
    AWS_REGION: str = "eu-south-2"
    S3_BUCKET: str = "insta-influencer-pipeline"
    S3_PREFIX: str = "jobs"
    EC2_AMI_ID: str = ""
    EC2_INSTANCE_TYPE: str = "g6.xlarge"
    EC2_KEY_NAME: str = ""
    EC2_SECURITY_GROUP_ID: str = ""
    EC2_SUBNET_ID: str = ""
    EC2_IAM_INSTANCE_PROFILE: str = ""
    USE_SPOT: bool = True
    EC2_SPOT_MAX_PRICE: float = 0.50
    # NoDecode: pydantic-settings would otherwise JSON-decode the env value;
    # we accept comma-separated strings, parsed by the validator below.
    SPOT_AZ_ROTATION: Annotated[tuple[str, ...], NoDecode] = (
        "eu-south-2a",
        "eu-south-2b",
        "eu-south-2c",
    )
    FALLBACK_TO_OD: bool = False
    # EBS Fast Snapshot Restore: enable per-AZ before generate, disable after.
    # Without FSR, fresh spot volumes lazy-load every block from S3 on first
    # read (~7-11 MB/s) regardless of gp3 throughput. With FSR, full gp3 speed
    # immediately. Cost: ~$0.75/AZ/hour while enabled — we enable just-in-time
    # before each generate batch and disable on completion.
    USE_FSR: bool = False
    # How long to wait for FSR state=enabled after enabling. AWS docs say
    # ~60 min/TB so for a 300 GB volume ~20 min is the typical upper bound.
    FSR_ENABLE_TIMEOUT_S: int = 1800

    # ── Storage backend ──────────────────────────────────────────────────
    STORAGE_BACKEND: Literal["s3", "local"] = "s3"
    LOCAL_STORE_ROOT: Path = REPO_ROOT / "volumes" / "store"

    # ── Model (SteadyDancer-14B only) ────────────────────────────────────
    DEFAULT_MODEL_QUANT: Literal[
        "fp16", "gguf-q4-s", "gguf-q4-m", "gguf-q5-m", "gguf-q6"
    ] = "gguf-q5-m"
    HF_STEADYDANCER: str = "MCG-NJU/SteadyDancer-14B"
    HF_STEADYDANCER_GGUF: str = "MCG-NJU/SteadyDancer-GGUF"
    STEADYDANCER_GIT_SHA: str = ""
    HF_DWPOSE: str = "yzd-v/DWPose"
    DEFAULT_SEED: int = 42

    # ── Output (DEFAULT_* seed Manifest.OutputSpec at prepare time) ──────
    DEFAULT_OUTPUT_FRAMES: int = 81
    DEFAULT_OUTPUT_FPS: int = 24
    REELS_OUTPUT_W: int = 1080
    REELS_OUTPUT_H: int = 1920
    DEFAULT_REELS_FORMAT_STRATEGY: Literal["letterbox", "pillarbox"] = "letterbox"
    # True so prepare-built manifests engage interp phase by default
    # (ffmpeg minterpolate 16fps -> 30fps Reels-native). Was False; v8.7
    # job 99a86c80604e shipped at 16fps despite OutputSpec default change
    # because build_manifest reads this cfg flag verbatim, overriding the
    # Pydantic default. Fixed at cfg layer in v8.8.
    DEFAULT_FRAME_INTERP: bool = True

    # ── Behavior ─────────────────────────────────────────────────────────
    DEFAULT_KEEP_REFERENCE_AUDIO: bool = True
    BACKGROUND_REPLACE: bool = False
    BACKGROUND_MATTE_MODEL: str = "birefnet"
    CONTENT_MODERATION_ENABLED: bool = False
    CONTENT_MODERATION_BINARY: str = ""

    # ── Branding ─────────────────────────────────────────────────────────
    INSTAGRAM_PAGE_HANDLE: str = ""
    INSTAGRAM_PAGE_NICHE: str = "dance"
    DEFAULT_HASHTAGS: str = "#reels #dance #trending"

    # ── Paths ────────────────────────────────────────────────────────────
    OUTPUT_DIR: Path = REPO_ROOT / "volumes" / "output" / "batch"
    ASSETS_DIR: Path = REPO_ROOT / "volumes" / "assets"
    LOG_DIR: Path = REPO_ROOT / "volumes" / "logs"
    EC2_WORK_DIR: Path = Path("/opt/insta-influencer/work")

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    LOG_FORMAT: Literal["text", "json"] = "text"

    # ── Observability ────────────────────────────────────────────────────
    HEARTBEAT_INTERVAL_S: int = 30
    RETRY_MAX_ATTEMPTS_DEFAULT: int = 3
    STDERR_TAIL_BYTES: int = 4096

    # ── Retention / cost ─────────────────────────────────────────────────
    RETENTION_DAYS: int = 14
    COST_REPORT_TIMEZONE: str = "UTC"

    @field_validator("SPOT_AZ_ROTATION", mode="before")
    @classmethod
    def _split_az(cls, v: object) -> tuple[str, ...]:
        if isinstance(v, str):
            return tuple(s.strip() for s in v.split(",") if s.strip())
        if isinstance(v, (list, tuple)):
            return tuple(str(x) for x in v)
        raise ValueError(f"SPOT_AZ_ROTATION must be string or sequence, got {type(v)}")

    def ensure_dirs(self) -> None:
        for p in (self.OUTPUT_DIR, self.ASSETS_DIR, self.LOG_DIR, self.LOCAL_STORE_ROOT):
            p.mkdir(parents=True, exist_ok=True)

    def to_subprocess_dict(self) -> dict[str, str]:
        """Serialize to env-var dict for child-process reconstruction.

        Pydantic Settings is not safely picklable across processes; instead
        we round-trip through env vars (the same source from which
        BaseSettings reads).
        """
        out: dict[str, str] = {}
        for name in type(self).model_fields:
            val = getattr(self, name)
            if val is None:
                continue
            if isinstance(val, tuple):
                out[name] = ",".join(map(str, val))
            elif isinstance(val, bool):
                out[name] = "true" if val else "false"
            elif isinstance(val, Path):
                out[name] = str(val)
            else:
                out[name] = str(val)
        return out


def load_config() -> Config:
    cfg = Config()
    # Bridge credentials to os.environ so downstream boto3 clients
    # (constructed without explicit creds) pick them up. setdefault means
    # we don't clobber values the operator already exported in their shell
    # or the test harness already set.
    import os as _os

    # Only bridge non-empty values; empty would clobber boto3's IAM-role
    # discovery on EC2 where the operator deliberately has no AWS keys set.
    if cfg.AWS_ACCESS_KEY_ID:
        _os.environ.setdefault("AWS_ACCESS_KEY_ID", cfg.AWS_ACCESS_KEY_ID)
    if cfg.AWS_SECRET_ACCESS_KEY:
        _os.environ.setdefault("AWS_SECRET_ACCESS_KEY", cfg.AWS_SECRET_ACCESS_KEY)
    _os.environ.setdefault("AWS_DEFAULT_REGION", cfg.AWS_REGION)
    if cfg.HF_TOKEN:
        _os.environ.setdefault("HF_TOKEN", cfg.HF_TOKEN)
        _os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", cfg.HF_TOKEN)
    cfg.ensure_dirs()
    return cfg


# Lazy init: importing this module shouldn't crash on missing env in tests.
# Production callers use `CONFIG = load_config()` explicitly. Tests usually
# build a `Config` directly with explicit kwargs.
_cached: Config | None = None


def get_config() -> Config:
    global _cached
    if _cached is None:
        _cached = load_config()
    return _cached
