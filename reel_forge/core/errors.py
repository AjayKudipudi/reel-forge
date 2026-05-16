"""Error classification and exception hierarchy.

Every exception thrown inside the pipeline is classified into an `ErrorClass`.
Bare exceptions are bugs; `classify(exc)` always returns a typed `ErrorInfo`.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .external_tool import ToolResult, ToolSpec


class ErrorClass(StrEnum):
    REEL_DOWNLOAD_FAILED = "reel_download_failed"
    REEL_DOWNLOAD_UNAVAILABLE = "reel_download_unavailable"
    LOCAL_VIDEO_INVALID = "local_video_invalid"
    PHOTO_INVALID = "photo_invalid"
    S3_UPLOAD_FAILED = "s3_upload_failed"
    S3_DOWNLOAD_FAILED = "s3_download_failed"
    SPOT_CAPACITY_UNAVAILABLE = "spot_capacity_unavailable"
    SPOT_MAX_PRICE_EXCEEDED = "spot_max_price_exceeded"
    INSTANCE_LAUNCH_TIMEOUT = "instance_launch_timeout"
    VCPU_LIMIT_EXCEEDED = "vcpu_limit_exceeded"
    POSE_EXTRACTION_NO_PERSON = "pose_extraction_no_person"
    POSE_EXTRACTION_LOW_CONF = "pose_extraction_low_confidence"
    MODEL_LOAD_FAILED = "model_load_failed"
    MODEL_OOM = "model_oom"
    INFERENCE_ERROR = "inference_error"
    FFMPEG_FAILED = "ffmpeg_failed"
    DISK_FULL = "disk_full"
    SPOT_RECLAIMED = "spot_reclaimed"
    CONTENT_MODERATION_REJECTED = "content_moderation_rejected"
    UNKNOWN = "unknown"


# Per-class retry budgets. Classes not in the dict are non-retryable.
RETRY_POLICY: dict[ErrorClass, int] = {
    ErrorClass.REEL_DOWNLOAD_FAILED: 3,
    ErrorClass.S3_UPLOAD_FAILED: 3,
    ErrorClass.S3_DOWNLOAD_FAILED: 3,
    ErrorClass.SPOT_CAPACITY_UNAVAILABLE: 5,
    ErrorClass.INSTANCE_LAUNCH_TIMEOUT: 2,
    ErrorClass.MODEL_OOM: 1,
    ErrorClass.INFERENCE_ERROR: 1,
    ErrorClass.SPOT_RECLAIMED: 999,  # always retry; spot interrupts shouldn't burn budget
}


def is_retryable(ec: ErrorClass) -> bool:
    return ec in RETRY_POLICY


def max_attempts(ec: ErrorClass) -> int:
    return RETRY_POLICY.get(ec, 1)


@dataclass(frozen=True)
class ErrorInfo:
    error_class: ErrorClass
    message: str
    retryable: bool
    stderr_tail: str | None = None
    attempt: int = 1


# ── Exception hierarchy ──────────────────────────────────────────────────


class PipelineError(Exception):
    """Base class. Concrete subclasses carry an ErrorClass."""

    error_class: ErrorClass = ErrorClass.UNKNOWN


class ReelDownloadFailed(PipelineError):
    error_class = ErrorClass.REEL_DOWNLOAD_FAILED


class ReelUnavailable(PipelineError):
    error_class = ErrorClass.REEL_DOWNLOAD_UNAVAILABLE


class LocalVideoInvalid(PipelineError):
    error_class = ErrorClass.LOCAL_VIDEO_INVALID


class PhotoInvalid(PipelineError):
    error_class = ErrorClass.PHOTO_INVALID


class ModelOOM(PipelineError):
    error_class = ErrorClass.MODEL_OOM


class PoseNoPerson(PipelineError):
    error_class = ErrorClass.POSE_EXTRACTION_NO_PERSON


class SpotReclaimed(PipelineError):
    error_class = ErrorClass.SPOT_RECLAIMED


class ContentModerationRejected(PipelineError):
    error_class = ErrorClass.CONTENT_MODERATION_REJECTED


class IllegalTransition(RuntimeError):
    """Raised when a status state transition is not in TRANSITIONS."""


class ToolFailed(PipelineError):
    """Raised by `run_tool`. Carries the spec, ToolResult, and pre-classified ErrorClass."""

    def __init__(
        self,
        *,
        spec: ToolSpec,
        result: ToolResult,
        error_class: ErrorClass,
        missing_artifact: Any = None,
    ) -> None:
        self.spec = spec
        self.result = result
        self.missing_artifact = missing_artifact
        self.error_class = error_class
        super().__init__(f"{spec.name} failed (rc={result.returncode}): {error_class}")


def classify(exc: BaseException) -> ErrorInfo:
    """Map an arbitrary exception to ErrorInfo.

    Catches torch OOM, botocore client errors, ToolFailed, anything subclass-of
    PipelineError. Unknowns get UNKNOWN/non-retryable with the type name in the
    message.
    """
    if isinstance(exc, ToolFailed):
        ec = exc.error_class
        return ErrorInfo(
            error_class=ec,
            message=str(exc),
            retryable=is_retryable(ec),
            stderr_tail=exc.result.stderr[-4096:] if exc.result.stderr else None,
        )
    if isinstance(exc, PipelineError):
        return ErrorInfo(
            error_class=exc.error_class,
            message=str(exc),
            retryable=is_retryable(exc.error_class),
        )
    name = type(exc).__name__
    msg = str(exc)
    if "OutOfMemory" in name or "OOM" in msg or "CUDA out of memory" in msg:
        return ErrorInfo(error_class=ErrorClass.MODEL_OOM, message=msg, retryable=True)
    if name == "ClientError":
        # botocore.exceptions.ClientError
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if "VcpuLimitExceeded" in code or "VcpuLimit" in code:
            return ErrorInfo(
                error_class=ErrorClass.VCPU_LIMIT_EXCEEDED,
                message=msg,
                retryable=False,
            )
        if "InsufficientInstanceCapacity" in code or "SpotMaxPriceTooLow" in code:
            return ErrorInfo(
                error_class=ErrorClass.SPOT_CAPACITY_UNAVAILABLE,
                message=msg,
                retryable=True,
            )
        if "NoSuchKey" in code or "NoSuchBucket" in code:
            return ErrorInfo(
                error_class=ErrorClass.S3_DOWNLOAD_FAILED,
                message=msg,
                retryable=True,
            )
    if name == "TimeoutExpired":
        return ErrorInfo(
            error_class=ErrorClass.INFERENCE_ERROR,
            message=f"timeout: {msg}",
            retryable=True,
        )
    return ErrorInfo(
        error_class=ErrorClass.UNKNOWN,
        message=f"{name}: {msg}",
        retryable=False,
    )
