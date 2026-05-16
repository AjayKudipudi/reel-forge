"""classify() coverage for known classes + unknown."""
from __future__ import annotations

from reel_forge.core.errors import (
    ErrorClass,
    ModelOOM,
    PhotoInvalid,
    PoseNoPerson,
    ReelDownloadFailed,
    classify,
    is_retryable,
    max_attempts,
)


def test_classify_pipeline_errors() -> None:
    assert classify(PhotoInvalid("bad")).error_class == ErrorClass.PHOTO_INVALID
    assert classify(ModelOOM("oom")).error_class == ErrorClass.MODEL_OOM
    assert classify(PoseNoPerson("no")).error_class == ErrorClass.POSE_EXTRACTION_NO_PERSON
    assert classify(ReelDownloadFailed("nope")).error_class == ErrorClass.REEL_DOWNLOAD_FAILED


def test_classify_torch_oom_string_match() -> None:
    info = classify(RuntimeError("CUDA out of memory: tried to allocate 4.2 GiB"))
    assert info.error_class == ErrorClass.MODEL_OOM
    assert info.retryable is True


def test_classify_unknown() -> None:
    info = classify(KeyError("nope"))
    assert info.error_class == ErrorClass.UNKNOWN
    assert info.retryable is False


def test_retry_policy_lookup() -> None:
    assert is_retryable(ErrorClass.MODEL_OOM)
    assert not is_retryable(ErrorClass.UNKNOWN)
    assert max_attempts(ErrorClass.MODEL_OOM) == 1
    assert max_attempts(ErrorClass.UNKNOWN) == 1
