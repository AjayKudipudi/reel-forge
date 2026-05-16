"""Filename and S3-key constants. Single source of truth.

Every module that constructs a path imports from here — never an inline
string literal. Renaming a file becomes a one-line change here, not a
grep-and-edit campaign.
"""
from __future__ import annotations

# ── Per-job filenames in work_dir / job_dir ──────────────────────────────
PHOTO = "photo.png"
REFERENCE_VIDEO = "reference.mp4"
MANIFEST = "manifest.json"
STATUS = "status.json"
POSE_DIR = "pose"  # subdir inside work_dir; passed to upstream as --cond_pos_folder
POSE_VIDEO = "pose/aligned_pose.mp4"  # produced by upstream pose_align.py
POSE_OVERLAY = "pose_overlay.mp4"
ANIMATED = "animated.mp4"
ANIMATED_60FPS = "animated_60fps.mp4"
ANIMATED_60FPS_FACE = "animated_60fps_face.mp4"
ANIMATED_W_AUDIO = "animated_with_audio.mp4"
FINAL = "final.mp4"
STATS = "stats.json"

# ── S3-key templates ─────────────────────────────────────────────────────
S3_JOB_DIR = "{prefix}/{job_id}"
S3_MARKER = "{prefix}/{job_id}/markers/{phase}.done"
S3_PENDING = "{prefix}/_pending.json"
S3_CANCEL = "{prefix}/{job_id}/_cancel"
S3_LOGS_DIR = "{prefix}/{job_id}/logs"
S3_LOG_FILE = "{prefix}/{job_id}/logs/{name}"
S3_FINAL = "{prefix}/{job_id}/" + FINAL
S3_MANIFEST = "{prefix}/{job_id}/" + MANIFEST
S3_STATUS = "{prefix}/{job_id}/" + STATUS


def s3_for(template: str, **kw: str) -> str:
    return template.format(**kw)


def s3_marker_key(prefix: str, job_id: str, phase: str) -> str:
    return S3_MARKER.format(prefix=prefix, job_id=job_id, phase=phase)


def s3_status_key(prefix: str, job_id: str) -> str:
    return S3_STATUS.format(prefix=prefix, job_id=job_id)


def s3_manifest_key(prefix: str, job_id: str) -> str:
    return S3_MANIFEST.format(prefix=prefix, job_id=job_id)


def s3_final_key(prefix: str, job_id: str) -> str:
    return S3_FINAL.format(prefix=prefix, job_id=job_id)


def s3_cancel_key(prefix: str, job_id: str) -> str:
    return S3_CANCEL.format(prefix=prefix, job_id=job_id)


def s3_pending_key(prefix: str) -> str:
    return S3_PENDING.format(prefix=prefix)


def s3_log_key(prefix: str, job_id: str, name: str) -> str:
    return S3_LOG_FILE.format(prefix=prefix, job_id=job_id, name=name)


def s3_job_prefix(prefix: str, job_id: str) -> str:
    return S3_JOB_DIR.format(prefix=prefix, job_id=job_id)
