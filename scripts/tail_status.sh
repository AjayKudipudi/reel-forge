#!/usr/bin/env bash
# Convenience: tail the per-job log file with structlog rendering.
set -euo pipefail
JOB_ID="${1:?usage: tail_status.sh <job_id>}"
LOG_DIR="${LOG_DIR:-volumes/logs}"
exec tail -f "${LOG_DIR}/${JOB_ID}.log"
