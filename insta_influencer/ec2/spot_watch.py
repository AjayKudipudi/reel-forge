"""Spot reclamation watcher.

Polls EC2 instance metadata for `spot/instance-action`. When AWS issues
the 2-minute warning, we set status to failed_recoverable with class
spot_reclaimed and exit cleanly so the orchestrator finishes its current
write before termination.

Uses `requests` rather than `urllib.request.urlopen` — IMDS endpoint is
a hardcoded RFC 3927 link-local address (169.254.169.254) and `requests`
makes that obvious to readers and to lint tools.
"""
from __future__ import annotations

import os
import threading
from datetime import UTC, datetime

import requests
import structlog

from ..core.errors import ErrorClass
from ..core.status import StatusManager
from ..core.status_models import FailureInfoModel

log = structlog.get_logger(__name__)

# AWS Instance Metadata Service v2 — link-local, hardcoded by AWS spec.
IMDS_HOST = "http://169.254.169.254"
IMDS_TOKEN_ENDPOINT = f"{IMDS_HOST}/latest/api/token"
IMDS_SPOT_ACTION_ENDPOINT = f"{IMDS_HOST}/latest/meta-data/spot/instance-action"


class SpotWatchThread:
    def __init__(self, status: StatusManager, *, interval_s: int = 5) -> None:
        self.status = status
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = os.getenv("INSTA_SPOT_WATCH", "1") == "1"

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="spot-watch")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                tok_resp = requests.put(
                    IMDS_TOKEN_ENDPOINT,
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
                    timeout=2,
                )
                if tok_resp.status_code != 200:
                    continue
                token = tok_resp.text
                action_resp = requests.get(
                    IMDS_SPOT_ACTION_ENDPOINT,
                    headers={"X-aws-ec2-metadata-token": token},
                    timeout=2,
                )
                if action_resp.status_code == 404:
                    # No spot action — instance is healthy.
                    continue
                if action_resp.status_code != 200:
                    log.warning("spot_watch.unexpected_status", code=action_resp.status_code)
                    continue
                if action_resp.text:
                    self._handle(action_resp.text)
                    return
            except requests.RequestException as exc:
                log.debug("spot_watch.request_error", err=str(exc))

    def _handle(self, body: str) -> None:
        log.warning("spot.reclaimed", body=body)
        failure = FailureInfoModel(
            phase=self.status.status.current_phase,
            error_class=ErrorClass.SPOT_RECLAIMED,
            message=f"spot/instance-action: {body}",
            retryable=True,
            occurred_at=datetime.now(UTC),
        )
        # Force an atomic flush so S3 has the reclaimed status before the
        # 2 minutes are up.
        self.status.status = self.status.status.model_copy(
            update={"failure": failure, "updated_at": datetime.now(UTC)}
        )
        self.status._flush()
