"""Background thread that pulses status.last_heartbeat_at and samples
GPU/CPU/RAM telemetry."""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import structlog

from ..core.status import StatusManager
from ..core.status_models import ResourceTelemetry

log = structlog.get_logger(__name__)


def _sample_gpu() -> tuple[float | None, float | None]:
    """Return (gpu_util_pct, gpu_mem_gb) — None when nvml unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(h).used / (1024**3)
        pynvml.nvmlShutdown()
        return float(util), float(mem)
    except Exception:
        return None, None


def _sample_cpu_ram() -> tuple[float | None, float | None]:
    try:
        import psutil

        return float(psutil.cpu_percent(interval=None)), float(
            psutil.virtual_memory().used / (1024**3)
        )
    except Exception:
        return None, None


class HeartbeatThread:
    def __init__(self, status: StatusManager, *, interval_s: int = 30) -> None:
        self.status = status
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        # Prime psutil so first cpu_percent isn't 0.0
        _sample_cpu_ram()
        while not self._stop.wait(self.interval_s):
            try:
                gpu_u, gpu_m = _sample_gpu()
                cpu_u, ram = _sample_cpu_ram()
                self.status.append_telemetry(
                    ResourceTelemetry(
                        timestamp=datetime.now(UTC),
                        gpu_util_pct=gpu_u,
                        gpu_mem_gb=gpu_m,
                        cpu_util_pct=cpu_u,
                        ram_gb=ram,
                    )
                )
            except Exception as exc:
                log.warning("heartbeat.error", err=str(exc))


_ = time  # keep
