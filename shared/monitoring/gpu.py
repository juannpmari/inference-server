"""Async pynvml wrapper with graceful degradation."""

import asyncio
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class GPUMonitor:
    """Polls GPU metrics via pynvml in a background task."""

    def __init__(
        self,
        poll_interval: float = 2.0,
        device_index: int = 0,
        on_sample: Optional[Callable[[dict], None]] = None,
    ):
        self._poll_interval = poll_interval
        self._device_index = device_index
        self._available = False
        self._handle = None
        self._task: asyncio.Task | None = None
        self._on_sample = on_sample

        # Current snapshot
        self.compute_utilization_pct: float = 0.0
        self.memory_used_bytes: int = 0
        self.memory_total_bytes: int = 0
        self.memory_utilization_pct: float = 0.0
        self.power_watts: float = 0.0

        # Session aggregates
        self._samples = 0
        self._sum_compute = 0.0
        self._sum_memory_pct = 0.0
        self._max_compute = 0.0
        self._max_memory_used = 0
        self._sum_power = 0.0
        self._max_power = 0.0

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._available = True
                logger.info(f"GPUMonitor initialized for device {device_index}")
            except Exception as e:
                logger.warning(f"pynvml init failed, GPU metrics unavailable: {e}")

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> None:
        if self._available and self._task is None:
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    async def _poll_loop(self) -> None:
        try:
            while True:
                self._sample()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            raise

    def _sample(self) -> None:
        if not self._available:
            return
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.power_watts = power_mw / 1000.0
            except Exception:
                self.power_watts = 0.0

            self.compute_utilization_pct = float(util.gpu)
            self.memory_used_bytes = mem.used
            self.memory_total_bytes = mem.total
            self.memory_utilization_pct = (mem.used / mem.total * 100.0) if mem.total else 0.0

            self._samples += 1
            self._sum_compute += self.compute_utilization_pct
            self._sum_memory_pct += self.memory_utilization_pct
            self._sum_power += self.power_watts
            if self.compute_utilization_pct > self._max_compute:
                self._max_compute = self.compute_utilization_pct
            if mem.used > self._max_memory_used:
                self._max_memory_used = mem.used
            if self.power_watts > self._max_power:
                self._max_power = self.power_watts

            # Fire callback with current snapshot
            if self._on_sample:
                snapshot = {
                    "compute_utilization_pct": self.compute_utilization_pct,
                    "memory_used_bytes": self.memory_used_bytes,
                    "memory_total_bytes": self.memory_total_bytes,
                    "memory_utilization_pct": self.memory_utilization_pct,
                    "power_watts": self.power_watts,
                }
                try:
                    self._on_sample(snapshot)
                except Exception as e:
                    logger.warning(f"GPU on_sample callback failed: {e}")

        except Exception as e:
            logger.warning(f"GPU sample failed: {e}")

    def get_summary(self) -> dict:
        if not self._available:
            return {"available": False}
        avg_compute = self._sum_compute / self._samples if self._samples else 0.0
        avg_memory = self._sum_memory_pct / self._samples if self._samples else 0.0
        avg_power = self._sum_power / self._samples if self._samples else 0.0
        return {
            "available": True,
            "device_index": self._device_index,
            "current": {
                "compute_utilization_pct": round(self.compute_utilization_pct, 1),
                "memory_used_bytes": self.memory_used_bytes,
                "memory_total_bytes": self.memory_total_bytes,
                "memory_utilization_pct": round(self.memory_utilization_pct, 1),
                "power_watts": round(self.power_watts, 1),
            },
            "session": {
                "avg_compute_utilization_pct": round(avg_compute, 1),
                "avg_memory_utilization_pct": round(avg_memory, 1),
                "max_compute_utilization_pct": round(self._max_compute, 1),
                "max_memory_used_bytes": self._max_memory_used,
                "avg_power_watts": round(avg_power, 1),
                "max_power_watts": round(self._max_power, 1),
                "samples": self._samples,
            },
        }
