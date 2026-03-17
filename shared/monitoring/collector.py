import collections
import statistics
import threading
import time
from typing import Callable, Dict, List, Optional

from shared.monitoring.models import RequestRecord


class SessionCollector:
    """Thread-safe in-process collector with a bounded ring buffer."""

    def __init__(self, maxlen: int = 10_000, lora_state_fn: Optional[Callable[[], dict]] = None):
        self._buffer: collections.deque[RequestRecord] = collections.deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._start_time = time.monotonic()
        self._start_wall = time.time()
        self._total_requests = 0
        self._total_errors = 0
        self._total_timeouts = 0
        self._max_batch_size = 0
        self._batch_sizes: collections.deque[int] = collections.deque(maxlen=10_000)

        # Queue depth tracking
        self._current_queue_depth = 0
        self._max_queue_depth = 0

        # Optional callback for live LoRA GPU state
        self._lora_state_fn = lora_state_fn

    def record_request(self, record: RequestRecord) -> None:
        with self._lock:
            self._buffer.append(record)
            self._total_requests += 1
            if record.status == "error":
                self._total_errors += 1
            elif record.status == "timeout":
                self._total_timeouts += 1

    def record_batch_size(self, size: int) -> None:
        with self._lock:
            self._batch_sizes.append(size)
            if size > self._max_batch_size:
                self._max_batch_size = size

    def inc_queue_depth(self) -> None:
        with self._lock:
            self._current_queue_depth += 1
            if self._current_queue_depth > self._max_queue_depth:
                self._max_queue_depth = self._current_queue_depth

    def dec_queue_depth(self) -> None:
        with self._lock:
            self._current_queue_depth = max(0, self._current_queue_depth - 1)

    def get_summary(self, model_id: Optional[str] = None) -> dict:
        with self._lock:
            records = list(self._buffer)
            batch_sizes = list(self._batch_sizes)
            total = self._total_requests
            errors = self._total_errors
            timeouts = self._total_timeouts
            max_batch = self._max_batch_size
            current_queue = self._current_queue_depth
            max_queue = self._max_queue_depth

        if model_id:
            records = [r for r in records if r.model_id == model_id]

        uptime = time.monotonic() - self._start_time
        rps = total / uptime if uptime > 0 else 0.0

        success_records = [r for r in records if r.status == "success"]

        lora = _lora_summary(records)
        # Merge live GPU state if available
        if self._lora_state_fn:
            try:
                gpu_state = self._lora_state_fn()
                lora["gpu_loaded_count"] = gpu_state.get("loaded_count", 0)
                lora["gpu_loaded_adapters"] = gpu_state.get("loaded_keys", [])
            except Exception:
                pass

        return {
            "session": {
                "start_time": self._start_wall,
                "uptime_seconds": round(uptime, 2),
                "total_requests": total,
                "requests_per_second": round(rps, 4),
            },
            "latency": {
                "ttft": _percentiles([r.ttft_s for r in success_records]),
                "prefill": _percentiles([r.prefill_s for r in success_records]),
                "queue_wait": _percentiles([r.queue_wait_s for r in success_records]),
                "inter_token": _percentiles([r.inter_token_latency_s for r in success_records]),
                "e2e": _percentiles([r.e2e_duration_s for r in success_records]),
            },
            "throughput": _throughput(success_records, uptime),
            "generation": {
                "input_length": _min_max_mean([r.input_tokens for r in success_records]),
                "output_length": _min_max_mean([r.output_tokens for r in success_records]),
            },
            "batching": {
                "avg_batch_size": round(statistics.mean(batch_sizes), 2) if batch_sizes else 0.0,
                "max_batch_size": max_batch,
            },
            "queue": {
                "current_depth": current_queue,
                "max_depth": max_queue,
            },
            "errors": {
                "error_rate": round(errors / total, 4) if total else 0.0,
                "timeout_rate": round(timeouts / total, 4) if total else 0.0,
                "total_errors": errors,
                "total_timeouts": timeouts,
            },
            "lora": lora,
            "gpu": {},
            "kv_cache": {},
        }

    def get_records_since(self, timestamp: float) -> List[RequestRecord]:
        """Return records newer than *timestamp* (for persistence flushing)."""
        with self._lock:
            return [r for r in self._buffer if r.timestamp >= timestamp]

    def get_all_records(self) -> List[RequestRecord]:
        with self._lock:
            return list(self._buffer)


def _percentiles(values: List[float]) -> dict:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)
    return {
        "avg": round(statistics.mean(sorted_v), 6),
        "p50": round(sorted_v[int(n * 0.50)], 6),
        "p95": round(sorted_v[min(int(n * 0.95), n - 1)], 6),
        "p99": round(sorted_v[min(int(n * 0.99), n - 1)], 6),
    }


def _min_max_mean(values: List[int]) -> dict:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(statistics.mean(values), 2),
    }


def _throughput(records: List[RequestRecord], uptime: float) -> dict:
    total_input = sum(r.input_tokens for r in records)
    total_output = sum(r.output_tokens for r in records)
    return {
        "requests_per_second": round(len(records) / uptime, 4) if uptime else 0.0,
        "input_tokens_per_second": round(total_input / uptime, 2) if uptime else 0.0,
        "output_tokens_per_second": round(total_output / uptime, 2) if uptime else 0.0,
    }


def _lora_summary(records: List[RequestRecord]) -> dict:
    adapter_records = [r for r in records if r.adapter_id]
    adapters = set(r.adapter_id for r in adapter_records)

    # Per-adapter request distribution
    per_adapter: Dict[str, int] = {}
    for r in adapter_records:
        per_adapter[r.adapter_id] = per_adapter.get(r.adapter_id, 0) + 1

    # Swap latency stats (only records with actual swaps)
    swap_records = [r for r in adapter_records if r.adapter_swap_latency_s > 0]
    swap_count = len(swap_records)
    avg_swap_ms = (
        round(statistics.mean(r.adapter_swap_latency_s for r in swap_records) * 1000, 2)
        if swap_records else 0.0
    )

    return {
        "active_adapters": len(adapters),
        "adapter_names": sorted(adapters),
        "total_adapter_requests": len(adapter_records),
        "per_adapter_requests": per_adapter,
        "swap_count": swap_count,
        "avg_swap_latency_ms": avg_swap_ms,
    }
