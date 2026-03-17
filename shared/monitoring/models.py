from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class TimingInfo:
    """Per-request timing data collected during the batching loop."""
    submitted_at: float = 0.0
    processing_started_at: float = 0.0
    first_token_at: float = 0.0
    last_step_at: float = 0.0
    finished_at: float = 0.0
    step_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    adapter_id: Optional[str] = None
    adapter_swap_latency_s: float = 0.0


@dataclass
class RequestRecord:
    """Completed request record for the session collector."""
    request_id: str
    model_id: str
    adapter_id: Optional[str]
    timestamp: float
    queue_wait_s: float
    prefill_s: float
    ttft_s: float
    decode_s: float
    e2e_duration_s: float
    input_tokens: int
    output_tokens: int
    inter_token_latency_s: float
    tokens_per_second: float
    status: str  # "success", "error", "timeout"
    adapter_swap_latency_s: float = 0.0

    @classmethod
    def from_timing(cls, request_id: str, model_id: str, timing: TimingInfo, status: str = "success") -> "RequestRecord":
        queue_wait = timing.processing_started_at - timing.submitted_at if timing.processing_started_at else 0.0
        prefill = timing.first_token_at - timing.processing_started_at if timing.first_token_at and timing.processing_started_at else 0.0
        ttft = timing.first_token_at - timing.submitted_at if timing.first_token_at else 0.0
        e2e = timing.finished_at - timing.submitted_at if timing.finished_at else 0.0
        decode = timing.finished_at - timing.first_token_at if timing.finished_at and timing.first_token_at else 0.0

        if timing.step_count > 1 and decode > 0:
            inter_token = decode / (timing.step_count - 1)
        else:
            inter_token = 0.0

        if decode > 0 and timing.output_tokens > 0:
            tps = timing.output_tokens / decode
        else:
            tps = 0.0

        return cls(
            request_id=request_id,
            model_id=model_id,
            adapter_id=timing.adapter_id,
            timestamp=timing.submitted_at,
            queue_wait_s=queue_wait,
            prefill_s=prefill,
            ttft_s=ttft,
            decode_s=decode,
            e2e_duration_s=e2e,
            input_tokens=timing.input_tokens,
            output_tokens=timing.output_tokens,
            inter_token_latency_s=inter_token,
            tokens_per_second=tps,
            status=status,
            adapter_swap_latency_s=timing.adapter_swap_latency_s,
        )
