"""Token-bucket rate limiter for the inference server gateway.

Provides a simple in-process ``TokenBucketRateLimiter`` that can be checked
on every incoming request to enforce a maximum requests-per-second rate.
"""

from __future__ import annotations

import time


class TokenBucketRateLimiter:
    """Token-bucket rate limiter.

    Parameters
    ----------
    rate:
        Sustained token refill rate (tokens per second).
    burst:
        Maximum number of tokens the bucket can hold (burst capacity).
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens: float = float(burst)
        self._last_refill: float = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    def allow(self) -> bool:
        """Return ``True`` and consume a token if available, else ``False``."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def retry_after(self) -> float:
        """Estimated seconds until a token becomes available."""
        self._refill()
        if self._tokens >= 1.0:
            return 0.0
        deficit = 1.0 - self._tokens
        if self._rate <= 0:
            return float("inf")
        return deficit / self._rate
