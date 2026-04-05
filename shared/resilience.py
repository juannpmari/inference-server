"""Circuit breaker and retry-with-backoff utilities for the inference server.

Provides:
- ``CircuitBreakerState`` enum (CLOSED, OPEN, HALF_OPEN)
- ``CircuitBreakerOpen`` exception
- ``CircuitBreaker`` class with configurable failure threshold and recovery
- ``retry_with_backoff`` async helper with exponential backoff and jitter
"""

from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import Any, Callable, Tuple, Type


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and calls are rejected."""

    def __init__(self, recovery_timeout: float, time_remaining: float) -> None:
        self.recovery_timeout = recovery_timeout
        self.time_remaining = time_remaining
        super().__init__(
            f"Circuit breaker is open. Recovery in {time_remaining:.1f}s"
        )


class CircuitBreaker:
    """Simple circuit breaker for async callables.

    States:
    - CLOSED: requests flow through normally. Failures increment the counter.
    - OPEN: all requests are rejected immediately. After *recovery_timeout*
      seconds the breaker transitions to HALF_OPEN.
    - HALF_OPEN: a limited number of probe requests are allowed. On success
      the breaker resets to CLOSED; on failure it re-opens.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitBreakerState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls: int = 0

    # -- internal helpers ----------------------------------------------------

    def _should_allow(self) -> bool:
        if self._state == CircuitBreakerState.CLOSED:
            return True

        if self._state == CircuitBreakerState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.recovery_timeout:
                self._state = CircuitBreakerState.HALF_OPEN
                self._half_open_calls = 0
                return True
            return False

        # HALF_OPEN
        if self._half_open_calls < self.half_open_max_calls:
            self._half_open_calls += 1
            return True
        return False

    def _record_success(self) -> None:
        self._failure_count = 0
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.CLOSED

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitBreakerState.HALF_OPEN:
            self._state = CircuitBreakerState.OPEN
            return

        if self._failure_count >= self.failure_threshold:
            self._state = CircuitBreakerState.OPEN

    # -- public API ----------------------------------------------------------

    async def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Invoke *func* through the circuit breaker.

        Raises ``CircuitBreakerOpen`` when the breaker is open.
        """
        if not self._should_allow():
            elapsed = time.monotonic() - self._last_failure_time
            remaining = max(0.0, self.recovery_timeout - elapsed)
            raise CircuitBreakerOpen(self.recovery_timeout, remaining)

        try:
            result = await func(*args, **kwargs)
        except Exception:
            self._record_failure()
            raise

        self._record_success()
        return result

    @property
    def state(self) -> CircuitBreakerState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count


# ---------------------------------------------------------------------------
# Retry with backoff
# ---------------------------------------------------------------------------

async def retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Tuple[Type[BaseException], ...] = (Exception,),
    **kwargs: Any,
) -> Any:
    """Call *func* with exponential backoff on transient failures.

    Parameters
    ----------
    func:
        Async callable to invoke.
    max_retries:
        Maximum number of retry attempts (total calls = max_retries + 1).
    base_delay:
        Initial delay in seconds before the first retry.
    max_delay:
        Upper bound on the backoff delay.
    retryable_exceptions:
        Tuple of exception types that trigger a retry.
    """
    last_exc: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except retryable_exceptions as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            delay = min(base_delay * (2 ** attempt), max_delay)
            jitter = random.uniform(0, delay * 0.5)
            await asyncio.sleep(delay + jitter)
    # Should not reach here, but satisfy type checkers
    raise last_exc  # type: ignore[misc]
