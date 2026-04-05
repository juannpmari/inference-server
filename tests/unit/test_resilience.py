"""Tests for shared.resilience — CircuitBreaker and retry_with_backoff."""

import asyncio
import time

import pytest

from shared.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    CircuitBreakerState,
    retry_with_backoff,
)


# ---------------------------------------------------------------------------
# CircuitBreaker tests
# ---------------------------------------------------------------------------

class TestCircuitBreakerStateTransitions:
    """Test the CLOSED -> OPEN -> HALF_OPEN -> CLOSED / OPEN cycle."""

    @pytest.mark.asyncio
    async def test_starts_closed(self):
        cb = CircuitBreaker(failure_threshold=3)
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_closed_to_open_after_threshold(self):
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)

        async def _fail():
            raise RuntimeError("boom")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await cb.call(_fail)

        assert cb.state == CircuitBreakerState.OPEN
        assert cb.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_rejects_immediately(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=100.0)

        async def _fail():
            raise RuntimeError("boom")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(_fail)

        assert cb.state == CircuitBreakerState.OPEN

        with pytest.raises(CircuitBreakerOpen) as exc_info:
            await cb.call(_fail)
        assert exc_info.value.recovery_timeout == 100.0
        assert exc_info.value.time_remaining > 0

    @pytest.mark.asyncio
    async def test_open_to_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)

        async def _fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(_fail)

        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout to elapse
        await asyncio.sleep(0.06)

        # The next call should be allowed (HALF_OPEN)
        async def _succeed():
            return "ok"

        result = await cb.call(_succeed)
        assert result == "ok"
        assert cb.state == CircuitBreakerState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)

        async def _fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await cb.call(_fail)

        await asyncio.sleep(0.02)

        async def _succeed():
            return 42

        result = await cb.call(_succeed)
        assert result == 42
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.failure_count == 0

    @pytest.mark.asyncio
    async def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)

        async def _fail():
            raise RuntimeError("boom")

        # Trip to OPEN
        with pytest.raises(RuntimeError):
            await cb.call(_fail)

        await asyncio.sleep(0.02)

        # In HALF_OPEN, another failure should re-open
        with pytest.raises(RuntimeError):
            await cb.call(_fail)

        assert cb.state == CircuitBreakerState.OPEN

    @pytest.mark.asyncio
    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)

        async def _fail():
            raise RuntimeError("boom")

        async def _succeed():
            return "ok"

        # 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await cb.call(_fail)
        assert cb.failure_count == 2

        # Success resets
        await cb.call(_succeed)
        assert cb.failure_count == 0
        assert cb.state == CircuitBreakerState.CLOSED


# ---------------------------------------------------------------------------
# retry_with_backoff tests
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        call_count = 0

        async def _succeed():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = await retry_with_backoff(_succeed, base_delay=0.01)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_success_after_transient_failures(self):
        attempts = 0

        async def _flaky():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("transient")
            return "recovered"

        result = await retry_with_backoff(
            _flaky,
            max_retries=3,
            base_delay=0.01,
            retryable_exceptions=(ConnectionError,),
        )
        assert result == "recovered"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self):
        async def _always_fail():
            raise ConnectionError("permanent")

        with pytest.raises(ConnectionError, match="permanent"):
            await retry_with_backoff(
                _always_fail,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )

    @pytest.mark.asyncio
    async def test_respects_retryable_exceptions(self):
        """Non-retryable exceptions should propagate immediately."""
        call_count = 0

        async def _fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError, match="not retryable"):
            await retry_with_backoff(
                _fail,
                max_retries=3,
                base_delay=0.01,
                retryable_exceptions=(ConnectionError,),
            )
        # Should not retry for ValueError
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_backoff_delay_increases(self):
        """Verify that successive retries take longer (exponential backoff)."""
        timestamps = []

        async def _fail():
            timestamps.append(time.monotonic())
            raise ConnectionError("fail")

        with pytest.raises(ConnectionError):
            await retry_with_backoff(
                _fail,
                max_retries=2,
                base_delay=0.05,
                max_delay=30.0,
                retryable_exceptions=(ConnectionError,),
            )

        # 3 calls total: initial + 2 retries
        assert len(timestamps) == 3
        gap1 = timestamps[1] - timestamps[0]
        gap2 = timestamps[2] - timestamps[1]
        # Second gap should be roughly >= first gap (exponential)
        assert gap2 >= gap1 * 0.8  # allow some jitter tolerance
