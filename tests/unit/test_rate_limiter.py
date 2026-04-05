"""Tests for shared.rate_limiter — TokenBucketRateLimiter."""

import time

import pytest

from shared.rate_limiter import TokenBucketRateLimiter


class TestTokenBucketRateLimiter:

    def test_allow_returns_true_up_to_burst(self):
        rl = TokenBucketRateLimiter(rate=10.0, burst=5)
        results = [rl.allow() for _ in range(5)]
        assert all(results), "All burst requests should be allowed"

    def test_allow_returns_false_after_drain(self):
        rl = TokenBucketRateLimiter(rate=10.0, burst=3)
        for _ in range(3):
            rl.allow()
        assert rl.allow() is False, "Should be denied after burst exhausted"

    def test_token_refill(self):
        rl = TokenBucketRateLimiter(rate=1000.0, burst=5)
        # Drain all tokens
        for _ in range(5):
            rl.allow()
        assert rl.allow() is False

        # Wait for refill (1000 tokens/sec → 1 token in 1ms)
        time.sleep(0.01)
        assert rl.allow() is True, "Token should have refilled"

    def test_burst_cap(self):
        rl = TokenBucketRateLimiter(rate=1000.0, burst=3)
        # Wait to potentially accumulate tokens beyond burst
        time.sleep(0.05)  # would accumulate 50 tokens at 1000/s
        # But burst cap is 3, so only 3 should be available
        results = [rl.allow() for _ in range(4)]
        assert results == [True, True, True, False]

    def test_retry_after_positive_when_empty(self):
        rl = TokenBucketRateLimiter(rate=10.0, burst=2)
        rl.allow()
        rl.allow()
        ra = rl.retry_after()
        assert ra > 0.0, "retry_after should be positive when bucket is empty"

    def test_retry_after_zero_when_available(self):
        rl = TokenBucketRateLimiter(rate=10.0, burst=5)
        assert rl.retry_after() == 0.0
