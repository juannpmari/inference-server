"""E2E test: circuit breaker behaviour when engine goes down.

Stop engine → send requests to gateway → verify 503 → restart → verify recovery.

Requires:
    docker compose -f docker-compose.yml \\
        -f tests/integration/docker-compose.test.yml up -d
"""

import subprocess
import time

import httpx
import pytest

GATEWAY_URL = "http://localhost:8000"
COMPOSE_FILES = [
    "-f", "docker-compose.yml",
    "-f", "tests/integration/docker-compose.test.yml",
]
RECOVERY_TIMEOUT = 90  # seconds


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *COMPOSE_FILES, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _wait_for_ready(timeout: float = RECOVERY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=3.0)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


def _chat_request() -> httpx.Response:
    return httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2-0.5B",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        },
        timeout=10.0,
    )


@pytest.fixture(scope="module", autouse=True)
def _ensure_stack_ready():
    try:
        resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=10.0)
        if resp.status_code != 200:
            pytest.skip("Stack not ready")
    except httpx.HTTPError:
        pytest.skip("Stack not running")


class TestCircuitBreaker:

    def test_engine_stop_triggers_503_then_recovery(self):
        # 1. Baseline: verify inference works
        resp = _chat_request()
        assert resp.status_code == 200, f"Baseline request failed: {resp.text}"

        # 2. Stop engine container
        _compose("stop", "engine")
        time.sleep(10)  # Let health cache expire

        # 3. Requests should fail with 503 (engine unreachable or circuit breaker)
        try:
            resp = _chat_request()
            assert resp.status_code in (502, 503, 504), \
                f"Expected 5xx with engine down, got {resp.status_code}"
        except httpx.HTTPError:
            pass  # Connection error is also acceptable

        # 4. Gateway /readyz should report 503
        resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=5.0)
        assert resp.status_code == 503

        # 5. Restart engine
        _compose("start", "engine")

        # 6. Wait for full recovery
        recovered = _wait_for_ready(timeout=RECOVERY_TIMEOUT)
        assert recovered, "Gateway should recover after engine restart"

        # 7. Verify inference works again
        resp = _chat_request()
        assert resp.status_code == 200, f"Post-recovery request failed: {resp.text}"
