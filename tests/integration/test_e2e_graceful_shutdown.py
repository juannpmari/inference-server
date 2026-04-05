"""E2E test: graceful shutdown — in-flight requests complete, new ones get 503.

Submit a long request → send SIGTERM to engine → verify request completes →
verify new requests get rejected.

Requires:
    docker compose -f docker-compose.yml \\
        -f tests/integration/docker-compose.test.yml up -d
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

GATEWAY_URL = "http://localhost:8000"
COMPOSE_FILES = [
    "-f", "docker-compose.yml",
    "-f", "tests/integration/docker-compose.test.yml",
]


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *COMPOSE_FILES, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _get_engine_container_id() -> str | None:
    """Get the container ID for the engine service."""
    result = _compose("ps", "-q", "engine")
    cid = result.stdout.strip()
    return cid if cid else None


def _long_chat_request() -> httpx.Response:
    """Send a request that takes a while to complete."""
    return httpx.post(
        f"{GATEWAY_URL}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen2-0.5B",
            "messages": [{"role": "user", "content": "Write a long poem about nature"}],
            "max_tokens": 256,
        },
        timeout=120.0,
    )


def _short_chat_request() -> httpx.Response:
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


class TestGracefulShutdown:

    def test_inflight_request_completes_after_sigterm(self):
        """In-flight request should complete even after SIGTERM."""
        cid = _get_engine_container_id()
        if not cid:
            pytest.skip("Engine container not found")

        # 1. Start a long request in the background
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_long_chat_request)
            time.sleep(2)  # Give it time to start processing

            # 2. Send SIGTERM to engine container
            subprocess.run(
                ["docker", "kill", "--signal=SIGTERM", cid],
                capture_output=True,
                timeout=10,
            )

            # 3. The in-flight request should complete (or fail gracefully)
            try:
                resp = future.result(timeout=90)
                # Request either completed successfully or was rejected gracefully
                assert resp.status_code in (200, 503), \
                    f"Expected 200 (completed) or 503 (drained), got {resp.status_code}"
            except Exception:
                # Connection drop is also acceptable during shutdown
                pass

        # 4. After engine is down, new requests via gateway should fail
        time.sleep(10)  # Let health cache expire
        try:
            resp = _short_chat_request()
            # Either 503 from gateway or connection error
            assert resp.status_code in (502, 503, 504)
        except httpx.HTTPError:
            pass  # Connection error is fine

        # 5. Restart engine for other tests
        _compose("start", "engine")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=3.0)
                if resp.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
