"""E2E test: health endpoint behaviour when containers start/stop.

Requires the test docker-compose stack to be running:
    docker compose -f docker-compose.yml \\
        -f tests/integration/docker-compose.test.yml up -d

Run with:
    pytest tests/integration/test_e2e_health.py -v
"""

import asyncio
import subprocess
import time

import httpx
import pytest

GATEWAY_URL = "http://localhost:8000"
COMPOSE_FILES = [
    "-f", "docker-compose.yml",
    "-f", "tests/integration/docker-compose.test.yml",
]
TIMEOUT = 60  # seconds to wait for recovery


def _compose(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", *COMPOSE_FILES, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _wait_for(url: str, expected_status: int, timeout: float = TIMEOUT) -> bool:
    """Poll *url* until it returns *expected_status* or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code == expected_status:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="module", autouse=True)
def _ensure_stack_running():
    """Verify the test stack is up before running integration tests."""
    try:
        resp = httpx.get(f"{GATEWAY_URL}/healthz", timeout=5.0)
        if resp.status_code != 200:
            pytest.skip("Integration test stack not running (gateway not healthy)")
    except httpx.HTTPError:
        pytest.skip("Integration test stack not running (gateway unreachable)")


class TestHealthEndpoints:
    """Basic health endpoint checks against the running stack."""

    def test_healthz_always_200(self):
        resp = httpx.get(f"{GATEWAY_URL}/healthz", timeout=5.0)
        assert resp.status_code == 200

    def test_health_always_200(self):
        resp = httpx.get(f"{GATEWAY_URL}/health", timeout=5.0)
        assert resp.status_code == 200

    def test_readyz_returns_200_when_healthy(self):
        resp = httpx.get(f"{GATEWAY_URL}/readyz", timeout=5.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"


class TestSidecarKillRecovery:
    """Kill sidecar → engine unhealthy → gateway /ready 503 → restart → recovery."""

    def test_sidecar_kill_and_recovery(self):
        # 1. Verify healthy baseline
        assert _wait_for(f"{GATEWAY_URL}/readyz", 200, timeout=30), \
            "Gateway should be ready before test starts"

        # 2. Kill sidecar
        _compose("stop", "sidecar")

        # 3. Wait for engine to become unhealthy (health cache TTL ~5s)
        # The engine health check should eventually fail since sidecar is gone.
        # Gateway /readyz may take a few cycles to reflect this.
        time.sleep(10)

        # 4. Restart sidecar
        _compose("start", "sidecar")

        # 5. Wait for recovery
        recovered = _wait_for(f"{GATEWAY_URL}/readyz", 200, timeout=TIMEOUT)
        assert recovered, "Gateway /readyz should recover after sidecar restart"
