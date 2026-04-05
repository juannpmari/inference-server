"""Locust load test for the inference server gateway.

Usage:
    # Headless mode (CI-friendly):
    locust -f tests/load/locustfile.py --headless -u 50 -r 5 -t 60s --host http://localhost:8000

    # With web UI:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Ramps from 1 to 100 users over 2 minutes, then holds for 1 minute.
"""

from locust import HttpUser, task, between, events
from locust.runners import MasterRunner, WorkerRunner


class InferenceUser(HttpUser):
    """Simulates a user sending chat completion requests."""

    wait_time = between(0.5, 2.0)

    @task(weight=5)
    def chat_completion(self):
        """Non-streaming chat completion."""
        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2-0.5B",
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 32,
                "stream": False,
            },
            timeout=60,
            name="/v1/chat/completions",
        )

    @task(weight=3)
    def chat_completion_streaming(self):
        """Streaming chat completion."""
        with self.client.post(
            "/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2-0.5B",
                "messages": [{"role": "user", "content": "Count to 5"}],
                "max_tokens": 64,
                "stream": True,
            },
            timeout=60,
            stream=True,
            name="/v1/chat/completions [stream]",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"Status {resp.status_code}")
                return
            # Consume the stream
            for _ in resp.iter_lines():
                pass
            resp.success()

    @task(weight=1)
    def text_completion(self):
        """Non-streaming text completion."""
        self.client.post(
            "/v1/completions",
            json={
                "model": "Qwen/Qwen2-0.5B",
                "prompt": "Once upon a time",
                "max_tokens": 32,
                "stream": False,
            },
            timeout=60,
            name="/v1/completions",
        )

    @task(weight=1)
    def health_check(self):
        """Lightweight health check to mix in some fast requests."""
        self.client.get("/readyz", name="/readyz")


# ---------------------------------------------------------------------------
# Custom load shape: ramp up then hold
# ---------------------------------------------------------------------------

from locust import LoadTestShape


class RampThenHold(LoadTestShape):
    """
    Ramp from 0 → 100 users over 120s, then hold at 100 for 60s.
    Total duration: 180s.
    """

    stages = [
        {"duration": 120, "users": 100, "spawn_rate": 1},   # Ramp
        {"duration": 180, "users": 100, "spawn_rate": 1},   # Hold
    ]

    def tick(self):
        run_time = self.get_run_time()
        for stage in self.stages:
            if run_time < stage["duration"]:
                return stage["users"], stage["spawn_rate"]
        return None  # Stop


# ---------------------------------------------------------------------------
# Assertions (for CI — check results after run)
# ---------------------------------------------------------------------------

@events.quitting.add_listener
def _check_results(environment, **kwargs):
    """Fail the process if error rate or latency is too high."""
    if isinstance(environment.runner, (MasterRunner, WorkerRunner)):
        return  # Only check on standalone/local runner

    stats = environment.runner.stats.total
    if stats.num_requests == 0:
        return

    error_rate = stats.num_failures / stats.num_requests
    if error_rate > 0.05:
        environment.process_exit_code = 1
        print(f"FAIL: Error rate {error_rate:.1%} exceeds 5% threshold")

    p99 = stats.get_response_time_percentile(0.99) or 0
    if p99 > 30_000:  # 30s — generous for mock engine
        environment.process_exit_code = 1
        print(f"FAIL: p99 latency {p99}ms exceeds 30s threshold")
