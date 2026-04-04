"""Realistic dispatcher -- Poisson-distributed arrivals over a fixed duration.

Simulates production-like traffic by sending requests at exponentially-
distributed inter-arrival times.  Unlike the sequential and concurrent
dispatchers which operate on fixed request counts, the realistic dispatcher
is duration-bounded with a stochastic request count determined by the
arrival rate (lambda) and duration.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import time

from benchmarks.dispatchers.base import BaseDispatcher, Prompt, ResponseRecord

logger = logging.getLogger(__name__)


class RealisticDispatcher(BaseDispatcher):
    """Duration-based dispatcher with Poisson arrivals and mixed prompt pool."""

    def __init__(
        self,
        client,
        prompts: list[Prompt],
        config: dict,
        rps: float,
        duration_seconds: int,
        drain_timeout_seconds: float = 60.0,
        seed: int | None = None,
    ):
        super().__init__(client, prompts, config)
        self.rps = rps
        self.duration_seconds = duration_seconds
        self.drain_timeout_seconds = drain_timeout_seconds
        self.seed = seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> list[ResponseRecord]:
        """Execute the Poisson arrival loop for ``duration_seconds``, then drain."""
        rng = random.Random(self.seed)
        tasks: list[asyncio.Task] = []
        start_time = time.monotonic()

        sent = await self._dispatch_loop(tasks, start_time, rng)
        records, timed_out = await self._drain_inflight(tasks, self.drain_timeout_seconds)

        actual_duration = time.monotonic() - start_time
        completed = len(records)

        metadata = self._build_run_metadata(sent, completed, actual_duration, records)
        validation = self._validate_arrival_rate(sent, actual_duration)

        if not validation["within_2sigma"]:
            logger.warning(
                "Arrival rate validation failed: expected ~%.0f requests, got %d (z=%.2f)",
                validation["expected_count"],
                validation["actual_count"],
                validation["z_score"],
            )
        if timed_out > 0:
            logger.warning(
                "Drain timeout: %d requests still in-flight after %.0fs",
                timed_out,
                self.drain_timeout_seconds,
            )

        logger.info(
            "Realistic run complete: sent=%d completed=%d timed_out=%d "
            "actual_rps=%.2f duration=%.1fs",
            sent, completed, timed_out,
            metadata["actual_rps"], actual_duration,
        )

        return records

    async def warmup_timed(self, duration_seconds: float) -> None:
        """Run Poisson traffic for *duration_seconds*, discarding all results.

        Uses the same dispatch loop as :meth:`run` so that warmup traffic
        mirrors the measurement phase (Poisson arrivals, random prompt
        selection).
        """
        rng = random.Random(self.seed)
        tasks: list[asyncio.Task] = []
        start = time.monotonic()
        sent = await self._dispatch_loop(tasks, start, rng, duration=int(duration_seconds))
        await self._drain_inflight(tasks, self.drain_timeout_seconds)
        logger.info(
            "Timed warmup complete: sent %d requests over %.1fs",
            sent, time.monotonic() - start,
        )

    # ------------------------------------------------------------------
    # Dispatch loop
    # ------------------------------------------------------------------

    async def _dispatch_loop(
        self,
        tasks: list[asyncio.Task],
        start_time: float,
        rng: random.Random,
        *,
        duration: int | None = None,
    ) -> int:
        """Inner loop: sleep for an exponentially-distributed interval,
        pick a random prompt, and fire the request.  Returns count sent.
        """
        effective_duration = duration if duration is not None else self.duration_seconds
        sent = 0
        while True:
            elapsed = time.monotonic() - start_time
            remaining = effective_duration - elapsed
            if remaining <= 0:
                break

            sleep_time = rng.expovariate(self.rps)
            if sleep_time > remaining:
                break

            await asyncio.sleep(sleep_time)

            prompt = rng.choice(self.prompts)
            task = asyncio.create_task(self.client.send_request(prompt))
            tasks.append(task)
            sent += 1

        return sent

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    async def _drain_inflight(
        self,
        tasks: list[asyncio.Task],
        timeout: float,
    ) -> tuple[list[ResponseRecord], int]:
        """Gather with timeout, mark timed-out tasks as errors."""
        if not tasks:
            return [], 0

        done, pending = await asyncio.wait(tasks, timeout=timeout)

        timed_out = len(pending)
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        records: list[ResponseRecord] = []
        for task in done:
            try:
                result = task.result()
                if isinstance(result, ResponseRecord):
                    records.append(result)
            except Exception as exc:
                records.append(ResponseRecord(
                    prompt_name="unknown",
                    input_tokens=0,
                    max_tokens=0,
                    ttft=0.0,
                    itl=[],
                    e2e=0.0,
                    output_tokens=0,
                    http_status=0,
                    error=f"task_exception: {exc}",
                ))

        return records, timed_out

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def _build_run_metadata(
        self,
        sent: int,
        completed: int,
        actual_duration: float,
        records: list[ResponseRecord],
    ) -> dict:
        """Compute actual_rps, expected vs actual counts."""
        successful = [r for r in records if r.error is None]
        return {
            "total_requests_sent": sent,
            "total_requests_completed": completed,
            "total_successful": len(successful),
            "actual_rps": sent / actual_duration if actual_duration > 0 else 0.0,
            "actual_duration_seconds": round(actual_duration, 2),
            "configured_rps": self.rps,
            "configured_duration_seconds": self.duration_seconds,
        }

    def _validate_arrival_rate(self, actual_sent: int, actual_duration: float) -> dict:
        """Validate that the observed request count is consistent with the
        configured Poisson rate.  Returns a dict including a z-score and
        whether the count falls within 2 standard deviations.
        """
        expected_count = self.rps * self.duration_seconds
        stdev = math.sqrt(expected_count) if expected_count > 0 else 0.0
        z_score = (actual_sent - expected_count) / stdev if stdev > 0 else 0.0
        return {
            "expected_count": expected_count,
            "actual_count": actual_sent,
            "z_score": round(z_score, 2),
            "within_2sigma": abs(z_score) <= 2.0,
            "actual_rps": actual_sent / actual_duration if actual_duration > 0 else 0.0,
        }
