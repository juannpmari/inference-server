"""Concurrent dispatcher -- fires N requests simultaneously via asyncio.gather.

Primary tool for characterising throughput scaling, batching efficiency,
queue behaviour, and error thresholds under controlled parallel load.

The engine has ``max_pending = 10`` by default, so concurrency levels of
4 and 8 stay within capacity while 16 and 20 deliberately exceed the
queue limit and trigger HTTP 429 rejections.
"""

from __future__ import annotations

import asyncio

from benchmarks.dispatchers.base import BaseDispatcher, Prompt, ResponseRecord


class ConcurrentDispatcher(BaseDispatcher):
    """Fire a fixed-size batch of requests simultaneously."""

    def __init__(
        self,
        client,
        prompts: list[Prompt],
        config: dict,
        concurrency: int,
    ):
        super().__init__(client, prompts, config)
        self.concurrency = concurrency

    # ------------------------------------------------------------------
    # Batch construction
    # ------------------------------------------------------------------

    def _build_batch(self) -> list[Prompt]:
        """Build a batch of exactly ``self.concurrency`` prompts via round-robin.

        Since the pool is 25 files and max concurrency tested is 20, every
        request gets a distinct prompt at N <= 25.  This avoids artificial
        prefix-cache hits within a single concurrent batch.
        """
        return [self.prompts[i % len(self.prompts)] for i in range(self.concurrency)]

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(self) -> list[ResponseRecord]:
        """Fire all N requests simultaneously, return all results including errors.

        No semaphore is used -- the whole point is to fire all N at once
        without throttling, to stress-test the server's queue.
        """
        batch = self._build_batch()

        async def _safe_send(prompt: Prompt) -> ResponseRecord:
            """Ensure every slot returns a ResponseRecord, never a bare exception."""
            try:
                return await self.client.send_request(prompt)
            except Exception as exc:
                return ResponseRecord(
                    prompt_name=prompt.name,
                    input_tokens=prompt.input_tokens,
                    max_tokens=prompt.max_tokens,
                    ttft=0.0,
                    itl=[],
                    e2e=0.0,
                    output_tokens=0,
                    http_status=0,
                    error=f"{type(exc).__name__}: {exc}",
                )

        tasks = [_safe_send(p) for p in batch]
        results = await asyncio.gather(*tasks)
        return list(results)
