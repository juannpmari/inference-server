"""Sequential dispatcher -- sends one request at a time (concurrency=1).

Establishes a contention-free baseline for latency metrics.  The dispatcher
itself is thin; most complexity lives in the shared modules (client, protocol,
metrics, results) that all three dispatch modes consume.
"""

from __future__ import annotations

from benchmarks.dispatchers.base import BaseDispatcher, ResponseRecord


class SequentialDispatcher(BaseDispatcher):
    """Iterate prompts serially -- one request at a time, no concurrency."""

    async def run(self) -> list[ResponseRecord]:
        """Send each prompt sequentially, returning all ResponseRecords.

        Failed requests (429, timeout, connection error) still produce a
        ResponseRecord with ``error`` populated and ``http_status`` set;
        they are not retried and not discarded.
        """
        results: list[ResponseRecord] = []
        for prompt in self.prompts:
            record = await self.client.send_request(prompt)
            results.append(record)
        return results
