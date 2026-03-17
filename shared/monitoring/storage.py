"""Storage backends for persisting monitoring records."""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import List

from shared.monitoring.models import RequestRecord

logger = logging.getLogger(__name__)


class MetricsStorageBackend(ABC):
    @abstractmethod
    async def store_records(self, records: List[RequestRecord]) -> None:
        ...

    @abstractmethod
    async def query_records(self, model_id: str | None = None, limit: int = 1000) -> List[dict]:
        ...


class LocalJSONLStore(MetricsStorageBackend):
    """Append-only JSONL file storage."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    async def store_records(self, records: List[RequestRecord]) -> None:
        if not records:
            return
        lines = [json.dumps(asdict(r)) + "\n" for r in records]
        await asyncio.to_thread(self._append, lines)

    def _append(self, lines: List[str]) -> None:
        with open(self._path, "a") as f:
            f.writelines(lines)

    async def query_records(self, model_id: str | None = None, limit: int = 1000) -> List[dict]:
        if not self._path.exists():
            return []
        lines = await asyncio.to_thread(self._read_lines)
        records = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if model_id and rec.get("model_id") != model_id:
                continue
            records.append(rec)
            if len(records) >= limit:
                break
        return records

    def _read_lines(self) -> List[str]:
        with open(self._path) as f:
            return f.readlines()


class BackgroundFlusher:
    """Periodically flushes new records from a SessionCollector to a storage backend."""

    def __init__(self, collector, backend: MetricsStorageBackend, interval: float = 30.0):
        self._collector = collector
        self._backend = backend
        self._interval = interval
        self._last_flush_ts = 0.0
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            # Final flush
            await self._flush()

    async def _flush_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._interval)
                await self._flush()
        except asyncio.CancelledError:
            raise

    async def _flush(self) -> None:
        records = self._collector.get_records_since(self._last_flush_ts)
        if records:
            try:
                await self._backend.store_records(records)
                self._last_flush_ts = max(r.timestamp for r in records) + 0.001
                logger.debug(f"Flushed {len(records)} records to storage")
            except Exception as e:
                logger.error(f"Failed to flush metrics: {e}")
