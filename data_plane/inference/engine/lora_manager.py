"""
LoRA adapter lifecycle manager for the inference engine.

Handles:
- Tracking which adapters are loaded on GPU (LRU order)
- Deduplicating concurrent downloads of the same adapter (leader-follower)
- Polling sidecar registry until adapter is ready (fire-and-forget pattern)
- Evicting LRU adapters when max_loras is reached
- Recording LoRA metrics
"""

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional

import httpx

from data_plane.inference.engine import metrics

logger = logging.getLogger(__name__)


class LoadedAdapter(NamedTuple):
    """Metadata for an adapter currently loaded on GPU."""
    lora_name: str
    lora_int_id: int
    lora_path: str
    adapter_identifier: str
    adapter_version: str


def _build_lora_request(name: str, int_id: int, path: str):
    """Build a vLLM LoRARequest or a simple namespace fallback for testing."""
    try:
        from vllm.lora.request import LoRARequest
        return LoRARequest(lora_name=name, lora_int_id=int_id, lora_path=path)
    except ImportError:
        from types import SimpleNamespace
        return SimpleNamespace(lora_name=name, lora_int_id=int_id, lora_path=path)


class LoRAManager:
    """
    Manages the full LoRA adapter lifecycle on the engine side.

    Thread safety: all state (_loaded, _pending_downloads) is accessed only from
    async coroutines on the event loop, protected by asyncio.Lock. The vLLM
    engine.step() runs in a separate thread but does not touch this state.
    engine.add_lora/remove_lora are called via asyncio.to_thread.
    """

    def __init__(self, engine, config, sidecar_url: str):
        self._engine = engine
        self._config = config
        self._sidecar_url = sidecar_url
        self._max_loras: int = config.max_loras
        self._poll_interval: float = config.adapter_poll_interval
        self._poll_timeout: float = config.adapter_poll_timeout

        # LRU tracking: OrderedDict, move_to_end on access, popitem(last=False) to evict
        self._loaded: OrderedDict[str, LoadedAdapter] = OrderedDict()

        # Dedup: one Event per in-flight adapter download
        self._pending_downloads: Dict[str, asyncio.Event] = {}

        self._lock = asyncio.Lock()

    @staticmethod
    def _adapter_key(identifier: str, version: Optional[str]) -> str:
        return f"{identifier}@{version or 'latest'}"

    @staticmethod
    def _adapter_int_id(identifier: str, version: Optional[str]) -> int:
        return hash(identifier + (version or "latest")) & 0x7FFFFFFF

    async def ensure_adapter_loaded(
        self,
        adapter_identifier: str,
        adapter_version: Optional[str] = None,
    ):
        """
        Ensure the requested adapter is loaded on GPU. Returns a LoRARequest
        suitable for passing to vLLM's engine.add_request().

        Safe to call concurrently — only one download + GPU load per adapter.
        """
        version = adapter_version or "latest"
        key = self._adapter_key(adapter_identifier, version)

        # Fast path: already loaded on GPU
        async with self._lock:
            if key in self._loaded:
                self._loaded.move_to_end(key)
                return self._make_lora_request(self._loaded[key])

            # Check if another coroutine is already fetching this adapter
            if key in self._pending_downloads:
                event = self._pending_downloads[key]
                is_leader = False
            else:
                event = asyncio.Event()
                self._pending_downloads[key] = event
                is_leader = True

        if is_leader:
            try:
                await self._trigger_and_poll(adapter_identifier, version, key)
            except Exception:
                # Clean up on failure so future requests can retry
                async with self._lock:
                    self._pending_downloads.pop(key, None)
                event.set()
                raise
            event.set()
        else:
            await event.wait()

        # Return the loaded adapter (or raise if it failed)
        async with self._lock:
            self._pending_downloads.pop(key, None)
            if key not in self._loaded:
                raise RuntimeError(
                    f"Adapter {adapter_identifier} v{version} failed to load"
                )
            self._loaded.move_to_end(key)
            return self._make_lora_request(self._loaded[key])

    async def _trigger_and_poll(self, identifier: str, version: str, key: str):
        """Trigger sidecar download, poll until ready, evict if needed, load to GPU."""

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Trigger download (fire-and-forget, returns 202 or 200 if cached)
            resp = await client.post(
                f"{self._sidecar_url}/adapter/load/{identifier}",
                params={"version": version},
            )
            resp.raise_for_status()

            # If already loaded, sidecar returns 200 with local_path
            if resp.status_code == 200:
                adapter_path = resp.json()["local_path"]
            else:
                # 2. Poll registry until status == "loaded"
                adapter_path = await self._poll_adapter_ready(client, identifier, version)

        # 3. Evict LRU if at capacity
        async with self._lock:
            while len(self._loaded) >= self._max_loras:
                evicted_key, evicted = self._loaded.popitem(last=False)
                logger.info(f"Evicting LRU adapter: {evicted_key}")
                await asyncio.to_thread(self._engine.remove_lora, evicted.lora_int_id)
                metrics.engine_lora_active.dec()

        # 4. Load onto GPU
        int_id = self._adapter_int_id(identifier, version)
        lora_name = f"{identifier}-{version}"
        lora_request = _build_lora_request(lora_name, int_id, adapter_path)

        start = time.time()
        await asyncio.to_thread(self._engine.add_lora, lora_request)
        duration = time.time() - start

        metrics.engine_lora_load_duration_seconds.labels(adapter=identifier).observe(duration)
        metrics.engine_lora_active.inc()

        loaded = LoadedAdapter(
            lora_name=lora_name,
            lora_int_id=int_id,
            lora_path=adapter_path,
            adapter_identifier=identifier,
            adapter_version=version,
        )
        async with self._lock:
            self._loaded[key] = loaded

        logger.info(
            f"Adapter {identifier} v{version} loaded to GPU in {duration:.2f}s "
            f"({len(self._loaded)}/{self._max_loras} slots used)"
        )

    async def _poll_adapter_ready(
        self, client: httpx.AsyncClient, identifier: str, version: str
    ) -> str:
        """Poll GET /registry/adapters until adapter status is 'loaded'. Returns local_path."""
        url = f"{self._sidecar_url}/registry/adapters"
        deadline = time.monotonic() + self._poll_timeout

        while True:
            await asyncio.sleep(self._poll_interval)

            try:
                resp = await client.get(url, timeout=5.0)
                resp.raise_for_status()
                registry = resp.json()
                entry = registry.get(identifier)

                if entry and entry.get("status") == "loaded":
                    return entry["local_path"]

                if entry and entry.get("status") == "failed":
                    error = entry.get("error", "unknown error")
                    raise RuntimeError(
                        f"Sidecar failed to fetch adapter {identifier}: {error}"
                    )

                if not entry:
                    # Entry disappeared — sidecar removed it on failure
                    raise RuntimeError(
                        f"Adapter {identifier} disappeared from sidecar registry (download likely failed)"
                    )

                logger.debug(
                    f"Adapter {identifier} status='{entry.get('status')}', polling..."
                )
            except httpx.RequestError as e:
                logger.warning(f"Sidecar poll failed ({e}), retrying...")

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Adapter {identifier} v{version} not ready within {self._poll_timeout}s"
                )

    def _make_lora_request(self, loaded: LoadedAdapter):
        return _build_lora_request(loaded.lora_name, loaded.lora_int_id, loaded.lora_path)

    @property
    def loaded_count(self) -> int:
        return len(self._loaded)

    @property
    def loaded_keys(self) -> list:
        return list(self._loaded.keys())
