"""KV Block Registry — unified metadata store for cached KV blocks.

Tracks where each block lives (L1 or L2) and exposes queries used by
cache-aware routing (Phase J) and the MultiTieredCacheManager.
"""

import json
import logging
import time
from typing import Dict, List, Optional

from shared.types import KVBlockEntry

logger = logging.getLogger(__name__)


class KVBlockRegistry:
    """In-memory registry of KV cache block locations with optional JSON persistence."""

    def __init__(self, persist_path: Optional[str] = None):
        self._blocks: Dict[str, KVBlockEntry] = {}
        self._persist_path = persist_path
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._eviction_window_start = time.time()
        self._l1_capacity_bytes: int = 0  # Set externally if known
        self._load_from_disk()

    def register(self, entry: KVBlockEntry) -> None:
        entry.created_at = entry.created_at or time.time()
        entry.last_accessed = entry.last_accessed or entry.created_at
        self._blocks[entry.key] = entry
        self._persist_to_disk()

    def unregister(self, key: str) -> Optional[KVBlockEntry]:
        entry = self._blocks.pop(key, None)
        if entry:
            self._evictions += 1
            self._persist_to_disk()
        return entry

    def lookup(self, key: str) -> Optional[KVBlockEntry]:
        entry = self._blocks.get(key)
        if entry:
            self._hits += 1
        else:
            self._misses += 1
        return entry

    def update_location(self, key: str, new_location: str, **kwargs) -> bool:
        entry = self._blocks.get(key)
        if not entry:
            return False
        entry.location = new_location
        for attr, val in kwargs.items():
            if hasattr(entry, attr):
                setattr(entry, attr, val)
        self._persist_to_disk()
        return True

    def record_access(self, key: str) -> None:
        entry = self._blocks.get(key)
        if entry:
            entry.last_accessed = time.time()
            entry.access_count += 1

    def query_by_prefix(self, prefix_hash: str, model_id: str = "") -> List[KVBlockEntry]:
        results = []
        for entry in self._blocks.values():
            if entry.prefix_hash == prefix_hash:
                if not model_id or entry.model_id == model_id:
                    results.append(entry)
        return results

    def all_entries(self) -> List[KVBlockEntry]:
        return list(self._blocks.values())

    def stats(self) -> dict:
        l1_blocks = [e for e in self._blocks.values() if e.location == "L1"]
        l2_blocks = [e for e in self._blocks.values() if e.location == "L2"]
        total_lookups = self._hits + self._misses
        l1_used = sum(e.size_bytes for e in l1_blocks)
        elapsed = time.time() - self._eviction_window_start
        eviction_rate = self._evictions / elapsed if elapsed > 0 else 0.0
        l1_util = l1_used / self._l1_capacity_bytes if self._l1_capacity_bytes > 0 else 0.0
        return {
            "total_blocks": len(self._blocks),
            "l1_blocks": len(l1_blocks),
            "l2_blocks": len(l2_blocks),
            "l1_used_bytes": l1_used,
            "l1_capacity_bytes": self._l1_capacity_bytes,
            "l1_utilization_ratio": round(l1_util, 4),
            "l2_used_bytes": sum(e.size_bytes for e in l2_blocks),
            "hit_rate": self._hits / total_lookups if total_lookups > 0 else 0.0,
            "eviction_count": self._evictions,
            "eviction_rate": round(eviction_rate, 4),
        }

    def _persist_to_disk(self) -> None:
        if not self._persist_path:
            return
        try:
            data = [e.to_dict() for e in self._blocks.values()]
            with open(self._persist_path, "w") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning(f"Failed to persist KV block registry: {exc}")

    def _load_from_disk(self) -> None:
        if not self._persist_path:
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for item in data:
                entry = KVBlockEntry.from_dict(item)
                self._blocks[entry.key] = entry
            logger.info(f"Restored {len(self._blocks)} KV block entries from disk")
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"Failed to load KV block registry: {exc}")
