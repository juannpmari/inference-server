from typing import Dict, Optional, List
from collections import OrderedDict

class EvictionPolicy:
    """Base interface for eviction strategies."""
    def record_access(self, key: str):
        """Mark a key as recently used."""
        raise NotImplementedError
    
    def track_new(self, key: str, size: int):
        """Track a new item in the cache."""
        raise NotImplementedError

    def select_victim(self) -> Optional[str]:
        """Selects a key to evict. Returns None if empty."""
        raise NotImplementedError

    def remove(self, key: str):
        """Removes a key from tracking (e.g. after manual deletion)."""
        raise NotImplementedError

class LRUPolicy(EvictionPolicy):
    """
    Least Recently Used (LRU) implementation using an OrderedDict.
    """
    def __init__(self):
        # Maps key -> size_bytes. Order dictates recency.
        self._cache_map: OrderedDict[str, int] = OrderedDict()

    def record_access(self, key: str):
        if key in self._cache_map:
            self._cache_map.move_to_end(key)

    def track_new(self, key: str, size: int):
        self._cache_map[key] = size
        self._cache_map.move_to_end(key)

    def select_victim(self) -> Optional[str]:
        if not self._cache_map:
            return None
        # Return the first item (least recently used)
        return next(iter(self._cache_map))

    def remove(self, key: str):
        if key in self._cache_map:
            del self._cache_map[key]