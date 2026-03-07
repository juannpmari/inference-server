"""Prometheus metrics for L1 KV cache operations."""

from prometheus_client import Counter, Gauge, Histogram

l1_cache_capacity_bytes = Gauge(
    "l1_cache_capacity_bytes",
    "Total L1 cache pool size in bytes",
)

l1_cache_used_bytes = Gauge(
    "l1_cache_used_bytes",
    "Currently allocated L1 bytes",
)

l1_cache_utilization_ratio = Gauge(
    "l1_cache_utilization_ratio",
    "L1 cache used / capacity ratio",
)

l1_cache_operations_total = Counter(
    "l1_cache_operations_total",
    "Total L1 cache operations",
    ["op", "status"],
)

l1_cache_operation_duration_seconds = Histogram(
    "l1_cache_operation_duration_seconds",
    "End-to-end L1 cache operation latency (includes transfer)",
    ["op"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

l1_cache_transfer_duration_seconds = Histogram(
    "l1_cache_transfer_duration_seconds",
    "Raw GPU transfer time",
    ["direction"],
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
)

l1_cache_transfer_bytes_total = Counter(
    "l1_cache_transfer_bytes_total",
    "Total bytes transferred",
    ["direction"],
)

l1_cache_evictions_total = Counter(
    "l1_cache_evictions_total",
    "Total L1 cache evictions",
    ["reason"],
)

l1_cache_blocks_stored = Gauge(
    "l1_cache_blocks_stored",
    "Current number of blocks stored in L1",
)
