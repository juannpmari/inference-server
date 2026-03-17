from prometheus_client import Counter, Histogram, Gauge

# Request counters
engine_requests_total = Counter(
    "engine_requests_total",
    "Total number of inference requests",
    ["model", "status"]
)

# Request duration histogram
engine_request_duration_seconds = Histogram(
    "engine_request_duration_seconds",
    "Time to complete inference request",
    ["model"]
)

# Time to first token
engine_time_to_first_token_seconds = Histogram(
    "engine_time_to_first_token_seconds",
    "Time from request to first token",
    ["model"]
)

# Tokens generated counter
engine_tokens_generated_total = Counter(
    "engine_tokens_generated_total",
    "Total tokens generated",
    ["model"]
)

# Tokens per second gauge
engine_tokens_per_second = Gauge(
    "engine_tokens_per_second",
    "Average tokens per second",
    ["model"]
)

# Batch size histogram
engine_batch_size = Histogram(
    "engine_batch_size",
    "Batch size per inference step",
    ["model"],
    buckets=[1, 2, 4, 8, 16, 32]
)

# Pending requests gauge
engine_pending_requests = Gauge(
    "engine_pending_requests",
    "Number of pending requests",
    ["model"]
)

# GPU memory usage gauge
engine_gpu_memory_used_bytes = Gauge(
    "engine_gpu_memory_used_bytes",
    "GPU memory used in bytes",
    ["device"]
)

# LoRA metrics
engine_lora_load_duration_seconds = Histogram(
    "engine_lora_load_duration_seconds",
    "Time to load LoRA adapter",
    ["adapter"]
)

engine_lora_active = Gauge(
    "engine_lora_active",
    "Number of active LoRA adapters"
)

# Stream cancelled counter
engine_stream_cancelled_total = Counter(
    "engine_stream_cancelled_total",
    "Total number of cancelled streams",
    ["model"]
)

# --- New metrics (monitoring system) ---

# Latency decomposition histograms
engine_queue_wait_seconds = Histogram(
    "engine_queue_wait_seconds",
    "Queue wait time per request",
    ["model"]
)

engine_prefill_seconds = Histogram(
    "engine_prefill_seconds",
    "Prefill latency per request",
    ["model"]
)

engine_inter_token_latency_seconds = Histogram(
    "engine_inter_token_latency_seconds",
    "Inter-token latency per request",
    ["model"]
)

# Token count histograms
engine_input_tokens_per_request = Histogram(
    "engine_input_tokens_per_request",
    "Input tokens per request",
    ["model"],
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
)

engine_output_tokens_per_request = Histogram(
    "engine_output_tokens_per_request",
    "Output tokens per request",
    ["model"],
    buckets=[16, 32, 64, 128, 256, 512, 1024, 2048]
)

# Decode throughput histogram
engine_decode_tokens_per_second = Histogram(
    "engine_decode_tokens_per_second",
    "Decode tokens per second per request",
    ["model"]
)

# LoRA per-adapter request counter
engine_lora_requests_total = Counter(
    "engine_lora_requests_total",
    "Total requests per LoRA adapter",
    ["adapter"]
)

# GPU gauges (Phase 2)
engine_gpu_compute_utilization_percent = Gauge(
    "engine_gpu_compute_utilization_percent",
    "GPU compute utilization percentage",
    ["device"]
)

engine_gpu_memory_total_bytes = Gauge(
    "engine_gpu_memory_total_bytes",
    "GPU total memory in bytes",
    ["device"]
)

engine_gpu_power_watts = Gauge(
    "engine_gpu_power_watts",
    "GPU power draw in watts",
    ["device"]
)
