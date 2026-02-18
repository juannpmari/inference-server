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
