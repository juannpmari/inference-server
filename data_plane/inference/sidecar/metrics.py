from prometheus_client import Counter, Gauge, Histogram

# Model loading
sidecar_model_load_duration_seconds = Histogram(
    "sidecar_model_load_duration_seconds",
    "Time to load a model artifact",
    ["model", "source"],
)

# Adapter loading
sidecar_adapter_load_duration_seconds = Histogram(
    "sidecar_adapter_load_duration_seconds",
    "Time to load a LoRA adapter",
    ["adapter"],
)

# Resident counts
sidecar_resident_models = Gauge(
    "sidecar_resident_models",
    "Number of models currently resident",
)

sidecar_resident_adapters = Gauge(
    "sidecar_resident_adapters",
    "Number of adapters currently resident",
)

# Download tracking
sidecar_download_bytes_total = Counter(
    "sidecar_download_bytes_total",
    "Total bytes downloaded",
    ["artifact_type"],
)
