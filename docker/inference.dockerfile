# ── Stage 1: Builder ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install Python 3.12 via deadsnakes PPA and build deps (single apt-get update)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy ONLY dependency metadata first (layer cache for deps)
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies (cached unless pyproject.toml/uv.lock change)
ENV UV_HTTP_TIMEOUT=300
RUN uv sync --frozen --no-dev --extra engine

# ── Stage 2: Runtime ────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-base-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Remove CUDA compat libraries that conflict with WSL2 driver stubs
RUN rm -rf /usr/local/cuda/compat

# Install only the Python runtime (no dev/build packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common gpg-agent && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev curl gcc build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv (needed for `uv run`)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy installed venv and project metadata from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/pyproject.toml /app/uv.lock /app/README.md ./

# Copy application code (changes here do NOT invalidate dependency cache)
COPY server_config.yaml ./
COPY shared/ ./shared/
COPY data_plane/ ./data_plane/

# Create model storage directory (shared volume with sidecar)
RUN mkdir -p /mnt/models

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "data_plane.inference.engine.api:app", \
     "--host", "0.0.0.0", "--port", "8080"]
