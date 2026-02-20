FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for grpcio and vllm compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy project metadata and dependency files
COPY pyproject.toml README.md uv.lock* ./

# Copy application code
COPY shared/ ./shared/
COPY data_plane/ ./data_plane/

# Install production dependencies
RUN uv sync --no-dev

# Create model storage directory (shared volume with sidecar)
RUN mkdir -p /mnt/models

EXPOSE 8080

CMD ["uv", "run", "uvicorn", "data_plane.inference.engine.api:app", \
     "--host", "0.0.0.0", "--port", "8080"]
