FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for grpcio compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for cache efficiency
COPY pyproject.toml uv.lock* ./

# Install production dependencies only
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY shared/ ./shared/
COPY data_plane/ ./data_plane/

# Install the project itself
RUN uv sync --no-dev

# Create model storage directory
RUN mkdir -p /mnt/models

EXPOSE 8001 50051

CMD ["uv", "run", "uvicorn", "data_plane.inference.sidecar.api:app", \
     "--host", "0.0.0.0", "--port", "8001"]
