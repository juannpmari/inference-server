FROM python:3.12-slim AS base

WORKDIR /app

# Install uv via curl-based installer
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Copy only dependency metadata first (cache-friendly layer)
COPY pyproject.toml uv.lock ./

# Create a minimal README.md so pyproject.toml metadata is valid
RUN touch README.md

# Install production dependencies (gateway only — no CUDA/ML libs)
RUN uv sync --no-dev --extra gateway

# Copy application code and config
COPY server_config.yaml ./
COPY shared/ ./shared/
COPY data_plane/__init__.py ./data_plane/
COPY data_plane/gateway/ ./data_plane/gateway/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "data_plane.gateway.routing:app", \
     "--host", "0.0.0.0", "--port", "8000"]
