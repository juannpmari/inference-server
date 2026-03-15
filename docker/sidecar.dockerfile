FROM python:3.12-slim AS base

WORKDIR /app

# Install uv via curl (faster than pip) and create model storage directory
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /mnt/models
ENV PATH="/root/.local/bin:$PATH"
ENV UV_HTTP_TIMEOUT=300

# Copy only dependency metadata first (cache-friendly layer)
COPY pyproject.toml uv.lock ./

# Create a minimal README.md so pyproject.toml metadata is valid
RUN touch README.md

# Install production dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --no-dev --extra sidecar

# Copy protobuf definitions and compile (cached unless proto files change)
COPY shared/proto/ ./shared/proto/
RUN uv run python -m grpc_tools.protoc -I shared/proto \
    --python_out=shared/proto --pyi_out=shared/proto \
    --grpc_python_out=shared/proto shared/proto/kv_cache.proto && \
    sed -i 's/^import kv_cache_pb2/from shared.proto import kv_cache_pb2/' shared/proto/kv_cache_pb2_grpc.py

# Copy application code and config (changes here don't invalidate layers above)
COPY shared/ ./shared/
COPY data_plane/ ./data_plane/
COPY server_config.yaml ./

EXPOSE 8001 50051

CMD ["uv", "run", "uvicorn", "data_plane.inference.sidecar.api:app", \
     "--host", "0.0.0.0", "--port", "8001"]
