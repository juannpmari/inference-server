FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy project metadata and dependency files
COPY pyproject.toml README.md uv.lock* ./

# Copy application code
COPY shared/ ./shared/
COPY data_plane/ ./data_plane/

# Install production dependencies
RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "data_plane.gateway.routing:app", \
     "--host", "0.0.0.0", "--port", "8000"]
