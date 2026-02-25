FROM python:3.12-slim AS base

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy project metadata and dependency files
COPY pyproject.toml README.md uv.lock* ./

# Copy application code
COPY shared/ ./shared/
COPY control_plane/ ./control_plane/

# Install production dependencies
RUN uv sync --no-dev

EXPOSE 8090

CMD ["uv", "run", "python", "-m", "control_plane.admission_controller"]
