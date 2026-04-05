FROM python:3.12-slim AS base

WORKDIR /app

# Install uv via curl-based installer
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.local/bin:$PATH"

# Copy only dependency files first (layer cache optimisation)
COPY pyproject.toml uv.lock ./

# Create a minimal README.md so pyproject.toml metadata is valid
RUN touch README.md

# Install production dependencies (cached unless pyproject.toml/uv.lock change)
RUN uv sync --no-dev

# Copy application code and config
COPY server_config.yaml ./
COPY shared/__init__.py ./shared/
COPY shared/config_loader.py ./shared/
COPY shared/types.py ./shared/
COPY control_plane/ ./control_plane/

EXPOSE 8090

CMD ["uv", "run", "python", "-m", "control_plane.admission_controller"]
