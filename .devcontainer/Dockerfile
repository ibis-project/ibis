FROM mcr.microsoft.com/vscode/devcontainers/python:3.11
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ARG USERNAME=vscode

RUN apt-get update && \
  apt-get install -y --no-install-recommends libgdal-dev && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY . /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --group dev --group tests \
        --extra duckdb --extra clickhouse  --extra examples --extra geospatial

ENV VENV_DIR=.venv
RUN chown -R $USERNAME $VENV_DIR && chmod -R 755 $VENV_DIR

ENV HYPOTHESIS_EXAMPLE_DB_DIR=.hypothesis
RUN chown -R $USERNAME $HYPOTHESIS_EXAMPLE_DB_DIR && chmod -R 755 $HYPOTHESIS_EXAMPLE_DB_DIR

ENV PYTEST_CACHE=.pytest_cache
RUN chown -R $USERNAME $PYTEST_CACHE && chmod -R 755 $PYTEST_CACHE

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

ENTRYPOINT []

USER $USERNAME