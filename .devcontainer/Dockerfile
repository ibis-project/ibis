FROM mcr.microsoft.com/vscode/devcontainers/python:3.13
COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /uvx /bin/
ARG USERNAME=vscode

RUN apt-get update && \
  apt-get install -y --no-install-recommends libgdal-dev python3-dev default-jdk && \
  rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install pipx --no-cache-dir
RUN python3 -m pipx ensurepath
RUN pipx install rust-just

RUN echo 'export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))' >> /etc/profile.d/java.sh && \
  echo 'export PATH=$JAVA_HOME/bin:$PATH' >> /etc/profile.d/java.sh

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY --chown=$USERNAME . /app

# Install the project's dependencies using the lockfile and settings
# Venv lives outside /app so it's not hidden when the workspace is bind-mounted at runtime
ENV VIRTUAL_ENV=/opt/venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv $VIRTUAL_ENV && just sync duckdb

RUN chown -R $USERNAME $VIRTUAL_ENV && chmod -R 755 $VIRTUAL_ENV
# Place executables in the environment at the front of the path
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

ENTRYPOINT []

USER $USERNAME
