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

# Install dependencies into the system interpreter rather than a venv. The
# workspace bind-mount replaces /app at runtime, which would hide a venv living
# under /app; the system site-packages lives outside /app and survives, and
# /usr/local/bin/python is already first on PATH in every (login) shell.
#
# ibis is installed editable (-e) against /app, which at runtime is the live
# bind-mounted workspace, so source edits are picked up without reinstalling.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -e '.[duckdb,examples,geospatial]' --group dev --group tests

ENTRYPOINT []

USER $USERNAME
