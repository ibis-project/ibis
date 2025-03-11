FROM ubuntu:22.04

# Set some default environment variables plus PYTHON_VERSION, REPO, TAG and WORKSPACE
ENV PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=UTF-8 \
    TZ=Etc/UTC \
    PYTHON_VERSION=3.12 \
    REPO=ibis-project/ibis \
    TAG=10.2.0 \
    WORKSPACE=/ibis

# Prevent interactive prompts during package installation
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y build-essential \
    libcurl4-openssl-dev \
    libffi-dev \
    tk-dev \
    xz-utils \
    curl \
    lsb-release \
    git \
    wget \
    libncursesw5-dev \
    zlib1g-dev \
    libpq-dev \
    libmysqlclient-dev \
    pkg-config \
    libssl-dev \
    unixodbc-dev \
    locales \
    locales-all \
    tzdata && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen

# Install uv (Uvicorn) for the ibis environment
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/bin" sh

# Create a non-root user for better security
RUN useradd -ms /bin/bash runner

# Create workspace directory and set permissions
RUN mkdir -p ${WORKSPACE} && \
    chown -R runner:runner ${WORKSPACE} && \
    chmod -R u+rwX,go+rX,go-w ${WORKSPACE}

# Switch to the non-root user and set the working directory
USER runner
WORKDIR $WORKSPACE

# Install Python version
RUN uv python install -v ${PYTHON_VERSION}

# Checkout the project at the given tag
RUN git clone "https://github.com/${REPO}.git" "${WORKSPACE}" && \
    cd "${WORKSPACE}" && \
    git checkout "${TAG}"


RUN uv venv
# Install other dependencies in the virtual environment
RUN uv pip install -r requirements-dev.txt

RUN uv run python -m pytest -v ibis/tests || true

# CMD to run tests
CMD ["uv", "run", "–offline", "python", "-m", "pytest", "-vv", "-rA", "--tb=short", "tests"]