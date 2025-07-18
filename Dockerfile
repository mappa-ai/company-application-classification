FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app/
# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    curl \
    pkg-config \
    cmake \ 
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && . ~/.cargo/env \
    && rustc --version
ENV PATH="/root/.cargo/bin:$PATH"
# Install uv
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#installing-uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Place executables in the environment at the front of the path
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#using-the-environment
ENV PATH="/app/.venv/bin:$PATH"

# Compile bytecode
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#compiling-bytecode
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

ENV PYTHONPATH=/app

COPY ./pyproject.toml ./uv.lock /app/

COPY ./src /app/src

# Sync the project
# Ref: https://docs.astral.sh/uv/guides/integration/docker/#intermediate-layers
# Sync the project - verificar que Rust esté disponible
RUN rustc --version && cargo --version
RUN uv sync

CMD ["fastapi", "run", "--workers", "4", "src/main.py","--host", "::", "--port", "8080"]
