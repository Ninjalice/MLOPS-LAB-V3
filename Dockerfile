
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml uv.lock README.md ./
COPY api ./api
COPY logic ./logic
COPY templates ./templates

# Use uv to sync dependencies and install the project into a virtualenv
RUN uv sync --frozen --no-dev


FROM base AS runtime

COPY api ./api
COPY logic ./logic
COPY templates ./templates

# Copy the .venv created by uv in the builder stage
COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

# Use uvicorn from the venv to run the FastAPI app
CMD ["/app/.venv/bin/uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
