
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


FROM base AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only the production requirements file
COPY requirements-docker.txt .

# Install only runtime dependencies (no PyTorch, no training libs)
RUN uv venv && \
    uv pip install --no-cache -r requirements-docker.txt


FROM base AS runtime

# Copy application code
COPY api ./api
COPY logic ./logic
COPY templates ./templates

# Copy ONNX model and class labels (required for inference)
COPY model.onnx ./
COPY class_labels.json ./

# Copy the .venv created by uv in the builder stage
COPY --from=builder /app/.venv /app/.venv

EXPOSE 8000

# Use uvicorn from the venv to run the FastAPI app
CMD ["/app/.venv/bin/uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
