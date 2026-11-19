# GateSentinel — Docker image
#
# Multi-stage build so that build tools (gcc, wheel build chains) do not
# bloat the final runtime image.
#
# Stage 1 (builder): install Python dependencies into a venv.
# Stage 2 (runtime): copy only the venv and application source.
#
# The OpenVINO runtime is installed as an optional dependency. If you are
# running on CUDA hardware instead, replace `openvino>=2024.0` with the
# appropriate torch+cuda wheel URL in the requirements step.
#
# Usage
# -----
#   docker build -t gatesentinel:latest .
#   docker run --rm \
#     -v $(pwd)/models:/app/models \
#     -v $(pwd)/data:/app/data \
#     -v $(pwd)/configs/local.yaml:/app/configs/local.yaml \
#     --device /dev/video0 \
#     gatesentinel:latest

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System packages needed only at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first so Docker layer caching works: the venv
# layer is only rebuilt when pyproject.toml changes, not every code edit.
COPY pyproject.toml requirements.lock.txt ./

# Create a venv and install all production dependencies (CPU-only Torch to
# keep image size reasonable; override in a local compose file for GPU).
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        numpy>=1.26 \
        opencv-contrib-python-headless>=4.10 \
        pydantic>=2.6,\<3.0 \
        pyyaml>=6.0 \
        sqlalchemy>=2.0,\<3.0 \
        "ultralytics>=8.3" \
        "paddleocr>=3.0" \
        "paddlepaddle>=3.0" \
        "openvino>=2024.0"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="GateSentinel ALPR" \
      org.opencontainers.image.description="Automatic license plate recognition for gate access control" \
      org.opencontainers.image.source="https://github.com/razerblade09/GateSentinel"

# Runtime-only system libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        # tesseract is optional but kept for the fallback OCR engine.
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the venv from builder — no compiler needed in the runtime stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source.
COPY src/      src/
COPY scripts/  scripts/
COPY configs/  configs/

# Data and model volumes — bind-mount at runtime, do not bake in.
VOLUME ["/app/data", "/app/models", "/app/logs"]

# Non-root user for the process — gate software should not run as root.
RUN useradd --system --no-create-home gatesentinel \
 && chown -R gatesentinel:gatesentinel /app
USER gatesentinel

# Health check: verify the config loads cleanly.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python scripts/show_config.py --profile deploy || exit 1

# Default command: run the pipeline with the deploy profile.
CMD ["python", "scripts/run.py", "--profile", "deploy"]
