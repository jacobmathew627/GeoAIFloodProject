# ──────────────────────────────────────────────
# Multi-stage Dockerfile for the GeoAI Flood Risk Dashboard
# ──────────────────────────────────────────────

# ═══════════════════════════════════════════════
# Stage 1: base with system libraries
# ═══════════════════════════════════════════════
FROM python:3.10-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
RUN chown -R appuser:appuser /app

# ═══════════════════════════════════════════════
# Stage 2: Python dependencies
# ═══════════════════════════════════════════════
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════
# Stage 3: application
# ═══════════════════════════════════════════════
FROM deps AS app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py serve.py align_data.py ./
COPY --chown=appuser:appuser static/ ./static/

# Model artefacts and rasters. `processed/` was copied by the previous
# version but is gitignored and absent from the build context, so the build
# failed at that line. `data_aligned/` is what the model actually reads, and
# is produced by `python align_data.py`.
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser outputs/ ./outputs/
COPY --chown=appuser:appuser GeoAI_New/ ./GeoAI_New/
COPY --chown=appuser:appuser data_aligned/ ./data_aligned/

RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app/data /app/logs

USER appuser
EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ═══════════════════════════════════════════════
# Stage 4: FastAPI variant
# ═══════════════════════════════════════════════
FROM deps AS fastapi

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser serve.py ./
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser models/ ./models/
COPY --chown=appuser:appuser outputs/ ./outputs/
COPY --chown=appuser:appuser data_aligned/ ./data_aligned/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["python", "serve.py", "--mode", "fastapi"]
