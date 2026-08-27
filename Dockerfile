# ──────────────────────────────────────────────
# Multi-stage Dockerfile for the GeoAI Flood Risk Dashboard
#
# Stages:
#   base   system libraries (GDAL/PROJ/GEOS) + non-root user
#   deps   Python dependencies
#   app    Streamlit dashboard          -> port 8501   (default target)
#   api    FastAPI scenario/stats API   -> port 8000
#
# On image size: an earlier version of this file COPYd GeoAI_New/ (3.7 GB) and
# data_aligned/ (1.3 GB) wholesale, producing an ~8 GB image that most hosts
# will not accept. Neither directory is read by the running application --
# both are inputs to the offline training pipeline. The dashboard reads
# pre-downsampled display rasters (display/, 21 MB, built by
# src/make_display_rasters.py) and the precomputed live_model.npz. See
# .dockerignore for the full accounting of what is excluded and why.
# ──────────────────────────────────────────────

# ═══════════════════════════════════════════════
# Stage 1: base
# ═══════════════════════════════════════════════
FROM python:3.10-slim AS base

# gdal-bin/libgdal-dev are required to build and run rasterio; libproj/libgeos
# back pyproj and shapely. curl is used by the HEALTHCHECKs below.
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
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
RUN chown -R appuser:appuser /app

# ═══════════════════════════════════════════════
# Stage 2: Python dependencies
# ═══════════════════════════════════════════════
FROM base AS deps

# Copied alone, before any source, so a code change does not invalidate the
# dependency layer.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ═══════════════════════════════════════════════
# Stage 3: Streamlit dashboard (default target)
# ═══════════════════════════════════════════════
FROM deps AS app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app.py ./
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

# Display-resolution static layers (21 MB) in place of GeoAI_New/ (3.7 GB).
# data_loading.get_layer_path() prefers this directory when it exists.
COPY --chown=appuser:appuser display/ ./display/

# Runtime model artefacts only:
#   live_model.npz                  precomputed susceptibility + routing basis
#   rainfall_forecast.joblib        the 3-day rainfall model
#   rainfall_forecast_latest.json   cached prediction, so the container never
#                                   reaches for the 874 MB IMD archive
#   *.json                          metric/threshold cards read by the UI
# susceptibility_model.joblib is deliberately absent: the app consumes the
# already-predicted surface inside live_model.npz, not the estimator.
COPY --chown=appuser:appuser models/live_model.npz ./models/
COPY --chown=appuser:appuser models/rainfall_forecast.joblib ./models/
COPY --chown=appuser:appuser models/*.json ./models/

# The conformal layer is the only outputs/ raster the dashboard opens.
COPY --chown=appuser:appuser outputs/conformal_sets.tif ./outputs/

RUN mkdir -p /app/data /app/logs && chown -R appuser:appuser /app/data /app/logs

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# python -m streamlit, not bare `streamlit`: the module form is resolved by the
# interpreter on PATH rather than by a console script, which keeps this working
# regardless of how the image's scripts directory is laid out.
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

# ═══════════════════════════════════════════════
# Stage 4: FastAPI backend
# ═══════════════════════════════════════════════
FROM deps AS api

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser serve.py ./
COPY --chown=appuser:appuser static/ ./static/
COPY --chown=appuser:appuser models/*.json ./models/

# Unlike the dashboard, the API serves the pre-generated per-scenario hazard
# rasters (~530 MB) -- see backend.load_hazard -- plus the conformal raster.
COPY --chown=appuser:appuser outputs/flood_hazard_*.tif ./outputs/
COPY --chown=appuser:appuser outputs/conformal_sets.tif ./outputs/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# uvicorn directly rather than through serve.py: serve.py shells out to a child
# process, which would leave uvicorn unable to receive SIGTERM as PID 1 and make
# container stops fall back to SIGKILL after the grace period.
WORKDIR /app/src
CMD ["python", "-m", "uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
