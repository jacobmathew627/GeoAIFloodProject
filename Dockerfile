# ──────────────────────────────────────────────
# Multi-stage Dockerfile for the GeoAI Flood Risk Dashboard
#
# Stages:
#   builder   compiler toolchain + headers, installs into /opt/venv
#   runtime   slim base with the venv copied in, no toolchain
#   app       Streamlit dashboard        -> port 8501   (default target)
#   api       FastAPI scenario/stats API -> port 8000
#
# Two things dominate an image like this: the data it carries and the build
# toolchain it forgets to drop. Both are addressed here.
#
# Data. The previous version COPYd GeoAI_New/ (3.7 GB) and data_aligned/
# (1.3 GB) wholesale. Neither is read by the running application -- both are
# inputs to the offline training pipeline. The dashboard reads pre-downsampled
# display rasters (display/, 21 MB, from src/make_display_rasters.py) and the
# precomputed live_model.npz, so the shipped data payload is ~33 MB rather than
# ~5.9 GB. See .dockerignore for the full accounting.
#
# Toolchain. build-essential and the -dev headers are needed to *install*
# rasterio/pyproj/scipy, not to run them, and they were ~1.1 GB of the final
# image. They now live only in `builder`. The wheels for rasterio and pyproj
# bundle their own GDAL and PROJ, so the runtime stage needs neither
# libgdal-dev nor gdal-bin -- verified by importing rasterio and opening a real
# raster in the runtime stage below, which fails the build if the bundled
# libraries are ever insufficient.
# ──────────────────────────────────────────────

# ═══════════════════════════════════════════════
# Stage 1: builder -- toolchain, discarded afterwards
# ═══════════════════════════════════════════════
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal \
    PIP_NO_CACHE_DIR=1

# A venv rather than the system site-packages, so the runtime stage can take
# the whole tree with one COPY and nothing else.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ═══════════════════════════════════════════════
# Stage 2: runtime -- no compilers, no headers
# ═══════════════════════════════════════════════
FROM python:3.10-slim AS runtime

# curl for the HEALTHCHECKs; libexpat1 because rasterio's bundled GDAL links
# against libexpat.so.1, which python:3.10-slim does not ship. That is the
# entire system-library debt of dropping libgdal-dev/gdal-bin here: ~100 kB
# instead of ~1.1 GB. Found by the probe below failing on it, which is what
# the probe is for.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

COPY --from=builder /opt/venv /opt/venv

# Fail the build here, not at runtime, if the wheel-bundled GDAL/PROJ are not
# self-sufficient without the system packages dropped above. A heredoc rather
# than `python -c "..."`: the escaping needed to keep a multi-statement -c
# argument intact across Docker's line-continuation parsing is its own source
# of bugs, and a silently mangled probe is worse than no probe.
RUN python - <<'PROBE'
import os
import tempfile

import numpy as np
import rasterio
from rasterio.transform import from_origin

# Import everything the app needs at runtime, so a missing shared library
# surfaces here rather than on the first request.
import fastapi  # noqa: F401
import pyproj  # noqa: F401
import sklearn  # noqa: F401
import streamlit  # noqa: F401

path = os.path.join(tempfile.gettempdir(), "probe.tif")
profile = dict(
    driver="GTiff", height=4, width=4, count=1, dtype="float32",
    crs="EPSG:32643", transform=from_origin(0, 100, 10, 10),
)
with rasterio.open(path, "w", **profile) as dst:
    dst.write(np.ones((4, 4), dtype="float32"), 1)
with rasterio.open(path) as src:
    assert src.read(1).sum() == 16.0, "raster round-trip lost data"
    assert str(src.crs) == "EPSG:32643", f"CRS not preserved: {src.crs}"

# Exercise a real reprojection: pyproj resolves this from its bundled PROJ
# database, which is the piece most likely to be missing without libproj-dev.
transformer = pyproj.Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
lon, lat = transformer.transform(660000.0, 1105000.0)
assert 75.0 < lon < 78.0 and 9.0 < lat < 11.5, f"reprojection wrong: {lon}, {lat}"

print(f"runtime probe OK: rasterio {rasterio.__version__}, GDAL {rasterio.__gdal_version__}")
PROBE

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app
RUN chown -R appuser:appuser /app

# ═══════════════════════════════════════════════
# Stage 3: Streamlit dashboard (default target)
# ═══════════════════════════════════════════════
FROM runtime AS app

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
FROM runtime AS api

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
