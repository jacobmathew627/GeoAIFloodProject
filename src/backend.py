"""
GeoAI Flood Risk Dashboard - FastAPI backend.

Routes
------
GET  /api/health              service and data readiness
GET  /api/scenarios           available rainfall scenarios
GET  /api/model               susceptibility model card and CV metrics
GET  /api/map/{mm}            hazard overlay (base64 PNG) + WGS84 bounds
GET  /api/risk_stats/{mm}     risk-class breakdown with real areas
GET  /api/runoff              SCS-CN runoff response for a curve number
GET  /api/places              known place lookup
GET  /                        static dashboard
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rasterio
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import (  # noqa: E402
    API,
    GEO,
    HYDRO,
    KNOWN_PLACES,
    MODELS_DIR,
    OUTPUT_DIR,
    RAINFALL,
    RASTER,
    RISK,
    STATIC_DIR,
    setup_logging,
)
from visualization import compute_risk_stats, prob_to_png_b64  # noqa: E402

LOGGER = setup_logging(logging.INFO)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GeoAI Flood Risk API", version="3.0")

# The dashboard is served from a different origin during development, and the
# CORS policy was configured in APIConfig but never actually applied.
app.add_middleware(
    CORSMiddleware,
    allow_origins=API.cors_origins,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Raster access
# ──────────────────────────────────────────────
#: Upper bound on a requested storm depth, matching the /api/runoff query
#: bound. Until load_hazard gained a live-evaluation fallback these routes were
#: implicitly bounded by which rasters happened to exist on disk, so an absurd
#: depth returned 404. Now that any depth can be evaluated, the bound has to be
#: stated: 2000 mm is already far beyond the 443 mm reference event and well
#: past anything in the Indian instrumental record.
MAX_RAINFALL_MM = 2000


def hazard_path(rainfall_mm: int) -> Path:
    return OUTPUT_DIR / f"flood_hazard_{int(rainfall_mm)}mm.tif"


@lru_cache(maxsize=1)
def _live_grid():
    """
    The precomputed live model, or None if it is not deployed.

    Cached at size 1 because it is ~7 MB of arrays and entirely
    rainfall-independent -- the whole point of live_model is that a new
    rainfall value costs array arithmetic, not a reload.
    """
    try:
        import live_model

        return live_model.load()
    except FileNotFoundError:
        return None


@lru_cache(maxsize=8)
def load_hazard(rainfall_mm: int) -> tuple:
    """
    Hazard surface for a storm depth.

    Two sources, in order:

    1. A pre-generated full-resolution raster in outputs/, if one exists for
       exactly this depth. That is the 42M-pixel product of src/hazard.py and
       is what a local checkout with the full pipeline will serve.
    2. Otherwise, evaluated live from models/live_model.npz on the display
       grid, exactly as the dashboard does.

    The fallback is what makes the API image deployable. Shipping the
    pre-generated rasters meant carrying ~530 MB for nine fixed depths, while
    the dashboard was already computing the same quantity on demand from a
    7 MB cache. It also removes a real limitation rather than just saving
    space: /api/map/{mm} used to 404 for any depth that had not been
    pre-generated, so the API answered nine questions and the dashboard
    answered all of them. Both now answer all of them.

    The two paths differ in resolution, not in formulation -- live_model's
    fluvial_probability() and hazard.py's combine() are the same routed call
    (see live_model.fluvial_probability). Statistics computed from the live
    path are therefore over ~0.39M display cells rather than ~42M full-
    resolution ones; the percentages agree closely but are not bit-identical,
    and `resolution` in the response says which path produced them.
    """
    path = hazard_path(rainfall_mm)
    if path.exists():
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            bounds = src.bounds
            crs = str(src.crs)
            transform = src.transform
            nd = src.nodata if src.nodata is not None else RASTER.nodata_value

        data[~np.isfinite(data)] = RASTER.nodata_value
        data[data == np.float32(nd)] = RASTER.nodata_value
        return data, bounds, crs, transform, "full"

    grid = _live_grid()
    if grid is None:
        raise FileNotFoundError(
            f"No hazard map for {rainfall_mm} mm and no live model to evaluate one. "
            "Run `python src/susceptibility.py --train --predict` then "
            "`python src/live_model.py --build`."
        )

    import live_model

    data = live_model.fluvial_probability(grid, float(rainfall_mm))
    # The live model marks the outside of the domain with NaN; the raster path
    # and everything downstream (mask_nodata, compute_risk_stats) expect the
    # -9999 sentinel.
    data = np.where(np.isfinite(data), data, RASTER.nodata_value).astype(np.float32)
    return data, grid.bounds, str(grid.crs), grid.transform, "display"


def available_scenarios() -> list:
    """
    Depths the API can answer for.

    With a live model deployed that is every configured scenario, because each
    one is evaluated on demand. Without it, only those with a pre-generated
    raster on disk.
    """
    if _live_grid() is not None:
        return [mm for mm in RAINFALL.scenarios]
    return [mm for mm in RAINFALL.scenarios if hazard_path(int(mm)).exists()]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/api/health")
def health() -> Dict[str, Any]:
    scenarios = available_scenarios()
    model_file = MODELS_DIR / "susceptibility_model.joblib"
    return {
        "status": "ok" if scenarios else "degraded",
        "hazard_scenarios_available": len(scenarios),
        "live_model_present": _live_grid() is not None,
        "susceptibility_model_present": model_file.exists(),
        "susceptibility_surface_present": (OUTPUT_DIR / "susceptibility.tif").exists(),
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/api/scenarios")
def get_scenarios() -> JSONResponse:
    available = set(available_scenarios())
    return JSONResponse(
        [{"rainfall_mm": mm, "available": mm in available} for mm in RAINFALL.scenarios]
    )


@app.get("/api/model")
def model_card() -> Dict[str, Any]:
    """Model provenance and held-out performance."""
    import json

    metrics_path = MODELS_DIR / "susceptibility_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run `python src/susceptibility.py --train`.",
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "susceptibility": metrics,
        "rainfall_response": {
            "method": "SCS Curve Number",
            "initial_abstraction_ratio": HYDRO.initial_abstraction_ratio,
            "antecedent_moisture_condition": HYDRO.amc,
            "reference_event_mm": RAINFALL.reference_event_mm,
        },
        "domain_note": (
            "Permanent water bodies are excluded from the model domain; they "
            "accounted for 80.3% of the raw Sentinel-1 flood inventory."
        ),
    }


@app.get("/api/map/{mm}")
def get_map(mm: int = PathParam(..., ge=0, le=MAX_RAINFALL_MM)) -> JSONResponse:
    """Hazard overlay as a base64 PNG plus WGS84 bounds for Leaflet."""
    try:
        data, bounds, crs, _, resolution = load_hazard(mm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    from pyproj import Transformer

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = transformer.transform(bounds.right, bounds.top)

    return JSONResponse(
        {
            "rainfall_mm": mm,
            "image_b64": prob_to_png_b64(data),
            "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
            "resolution": resolution,
        }
    )


@app.get("/api/risk_stats/{mm}")
def risk_stats(mm: int = PathParam(..., ge=0, le=MAX_RAINFALL_MM)) -> Dict[str, Any]:
    try:
        data, _, _, transform, resolution = load_hazard(mm)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    stats = compute_risk_stats(data, RISK, transform)
    stats["rainfall_mm"] = mm
    # Areas are derived from the transform, so they are correct either way, but
    # a caller comparing two responses should know which grid produced them.
    stats["resolution"] = resolution
    return stats


@app.get("/api/runoff")
def runoff(
    rainfall_mm: float = Query(150.0, ge=0.0, le=2000.0),
    curve_number: Optional[float] = Query(None, ge=30.0, le=100.0),
) -> Dict[str, Any]:
    """SCS-CN runoff depth for a given storm and curve number."""
    from hydrology import adjust_cn_for_amc, potential_retention, runoff_depth

    cn_ii = curve_number if curve_number is not None else HYDRO.default_curve_number
    cn = adjust_cn_for_amc(np.array([[cn_ii]], dtype=np.float32), HYDRO.amc)
    q = float(runoff_depth(rainfall_mm, cn)[0, 0])

    return {
        "rainfall_mm": rainfall_mm,
        "curve_number_amc_ii": cn_ii,
        "curve_number_adjusted": float(cn[0, 0]),
        "antecedent_moisture_condition": HYDRO.amc,
        "potential_retention_mm": float(potential_retention(cn)[0, 0]),
        "runoff_depth_mm": q,
        "runoff_coefficient": q / rainfall_mm if rainfall_mm > 0 else 0.0,
    }


@app.get("/api/conformal")
def conformal_summary() -> Dict[str, Any]:
    """
    Distribution-free coverage guarantee for the susceptibility map.

    Returns the calibrated prediction-set thresholds plus the achieved
    coverage, both marginally and per probability stratum. The per-stratum
    numbers matter more than the headline: marginal coverage can meet its
    target while the high-risk band fails.
    """
    import json

    metrics_path = MODELS_DIR / "susceptibility_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Model metrics not found. Run `python src/susceptibility.py --train`.",
        )

    summary = json.loads(metrics_path.read_text(encoding="utf-8")).get("conformal")
    if not summary:
        raise HTTPException(
            status_code=404, detail="This model was trained without conformal calibration."
        )

    from conformal import SET_LABELS

    return {
        **summary,
        "set_codes": {str(k): v for k, v in SET_LABELS.items()},
        "raster": "outputs/conformal_sets.tif",
        "note": (
            "Coverage is guaranteed marginally over district pixels exchangeable "
            "with the calibration sample. Check conditional_coverage before acting "
            "on the high-probability band."
        ),
    }


@app.get("/api/places")
def get_places() -> Dict[str, Any]:
    return {
        "places": KNOWN_PLACES,
        "map_center": list(GEO.map_center),
        "zoom_start": GEO.zoom_start,
    }


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse(
            "<h1>GeoAI Flood Risk API</h1>"
            "<p>No static dashboard bundled. See <a href='/docs'>/docs</a>.</p>",
            status_code=200,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host=API.host, port=API.port, reload=API.reload)
