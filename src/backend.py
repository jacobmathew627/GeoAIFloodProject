"""
GeoAI Flood Risk Dashboard – FastAPI Backend
Serves:
  GET /api/scenarios      → available rainfall scenarios
  POST /api/predict       → run inference for a given rainfall value
  GET /api/map/{mm}       → return flood probability raster as PNG tiles
  GET /api/risk_stats/{mm} → risk class stats for the given rainfall
  GET /                   → serve the Leaflet frontend
"""

import os
import io
import json
import sys
import numpy as np
import rasterio
from rasterio.enums import Resampling
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import base64

# ----- Path setup -----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "geoai_flood_final.pth")
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ----- FastAPI App -----
app = FastAPI(title="GeoAI Flood Risk API", version="1.0")

# ----- Colour ramp for probability -----
# Low (green) → Moderate (yellow) → High (orange) → Critical (red)
COLORMAP = [
    (0.00, (26, 152, 80)),
    (0.10, (145, 207, 96)),
    (0.20, (254, 224, 139)),
    (0.35, (253, 174, 97)),
    (0.55, (215, 48, 39)),
    (1.00, (165, 0, 38)),
]

PLACES = {
    "Ernakulam": [9.980, 76.280],
    "MG Road": [9.966, 76.287],
    "Edappally": [10.024, 76.308],
    "Kaloor": [9.994, 76.292],
    "Vyttila": [9.966, 76.318],
    "Aluva": [10.108, 76.357],
    "Kakkanad": [10.011, 76.340],
    "Perumbavoor": [10.109, 76.475],
    "Muvattupuzha": [9.982, 76.582],
    "North Paravur": [10.158, 76.214],
}


# ----- Helpers -----
def apply_colormap(prob: np.ndarray) -> np.ndarray:
    """Convert (H,W) float32 [0-1] flood probability → (H,W,4) RGBA uint8."""
    rgba = np.zeros((*prob.shape, 4), dtype=np.uint8)
    for i in range(len(COLORMAP) - 1):
        v0, c0 = COLORMAP[i]
        v1, c1 = COLORMAP[i + 1]
        mask = (prob >= v0) & (prob <= v1)
        if not mask.any():
            continue
        t = (prob[mask] - v0) / (v1 - v0 + 1e-6)
        for ch in range(3):
            rgba[mask, ch] = (c0[ch] * (1 - t) + c1[ch] * t).astype(np.uint8)
        rgba[mask, 3] = 200  # semi-transparent
    return rgba


def load_prob_tif(rainfall_mm: int) -> tuple[np.ndarray, rasterio.profiles.Profile]:
    p = os.path.join(OUTPUT_DIR, f"flood_prob_final_{rainfall_mm}mm.tif")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No pre-computed map for {rainfall_mm}mm. Run inference first.")
    with rasterio.open(p) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        bounds = src.bounds
        crs = str(src.crs)
    return data, profile, bounds, crs


def prob_to_png_b64(prob: np.ndarray, max_dim: int = 1024) -> str:
    """Downscale → RGBA → base64 PNG."""
    h, w = prob.shape
    scale = min(max_dim / max(h, w), 1.0)
    new_h, new_w = int(h * scale), int(w * scale)
    prob_small = np.array(Image.fromarray(prob).resize((new_w, new_h), Image.BILINEAR))
    rgba = apply_colormap(prob_small)
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ----- API Routes -----
@app.get("/api/scenarios")
def get_scenarios():
    available = []
    for mm in [100, 150, 200]:
        p = os.path.join(OUTPUT_DIR, f"flood_prob_final_{mm}mm.tif")
        available.append({"rainfall_mm": mm, "available": os.path.exists(p)})
    return JSONResponse(available)


class InferRequest(BaseModel):
    rainfall_mm: float = 150.0


@app.post("/api/predict")
def predict(req: InferRequest):
    try:
        from inference_final import run_inference
        prob, profile = run_inference(req.rainfall_mm)
        return {"status": "ok", "rainfall_mm": req.rainfall_mm,
                "max_prob": float(prob.max()), "mean_prob": float(prob.mean())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/map/{mm}")
def get_map(mm: int):
    """Return overlay image (base64 PNG) + bounds for Leaflet."""
    try:
        prob, profile, bounds, crs = load_prob_tif(mm)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    from pyproj import Transformer
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon_min, lat_min = tr.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = tr.transform(bounds.right, bounds.top)

    img_b64 = prob_to_png_b64(prob)
    return JSONResponse({
        "image_b64": img_b64,
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]]
    })


@app.get("/api/risk_stats/{mm}")
def risk_stats(mm: int):
    try:
        prob, _, _, _ = load_prob_tif(mm)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    valid = prob[prob >= 0]
    total = valid.size
    return {
        "rainfall_mm": mm,
        "safe_pct":     round(float((valid < 0.10).mean() * 100), 2),
        "moderate_pct": round(float(((valid >= 0.10) & (valid < 0.25)).mean() * 100), 2),
        "high_pct":     round(float(((valid >= 0.25) & (valid < 0.50)).mean() * 100), 2),
        "critical_pct": round(float((valid >= 0.50).mean() * 100), 2),
        "mean_prob":    round(float(valid.mean()), 4),
        "max_prob":     round(float(valid.max()), 4),
    }


@app.get("/api/places")
def get_places():
    return PLACES


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if not os.path.exists(html_path):
        return HTMLResponse("<h1>Dashboard not built yet. Run serve.py first.</h1>")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
