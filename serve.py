#!/usr/bin/env python3
"""
serve.py - launcher for the GeoAI Flood Risk Dashboard.

    python serve.py                  # Streamlit on :8501
    python serve.py --mode fastapi   # FastAPI on :8000
    python serve.py --mode both      # both
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALIGNED_DIR = PROJECT_ROOT / "data_aligned"

SCENARIOS = (50, 100, 150, 200, 250, 300, 400)


def check_readiness() -> bool:
    """Report what is present and what needs generating. Never fatal."""
    ready = True

    if not ALIGNED_DIR.exists() or not any(ALIGNED_DIR.glob("*_aligned.tif")):
        print("[MISSING] Aligned rasters.       -> python align_data.py")
        ready = False

    derived = ["upstream_cn_aligned.tif", "dem_rel_1km_aligned.tif"]
    if not all((ALIGNED_DIR / f).exists() for f in derived):
        print("[MISSING] Context features.      -> python src/derive_features.py")
        ready = False

    if not (MODELS_DIR / "susceptibility_model.joblib").exists():
        print("[MISSING] Susceptibility model.  -> python src/susceptibility.py --train")
        ready = False

    if not (OUTPUT_DIR / "susceptibility.tif").exists():
        print("[MISSING] Susceptibility raster. -> python src/susceptibility.py --predict")
        ready = False

    if not (OUTPUT_DIR / "conformal_sets.tif").exists():
        print("[MISSING] Conformal raster.      -> python src/susceptibility.py --conformal")
        ready = False

    missing = [mm for mm in SCENARIOS if not (OUTPUT_DIR / f"flood_hazard_{mm}mm.tif").exists()]
    if missing:
        print(f"[MISSING] Hazard maps for {missing} mm. -> python src/hazard.py")
        ready = False

    if ready:
        print("[OK] All model artefacts present.")
    return ready


def _banner(title: str, url: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    check_readiness()
    print(f"\nStarting on {url} ...  (Ctrl+C to stop)\n")


def streamlit_command() -> list:
    return [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port=8501", "--server.address=0.0.0.0",
    ]


def fastapi_command(reload: bool = False) -> list:
    cmd = [
        sys.executable, "-m", "uvicorn", "backend:app",
        "--host", "0.0.0.0", "--port", "8000",
    ]
    if reload:
        cmd.append("--reload")
    return cmd


def run_streamlit() -> int:
    _banner("GeoAI Flood Risk Dashboard - Ernakulam, Kerala", "http://localhost:8501")
    # cwd is passed per-process rather than via os.chdir, which would mutate
    # global interpreter state and race with the other server in --mode both.
    return subprocess.run(streamlit_command(), cwd=PROJECT_ROOT).returncode


def run_fastapi(reload: bool = False) -> int:
    _banner("GeoAI Flood Risk API - Ernakulam, Kerala", "http://localhost:8000")
    return subprocess.run(fastapi_command(reload), cwd=SRC_DIR).returncode


def run_both() -> int:
    """Start the API in the background, then run Streamlit in the foreground."""
    api = subprocess.Popen(fastapi_command(reload=False), cwd=SRC_DIR)
    try:
        return subprocess.run(streamlit_command(), cwd=PROJECT_ROOT).returncode
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover
            api.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="GeoAI Flood Risk Dashboard launcher")
    parser.add_argument(
        "--mode", choices=["streamlit", "fastapi", "both"], default="streamlit"
    )
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload (dev only)")
    parser.add_argument("--check", action="store_true", help="Report readiness and exit")
    args = parser.parse_args()

    if args.check:
        return 0 if check_readiness() else 1
    if args.mode == "streamlit":
        return run_streamlit()
    if args.mode == "fastapi":
        return run_fastapi(args.reload)
    return run_both()


if __name__ == "__main__":
    raise SystemExit(main())
