"""
serve.py – One-command launch for the GeoAI Flood Risk Dashboard.
Run:  python serve.py
Then open:  http://localhost:8000
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "geoai_flood_final.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


def check_model():
    if not os.path.exists(MODEL_PATH):
        print("[WARNING] Trained model not found at models/geoai_flood_final.pth")
        print("  → Run:  python src/train_final.py")
        print("  → Then: python src/inference_final.py")
        return False
    return True


def check_outputs():
    missing = []
    for mm in [100, 150, 200]:
        p = os.path.join(OUTPUT_DIR, f"flood_prob_final_{mm}mm.tif")
        if not os.path.exists(p):
            missing.append(mm)
    if missing:
        print(f"[WARNING] Missing inference outputs for: {missing} mm scenarios.")
        print("  → Run:  python src/inference_final.py")
    return len(missing) == 0


def run():
    print("=" * 60)
    print("  GeoAI Flood Risk Dashboard – Ernakulam, Kerala")
    print("=" * 60)
    check_model()
    check_outputs()
    print("\nStarting FastAPI server on http://localhost:8000 ...")
    print("Press Ctrl+C to stop.\n")
    os.chdir(SRC_DIR)
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "backend:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


if __name__ == "__main__":
    run()
