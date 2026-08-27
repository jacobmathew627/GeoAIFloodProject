"""
Legacy PyTorch U-Net inference.

STATUS: superseded. The supported prediction path is
`src/susceptibility.py` (terrain susceptibility) combined with
`src/hazard.py` (SCS-CN rainfall response). This module is kept so the
archived .pth checkpoints remain loadable and reproducible.

Important caveat: these checkpoints were trained against the *pre-fix*
aligned rasters, in which HAND, TWI, TPI, SPI, NDVI and NDWI all had their
nodata sentinels clipped into the valid range (see align_data.py). Their
learned normalisation therefore does not match the corrected feature stack,
and their output should be treated as a historical artefact, not a
prediction. `geoai_flood_final.pth` is not loadable here at all: it is a
64-base-channel architecture with `inc.double_conv.*` keys, whereas this
UNet is 32-base-channel with `inc.*` keys.
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
import torch.nn as nn

from config import (
    ALIGNED_DIR,
    MODEL,
    MODEL_FILES,
    MODELS_DIR,
    OUTPUT_DIR,
    RASTER,
    setup_logging,
)

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value


# ──────────────────────────────────────────────
# Architecture (matches the archived checkpoints)
# ──────────────────────────────────────────────
class UNet(nn.Module):
    """U-Net matching the archived checkpoints: 32 base channels, bias=True."""

    def __init__(self, n_channels: int = 6, n_classes: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        def double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.inc = double_conv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), double_conv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), double_conv(64, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv2 = double_conv(64, 32)
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        x = self.up1(x3)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)

        return self.sigmoid(self.outc(x))


# ──────────────────────────────────────────────
# Model registry
# ──────────────────────────────────────────────
# Channel orders verified against the checkpoints' inc.0.weight shapes:
# real2018 = 4ch, robust_sar = 6ch, supercharged = 9ch.
_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "pytorch_standard": {
        "model_file": MODEL_FILES["pytorch_standard"],
        "features": ["dem", "slope", "flow", "lulc"],
        "suffix": "_legacy_standard",
    },
    "pytorch_robust": {
        "model_file": MODEL_FILES["pytorch_robust"],
        "features": ["dem", "slope", "flow", "lulc", "sar_vv", "sar_vh"],
        "suffix": "_legacy_robust",
    },
    "pytorch_supercharged": {
        "model_file": MODEL_FILES["pytorch_supercharged"],
        "features": [
            "dem",
            "slope",
            "flow",
            "lulc",
            "sar_vv",
            "sar_vh",
            "twi",
            "river_dist",
            "urban_dist",
        ],
        "suffix": "_legacy_supercharged",
    },
}

AVAILABLE_MODELS: List[str] = list(_MODEL_CONFIGS)


def get_model_config(model_type: str) -> Dict[str, Any]:
    if model_type not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown model_type {model_type!r}. Available: {AVAILABLE_MODELS}")
    return _MODEL_CONFIGS[model_type]


# ──────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────
def normalize_channel(arr: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Scale to [0, 1] using precomputed percentile statistics."""
    mn, mx = stats["min"], stats["max"]
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr - mn) / (mx - mn), 0.0, 1.0).astype(np.float32)


def compute_normalization_stats(
    features: List[str], aligned_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    """Robust (1st/99th percentile) statistics per channel."""
    aligned_dir = aligned_dir or ALIGNED_DIR
    stats: Dict[str, Dict[str, float]] = {}

    for name in features:
        path = aligned_dir / f"{name}_aligned.tif"
        if not path.exists():
            LOGGER.warning("Feature file not found: %s; using identity stats", path)
            stats[name] = {"min": 0.0, "max": 1.0}
            continue

        try:
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                nd = src.nodata if src.nodata is not None else NODATA
            arr = arr[np.isfinite(arr) & (arr != np.float32(nd))]

            if arr.size:
                stats[name] = {
                    "min": float(np.percentile(arr, 1)),
                    "max": float(np.percentile(arr, 99)),
                }
            else:
                stats[name] = {"min": 0.0, "max": 1.0}
        except Exception as exc:  # pragma: no cover - I/O failure path
            LOGGER.error("Error computing stats for %s: %s", name, exc)
            stats[name] = {"min": 0.0, "max": 1.0}

    return stats


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
def _resolve_device() -> torch.device:
    if MODEL.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(MODEL.device)


def _build_stack(
    features: List[str], stats: Dict[str, Dict[str, float]], aligned_dir: Path
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Assemble the normalised input stack and the valid-data mask.

    Nodata pixels are set to 0 *after* normalisation. The previous version
    normalised only the valid pixels and left -9999 in place everywhere else,
    so the network was fed a sentinel four orders of magnitude outside its
    training range at every pixel outside the district.
    """
    ref_path = aligned_dir / "lulc_aligned.tif"
    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference grid not found: {ref_path}. Run `python align_data.py` first."
        )

    with rasterio.open(ref_path) as ref:
        height, width = ref.shape
        profile = ref.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=NODATA, compress="lzw")

    stack = np.zeros((len(features), height, width), dtype=np.float32)
    valid_all = np.ones((height, width), dtype=bool)

    for idx, name in enumerate(features):
        path = aligned_dir / f"{name}_aligned.tif"
        if not path.exists():
            LOGGER.warning("Feature %s missing; channel left at zero", name)
            continue

        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            nd = src.nodata if src.nodata is not None else NODATA

        if data.shape != (height, width):
            raise ValueError(
                f"Feature {name} has shape {data.shape}, expected {(height, width)}. "
                "Re-run align_data.py."
            )

        valid = np.isfinite(data) & (data != np.float32(nd))
        normalised = np.zeros_like(data, dtype=np.float32)
        if valid.any():
            normalised[valid] = normalize_channel(data[valid], stats[name])

        stack[idx] = normalised
        valid_all &= valid

    return stack, valid_all, profile


def _tiled_inference(model: nn.Module, stack: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Tiled inference with overlap averaging.

    Overlapping predictions are averaged, not maximised. Taking the maximum
    biases every seam upward, which showed up as a visible grid of hot lines
    across the old output rasters.
    """
    n_channels, height, width = stack.shape
    tile = MODEL.tile_size
    stride = max(1, tile - MODEL.tile_overlap)

    total = np.zeros((height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for i in range(0, height, stride):
            for j in range(0, width, stride):
                h_end = min(i + tile, height)
                w_end = min(j + tile, width)
                curr_h, curr_w = h_end - i, w_end - j

                patch = stack[:, i:h_end, j:w_end]
                if curr_h < tile or curr_w < tile:
                    padded = np.zeros((n_channels, tile, tile), dtype=np.float32)
                    padded[:, :curr_h, :curr_w] = patch
                    patch = padded

                tensor = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0).to(device)
                # Index rather than squeeze(): squeeze() would also collapse a
                # spatial axis of length 1 on edge tiles.
                prob = model(tensor)[0, 0].cpu().numpy()

                total[i:h_end, j:w_end] += prob[:curr_h, :curr_w]
                count[i:h_end, j:w_end] += 1.0

    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)


def run_inference(
    model_type: str = "pytorch_supercharged",
    aligned_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    write: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Score the archived U-Net over the aligned grid.

    Returns the raw network probability. No rainfall factor is applied: the
    old multiplier table ({100mm: 0.8, 150mm: 1.6, 200mm: 2.6, ...}) was
    applied directly to a probability, which both saturated at 1.0 above
    ~150 mm and produced non-monotonic maps. Rainfall conditioning now lives
    in hazard.py where it is derived from SCS-CN runoff.
    """
    aligned_dir = aligned_dir or ALIGNED_DIR
    output_dir = output_dir or OUTPUT_DIR

    config = get_model_config(model_type)
    model_path = MODELS_DIR / config["model_file"]
    features = config["features"]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = _resolve_device()
    LOGGER.info(
        "Legacy inference: model=%s channels=%d device=%s",
        model_type,
        len(features),
        device,
    )

    model = UNet(n_channels=len(features), n_classes=1).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    stats = compute_normalization_stats(features, aligned_dir)
    stack, valid, profile = _build_stack(features, stats, aligned_dir)

    probability = _tiled_inference(model, stack, device)

    # Restore the nodata mask: the network happily predicts over padding and
    # over pixels outside the district, and the old pipeline wrote those
    # predictions into the raster.
    output = np.where(valid, probability, NODATA).astype(np.float32)

    if write:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"flood_prob{config['suffix']}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(output, 1)
        LOGGER.info("Wrote %s", path)

    inside = probability[valid]
    if inside.size:
        LOGGER.info(
            "Valid pixels %.2fM | min=%.4f max=%.4f mean=%.4f",
            inside.size / 1e6,
            inside.min(),
            inside.max(),
            inside.mean(),
        )

    del stack
    gc.collect()
    return output, profile


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Legacy U-Net inference")
    parser.add_argument("--model", choices=AVAILABLE_MODELS, default="pytorch_supercharged")
    parser.add_argument("--all", action="store_true", help="Run every archived model")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    LOGGER.warning(
        "This is the legacy path. The supported pipeline is "
        "`python src/susceptibility.py --train --predict` then `python src/hazard.py`."
    )

    for model_type in AVAILABLE_MODELS if args.all else [args.model]:
        try:
            output, _ = run_inference(model_type)
            print(f"{model_type}: shape={output.shape}")
        except Exception as exc:
            print(f"{model_type}: FAILED — {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
