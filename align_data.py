#!/usr/bin/env python3
"""
Data Alignment & Validation Script for GeoAI Flood Risk Project.

Resamples every conditioning factor onto a single master grid, normalises
nodata to one sentinel, and writes a district mask plus a presence/absence
label raster that downstream training can actually use.

Design rules (these exist because the previous version violated all three):
  1. The nodata mask is carried separately from the values. Post-processing
     (clip / log / normalise) is only ever applied to valid pixels, so a
     sentinel such as -99999 can never be silently clipped into the valid
     range.
  2. The district footprint is defined once, by the LULC master grid, and is
     re-applied to every layer. Layers whose source extends past the district
     (e.g. river distance, NDVI) are trimmed to it.
  3. The flood inventory is presence-only (SAR detects water, it does not
     detect "dry"). It is therefore written as an explicit 1 / 0 / nodata
     raster where 0 means "inside the district and not observed flooded",
     never "outside the study area".

Run:  python align_data.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
GEOAI_NEW = PROJECT_ROOT / "GeoAI_New"
OUTPUT_DIR = PROJECT_ROOT / "data_aligned"
OUTPUT_DIR.mkdir(exist_ok=True)

# LULC is the finest grid available and defines the district footprint.
MASTER_GRID = GEOAI_NEW / "Ernakulam_LULC_2018.tif"

NODATA = -9999.0

# name -> (source file, is_categorical, source nodata override)
# The override is needed where the GeoTIFF header lies about its own nodata
# or omits it entirely.
FEATURES = {
    "dem": ("Ernakulam_Clipped_DEM.tif", False, -9999.0),
    "slope": ("Ernakulam_Slope.tif", False, -9999.0),
    "river_dist": ("Ernakulam_River_Distance.tif", False, None),
    "lulc": ("Ernakulam_LULC_2018.tif", True, 0.0),
    "urban_dist": ("Distance_to_Builtup_Final.tif", False, -9999.0),
    "ndvi": ("NDVI_Aligned.tif", False, None),
    "ndwi": ("NDWI_Aligned.tif", False, None),
    "hand": ("Ernakulam_HAND.tif", False, -99999.0),
    "twi": ("Ernakulam_TWI.tif", False, -99999.0),
    "tpi": ("Ernakulam_TPI.tif", False, -99999.0),
    "spi": ("Ernakulam_SPI.tif", False, -99999.0),
    "flow": ("Ernakulam_Flow_Accumulation.tif", False, 0.0),
    "urban_mask": ("Urban_Mask.tif", True, -3.4028234663852886e38),
}

FLOOD_INVENTORY = "Ground_Truth_Final.tif"

# Layers whose native extent is larger than the district. Everything outside
# the LULC footprint is meaningless for this study area and gets masked.
CLIP_TO_DISTRICT = set(FEATURES) | {"ground_truth"}


# ──────────────────────────────────────────────
# Master grid
# ──────────────────────────────────────────────
def get_master_grid() -> Tuple[dict, Tuple[int, int], object, object]:
    """Return (profile, shape, crs, transform) of the master grid."""
    with rasterio.open(MASTER_GRID) as src:
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype=rasterio.float32,
            nodata=NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        return profile, src.shape, src.crs, src.transform


def build_district_mask(master_shape, master_crs, master_transform) -> np.ndarray:
    """District footprint: pixels where the LULC master grid has a real class."""
    with rasterio.open(MASTER_GRID) as src:
        lulc = src.read(1).astype(np.float32)
    if lulc.shape != master_shape:  # pragma: no cover - master is its own grid
        raise RuntimeError(f"LULC shape {lulc.shape} != master {master_shape}")
    mask = lulc > 0
    LOGGER.info(
        "District mask: %.2fM of %.2fM pixels (%.1f%%)",
        mask.sum() / 1e6,
        mask.size / 1e6,
        100 * mask.sum() / mask.size,
    )
    return mask


# ──────────────────────────────────────────────
# Reprojection
# ──────────────────────────────────────────────
def reproject_to_master(
    src_path: Path,
    master_shape,
    master_crs,
    master_transform,
    is_categorical: bool = False,
    src_nodata_override: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Reproject a raster onto the master grid.

    Returns (values, valid_mask). Values at invalid pixels are undefined and
    must never be read without consulting the mask.
    """
    if not src_path.exists():
        LOGGER.warning("Source not found: %s", src_path)
        return None

    with rasterio.open(src_path) as src:
        data = src.read(1).astype(np.float32)

        src_nodata = src_nodata_override if src_nodata_override is not None else src.nodata
        if src_nodata is not None:
            data[data == np.float32(src_nodata)] = np.nan
        # NaN is always nodata regardless of what the header claims.
        # (NDVI/NDWI carry no nodata tag and encode gaps as NaN.)

        dst = np.full(master_shape, np.nan, dtype=np.float32)
        reproject(
            source=data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master_transform,
            dst_crs=master_crs,
            resampling=Resampling.nearest if is_categorical else Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    valid = np.isfinite(dst)
    return dst, valid


# ──────────────────────────────────────────────
# Per-layer post-processing (valid pixels only)
# ──────────────────────────────────────────────
def post_process(name: str, values: np.ndarray, valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply layer-specific conditioning to valid pixels only.

    Returns (values, valid) — a rule may *shrink* the valid mask (e.g. a
    physically impossible value is dropped) but never grows it, and never
    writes a transformed value into an invalid pixel.
    """
    v = values.copy()
    m = valid.copy()

    def vals():
        return v[m]

    if name == "lulc":
        # ESA WorldCover classes 1..11. Anything else is not a land class.
        v[m] = np.round(v[m])
        m &= (v >= 1) & (v <= 11)

    elif name == "urban_mask":
        v[m] = (v[m] > 0.5).astype(np.float32)

    elif name == "hand":
        # Height Above Nearest Drainage is a height; small negatives are DEM
        # noise, large negatives are not physical.
        m &= v > -50
        v[m] = np.clip(v[m], 0.0, 100.0)

    elif name == "twi":
        # Topographic Wetness Index: ln(a / tan(beta)). Real range ~ -5..35.
        m &= (v > -20) & (v < 60)
        v[m] = np.clip(v[m], -5.0, 35.0)

    elif name == "tpi":
        m &= (v > -1000) & (v < 1000)
        v[m] = np.clip(v[m], -50.0, 50.0)

    elif name == "spi":
        # Stream Power Index spans ~12 orders of magnitude and carries huge
        # negatives from flat-cell division. Use a signed log and robust
        # percentile clipping computed on the valid pixels alone.
        if m.any():
            sl = np.sign(v[m]) * np.log1p(np.abs(v[m]))
            lo, hi = np.percentile(sl, [1, 99])
            v[m] = np.clip(sl, lo, hi)

    elif name == "flow":
        # Flow accumulation is heavily right-skewed; log-compress it but keep
        # physical units interpretable by not rescaling to 0-1 here (that is a
        # modelling decision, not an alignment one).
        if m.any():
            v[m] = np.log1p(np.maximum(v[m], 0.0))

    elif name in ("ndvi", "ndwi"):
        m &= (v >= -1.0) & (v <= 1.0)

    elif name in ("river_dist", "urban_dist"):
        m &= v >= 0
        if m.any():
            p999 = np.percentile(v[m], 99.9)
            v[m] = np.clip(v[m], 0.0, p999)

    elif name == "dem":
        # Ernakulam tops out around 800 m; below -30 m is not land here.
        m &= (v > -30) & (v < 1500)

    elif name == "slope":
        m &= (v >= 0) & (v <= 90)

    return v, m


# ──────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────
def write_layer(path: Path, values: np.ndarray, valid: np.ndarray, profile: dict) -> None:
    out = np.full(values.shape, NODATA, dtype=np.float32)
    out[valid] = values[valid].astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def align_all() -> dict:
    profile, master_shape, master_crs, master_transform = get_master_grid()
    LOGGER.info("Master grid: %s CRS=%s", master_shape, master_crs)

    district = build_district_mask(master_shape, master_crs, master_transform)
    write_layer(
        OUTPUT_DIR / "district_mask.tif",
        district.astype(np.float32),
        np.ones_like(district, dtype=bool),
        profile,
    )

    results = {}
    for name, (fname, is_cat, nd_override) in FEATURES.items():
        LOGGER.info("Aligning %-12s <- %s", name, fname)
        got = reproject_to_master(
            GEOAI_NEW / fname,
            master_shape,
            master_crs,
            master_transform,
            is_categorical=is_cat,
            src_nodata_override=nd_override,
        )
        if got is None:
            results[name] = False
            continue

        values, valid = got
        values, valid = post_process(name, values, valid)

        if name in CLIP_TO_DISTRICT:
            valid &= district

        out_path = OUTPUT_DIR / f"{name}_aligned.tif"
        write_layer(out_path, values, valid, profile)

        if valid.any():
            vv = values[valid]
            LOGGER.info(
                "  -> valid=%.2fM (%.1f%% of district) range=[%.3f, %.3f] mean=%.3f",
                valid.sum() / 1e6,
                100 * valid.sum() / max(district.sum(), 1),
                vv.min(),
                vv.max(),
                vv.mean(),
            )
        else:
            LOGGER.error("  -> NO VALID DATA for %s", name)
        results[name] = bool(valid.any())

    # ── Flood inventory: presence-only -> explicit presence / absence ──
    LOGGER.info("Aligning flood inventory <- %s", FLOOD_INVENTORY)
    got = reproject_to_master(
        GEOAI_NEW / FLOOD_INVENTORY,
        master_shape,
        master_crs,
        master_transform,
        is_categorical=True,
        src_nodata_override=-9999.0,
    )
    if got is not None:
        gt_values, gt_valid = got
        flood = gt_valid & (gt_values > 0.5)
        # Label domain is the district; inside it, "not detected as water" is
        # our best available absence evidence. Outside it we know nothing.
        labels = np.where(flood, 1.0, 0.0).astype(np.float32)
        write_layer(OUTPUT_DIR / "ground_truth_aligned.tif", labels, district, profile)
        LOGGER.info(
            "  -> flood pixels=%.3fM (%.2f%% of district), absence pixels=%.2fM",
            flood.sum() / 1e6,
            100 * flood.sum() / max(district.sum(), 1),
            (district & ~flood).sum() / 1e6,
        )
        results["ground_truth"] = bool(flood.any())
    else:
        results["ground_truth"] = False

    return results


# ──────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────
def verify() -> bool:
    LOGGER.info("=== VERIFICATION ===")
    shapes, crs_set = set(), set()
    ok = True

    for name in list(FEATURES) + ["ground_truth", "district_mask"]:
        path = OUTPUT_DIR / (f"{name}.tif" if name == "district_mask" else f"{name}_aligned.tif")
        if not path.exists():
            LOGGER.error("MISSING: %s", path)
            ok = False
            continue

        with rasterio.open(path) as src:
            shapes.add(src.shape)
            crs_set.add(str(src.crs))
            data = src.read(1)
            valid = data[data != NODATA]
            valid = valid[np.isfinite(valid)]

        if valid.size == 0:
            LOGGER.error("NO VALID DATA: %s", name)
            ok = False
            continue

        LOGGER.info(
            "%-14s valid=%6.2fM range=[%10.3f, %10.3f] mean=%9.3f std=%8.3f",
            name,
            valid.size / 1e6,
            valid.min(),
            valid.max(),
            valid.mean(),
            valid.std(),
        )

    LOGGER.info("Unique shapes: %s", shapes)
    LOGGER.info("Unique CRS: %s", crs_set)

    if len(shapes) != 1 or len(crs_set) != 1:
        LOGGER.error("ALIGNMENT FAILED - shape/CRS mismatch")
        ok = False
    else:
        LOGGER.info("All rasters share one grid.")
    return ok


if __name__ == "__main__":
    LOGGER.info("=" * 60)
    LOGGER.info("GeoAI Flood Risk - Data Alignment & Validation")
    LOGGER.info("=" * 60)

    align_all()
    if verify():
        LOGGER.info("DATA PREPARATION COMPLETE -> %s", OUTPUT_DIR)
    else:
        LOGGER.error("ALIGNMENT FAILED - see errors above")
        sys.exit(1)
