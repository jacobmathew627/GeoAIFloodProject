"""
Data Loading Module for GeoAI Flood Risk Project.

Handles raster I/O, downsampling for display, and nodata normalisation.

Note on nodata: every reader here returns a `nodata` value of
RASTER.nodata_value and guarantees that invalid pixels hold exactly that
value. Callers test `data > -9000` rather than comparing floats for equality,
because bilinear downsampling can perturb the sentinel slightly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from config import GEOAI_NEW_DIR, OUTPUT_DIR, RAINFALL, RASTER

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

# Display-name -> (filename, layer kind). The layer kind drives both the
# resampling method and the nodata rules, and is a real layer identity rather
# than the string "continuous"/"categorical" the previous version passed in
# (which meant none of the per-layer nodata rules ever matched).
LAYER_REGISTRY: Dict[str, Tuple[str, str]] = {
    "DEM": ("Ernakulam_Clipped_DEM.tif", "dem"),
    "Slope": ("Ernakulam_Slope.tif", "slope"),
    "LULC": ("Ernakulam_LULC_2018.tif", "lulc"),
    "TWI": ("Ernakulam_TWI.tif", "twi"),
    "SPI": ("Ernakulam_SPI.tif", "spi"),
    "HAND": ("Ernakulam_HAND.tif", "hand"),
    "TPI": ("Ernakulam_TPI.tif", "tpi"),
    "Distance to Water": ("Ernakulam_River_Distance.tif", "distance_water"),
    "Distance to Built-up": ("Distance_to_Builtup_Final.tif", "distance_urban"),
    "NDVI (Vegetation)": ("NDVI_Aligned.tif", "ndvi"),
    "NDWI (Water)": ("NDWI_Aligned.tif", "ndwi"),
    "Sentinel-1 Ground Truth": ("Ground_Truth_Final.tif", "ground_truth"),
    "Flow Accumulation": ("Ernakulam_Flow_Accumulation.tif", "flow"),
    "Urban Mask": ("Urban_Mask.tif", "urban_mask"),
}

CATEGORICAL_KINDS = {"lulc", "urban_mask", "ground_truth", "conformal"}

# Sentinels used by the source rasters that GDAL does not report.
_EXTRA_SENTINELS = (-99999.0, -9999.0, -3.4028234663852886e38)


# ──────────────────────────────────────────────
# Core reader
# ──────────────────────────────────────────────
def read_downsampled(
    path: Union[str, Path],
    layer_kind: str = "continuous",
    max_dim: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Read a raster, downsampling for display.

    Args:
        path: Raster path.
        layer_kind: Layer identity (see LAYER_REGISTRY), used for nodata rules
            and resampling choice.
        max_dim: Longest output dimension; defaults to RASTER.max_dimension.

    Returns:
        (data, metadata) or (None, None) on failure. `metadata` is a dict with
        keys bounds, crs, transform, nodata, original_shape.
    """
    max_dim = max_dim or RASTER.max_dimension
    path = Path(path)
    if not path.exists():
        LOGGER.warning("File not found: %s", path)
        return None, None

    categorical = layer_kind in CATEGORICAL_KINDS

    try:
        with rasterio.open(path) as src:
            scale = max_dim / max(src.width, src.height)
            downsampling = scale < 1.0

            if downsampling:
                new_h = max(1, int(src.height * scale))
                new_w = max(1, int(src.width * scale))
                out_shape = (new_h, new_w)

                # Always take a nearest-neighbour read. Every output pixel then
                # holds a genuine source value, which is what identifies nodata
                # reliably. Categorical layers stop here: any averaging would
                # invent class codes that do not exist (a "class 3.5").
                nearest = src.read(1, out_shape=out_shape, resampling=Resampling.nearest).astype(
                    np.float32
                )
                # Continuous layers use AVERAGE, not bilinear. At this grid's
                # downsampling ratio (7374 -> 1000) bilinear interpolates from
                # a handful of source pixels and behaves close to sampling, so
                # it over-represents the tail of a skewed field: the hazard
                # map's area above the critical threshold came out 3x the
                # full-resolution value. Averaging preserves the integral of a
                # probability surface, so expected-area statistics computed on
                # the display array agree with the full-resolution raster.
                smooth = (
                    None
                    if categorical
                    else src.read(1, out_shape=out_shape, resampling=Resampling.average).astype(
                        np.float32
                    )
                )
                transform = src.transform * src.transform.scale(
                    src.width / new_w, src.height / new_h
                )
            else:
                nearest = src.read(1).astype(np.float32)
                smooth = None
                transform = src.transform

            file_nodata = src.nodata
            bounds = src.bounds
            crs = str(src.crs)
            original_shape = (src.height, src.width)

        invalid = _invalid_mask(nearest, file_nodata)

        if smooth is not None:
            # Bilinear resampling averages across the nodata boundary, dragging
            # a -9999 or -99999 sentinel into neighbouring output pixels. That
            # is how HAND -- a height, so necessarily non-negative -- came out
            # of the display path with a minimum of -36.7 m. Take the smooth
            # values only where the nearest read says the pixel is real, and
            # clamp them to the range the real values actually span so a
            # partially-blended edge pixel cannot escape it.
            valid_near = ~invalid
            if valid_near.any():
                lo = float(nearest[valid_near].min())
                hi = float(nearest[valid_near].max())
                data = np.where(valid_near, np.clip(smooth, lo, hi), NODATA).astype(np.float32)
            else:
                data = np.full(nearest.shape, NODATA, dtype=np.float32)
        else:
            data = nearest
            data[invalid] = NODATA

        data = _apply_layer_nodata_rules(data, layer_kind)

        metadata = {
            "bounds": bounds,
            "crs": crs,
            "transform": transform,
            "nodata": NODATA,
            "original_shape": original_shape,
            "original_nodata": file_nodata,
        }
        return data, metadata

    except Exception as exc:  # pragma: no cover - I/O failure path
        LOGGER.error("Error reading %s: %s", path, exc)
        return None, None


def _invalid_mask(data: np.ndarray, file_nodata: Optional[float]) -> np.ndarray:
    """
    Identify nodata pixels.

    Several source rasters carry a sentinel that GDAL does not report in the
    header (-99999 for the terrain indices, -3.4e38 for the urban mask), so
    the declared nodata alone is not enough.
    """
    invalid = ~np.isfinite(data)
    if file_nodata is not None:
        invalid |= np.isclose(data, file_nodata, rtol=1e-6, atol=1e-3)
    for sentinel in _EXTRA_SENTINELS:
        invalid |= np.isclose(data, sentinel, rtol=1e-6, atol=1e-3)
    return invalid


def _apply_layer_nodata_rules(data: np.ndarray, layer_kind: str) -> np.ndarray:
    """
    Mask values that are physically impossible for the given layer.

    Applied only to pixels that are currently valid, so a rule can never
    resurrect a nodata pixel.
    """
    valid = data > -9000

    def drop(condition: np.ndarray) -> None:
        data[valid & condition] = NODATA

    if layer_kind == "lulc":
        drop((data < 1) | (data > 11))
    elif layer_kind == "hand":
        # HAND is a height above drainage: negative values are DEM noise.
        drop((data < -5) | (data > 500))
    elif layer_kind == "spi":
        # Raw SPI spans ~12 orders of magnitude from flat-cell division; only
        # the physically absurd tail is dropped, and the display colormap
        # normalises on percentiles.
        drop(np.abs(data) > 1e12)
    elif layer_kind == "flow":
        drop(data < 0)
    elif layer_kind == "ground_truth":
        drop((data < 0) | (data > 1))
    elif layer_kind == "dem":
        drop((data < -30) | (data > 1500))
    elif layer_kind == "slope":
        drop((data < 0) | (data > 90))
    elif layer_kind in ("ndvi", "ndwi"):
        drop((data < -1) | (data > 1))
    elif layer_kind == "twi":
        drop((data < -20) | (data > 60))
    elif layer_kind == "tpi":
        drop((data < -1000) | (data > 1000))
    elif layer_kind in ("distance_water", "distance_urban"):
        drop(data < 0)
        remaining = data > -9000
        if remaining.any():
            p999 = np.percentile(data[remaining], 99.9)
            data[remaining & (data > p999 * 1.5)] = NODATA

    return data


# ──────────────────────────────────────────────
# Hazard maps
# ──────────────────────────────────────────────
def load_hazard_maps(
    output_dir: Optional[Path] = None,
    max_dim: Optional[int] = None,
) -> Tuple[Dict[float, np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load the rainfall-conditioned hazard rasters written by hazard.py.

    Returns ({rainfall_mm: array}, metadata). All arrays share one shape; a
    scenario whose shape disagrees with the first one loaded is skipped with a
    warning rather than silently cropped, because cropping misaligns the
    geographic bounds that the display uses.
    """
    output_dir = output_dir or OUTPUT_DIR
    maps: Dict[float, np.ndarray] = {}
    meta: Optional[Dict[str, Any]] = None
    reference_shape: Optional[Tuple[int, int]] = None

    for mm in RAINFALL.scenarios:
        path = output_dir / f"flood_hazard_{int(mm)}mm.tif"
        if not path.exists():
            continue

        data, m = read_downsampled(path, layer_kind="hazard", max_dim=max_dim)
        if data is None:
            continue

        if reference_shape is None:
            reference_shape, meta = data.shape, m
        elif data.shape != reference_shape:
            LOGGER.warning("Skipping %s: shape %s != %s", path.name, data.shape, reference_shape)
            continue

        maps[float(mm)] = data

    if not maps:
        LOGGER.warning("No hazard rasters found in %s", output_dir)

    return maps, meta


def load_susceptibility(
    output_dir: Optional[Path] = None,
    max_dim: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load the rainfall-independent susceptibility surface."""
    output_dir = output_dir or OUTPUT_DIR
    return read_downsampled(output_dir / "susceptibility.tif", layer_kind="hazard", max_dim=max_dim)


def load_conformal_sets(
    output_dir: Optional[Path] = None,
    max_dim: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """
    Load the conformal decision raster.

    Categorical: the values are prediction-set codes, so it must be resampled
    with nearest neighbour or the display would invent codes between classes.
    """
    output_dir = output_dir or OUTPUT_DIR
    return read_downsampled(
        output_dir / "conformal_sets.tif", layer_kind="conformal", max_dim=max_dim
    )


# ──────────────────────────────────────────────
# Static layers
# ──────────────────────────────────────────────
def get_layer_path(layer_name: str, geoai_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve a display layer name to a file path, or None if unavailable."""
    geoai_dir = geoai_dir or GEOAI_NEW_DIR
    entry = LAYER_REGISTRY.get(layer_name)
    if entry is None:
        return None
    path = geoai_dir / entry[0]
    return path if path.exists() else None


def load_static_layer(
    layer_name: str,
    geoai_dir: Optional[Path] = None,
    max_dim: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Load a static conditioning-factor layer by its display name."""
    entry = LAYER_REGISTRY.get(layer_name)
    if entry is None:
        LOGGER.warning("Unknown layer: %s", layer_name)
        return None, None

    path = get_layer_path(layer_name, geoai_dir)
    if path is None:
        LOGGER.warning("Layer file missing for %s", layer_name)
        return None, None

    return read_downsampled(path, layer_kind=entry[1], max_dim=max_dim)


def read_windowed(
    path: Union[str, Path],
    window: Window,
    layer_kind: str = "continuous",
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """Read one window of a raster, for large-file processing."""
    try:
        with rasterio.open(path) as src:
            data = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
            nd = src.nodata

        invalid = ~np.isfinite(data)
        if nd is not None:
            invalid |= np.isclose(data, nd, rtol=0, atol=1e-3)
        data[invalid] = NODATA
        return _apply_layer_nodata_rules(data, layer_kind), transform
    except Exception as exc:  # pragma: no cover - I/O failure path
        LOGGER.error("Error reading window from %s: %s", path, exc)
        return None, None
