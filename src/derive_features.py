"""
Derive the context features that a pixel-independent model cannot see.

Two kinds of context are added, both motivated by the same finding: grid
models score each cell in isolation, and flooding is not an isolated-cell
phenomenon. A GraphSAGE model over a watershed connectivity graph reached
AUC 0.978 against 0.881 for the best pixel ensemble on the same kind of
Sentinel-1 inventory, the gap attributed to upstream-downstream propagation
(arXiv:2603.15681).

  1. DRAINAGE-NETWORK CONTEXT -- `upstream_cn`
     The catchment-average curve number of everything that drains into a
     cell. This answers "what kind of land sheds water onto me", which is a
     property of the flow network, not of Euclidean distance: a cell 200 m
     from dense Kochi rooftops but on the far side of a divide receives none
     of their runoff, and no neighbourhood filter can express that.

  2. MULTI-SCALE TOPOGRAPHIC CONTEXT -- `dem_rel_1km`
     Elevation minus the mean elevation within ~1 km. TPI is already in the
     stack but is computed at a much shorter range, so it captures
     micro-relief. This captures "am I in a regional basin", which is the
     scale at which the coastal plain actually drowns.

Both are written into data_aligned/ as ordinary aligned rasters, so
feature_stack reads them exactly like any other conditioning factor.

Run:  python src/derive_features.py
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import ALIGNED_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

# Radius of the regional-relief window, in master-grid cells. The master grid
# is 10 m, so 101 cells is roughly 1 km across.
FOCAL_WINDOW_CELLS = 101


def _write(path: Path, values: np.ndarray, valid: np.ndarray, profile: dict) -> None:
    out = np.full(values.shape, NODATA, dtype=np.float32)
    out[valid] = values[valid].astype(np.float32)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)
    v = values[valid]
    LOGGER.info(
        "  wrote %s  valid=%.2fM range=[%.3f, %.3f] mean=%.3f",
        path.name, valid.sum() / 1e6, v.min(), v.max(), v.mean(),
    )


# ──────────────────────────────────────────────
# 1. Drainage-network context
# ──────────────────────────────────────────────
def build_upstream_cn(aligned_dir: Optional[Path] = None) -> Path:
    """Catchment-average curve number, routed on the 30 m filled DEM."""
    from feature_stack import compute_curve_number, grid_profile
    from routing import FlowNetwork, to_master_grid

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)

    LOGGER.info("Building flow network...")
    net = FlowNetwork()

    LOGGER.info("Projecting curve number onto the routing grid...")
    cn_master, cn_valid = compute_curve_number(aligned_dir=aligned_dir)

    cn_route = np.full(net.elev.shape, np.nan, dtype=np.float32)
    reproject(
        source=np.where(cn_valid, cn_master, np.nan).astype(np.float32),
        destination=cn_route,
        src_transform=master["transform"],
        src_crs=master["crs"],
        dst_transform=net.profile["transform"],
        dst_crs=net.profile["crs"],
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    del cn_master, cn_valid

    LOGGER.info("Accumulating curve number down the drainage network...")
    upstream = net.upstream_mean(cn_route)

    LOGGER.info("Resampling back to the master grid...")
    out = to_master_grid(upstream, net.profile, master)

    # Constrain to the model domain so the feature never invents data outside
    # the district.
    from feature_stack import domain_mask

    valid = domain_mask(aligned_dir=aligned_dir) & np.isfinite(out)
    path = aligned_dir / "upstream_cn_aligned.tif"
    _write(path, out, valid, master)
    return path


# ──────────────────────────────────────────────
# 2. Multi-scale topographic context
# ──────────────────────────────────────────────
def build_regional_relief(aligned_dir: Optional[Path] = None) -> Path:
    """Elevation relative to the ~1 km neighbourhood mean."""
    from scipy.ndimage import uniform_filter

    from feature_stack import grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)

    dem, valid = read_raster("dem", aligned_dir=aligned_dir)

    # A plain uniform_filter would average NaNs into every window that touches
    # the district edge. Filter the filled values and the mask separately and
    # divide, which is a correct nodata-aware mean.
    filled = np.where(valid, dem, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)

    LOGGER.info("Computing %d-cell focal mean...", FOCAL_WINDOW_CELLS)
    total = uniform_filter(filled, size=FOCAL_WINDOW_CELLS, mode="constant", cval=0.0)
    count = uniform_filter(weight, size=FOCAL_WINDOW_CELLS, mode="constant", cval=0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        focal_mean = np.where(count > 1e-6, total / count, np.nan)

    relief = dem - focal_mean
    out_valid = valid & np.isfinite(relief)

    path = aligned_dir / "dem_rel_1km_aligned.tif"
    _write(path, relief, out_valid, master)
    return path


if __name__ == "__main__":  # pragma: no cover
    setup_logging(logging.INFO)
    LOGGER.info("=" * 60)
    LOGGER.info("Deriving drainage-network and multi-scale context features")
    LOGGER.info("=" * 60)
    build_regional_relief()
    build_upstream_cn()
    LOGGER.info("Done. Re-run `python src/susceptibility.py --train` to use them.")
