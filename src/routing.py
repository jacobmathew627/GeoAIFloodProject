"""
D8 flow routing over the drainage network.

Why this exists
---------------
Every model in this project up to now scored each pixel independently. That
is the single largest structural weakness of grid-based flood susceptibility
mapping: flooding is not a per-pixel property, it is what happens when runoff
generated upstream arrives somewhere with nowhere to go. Recent work makes
the point directly -- a GraphSAGE model over a watershed connectivity graph
reached AUC 0.978 against 0.881 for the best pixel-independent ensemble on
the same inventory, and the authors attribute the gap to upstream-downstream
propagation that raster models structurally cannot see
(arXiv:2603.15681, Himachal Pradesh flash floods).

Rather than swap the whole model for a GNN, this module recovers the same
signal in the feature and physics layers:

  * `accumulate` routes any weight grid down the D8 network, so runoff
    generated upstream can be summed at the receiving cell.
  * `upstream_mean` gives the catchment-average of any grid -- used for
    "what kind of land drains into me", which is drainage-network context a
    Euclidean neighbourhood cannot express.

Routing runs on the depression-filled 30 m DEM (`Ernakulam_Filled_DEM.tif`),
which is the correct grid for it: routing on the 10 m master grid would be
four times the work for no extra hydrological information, since the DEM it
derives from is 30 m anyway.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import GEOAI_NEW_DIR, RASTER

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

FILLED_DEM = "Ernakulam_Filled_DEM.tif"

# D8 neighbour offsets (row, col) and their centre-to-centre distances in
# cell widths. Diagonals are sqrt(2) further away, which matters: without the
# distance weighting the steepest-descent choice is biased toward diagonals.
_D8 = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, np.sqrt(2)),
    (-1, 1, np.sqrt(2)),
    (1, -1, np.sqrt(2)),
    (1, 1, np.sqrt(2)),
]


# ──────────────────────────────────────────────
# Flow network
# ──────────────────────────────────────────────
def load_filled_dem(geoai_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load the depression-filled DEM. Returns (elevation, valid, profile)."""
    geoai_dir = geoai_dir or GEOAI_NEW_DIR
    path = geoai_dir / FILLED_DEM
    if not path.exists():
        raise FileNotFoundError(f"Filled DEM not found: {path}")

    with rasterio.open(path) as src:
        elev = src.read(1).astype(np.float32)
        nd = src.nodata
        profile = src.profile.copy()

    valid = np.isfinite(elev)
    if nd is not None:
        valid &= elev != np.float32(nd)
    # The filled DEM carries -99999 for nodata; anything far below sea level
    # in a coastal district is that sentinel, not terrain.
    valid &= elev > -100.0
    elev = np.where(valid, elev, np.nan).astype(np.float32)

    LOGGER.info(
        "Filled DEM %s: %.2fM valid cells, range [%.2f, %.2f] m",
        elev.shape,
        valid.sum() / 1e6,
        np.nanmin(elev),
        np.nanmax(elev),
    )
    return elev, valid, profile


def _shift(grid: np.ndarray, dr: int, dc: int, fill) -> np.ndarray:
    """Align the (dr, dc) neighbour of every cell, padding the edge with `fill`."""
    h, w = grid.shape
    out = np.full((h, w), fill, dtype=grid.dtype)
    r0, r1 = max(0, -dr), h - max(0, dr)
    c0, c1 = max(0, -dc), w - max(0, dc)
    sr0, sr1 = max(0, dr), h - max(0, -dr)
    sc0, sc1 = max(0, dc), w - max(0, -dc)
    out[r0:r1, c0:c1] = grid[sr0:sr1, sc0:sc1]
    return out


def d8_receivers(
    elev: np.ndarray,
    valid: np.ndarray,
    flat_tiebreak: Optional[np.ndarray] = None,
    rank: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Steepest-descent receiver for every cell, as a flat index.

    Two passes:

    1. Strict steepest descent among the eight neighbours, distance-weighted.

    2. Flat resolution. A depression-filled DEM is *exactly* level across every
       filled area, and strict descent has nothing to choose between
       neighbours there, so drainage never concentrates into a channel. That
       is not a corner case in this district -- the Vembanad coastal plain,
       which is precisely where flooding matters, is mostly flat. Measured on
       the raw first pass, the median upslope area on mapped stream cells was
       only 7.7x the off-channel median; the shipped accumulation raster
       achieves 52.6x.

       Cells left without a strictly lower neighbour therefore route to the
       non-higher neighbour carrying the most drainage according to
       `flat_tiebreak` (the validated flow-accumulation raster). Routing is
       still never uphill; the tie-break only orders choices that gravity
       alone leaves undetermined.

       `rank` is the cell's position in the processing order. On a flat, two
       equal-elevation cells could otherwise choose each other, and a 2-cycle
       double-counts mass in the single-pass accumulation. Requiring the
       receiver to come strictly later in the processing order makes the
       network a DAG by construction.
    """
    h, w = elev.shape
    flat_idx = np.arange(h * w, dtype=np.int64).reshape(h, w)

    best_slope = np.zeros((h, w), dtype=np.float32)
    receiver = flat_idx.copy()

    for dr, dc, dist in _D8:
        n_elev = _shift(elev, dr, dc, np.nan)
        n_idx = _shift(flat_idx, dr, dc, -1)

        slope = (elev - n_elev) / dist
        better = valid & np.isfinite(slope) & (slope > best_slope)
        best_slope = np.where(better, slope, best_slope)
        receiver = np.where(better, n_idx, receiver)

    if flat_tiebreak is not None:
        unresolved = valid & (best_slope <= 0)
        n_unresolved = int(unresolved.sum())
        if n_unresolved:
            tb = np.where(np.isfinite(flat_tiebreak), flat_tiebreak, -np.inf).astype(np.float64)
            best_tb = np.full((h, w), -np.inf)

            for dr, dc, _ in _D8:
                n_elev = _shift(elev, dr, dc, np.nan)
                n_idx = _shift(flat_idx, dr, dc, -1)
                n_tb = _shift(tb, dr, dc, -np.inf)

                # Non-higher only, so this can never create an uphill edge.
                downhill_ok = np.isfinite(n_elev) & (n_elev <= elev)
                if rank is not None:
                    n_rank = _shift(rank, dr, dc, np.int64(-1))
                    downhill_ok &= n_rank > rank
                better = unresolved & downhill_ok & (n_tb > best_tb)
                best_tb = np.where(better, n_tb, best_tb)
                receiver = np.where(better, n_idx, receiver)

            LOGGER.info(
                "  flat resolution applied to %d cells (%.1f%% of valid)",
                n_unresolved,
                100 * n_unresolved / max(valid.sum(), 1),
            )

    receiver[~valid] = flat_idx[~valid]
    return receiver.ravel()


def _topological_order(elev: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Cells sorted from highest to lowest.

    Because water only ever moves to a strictly lower cell, descending
    elevation is a valid topological order for the D8 network: when a cell is
    processed, every cell that drains into it has already been processed.
    This is what makes accumulation a single pass instead of an iteration to
    convergence.
    """
    flat_elev = np.where(valid, elev, -np.inf).ravel()
    return np.argsort(-flat_elev, kind="stable")


def accumulate(
    weights: np.ndarray,
    receiver: np.ndarray,
    order: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Route `weights` down the D8 network and return the accumulated total.

    The result at a cell is its own weight plus the weight of everything
    upstream of it. Invalid cells contribute nothing.
    """
    acc = np.where(valid, weights, 0.0).astype(np.float64).ravel()
    valid_flat = valid.ravel()

    for i in order:
        if not valid_flat[i]:
            continue
        j = receiver[i]
        if j != i:
            acc[j] += acc[i]

    out = acc.reshape(weights.shape).astype(np.float32)
    return np.where(valid, out, np.nan)


def upstream_mean(
    values: np.ndarray,
    receiver: np.ndarray,
    order: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    Catchment-average of `values` over each cell's upstream area.

    Computed as accumulated(values) / accumulated(1), so it answers "what is
    the average character of the land draining into me" -- drainage-network
    context, not Euclidean neighbourhood context.
    """
    finite = valid & np.isfinite(values)
    total = accumulate(np.where(finite, values, 0.0), receiver, order, finite)
    count = accumulate(np.ones_like(values, dtype=np.float32), receiver, order, finite)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(count > 0, total / count, np.nan)
    return np.where(valid, out, np.nan).astype(np.float32)


# ──────────────────────────────────────────────
# Reprojection onto the master grid
# ──────────────────────────────────────────────
def to_master_grid(
    data: np.ndarray,
    src_profile: dict,
    master_profile: dict,
) -> np.ndarray:
    """Resample a routing-grid result onto the 10 m master grid."""
    dst = np.full((master_profile["height"], master_profile["width"]), np.nan, dtype=np.float32)
    reproject(
        source=np.where(np.isfinite(data), data, np.nan).astype(np.float32),
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=master_profile["transform"],
        dst_crs=master_profile["crs"],
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def _load_flow_accumulation(
    geoai_dir: Optional[Path], shape: Tuple[int, int]
) -> Optional[np.ndarray]:
    """Load the shipped flow-accumulation raster, used only to resolve flats."""
    geoai_dir = geoai_dir or GEOAI_NEW_DIR
    path = geoai_dir / "Ernakulam_Flow_Accumulation.tif"
    if not path.exists():
        LOGGER.warning("No flow-accumulation raster; flats will not be resolved")
        return None

    with rasterio.open(path) as src:
        acc = src.read(1).astype(np.float64)
        nd = src.nodata
    if acc.shape != shape:
        LOGGER.warning(
            "Flow accumulation shape %s != DEM %s; flats will not be resolved",
            acc.shape,
            shape,
        )
        return None

    acc = np.where(np.isfinite(acc) & (acc != nd) & (acc > 0), acc, np.nan)
    return np.log1p(acc)


class FlowNetwork:
    """The D8 network of the filled DEM, built once and reused."""

    def __init__(self, geoai_dir: Optional[Path] = None, resolve_flats: bool = True):
        self.elev, self.valid, self.profile = load_filled_dem(geoai_dir)

        # Order first: the flat-resolution pass needs each cell's rank in it
        # to guarantee an acyclic network.
        LOGGER.info("Sorting cells into topological order...")
        self.order = _topological_order(self.elev, self.valid)
        rank_flat = np.empty(self.order.size, dtype=np.int64)
        rank_flat[self.order] = np.arange(self.order.size, dtype=np.int64)
        rank = rank_flat.reshape(self.elev.shape)

        tiebreak = _load_flow_accumulation(geoai_dir, self.elev.shape) if resolve_flats else None

        LOGGER.info("Building D8 receivers...")
        self.receiver = d8_receivers(self.elev, self.valid, tiebreak, rank)

        valid_flat = np.flatnonzero(self.valid.ravel())
        pits = int(np.sum(self.receiver[valid_flat] == valid_flat))
        LOGGER.info(
            "Flow network ready: %.2fM cells, %d terminal cells (pits/outlets)",
            self.valid.sum() / 1e6,
            pits,
        )

    @property
    def cell_area_m2(self) -> float:
        t = self.profile["transform"]
        return abs(t.a * t.e)

    def accumulate(self, weights: np.ndarray) -> np.ndarray:
        return accumulate(weights, self.receiver, self.order, self.valid)

    def upstream_mean(self, values: np.ndarray) -> np.ndarray:
        return upstream_mean(values, self.receiver, self.order, self.valid)

    def contributing_area_m2(self) -> np.ndarray:
        """Upslope contributing area. Used to validate against the shipped raster."""
        return self.accumulate(np.full(self.elev.shape, self.cell_area_m2, dtype=np.float32))
