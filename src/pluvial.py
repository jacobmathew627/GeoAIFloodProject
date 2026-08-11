"""
Pluvial (rain-driven) waterlogging: routed runoff against depression storage.

Why this module exists
----------------------
The learned susceptibility surface predicts *riverine and backwater inundation
extent*, because that is what its label contains: Sentinel-1 open water during
an event driven by Periyar flooding and reservoir releases. Sampled at Kochi's
urban core it returns ~0.001 -- the top 0.1% of its hazard has median urban
fraction 0.0 against a district median of 1.0. It is not a waterlogging model
and cannot become one by re-weighting.

Urban waterlogging is a different mechanism: rain falls, some fraction becomes
runoff, that runoff travels downslope, and it ponds where the terrain has a
depression and the drainage cannot carry it away. This module models that
mechanism directly. It needs no flood labels, which is what makes it usable
here -- there are none for waterlogging.

The approach follows the DEM-based fill-spill family (RUFIDAM, Safer_RAIN),
which the literature finds outperforms topographic indices for pluvial
flooding precisely because it accounts for depressions and their connectivity,
and needs only a DEM and net rainfall. Notably, one Guangzhou study found TWI
*could not* map frequently flooded areas -- consistent with TWI ranking near
the bottom of this project's own feature importances (0.008).

What this module does NOT do, and why
-------------------------------------
The obvious approach is fill-spill: find depressions, fill them with routed
runoff. It was implemented, measured, and abandoned. `Ernakulam_Clipped_DEM`
is 30 m horizontally with **1 m vertical quantisation**, and street ponding is
0.1-0.5 m deep. Filling it yields depressions with a median depth of 3.0 m, a
p90 of 7.0 m and a maximum of 28 m: these are regional basins on the coastal
plain, not urban hollows. The model saturated immediately, putting 2 m of
standing water in central Ernakulam at 50 mm of rain, and it simply
re-expressed the same fluvial signal the learned model already carries.
`fill_depressions` is kept below because it is correct and reusable, but
nothing here depends on it. Resolving street-scale ponding needs a LiDAR or
photogrammetric DEM, ideally 1 m with sub-decimetre vertical accuracy.

What it does instead
--------------------
A rainfall-driven wetness index. The classic topographic wetness index,
ln(a / tan b), uses specific catchment area `a` as a stand-in for how much
water arrives. Here the actual routed runoff volume is used in its place, so
the index responds to the storm and to land cover rather than to terrain
alone:

    V(x, P)  = sum_k  Q_k(P) * A_k(x)        routed runoff volume, m3
    a(x, P)  = V(x, P) / cell_width          specific runoff, m2
    W(x, P)  = ln( a(x, P) / (tan b + eps) ) dynamic wetness

`Q_k` is SCS-CN runoff for land-cover class k and `A_k(x)` the upstream area
of that class draining through x. Unlike static TWI, W rises when it rains
harder, and rises faster where the upstream catchment is impervious -- which
is the mechanism behind urban waterlogging. Static TWI cannot express either;
one Guangzhou study found it could not map frequently flooded areas at all,
consistent with TWI ranking near the bottom of this project's own feature
importances (0.008).

Real-time evaluation
--------------------
SCS-CN runoff depends on the pixel only through its curve number, and the
curve number takes one value per land-cover class -- seven in this district.
So for a storm of depth P there are only seven distinct runoff depths, and

    arriving(x, P) = sum_k  Q_k(P) * N_k(x)

where N_k(x) is the number of class-k cells draining through x. The N_k are
rainfall-independent and accumulated once. Evaluating a new rainfall is then
seven multiply-adds over the grid -- milliseconds, exact, no approximation.
That is what makes the slider live rather than an interpolation between
pre-rendered scenarios.
"""
from __future__ import annotations

import heapq
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, HYDRO, RASTER

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

#: Storm-drain conveyance allowance, in mm over the storm. Runoff up to this
#: depth is assumed to be carried away rather than ponding. Kochi's drains are
#: widely reported as under-capacity and frequently blocked, so this is
#: deliberately modest. It is an assumption, not a measurement: there is no
#: drainage-network dataset in this repository.
DEFAULT_DRAINAGE_MM = 40.0

#: Ponding depth at which waterlogging becomes disruptive (ankle-deep, stalls
#: traffic). Used to turn a depth into a yes/no indicator.
NUISANCE_DEPTH_MM = 100.0


# ──────────────────────────────────────────────
# Depression storage
# ──────────────────────────────────────────────
def fill_depressions(elev: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Priority-flood depression fill (Barnes et al.).

    Returns the filled surface. `filled - elev` is the depression storage
    depth: how deep water can pond before it spills.

    NOTE: this is computed here rather than read from
    `GeoAI_New/Ernakulam_Filled_DEM.tif`. That raster is not a filled version
    of `Ernakulam_Clipped_DEM.tif` -- its maximum (786.53 m) is *below* the
    raw maximum (790.00 m), which a fill cannot produce. Differencing the two
    gives a median "depression" of 4.65 m and 8.8 m of equivalent storage
    across the district, which is not terrain.
    """
    h, w = elev.shape
    filled = np.full((h, w), np.inf, dtype=np.float64)
    closed = ~valid

    heap = []
    # Seed with every valid cell on the grid edge or adjacent to nodata: those
    # are the points water can leave from.
    edge = np.zeros((h, w), dtype=bool)
    edge[0, :] = edge[-1, :] = True
    edge[:, 0] = edge[:, -1] = True
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.ones((h, w), dtype=bool)
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        sr0, sr1 = max(0, dr), h - max(0, -dr)
        sc0, sc1 = max(0, dc), w - max(0, -dc)
        shifted[r0:r1, c0:c1] = closed[sr0:sr1, sc0:sc1]
        edge |= shifted
    seeds = np.flatnonzero((valid & edge).ravel())

    for i in seeds:
        r, c = divmod(int(i), w)
        filled[r, c] = elev[r, c]
        heap.append((float(elev[r, c]), r, c))
    heapq.heapify(heap)

    LOGGER.info("  priority-flood from %d seed cells...", len(heap))
    push, pop = heapq.heappush, heapq.heappop
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1))

    while heap:
        level, r, c = pop(heap)
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if closed[nr, nc] or filled[nr, nc] != np.inf:
                continue
            # Water cannot sit lower than the level it must spill over.
            nv = elev[nr, nc] if elev[nr, nc] > level else level
            filled[nr, nc] = nv
            push(heap, (float(nv), nr, nc))

    filled[~valid] = np.nan
    return filled


def depression_depth_m(elev: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-cell depression depth in metres, from a priority-flood fill."""
    filled = fill_depressions(elev, valid)
    depth = np.where(valid, filled - elev, np.nan)
    return np.clip(depth, 0.0, None)


def label_depressions(depth_m: np.ndarray, min_depth_m: float = 0.05):
    """
    Label connected depressions.

    Returns (labels, n). Cells shallower than `min_depth_m` are background:
    at 30 m resolution, sub-5 cm "depressions" are DEM noise rather than
    terrain that holds water.
    """
    from scipy.ndimage import label

    mask = np.isfinite(depth_m) & (depth_m > min_depth_m)
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labels, n = label(mask, structure=structure)
    return labels, int(n)


# ──────────────────────────────────────────────
# Rainfall-independent routing basis
# ──────────────────────────────────────────────
def build_runoff_basis(net, lulc_route: np.ndarray) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Accumulate, per land-cover class, the number of cells draining through
    each cell.

    Returns ({class: N_k}, classes_present). Rainfall-independent, so this is
    computed once and reused for every rainfall value.
    """
    classes = sorted(
        int(c) for c in np.unique(lulc_route[np.isfinite(lulc_route)])
        if 1 <= int(c) <= 11
    )
    basis = {}
    for k in classes:
        weight = ((np.round(lulc_route) == k) & net.valid).astype(np.float32)
        basis[k] = net.accumulate(weight)
        LOGGER.info(
            "  class %2d: %8d cells, max upstream count %.0f",
            k, int(weight.sum()), np.nanmax(basis[k]),
        )
    return basis, np.array(classes)


def class_runoff_depths(rainfall_mm: float, classes: np.ndarray) -> Dict[int, float]:
    """SCS-CN runoff depth (mm) for each land-cover class at this rainfall."""
    from hydrology import adjust_cn_for_amc, runoff_depth

    out = {}
    for k in classes:
        cn_ii = HYDRO.curve_numbers.get(int(k), HYDRO.default_curve_number)
        cn = adjust_cn_for_amc(np.array([[cn_ii]], dtype=np.float32), HYDRO.amc)
        out[int(k)] = float(runoff_depth(rainfall_mm, cn)[0, 0])
    return out


# ──────────────────────────────────────────────
# The model
# ──────────────────────────────────────────────
class PluvialModel:
    """
    Rainfall-driven wetness, evaluated in milliseconds for any storm depth.

    Speed comes from the class decomposition: SCS-CN runoff depends on the
    pixel only through its curve number, and there are seven curve numbers in
    this district. So a new rainfall value costs seven multiply-adds over the
    grid, not a re-route.
    """

    def __init__(
        self,
        basis: Dict[int, np.ndarray],
        classes: np.ndarray,
        tan_slope: np.ndarray,
        valid: np.ndarray,
        cell_area_m2: float,
        cell_width_m: float,
    ):
        self.basis = basis
        self.classes = classes
        self.tan_slope = tan_slope
        self.valid = valid
        self.cell_area_m2 = cell_area_m2
        self.cell_width_m = cell_width_m

    # -- construction ------------------------------------------------------
    @classmethod
    def build(cls, aligned_dir: Optional[Path] = None):
        """Build from the aligned rasters. Expensive; cache the result."""
        from rasterio.enums import Resampling
        from rasterio.warp import reproject

        from feature_stack import grid_profile, read_raster
        from routing import FlowNetwork

        aligned_dir = aligned_dir or ALIGNED_DIR
        master = grid_profile(aligned_dir)

        LOGGER.info("Building flow network...")
        net = FlowNetwork()

        def to_route(name, resampling=Resampling.average):
            values, ok = read_raster(name, aligned_dir=aligned_dir)
            out = np.full(net.elev.shape, np.nan, dtype=np.float32)
            reproject(
                source=np.where(ok, values, np.nan).astype(np.float32),
                destination=out,
                src_transform=master["transform"], src_crs=master["crs"],
                dst_transform=net.profile["transform"], dst_crs=net.profile["crs"],
                resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan,
            )
            return out

        LOGGER.info("Projecting land cover and slope onto the routing grid...")
        lulc_route = to_route("lulc", Resampling.nearest)
        slope_deg = to_route("slope")

        LOGGER.info("Accumulating the rainfall-independent runoff basis...")
        basis, classes = build_runoff_basis(net, lulc_route)

        # Guard the gradient away from zero: a perfectly flat cell would send
        # the wetness index to infinity, and 0.1% is a realistic floor for a
        # surface that still drains.
        tan_slope = np.tan(np.radians(np.clip(slope_deg, 0.06, 60.0)))
        tan_slope = np.where(np.isfinite(tan_slope), tan_slope, np.nan)

        transform = net.profile["transform"]
        model = cls(
            basis=basis, classes=classes, tan_slope=tan_slope, valid=net.valid,
            cell_area_m2=net.cell_area_m2, cell_width_m=abs(transform.a),
        )
        model.profile = net.profile
        model.master_profile = master
        return model

    # -- evaluation --------------------------------------------------------
    def routed_runoff_m3(self, rainfall_mm: float) -> np.ndarray:
        """Runoff volume draining through each cell, in cubic metres."""
        q = class_runoff_depths(rainfall_mm, self.classes)
        total = np.zeros(self.tan_slope.shape, dtype=np.float32)
        for k, n_k in self.basis.items():
            depth_mm = q.get(int(k), 0.0)
            if depth_mm > 0:
                total += np.float32(depth_mm) * np.nan_to_num(n_k, nan=0.0)
        # depth (mm) x upstream cell count -> volume
        volume = total / 1000.0 * self.cell_area_m2
        return np.where(self.valid, volume, np.nan)

    def dynamic_wetness(self, rainfall_mm: float) -> np.ndarray:
        """
        ln( specific runoff / tan(slope) ) -- TWI with real runoff in place of
        catchment area, so it responds to the storm and to land cover.
        """
        volume = self.routed_runoff_m3(rainfall_mm)
        specific = volume / self.cell_width_m
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.log(np.maximum(specific, 1e-6) / self.tan_slope)
        return np.where(np.isfinite(volume) & np.isfinite(self.tan_slope), w, np.nan)
