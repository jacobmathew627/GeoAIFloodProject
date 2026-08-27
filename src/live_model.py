"""
Live evaluation: any rainfall depth, computed on demand, in milliseconds.

Nothing here interpolates between pre-rendered scenarios. Both surfaces are
closed-form functions of rainfall, so moving a slider recomputes them from the
model rather than blending stored rasters:

    fluvial(x, P)  = sigma( logit(S(x)) + beta * ln( Q(x,P) / Q(x,P_ref) ) )
    pluvial(x, P)  = f( routed SCS-CN runoff, local slope )

`S` is the learned susceptibility surface and is rainfall-independent, so it is
loaded once. `Q` is SCS-CN runoff, which depends on the pixel only through its
curve number. Everything rainfall-dependent is therefore a handful of scalar
evaluations followed by array arithmetic on a ~1000x771 display grid.

The two layers are deliberately NOT blended. They answer different questions,
one is calibrated and one is a proxy-validated physics index, and averaging
them would launder the weaker validation into the stronger one's credibility:

  fluvial   Probability of riverine/backwater inundation. Trained on the NDEM
            2018 inundation inventory, calibrated, spatial-block AUC 0.824,
            with a conformal coverage guarantee. Trust the number. Includes
            OSM drainage proximity/density as inputs (ranked 5th/6th of 16 by
            permutation importance) but still scores near chance, AUC 0.388,
            against the 14 documented urban waterlogging hotspots -- it is
            answering a different question (basin-scale inundation) than the
            one those hotspots pose (street-level ponding).
  pluvial   Relative index of rain-driven waterlogging pressure, from routed
            runoff and local gradient. Physics only, not probability-
            calibrated -- but no longer untested: against the 14 documented
            hotspots vs. an elevation-matched urban background it scores
            AUC 0.807 (95% CI 0.698-0.908), and proximity to a mapped drain
            or canal alone gets AUC 0.713 (canals in this city are tidal and
            back up, so *closer* is worse, not better -- see
            src/osm_drainage.py). The control is a proxy, not real incident
            records, since none exist for this district yet. Use it to rank,
            never as a probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, HYDRO, MODELS_DIR, OUTPUT_DIR, RAINFALL, RASTER

if TYPE_CHECKING:
    # Only for type-checking: importing rasterio at module scope here would
    # be a real dependency just to name two types. `transform`/`bounds` were
    # annotated as `object` before, which is why mypy could not see
    # `.bottom`/`.right`/`.top`/`~` on them despite both working at runtime.
    from affine import Affine
    from rasterio.coords import BoundingBox

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

#: Cached precomputation, so the app starts fast.
#:
#: Stored as a plain .npz rather than a pickle. Pickling the LiveGrid
#: dataclass records the module it was defined in, and building the cache via
#: `python src/live_model.py --build` defines it in `__main__` -- so the app,
#: which imports `live_model`, could not resolve the class and failed with
#: "Can't get attribute 'LiveGrid'". Arrays plus a small JSON header have no
#: such identity problem and load faster.
CACHE_NAME = "live_model.npz"


@dataclass
class LiveGrid:
    """Everything rainfall-independent, on the display grid."""

    susceptibility: np.ndarray  # (H, W) in [0,1], NaN outside the domain
    curve_number: np.ndarray  # (H, W) AMC-adjusted CN
    basis: Dict[int, np.ndarray]  # class -> upstream cell count
    classes: np.ndarray
    tan_slope: np.ndarray
    population: np.ndarray  # (H, W) people per display cell (WorldPop), NaN outside the domain
    building_area: np.ndarray  # (H, W) building footprint m2 per display cell (OSM)
    cell_area_m2: float
    cell_width_m: float
    transform: "Affine"
    crs: str
    bounds: "BoundingBox"
    shape: Tuple[int, int]

    # Percentiles of the pluvial raw score at the reference event, used to
    # express the index on a 0-1 scale that means "relative to the reference
    # storm" rather than an arbitrary log value.
    pluvial_lo: float = 0.0
    pluvial_hi: float = 1.0


# ──────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────
def build(max_dim: int = 1000, aligned_dir: Optional[Path] = None) -> LiveGrid:
    """Precompute the rainfall-independent layers on the display grid."""
    import rasterio
    from rasterio.enums import Resampling

    from hydrology import adjust_cn_for_amc
    from pluvial import PluvialModel

    aligned_dir = aligned_dir or ALIGNED_DIR

    susc_path = OUTPUT_DIR / "susceptibility.tif"
    if not susc_path.exists():
        raise FileNotFoundError(
            f"{susc_path} not found. Run `python src/susceptibility.py --predict`."
        )

    LOGGER.info("Loading susceptibility at display resolution...")
    with rasterio.open(susc_path) as src:
        scale = min(max_dim / max(src.width, src.height), 1.0)
        out_h = max(1, int(src.height * scale))
        out_w = max(1, int(src.width * scale))
        # Nearest picks a real pixel, so nodata stays identifiable; average
        # over the -9999 sentinel would smear it into the valid range.
        near = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        avg = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.average)
        nd = src.nodata if src.nodata is not None else NODATA
        transform = src.transform * src.transform.scale(src.width / out_w, src.height / out_h)
        crs, bounds = str(src.crs), src.bounds

    ok = np.isfinite(near) & (near != np.float32(nd))
    susceptibility = np.where(ok, np.clip(avg, 0.0, 1.0), np.nan).astype(np.float32)
    shape = susceptibility.shape
    LOGGER.info(
        "  display grid %s, %.2fM valid cells", shape, np.isfinite(susceptibility).sum() / 1e6
    )

    # -- land cover and slope on the same grid --
    def load(name, resampling):
        path = aligned_dir / f"{name}_aligned.tif"
        with rasterio.open(path) as src:
            a = src.read(1, out_shape=shape, resampling=resampling).astype(np.float32)
            n = src.nodata if src.nodata is not None else NODATA
        return np.where(np.isfinite(a) & (a != np.float32(n)), a, np.nan)

    lulc = load("lulc", Resampling.nearest)
    slope_deg = load("slope", Resampling.average)

    # -- population and building exposure, resampled with count conservation --
    # `population_aligned.tif` and `building_area_aligned.tif` each hold a
    # *count* (people, m2 of footprint) per cell, not a density. An ordinary
    # average -- what `load()` above uses for slope -- would report the mean
    # count of the source cells a display cell covers, silently shrinking the
    # district total by the same downsampling factor used to build `shape`.
    # Resampling.average followed by rescaling by the cell-area ratio turns
    # that mean back into a sum -- the same area-ratio trick population.py
    # uses in the other direction (its 100 m source to this project's 10 m
    # master grid).
    def load_conserved(name, fallback_hint):
        path = aligned_dir / f"{name}_aligned.tif"
        if not path.exists():
            LOGGER.warning(
                "No %s -- figures depending on it fall back to a coarser " "estimate. Run `%s`.",
                path,
                fallback_hint,
            )
            return np.full(shape, np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            raw = src.read(1, out_shape=shape, resampling=Resampling.average).astype(np.float32)
            nd = src.nodata if src.nodata is not None else NODATA
            src_cell_m2 = abs(src.transform.a * src.transform.e)
        dst_cell_m2 = abs(transform.a * transform.e)
        ok = np.isfinite(raw) & (raw != np.float32(nd))
        return np.where(ok, raw * (dst_cell_m2 / src_cell_m2), np.nan).astype(np.float32)

    population = load_conserved("population", "python src/population.py --project <id>")
    building_area = load_conserved("building_area", "python src/building_exposure.py --build")

    LOGGER.info("Deriving the curve number grid...")
    cn = np.full(shape, np.nan, dtype=np.float32)
    valid_lulc = np.isfinite(lulc)
    cn[valid_lulc] = HYDRO.default_curve_number
    for cls, value in HYDRO.curve_numbers.items():
        cn[valid_lulc & (np.round(lulc) == cls)] = value
    cn = adjust_cn_for_amc(cn, HYDRO.amc)

    tan_slope = np.tan(np.radians(np.clip(slope_deg, 0.06, 60.0)))

    # -- routed runoff basis, accumulated at 30 m then brought to the display grid --
    LOGGER.info("Building the routed runoff basis...")
    from pluvial import reproject_basis_to_grid

    pm = PluvialModel.build(aligned_dir=aligned_dir)
    basis = reproject_basis_to_grid(pm.basis, pm.profile, transform, crs, shape)

    px_w = abs(transform.a)
    grid = LiveGrid(
        susceptibility=susceptibility,
        curve_number=cn,
        basis=basis,
        classes=pm.classes,
        tan_slope=tan_slope.astype(np.float32),
        population=population,
        building_area=building_area,
        cell_area_m2=abs(transform.a * transform.e),
        cell_width_m=px_w,
        transform=transform,
        crs=crs,
        bounds=bounds,
        shape=shape,
    )

    # Fix the pluvial 0-1 scale against the reference storm so the index is
    # comparable between rainfall values instead of being re-stretched each time.
    ref = _pluvial_raw(grid, RAINFALL.reference_event_mm)
    finite = ref[np.isfinite(ref)]
    grid.pluvial_lo = float(np.percentile(finite, 5))
    grid.pluvial_hi = float(np.percentile(finite, 99.5))
    LOGGER.info(
        "  pluvial scale anchored at the reference storm: [%.2f, %.2f]",
        grid.pluvial_lo,
        grid.pluvial_hi,
    )
    return grid


def save(grid: LiveGrid, model_dir: Optional[Path] = None) -> Path:
    """Persist as arrays plus a JSON header -- no pickled classes."""
    import json

    model_dir = model_dir or MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / CACHE_NAME

    arrays = {
        "susceptibility": grid.susceptibility,
        "curve_number": grid.curve_number,
        "tan_slope": grid.tan_slope,
        "population": grid.population,
        "building_area": grid.building_area,
        "classes": np.asarray(grid.classes),
    }
    for k, v in grid.basis.items():
        arrays[f"basis_{int(k)}"] = v

    t = grid.transform
    header = {
        "cell_area_m2": float(grid.cell_area_m2),
        "cell_width_m": float(grid.cell_width_m),
        "pluvial_lo": float(grid.pluvial_lo),
        "pluvial_hi": float(grid.pluvial_hi),
        "shape": list(grid.shape),
        "crs": str(grid.crs),
        "transform": [t.a, t.b, t.c, t.d, t.e, t.f],
        "bounds": [grid.bounds.left, grid.bounds.bottom, grid.bounds.right, grid.bounds.top],
    }
    arrays["header_json"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)

    np.savez_compressed(path, **arrays)
    LOGGER.info("Cached live model -> %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


def load(model_dir: Optional[Path] = None) -> LiveGrid:
    import json

    from affine import Affine
    from rasterio.coords import BoundingBox

    model_dir = model_dir or MODELS_DIR
    path = model_dir / CACHE_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/live_model.py --build`.")

    with np.load(path, allow_pickle=False) as z:
        header = json.loads(bytes(z["header_json"]).decode("utf-8"))
        classes = z["classes"]
        basis = {int(k): z[f"basis_{int(k)}"] for k in classes}
        # "population" postdates the rest of this cache format; a cache built
        # before src/population.py existed won't have it. Falling back to
        # all-NaN degrades gracefully to the density-estimate fallback in
        # create_alert_message() rather than crashing the app on a stale cache.
        shape = tuple(header["shape"])
        population = (
            z["population"] if "population" in z.files else np.full(shape, np.nan, dtype=np.float32)
        )
        building_area = (
            z["building_area"]
            if "building_area" in z.files
            else np.full(shape, np.nan, dtype=np.float32)
        )
        grid = LiveGrid(
            susceptibility=z["susceptibility"],
            curve_number=z["curve_number"],
            basis=basis,
            classes=classes,
            tan_slope=z["tan_slope"],
            population=population,
            building_area=building_area,
            cell_area_m2=header["cell_area_m2"],
            cell_width_m=header["cell_width_m"],
            transform=Affine(*header["transform"]),
            crs=header["crs"],
            bounds=BoundingBox(*header["bounds"]),
            shape=tuple(header["shape"]),
            pluvial_lo=header["pluvial_lo"],
            pluvial_hi=header["pluvial_hi"],
        )
    return grid


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def fluvial_probability(grid: LiveGrid, rainfall_mm: float) -> np.ndarray:
    """
    Calibrated probability of riverine/backwater inundation at this rainfall.

    Identical formulation to hazard.combine's routed path -- same
    pluvial.routed_runoff_ratio() call, same basis grid.basis already
    computed for the pluvial index, so the live slider and the batch-
    generated hazard rasters answer the same question instead of drifting
    into two different formulas that happen to share a name.
    """
    from hazard import combine
    from pluvial import routed_runoff_ratio

    ratio = routed_runoff_ratio(
        grid.basis,
        grid.classes,
        float(rainfall_mm),
        RAINFALL.reference_event_mm,
        grid.cell_area_m2,
        np.isfinite(grid.susceptibility),
    )
    return combine(grid.susceptibility, grid.curve_number, float(rainfall_mm), runoff_ratio=ratio)


def _pluvial_raw(grid: LiveGrid, rainfall_mm: float) -> np.ndarray:
    """Dynamic wetness: ln(routed runoff per unit width / tan slope)."""
    from pluvial import class_runoff_depths

    q = class_runoff_depths(rainfall_mm, grid.classes)
    total = np.zeros(grid.shape, dtype=np.float32)
    for k, n_k in grid.basis.items():
        depth_mm = q.get(int(k), 0.0)
        if depth_mm > 0:
            total += np.float32(depth_mm) * n_k

    volume = total / 1000.0 * grid.cell_area_m2
    specific = volume / grid.cell_width_m
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.log(np.maximum(specific, 1e-6) / grid.tan_slope)
    return np.where(np.isfinite(grid.susceptibility) & np.isfinite(grid.tan_slope), w, np.nan)


def pluvial_index(grid: LiveGrid, rainfall_mm: float) -> np.ndarray:
    """
    Rain-driven waterlogging pressure, 0-1, anchored to the reference storm.

    UNVALIDATED. Physics only -- routed SCS-CN runoff over local gradient.
    There are no urban waterlogging labels for this district, so this index
    has never been tested against the phenomenon it names. It is a ranking,
    not a probability.
    """
    raw = _pluvial_raw(grid, rainfall_mm)
    span = max(grid.pluvial_hi - grid.pluvial_lo, 1e-6)
    return np.clip((raw - grid.pluvial_lo) / span, 0.0, 1.0)


# ──────────────────────────────────────────────
# Point query
# ──────────────────────────────────────────────
def query(grid: LiveGrid, lat: float, lon: float, rainfall_mm: float) -> Optional[Dict]:
    """Both surfaces, plus the physical quantities behind them, at one point."""
    from pyproj import Transformer

    from config import LULC_CLASS_NAMES
    from hydrology import runoff_depth

    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(lon, lat)
    col, row = ~grid.transform * (x, y)
    row, col = int(row), int(col)

    h, w = grid.shape
    if not (0 <= row < h and 0 <= col < w):
        return None
    if not np.isfinite(grid.susceptibility[row, col]):
        return None

    cn = float(grid.curve_number[row, col])
    q = float(runoff_depth(rainfall_mm, np.array([[cn]], dtype=np.float32))[0, 0])

    fluvial = fluvial_probability(grid, rainfall_mm)[row, col]
    pluvial = pluvial_index(grid, rainfall_mm)[row, col]

    # Which land-cover class does this curve number correspond to?
    cover = None
    for cls, cn_ii in HYDRO.curve_numbers.items():
        from hydrology import adjust_cn_for_amc

        adj = float(adjust_cn_for_amc(np.array([[cn_ii]], dtype=np.float32), HYDRO.amc)[0, 0])
        if abs(adj - cn) < 0.51:
            cover = LULC_CLASS_NAMES.get(cls, f"class {cls}")
            break

    return {
        "lat": lat,
        "lon": lon,
        "row": row,
        "col": col,
        "rainfall_mm": float(rainfall_mm),
        "land_cover": cover,
        "curve_number": cn,
        "runoff_mm": q,
        "runoff_coefficient": q / rainfall_mm if rainfall_mm > 0 else 0.0,
        "fluvial_probability": float(fluvial),
        "pluvial_index": float(pluvial),
        "susceptibility": float(grid.susceptibility[row, col]),
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:  # pragma: no cover
    import argparse
    import time

    from config import setup_logging

    parser = argparse.ArgumentParser(description="Live rainfall model")
    parser.add_argument("--build", action="store_true", help="Precompute and cache")
    parser.add_argument("--max-dim", type=int, default=1000)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.INFO)

    if args.build:
        grid = build(max_dim=args.max_dim)
        save(grid)
    else:
        grid = load()

    if args.benchmark:
        for mm in (50, 120, 200, 337):
            t0 = time.perf_counter()
            f = fluvial_probability(grid, mm)
            p = pluvial_index(grid, mm)
            dt = (time.perf_counter() - t0) * 1000
            LOGGER.info(
                "%4d mm -> %5.1f ms | fluvial mean %.4f | pluvial mean %.3f",
                mm,
                dt,
                np.nanmean(f),
                np.nanmean(p),
            )


if __name__ == "__main__":  # pragma: no cover
    main()
