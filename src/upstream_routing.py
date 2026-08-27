"""
Route the upstream DEM so the model can see water entering the district.

The gap this closes
-------------------
`src/upstream_dem.py` built a 25,085 km2 DEM over the full contributing area,
but routing it failed. Hand-rolled priority-flood filling across 25,000 km2 of
Western Ghats produced very large flats (99th-percentile fill depth 7.7 m,
maximum 124.8 m) and the flat-resolution tie-break dispersed flow instead of
concentrating it. Probed against published catchment areas the result was
useless: 0 km2 for the Periyar at Aluva against roughly 5,000 expected, and the
largest accumulation sitting on the southern grid edge rather than on a channel.

The fix is the one named in that module's docstring: condition the DEM by
*breaching* rather than filling. Breaching carves a channel down through the
lip of a depression, which preserves the drainage direction; filling raises the
depression to its lip, which creates a flat with no direction at all. Over a
mountainous 25,000 km2 grid that difference decides whether flow concentrates
into rivers.

WhiteboxTools does this correctly and fast, so this module drives it rather than
reimplementing it:

    breach_depressions_least_cost  (Lindsay 2016)
    d8_flow_accumulation           in catchment-area units

Validation is the point
-----------------------
A flow network is easy to produce and easy to get wrong -- the previous attempt
looked like a river network at a glance and was nonsense. So this module refuses
to declare success on appearance. It probes the accumulation at gauging points
with *published* catchment areas and reports the ratio. `validate()` returns
those ratios and `build()` logs them; if they are not near 1, the routing is
still wrong and the output should not be used.

Snapping matters: flow accumulation is enormous on a channel cell and tiny one
cell off it, so a probe at a hand-typed coordinate will read near zero even
when the network is perfect. Each probe therefore takes the maximum within
`SNAP_RADIUS_M` and reports how far it had to move.

Output
------
    upstream_area_aligned.tif   contributing area in km2, on the master grid

This is the first feature in the project that carries information from outside
the district boundary.

Run:  python src/upstream_routing.py --build
      python src/upstream_routing.py --validate
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, GEOAI_NEW_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

DEM_NAME = "Upstream_DEM.tif"

#: Working directory for the intermediate rasters. These are large (the
#: breached DEM alone is ~100 MB) and entirely regenerable, so they go outside
#: the tracked tree.
WORK_DIR = GEOAI_NEW_DIR / "routing_work"

#: Maximum breach search distance, in cells. Lindsay's least-cost breaching
#: looks this far for a path out of a depression before giving up and filling.
#: At 30 m, 200 cells is 6 km -- enough to carve out of a Ghats valley, and the
#: cost is only runtime.
BREACH_DIST_CELLS = 200

#: How far a probe may move to find the channel, in metres. Flow accumulation
#: falls off a cliff one cell off the channel, so a hand-typed coordinate needs
#: snapping. 1 km is generous for a named river and still small enough that it
#: cannot jump to a different basin.
SNAP_RADIUS_M = 1000.0

#: Gauging and confluence points with catchment areas from published basin
#: studies (CWC / Kerala State Water Resources). These are the check on whether
#: the routing is real. Areas are the contributing catchment *at that point*,
#: which is smaller than the basin total for anything upstream of the mouth.
PROBES: Tuple[Dict, ...] = (
    {
        "name": "Periyar at Aluva",
        "lat": 10.1076,
        "lon": 76.3516,
        "expected_km2": 5000.0,
        "source": "Periyar basin 5,398 km2 total; Aluva is near the downstream limit",
    },
    {
        "name": "Periyar at Neriamangalam",
        "lat": 10.0631,
        "lon": 76.7811,
        "expected_km2": 3300.0,
        "source": "Upstream of the Muvattupuzha confluence",
    },
    {
        "name": "Chalakudy at Chalakudy town",
        "lat": 10.3070,
        "lon": 76.3320,
        "expected_km2": 1400.0,
        "source": "Chalakudy basin 1,704 km2 total",
    },
    {
        "name": "Muvattupuzha at Muvattupuzha",
        "lat": 9.9790,
        "lon": 76.5790,
        "expected_km2": 1100.0,
        "source": "Muvattupuzha basin 1,554 km2 total",
    },
)

#: A ratio inside this band counts as agreement. Wide on purpose: the published
#: areas are basin totals read against approximate gauge coordinates, and the
#: DEM is 30 m Terrarium rather than a surveyed product. The failure this is
#: built to catch is off by a factor of 100, not 2.
RATIO_OK = (0.5, 2.0)


def _wbt(work_dir: Path):
    """A configured WhiteboxTools instance."""
    try:
        import whitebox
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "WhiteboxTools is required for upstream routing. Install it with:\n"
            "    python -m pip install whitebox"
        ) from exc

    wbt = whitebox.WhiteboxTools()
    wbt.set_verbose_mode(False)
    wbt.set_working_dir(str(work_dir))
    return wbt


def route(
    dem_path: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    breach_dist: int = BREACH_DIST_CELLS,
    force: bool = False,
) -> Path:
    """
    Condition the DEM by breaching and compute D8 contributing area.

    Returns the path to the accumulation raster, in m2 per cell.
    """
    dem_path = dem_path or (GEOAI_NEW_DIR / DEM_NAME)
    work_dir = work_dir or WORK_DIR
    if not dem_path.exists():
        raise FileNotFoundError(f"{dem_path} not found. Run: python src/upstream_dem.py --build")
    work_dir.mkdir(parents=True, exist_ok=True)

    accum = work_dir / "accum_area.tif"
    if accum.exists() and not force:
        LOGGER.info("Using existing %s", accum.name)
        return accum

    # WhiteboxTools resolves relative names against its working directory, so
    # the source DEM is copied in rather than referenced across directories.
    local_dem = work_dir / DEM_NAME
    if not local_dem.exists() or force:
        LOGGER.info("Copying DEM into %s", work_dir.name)
        shutil.copy2(dem_path, local_dem)

    wbt = _wbt(work_dir)
    breached = "breached.tif"

    LOGGER.info(
        "Breaching depressions (least cost, max %d cells = %.1f km)...",
        breach_dist,
        breach_dist * 30.0 / 1000.0,
    )
    rc = wbt.breach_depressions_least_cost(
        dem=DEM_NAME, output=breached, dist=breach_dist, fill=True
    )
    if rc != 0:
        raise RuntimeError(f"breach_depressions_least_cost failed (code {rc})")

    LOGGER.info("Computing D8 flow accumulation in catchment-area units...")
    rc = wbt.d8_flow_accumulation(i=breached, output=accum.name, out_type="catchment area")
    if rc != 0:
        raise RuntimeError(f"d8_flow_accumulation failed (code {rc})")

    LOGGER.info("  wrote %s", accum.name)
    return accum


def validate(
    accum_path: Optional[Path] = None,
    probes: Tuple[Dict, ...] = PROBES,
    snap_radius_m: float = SNAP_RADIUS_M,
) -> List[Dict]:
    """
    Probe the accumulation at points with published catchment areas.

    This is the test that the previous routing attempt failed. Reports the
    snapped area, the published area, and the ratio.
    """
    import rasterio
    from pyproj import Transformer

    accum_path = accum_path or (WORK_DIR / "accum_area.tif")
    if not accum_path.exists():
        raise FileNotFoundError(f"{accum_path} not found. Run route() first.")

    with rasterio.open(accum_path) as src:
        area = src.read(1).astype(np.float64)
        transform = src.transform
        crs = src.crs
        nd = src.nodata
        H, W = src.height, src.width

    if nd is not None:
        area = np.where(area == nd, np.nan, area)
    # WhiteboxTools emits large negative sentinels for nodata in places.
    area = np.where(area < 0, np.nan, area)

    to_grid = Transformer.from_crs("EPSG:4326", str(crs), always_xy=True)
    cell_m = abs(transform.a)
    rad = max(1, int(round(snap_radius_m / cell_m)))

    results: List[Dict] = []
    for p in probes:
        x, y = to_grid.transform(p["lon"], p["lat"])
        col, row = ~transform * (x, y)
        row, col = int(row), int(col)

        if not (0 <= row < H and 0 <= col < W):
            results.append(
                {**p, "found_km2": None, "ratio": None, "note": "outside the routed grid"}
            )
            continue

        r0, r1 = max(0, row - rad), min(H, row + rad + 1)
        c0, c1 = max(0, col - rad), min(W, col + rad + 1)
        win = area[r0:r1, c0:c1]
        if not np.isfinite(win).any():
            results.append(
                {
                    **p,
                    "found_km2": None,
                    "ratio": None,
                    "note": "no valid accumulation in the snap window",
                }
            )
            continue

        flat = int(np.nanargmax(win))
        wr, wc = np.unravel_index(flat, win.shape)
        found_m2 = float(win[wr, wc])
        found_km2 = found_m2 / 1e6
        snap_m = float(np.hypot((r0 + wr) - row, (c0 + wc) - col) * cell_m)
        ratio = found_km2 / p["expected_km2"] if p["expected_km2"] else None

        results.append(
            {
                "name": p["name"],
                "expected_km2": p["expected_km2"],
                "found_km2": round(found_km2, 1),
                "ratio": round(ratio, 3) if ratio is not None else None,
                "snapped_m": round(snap_m, 0),
                "source": p["source"],
            }
        )

    LOGGER.info("")
    LOGGER.info("%-32s %12s %12s %8s %9s", "probe", "published", "routed", "ratio", "snapped")
    LOGGER.info("-" * 78)
    for r in results:
        if r.get("found_km2") is None:
            LOGGER.info(
                "%-32s %9.0f km2 %12s", r["name"], r["expected_km2"], r.get("note", "failed")
            )
            continue
        flag = "" if RATIO_OK[0] <= r["ratio"] <= RATIO_OK[1] else "   <- OUT OF RANGE"
        LOGGER.info(
            "%-32s %9.0f km2 %9.0f km2 %8.2f %7.0f m%s",
            r["name"],
            r["expected_km2"],
            r["found_km2"],
            r["ratio"],
            r["snapped_m"],
            flag,
        )
    return results


def align(
    accum_path: Optional[Path] = None,
    aligned_dir: Optional[Path] = None,
) -> Path:
    """
    Resample contributing area onto the master grid, in km2.

    Nearest-neighbour rather than bilinear: contributing area is a channel
    property that jumps by orders of magnitude between adjacent cells, and
    averaging across a channel edge would invent intermediate catchments that
    do not exist.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    from feature_stack import grid_profile, read_raster

    accum_path = accum_path or (WORK_DIR / "accum_area.tif")
    aligned_dir = aligned_dir or ALIGNED_DIR

    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]

    dst = np.full((H, W), np.nan, dtype=np.float32)
    with rasterio.open(accum_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )

    # m2 -> km2, and negative sentinels out.
    dst = np.where(np.isfinite(dst) & (dst >= 0), dst / 1e6, np.nan)

    _, district = read_raster("lulc", aligned_dir=aligned_dir)
    out = np.where(district & np.isfinite(dst), dst, NODATA).astype(np.float32)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")
    out_path = aligned_dir / "upstream_area_aligned.tif"
    with rasterio.open(out_path, "w", **profile) as sink:
        sink.write(out, 1)

    vals = dst[district & np.isfinite(dst)]
    if vals.size:
        LOGGER.info(
            "  %s -> median %.2f km2, p90 %.1f, p99.9 %.0f, max %.0f",
            out_path.name,
            np.median(vals),
            np.percentile(vals, 90),
            np.percentile(vals, 99.9),
            vals.max(),
        )
    return out_path


def build(force: bool = False) -> Dict:
    """Route, validate, and align. Refuses to align if validation fails."""
    accum = route(force=force)
    checks = validate(accum)

    usable = [c for c in checks if c.get("ratio") is not None]
    passing = [c for c in usable if RATIO_OK[0] <= c["ratio"] <= RATIO_OK[1]]
    LOGGER.info("")
    LOGGER.info(
        "%d of %d probes within %.1fx-%.1fx of published", len(passing), len(checks), *RATIO_OK
    )

    aligned = None
    if len(passing) >= max(2, len(checks) // 2):
        LOGGER.info("Routing validated; aligning onto the master grid.")
        aligned = align(accum)
    else:
        LOGGER.warning(
            "Routing did NOT validate. Not writing an aligned raster -- a "
            "plausible-looking flow network that disagrees with published "
            "catchment areas is worse than none, because it would be trusted."
        )

    summary = {
        "dem": str(GEOAI_NEW_DIR / DEM_NAME),
        "breach_dist_cells": BREACH_DIST_CELLS,
        "snap_radius_m": SNAP_RADIUS_M,
        "probes": checks,
        "n_passing": len(passing),
        "n_probes": len(checks),
        "aligned_raster": aligned.name if aligned else None,
        "method": (
            "WhiteboxTools breach_depressions_least_cost (Lindsay 2016) then "
            "d8_flow_accumulation in catchment-area units. Breaching rather "
            "than filling because filling a 25,000 km2 mountainous grid "
            "creates flats whose tie-break disperses flow instead of "
            "concentrating it."
        ),
    }
    (WORK_DIR / "routing_validation.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Route the upstream DEM")
    parser.add_argument("--build", action="store_true", help="Route, validate, align")
    parser.add_argument("--validate", action="store_true", help="Probe an existing run")
    parser.add_argument("--force", action="store_true", help="Recompute the routing")
    parser.add_argument("--breach-dist", type=int, default=BREACH_DIST_CELLS)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.validate:
        validate()
    elif args.build:
        build(force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
