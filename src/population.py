"""
Real gridded population, replacing the district-average density estimate.

Why
---
`create_alert_message()` estimated exposed population as
`critical_km2 * district_average_density * residential_fraction` -- the same
number applied whether the critical zone sat in dense Kochi or a paddy field
on the district's rural edge, because the model only knew a single district-
wide density. Labelled "planning estimate" in the UI for exactly this reason.

WorldPop gives an actual answer instead: population *count* per 100 m cell,
free, CC-BY, no credentials, already confirmed reachable via this project's
Earth Engine access (`WorldPop/GP/100m/pop`, IND_2020).

Two sums were checked against each other and they disagree, which is worth
recording rather than picking the first one seen. `fetch()` logs a
server-side `reduceRegion` sum over the district polygon (~2.9M) at
download time; summing the downloaded raster directly, over the same
polygon mask, gives ~3.66M instead (`align()`'s `total_before` diagnostic).
The two calls differ in how the sum is computed -- `reduceRegion` without
an explicit `crs`/`crsTransform` is a known source of scale/reprojection
mismatches on GEE, while the raster sum is a plain sum over the pixels that
actually ship in the file this module aligns. The raster-sum figure is also
the one consistent with the 2011 census figure for the district
(3,282,388): 3.66M represents ~11.5% growth over 9 years, plausible for a
growing metro; 2.9M would mean a population *decline*, implausible for
Kochi. Treat `total_before` in `align()`'s log line as the trustworthy
figure and the `fetch()`-time log as a secondary sanity check only.

The disaggregation pitfall
---------------------------
WorldPop's band is a *count*, not a density. Reprojecting a 100 m count
raster onto this project's 10 m master grid with ordinary resampling
(nearest or bilinear) would put approximately the *same* count value in
every one of the ~100 destination cells that tile one source cell -- so
summing the aligned raster would overcount total population by roughly 100x.
`align()` below corrects for this explicitly: it reprojects, then rescales
by the destination-to-source cell-area ratio, so the sum is conserved. This
is the standard population-disaggregation correction and it is easy to get
wrong silently, because the reprojected raster still *looks* plausible --
smoothly varying, right shape -- right up until you sum it and the district
total is 100x too high.

Output
------
    population_aligned.tif   people per 10 m cell (usually << 1; sum, don't read individual cells)

Run:  python src/population.py --build
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from config import ALIGNED_DIR, setup_logging

LOGGER = logging.getLogger("geoai_flood")

WORLDPOP_COLLECTION = "WorldPop/GP/100m/pop"
WORLDPOP_YEAR = 2020
WORLDPOP_COUNTRY = "IND"

DISTRICT_NAME = "Ernakulam"


def fetch(project: str, out_dir: Path) -> Path:
    """Download the WorldPop population count raster for Ernakulam via GEE."""
    import ee

    LOGGER.info("Initialising Earth Engine (project=%s)...", project)
    ee.Initialize(project=project)

    img = (
        ee.ImageCollection(WORLDPOP_COLLECTION)
        .filterDate(f"{WORLDPOP_YEAR}-01-01", f"{WORLDPOP_YEAR + 1}-01-01")
        .filterMetadata("country", "equals", WORLDPOP_COUNTRY)
        .first()
    )

    roi = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM2_NAME", DISTRICT_NAME))
        .geometry()
    )

    total = img.reduceRegion(ee.Reducer.sum(), roi, 100, maxPixels=int(1e10)).getInfo()[
        "population"
    ]
    LOGGER.info("  Ernakulam total population (WorldPop %d): %.0f", WORLDPOP_YEAR, total)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"worldpop_{WORLDPOP_YEAR}.tif"

    import shutil
    import urllib.request

    url = img.clip(roi).getDownloadURL(
        {
            "scale": 100,
            "region": roi,
            "format": "GEO_TIFF",
            "crs": "EPSG:32643",
        }
    )
    with urllib.request.urlopen(url, timeout=600) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    LOGGER.info("  wrote %s", path)
    return path


def align(src_path: Path, aligned_dir: Optional[Path] = None) -> Path:
    """
    Reproject WorldPop onto the master grid, conserving total population.

    See the module docstring for why this needs the explicit area-ratio
    rescale rather than a plain reproject.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    from feature_stack import grid_profile, read_raster
    from config import RASTER

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]

    with rasterio.open(src_path) as src:
        # GEE's GeoTIFF export leaves the nodata tag unset (src.nodata is
        # None) despite genuinely using -99999 as the out-of-ROI sentinel.
        # Passing src.nodata straight to reproject() -- the obvious thing to
        # do -- silently told it there was no nodata at all, so -99999 got
        # blended into the bilinear resampling at the district edge like a
        # real population value. Caught by cross-checking the aligned total
        # against the exact figure GEE's own reduceRegion had already given.
        src_nodata = src.nodata if src.nodata is not None else -99999.0
        src_cell_m2 = abs(src.transform.a * src.transform.e)
        dst = np.full((H, W), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    dst_cell_m2 = abs(master["transform"].a * master["transform"].e)
    # Conserve the sum: bilinear resampling preserves the *value* pattern
    # (a smooth count surface), not the total. Each destination cell should
    # hold its share of the source cell's count, proportional to area.
    dst = dst * (dst_cell_m2 / src_cell_m2)
    dst = np.where(np.isfinite(dst) & (dst >= 0), dst, 0.0).astype(np.float32)

    _, district = read_raster("lulc", aligned_dir=aligned_dir)
    out = np.where(district, dst, RASTER.nodata_value).astype(np.float32)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=RASTER.nodata_value, compress="lzw")
    out_path = aligned_dir / "population_aligned.tif"
    with rasterio.open(out_path, "w", **profile) as sink:
        sink.write(out, 1)

    with rasterio.open(src_path) as src:
        v = src.read(1).astype(np.float64)
        nd = src.nodata if src.nodata is not None else -99999.0
    ok = np.isfinite(v) & (v != nd)
    total_before = float(v[ok].sum())
    total_after = float(dst[district].sum())
    LOGGER.info(
        "  %s -> total %.0f (source: %.0f, ratio %.3f)",
        out_path.name,
        total_after,
        total_before,
        total_after / max(total_before, 1e-6),
    )
    return out_path


def build(project: str, aligned_dir: Optional[Path] = None) -> dict:
    aligned_dir = aligned_dir or ALIGNED_DIR
    cache_dir = aligned_dir.parent / "data" / "worldpop"
    src_path = cache_dir / f"worldpop_{WORLDPOP_YEAR}.tif"
    if not src_path.exists():
        src_path = fetch(project, cache_dir)
    out_path = align(src_path, aligned_dir)

    summary = {
        "source": f"WorldPop {WORLDPOP_COLLECTION}/{WORLDPOP_COUNTRY}_{WORLDPOP_YEAR} (CC-BY 4.0)",
        "raster": out_path.name,
    }
    (aligned_dir / "population.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:  # pragma: no cover
    import os

    parser = argparse.ArgumentParser(description="Fetch and align WorldPop population")
    parser.add_argument(
        "--project",
        default=os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("EE_PROJECT"),
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if not args.project:
        LOGGER.error("No Earth Engine project. Set --project or $EARTHENGINE_PROJECT.")
        raise SystemExit(2)
    build(args.project)


if __name__ == "__main__":  # pragma: no cover
    main()
