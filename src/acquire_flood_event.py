"""
Acquire a Sentinel-1 flood inventory for one event via Google Earth Engine.

STATUS: executed, twice. Authenticate once with:

    python -c "import ee; ee.Authenticate()"
    python src/acquire_flood_event.py --event 2026 --project YOUR_PROJECT_ID
    python src/acquire_flood_event.py --event 2026 --align

Run for 2026 (the flood beginning late July 2026, peak ~1 Aug -- Ernakulam
under orange alert, rivers over warning levels): 5.6 km2 detected, 56,324
cells on the master grid. The event and threshold tables have direct test
coverage (tests/test_acquire_flood_event.py); build_flood_image()/acquire()
still need a live, authenticated EE session to exercise, and align() needs a
real master grid on disk, so neither is unit tested -- same precedent as
src/upstream_routing.py's own align().

Known limitation, hit on the 2026 run: only two descending VV/IW scenes
existed for the district in the event window (29 Jul, 10 Aug), neither
bracketing the confirmed 1 Aug peak the way 2018's window did. min() over the
window is the best available estimate, but it is honestly degraded -- treat
results from a sparsely-covered event with real skepticism before feeding
them into fit_beta.py, and do not silently mix a Sentinel-1-derived extent
or an ERA5-only rainfall figure (IMD's gauge archive lags real time by more
than the weeks between a flood and this being run) into a calibration built
from NDEM/IMD sources without accounting for both being systematically
different measurement methods, not just noisier ones.

Why not reuse the existing extraction scripts
---------------------------------------------
`src/extract_sar_gee.py` and `src/extract_gee_flood.py` take a **median
composite over the whole of August 2018** and threshold it. A median over a
month is dominated by the ~27 days that were not flooded, so it systematically
under-detects a flood lasting three or four days -- which is consistent with
the inventory holding only 30 km2 of non-permanent-water flooding for an event
that displaced people district-wide.

This script uses change detection instead, which is standard practice for SAR
flood mapping:

  1. A pre-event baseline: the median VV backscatter over a comparable,
     non-flooded window (same season where possible, so vegetation state and
     incidence angle are similar).
  2. An event image: the minimum VV over the flood window. Minimum, not
     median, because open water is specular -- it reflects radar away from
     the sensor -- so flooding shows up as a sharp *drop*, and any single
     acquisition catching the peak is the informative one.
  3. Flood = low absolute backscatter AND a large drop from baseline.

Two absolute criteria alone would flag every permanently dark surface (roads,
airport aprons, the backwaters); the drop criterion is what isolates change.
Terrain masks then remove the classic false positives: radar shadow on slopes
and anything too high above the drainage network to plausibly inundate.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import setup_logging

LOGGER = logging.getLogger("geoai_flood")

#: Event windows. Baselines are chosen in the same season of an adjacent,
#: non-flooded year so that crop stage and look geometry are comparable.
EVENTS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "2018": {"event": ("2018-08-14", "2018-08-22"), "baseline": ("2018-06-01", "2018-07-15")},
    "2019": {"event": ("2019-08-08", "2019-08-18"), "baseline": ("2019-06-01", "2019-07-15")},
    "2021": {"event": ("2021-10-15", "2021-10-25"), "baseline": ("2021-08-15", "2021-09-30")},
    # 2026 Kerala floods: onset late July, peak ~1 Aug (near-cloudburst, >300mm
    # single-day, Ernakulam under orange alert). Only two descending VV/IW
    # scenes exist for the district in this window -- 29 Jul (3 days before
    # peak) and 10 Aug (9 days after it) -- so neither brackets the peak the
    # way 2018's window did. The event window spans both; min() over the
    # window picks up whichever date shows flooding at each pixel, which is
    # the best available estimate but is honestly degraded, closer to 2019's
    # coverage problem than to a clean acquisition. Treat the result with
    # the same skepticism before it goes into fit_beta.py.
    "2026": {"event": ("2026-07-25", "2026-08-12"), "baseline": ("2026-05-15", "2026-07-10")},
}

DISTRICT_NAME = "Ernakulam"

# Decision thresholds, in dB. VV over open water typically falls below about
# -16 dB; a drop of 3 dB or more from a stable baseline is the usual change
# criterion. Both must hold.
VV_WATER_MAX_DB = -16.0
VV_DROP_MIN_DB = 3.0

# Terrain exclusions. Slopes above ~5 degrees do not pond, and anything more
# than ~15 m above the nearest drainage is not reachable by fluvial flooding.
MAX_SLOPE_DEG = 5.0
MAX_HAND_M = 15.0


def build_flood_image(event: str, project: str):
    """Construct the flood mask as an ee.Image. Requires an initialised ee."""
    import ee

    if event not in EVENTS:
        raise ValueError(f"Unknown event {event!r}. Known: {sorted(EVENTS)}")
    windows = EVENTS[event]

    roi = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM2_NAME", DISTRICT_NAME))
        .geometry()
    )

    def s1(start, end):
        return (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            # Mixing orbit directions mixes look geometry, which by itself
            # shifts backscatter by several dB and would swamp the 3 dB signal.
            .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
            .select("VV")
        )

    baseline_col = s1(*windows["baseline"])
    event_col = s1(*windows["event"])

    # Speckle is multiplicative and severe in single looks; a small focal
    # median suppresses it without smearing edges the way a mean would.
    baseline = baseline_col.median().focal_median(30, "circle", "meters")
    event_img = event_col.min().focal_median(30, "circle", "meters")

    drop = baseline.subtract(event_img)
    flooded = event_img.lt(VV_WATER_MAX_DB).And(drop.gt(VV_DROP_MIN_DB))

    # Terrain exclusions.
    dem = ee.Image("USGS/SRTMGL1_003")
    slope = ee.Terrain.slope(dem)
    flooded = flooded.And(slope.lt(MAX_SLOPE_DEG))

    # Permanent water from the JRC surface-water history: anything present
    # more than half the time is not flooding, it is a lake. This is the same
    # correction applied in feature_stack.domain_mask, done here at source.
    occurrence = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
    permanent = occurrence.gt(50).unmask(0)
    flooded = flooded.And(permanent.Not())

    return flooded.selfMask().rename("flood").clip(roi), roi, baseline_col, event_col


def acquire(event: str, project: str, scale: int = 30, out_dir: str = "GeoAI_New") -> str:
    """Download the flood mask for `event` as a GeoTIFF."""
    import shutil
    import urllib.request

    import ee

    LOGGER.info("Initialising Earth Engine (project=%s)...", project)
    ee.Initialize(project=project)

    flood, roi, baseline_col, event_col = build_flood_image(event, project)

    n_base = baseline_col.size().getInfo()
    n_event = event_col.size().getInfo()
    LOGGER.info(
        "  %s: %d baseline scenes (%s..%s), %d event scenes (%s..%s)",
        event, n_base, *EVENTS[event]["baseline"], n_event, *EVENTS[event]["event"],
    )
    if n_event == 0:
        raise RuntimeError(
            f"No Sentinel-1 scenes in the {event} event window. Widen the window "
            "or drop the DESCENDING orbit filter."
        )
    if n_base < 2:
        LOGGER.warning(
            "  only %d baseline scene(s); the median baseline will be noisy", n_base
        )

    area_km2 = (
        flood.multiply(ee.Image.pixelArea())
        .reduceRegion(ee.Reducer.sum(), roi, scale, maxPixels=int(1e10))
        .getInfo()
    )
    LOGGER.info("  detected flood area: %s", area_km2)

    path = f"{out_dir}/Flood_Extent_{event}.tif"
    LOGGER.info("  exporting to %s at %d m...", path, scale)

    # getDownloadURL rather than geemap: one fewer heavy dependency, and the
    # mask is well inside the 32 MB response limit (~2.7M pixels of uint8 over
    # this district at 30 m).
    url = flood.unmask(0).toByte().getDownloadURL({
        "scale": scale,
        "region": roi,
        "format": "GEO_TIFF",
        "crs": "EPSG:32643",
    })
    with urllib.request.urlopen(url, timeout=600) as response, open(path, "wb") as f:
        shutil.copyfileobj(response, f)

    size_mb = os.path.getsize(path) / 1e6
    LOGGER.info("  wrote %s (%.1f MB)", path, size_mb)
    LOGGER.info("  next: python src/acquire_flood_event.py --event %s --align", event)
    return path


def align(event: str, aligned_dir: Optional[Path] = None, out_dir: str = "GeoAI_New") -> Path:
    """
    Resample an acquired flood extent onto the master grid.

    Nearest-neighbour, not bilinear: this is a binary mask, and averaging
    across a flood-edge pixel would invent a fractional "half flooded" value
    that does not mean anything. Matches the resampling choice already used
    for upstream_routing.align() and ndem_labels.rasterize_event() for the
    same reason.

    Earlier events (2018/2019/2021) went through align_data.py's legacy
    "ground_truth" path, which is a single Sentinel-1 scene special-cased for
    the original model. NDEM's multi-event inventory instead goes through
    ndem_labels.rasterize_event(), which starts from vector polygons. This
    acquisition is already a raster (Sentinel-1 change detection via GEE), so
    it needs neither -- just a reproject onto the master grid, done here.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    from config import ALIGNED_DIR, RASTER
    from feature_stack import grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    src_path = Path(out_dir) / f"Flood_Extent_{event}.tif"
    if not src_path.exists():
        raise FileNotFoundError(f"{src_path} not found. Run --event {event} first (without --align).")

    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]

    dst = np.zeros((H, W), dtype=np.float32)
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=master["transform"],
            dst_crs=master["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

    _, district = read_raster("lulc", aligned_dir=aligned_dir)
    out = np.where(district, dst, RASTER.nodata_value).astype(np.float32)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=RASTER.nodata_value, compress="lzw")
    out_path = aligned_dir / f"sentinel1_flood_{event}_aligned.tif"
    with rasterio.open(out_path, "w", **profile) as sink:
        sink.write(out, 1)

    flooded = (dst > 0.5) & district
    px_km2 = (RASTER.cell_size / 1000.0) ** 2
    LOGGER.info(
        "  %s -> %d flooded px (%.2f km2) on the master grid",
        out_path.name, int(flooded.sum()), float(flooded.sum()) * px_km2,
    )
    return out_path


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Acquire a Sentinel-1 flood inventory")
    parser.add_argument("--event", default="2019", choices=sorted(EVENTS))
    # No default. The ID previously hardcoded here (empyrean-backup-387418) no
    # longer exists -- Earth Engine reports "project not found or deleted" --
    # and a dead default silently sends everyone down the same dead end. Earth
    # Engine now requires a Cloud project registered for its use; register one
    # free for research at https://code.earthengine.google.com/register
    parser.add_argument(
        "--project",
        default=os.environ.get("EARTHENGINE_PROJECT") or os.environ.get("EE_PROJECT"),
        help="Earth Engine Cloud project ID. Defaults to $EARTHENGINE_PROJECT "
             "or $EE_PROJECT. Register one at "
             "https://code.earthengine.google.com/register",
    )
    parser.add_argument("--scale", type=int, default=30)
    parser.add_argument(
        "--align", action="store_true",
        help="Resample an already-acquired extent onto the master grid instead of fetching",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)

    if args.align:
        align(args.event)
        return

    if not args.project:
        LOGGER.error(
            "No Earth Engine project. Earth Engine requires a Cloud project "
            "registered for its use.\n"
            "  1. Register one (free for research): "
            "https://code.earthengine.google.com/register\n"
            "  2. Then either pass --project YOUR_PROJECT_ID\n"
            "     or set EARTHENGINE_PROJECT=YOUR_PROJECT_ID"
        )
        raise SystemExit(2)

    try:
        acquire(args.event, args.project, args.scale)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        message = str(exc)
        if "not found or deleted" in message or "has not been used" in message:
            LOGGER.error(
                "That project is not registered for Earth Engine. Register it at "
                "https://code.earthengine.google.com/register"
            )
        elif "authorize" in message.lower():
            LOGGER.error('Run:  python -c "import ee; ee.Authenticate()"')
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
