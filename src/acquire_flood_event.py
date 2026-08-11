"""
Acquire a Sentinel-1 flood inventory for one event via Google Earth Engine.

STATUS: written but NOT executed. Earth Engine needs interactive browser
authentication bound to a Google account, and the stored refresh token in
~/.config/earthengine/credentials is expired. Run this once:

    python -c "import ee; ee.Authenticate()"
    python src/acquire_flood_event.py --event 2019

Everything below the GEE calls is exercised by tests; the GEE calls themselves
have not been run against the live API, so treat the first run as a dry run
and check the printed diagnostics before trusting the output.

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
from typing import Dict, Tuple

from config import setup_logging

LOGGER = logging.getLogger("geoai_flood")

#: Event windows. Baselines are chosen in the same season of an adjacent,
#: non-flooded year so that crop stage and look geometry are comparable.
EVENTS: Dict[str, Dict[str, Tuple[str, str]]] = {
    "2018": {"event": ("2018-08-14", "2018-08-22"), "baseline": ("2018-06-01", "2018-07-15")},
    "2019": {"event": ("2019-08-08", "2019-08-18"), "baseline": ("2019-06-01", "2019-07-15")},
    "2021": {"event": ("2021-10-15", "2021-10-25"), "baseline": ("2021-08-15", "2021-09-30")},
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
    import ee
    import geemap

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
    geemap.ee_export_image(
        flood.unmask(0), filename=path, scale=scale, region=roi, file_per_band=False
    )
    LOGGER.info("  done. Re-run align_data.py to bring it onto the master grid.")
    return path


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Acquire a Sentinel-1 flood inventory")
    parser.add_argument("--event", default="2019", choices=sorted(EVENTS))
    parser.add_argument("--project", default="empyrean-backup-387418")
    parser.add_argument("--scale", type=int, default=30)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    try:
        acquire(args.event, args.project, args.scale)
    except Exception as exc:
        LOGGER.error("%s: %s", type(exc).__name__, exc)
        LOGGER.error(
            "If this is an authentication error, run:  "
            'python -c "import ee; ee.Authenticate()"'
        )
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
