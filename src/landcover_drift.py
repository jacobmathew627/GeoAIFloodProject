"""
Measure how far the model's land cover has drifted from the present day.

Why this is not simply "the data is stale"
-------------------------------------------
The susceptibility model is trained on the August 2018 flood inventory, so
2018 land cover is the *correct* land cover for training: the curve numbers,
`urban_dist` and the LULC-derived features all describe the surface that
produced the floods being learned from. Swapping in 2025 land cover and
retraining would misalign the features from the labels, which is a worse
error than the one it fixes.

The drift matters for *inference*, not training. When the app evaluates a
storm today, it routes runoff over a 2018 surface. Land that has been built
on since sheds far more water than the model believes, so present-day risk is
understated in exactly the places where it has grown fastest.

This module measures the size of that gap rather than leaving it as an
adjective. Google Dynamic World gives a consistent 10 m land-cover series from
2015 to now, so the same class (built-up, code 6) can be counted on the same
footprint in different years -- which ESA WorldCover, with only 2020 and 2021,
cannot support.

Measured 2026-08-27 (cached in models/landcover_drift.json):

    2018    691.0 km2   built-up
    2021    795.7 km2   +15.2%
    2024    866.1 km2   +25.3%
    2025    852.1 km2   +23.3%   (+161 km2 since 2018)

The 2024->2025 dip is not a real reversal; it is the modal-class reducer
responding to different cloud-free coverage between windows. Treat the trend,
not any single year, as the signal.

What to do about it is a temporal-transfer problem, not a data refresh -- see
the "Known limitations" section of README.md.

Run:  python src/landcover_drift.py --project <earthengine-project>
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import MODELS_DIR, setup_logging

LOGGER = logging.getLogger("geoai_flood")

DISTRICT_NAME = "Ernakulam"

#: Dynamic World's built-up class. The full schema is
#: 0 water, 1 trees, 2 grass, 3 flooded_vegetation, 4 crops, 5 shrub_and_scrub,
#: 6 built, 7 bare, 8 snow_and_ice.
BUILT_CLASS = 6

#: The model's land cover epoch -- the year of the LULC raster the pipeline was
#: aligned and trained on, and the year of the flood inventory it learned from.
MODEL_EPOCH = 2018

#: One-year windows to count over. A single year is short enough to be a
#: meaningful snapshot and long enough to survive monsoon cloud cover.
WINDOWS: Tuple[int, ...] = (2018, 2021, 2024, 2025)

CACHE_NAME = "landcover_drift.json"


def measure(project: str, windows: Optional[Tuple[int, ...]] = None) -> Dict:
    """Built-up area per window, from Dynamic World, over the district."""
    import ee

    windows = windows or WINDOWS
    LOGGER.info("Initialising Earth Engine (project=%s)...", project)
    ee.Initialize(project=project)

    roi = (
        ee.FeatureCollection("FAO/GAUL/2015/level2")
        .filter(ee.Filter.eq("ADM2_NAME", DISTRICT_NAME))
        .geometry()
    )

    def built_km2(year: int):
        # Modal class over the window rather than a single scene: an individual
        # Dynamic World image is a per-scene classification and is noisy under
        # monsoon cloud. The mode is the stable label for that year.
        label = (
            ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
            .filterBounds(roi)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .select("label")
            .mode()
        )
        area = (
            label.eq(BUILT_CLASS)
            .multiply(ee.Image.pixelArea())
            .reduceRegion(ee.Reducer.sum(), roi, 30, maxPixels=int(1e10))
            .get("label")
        )
        return ee.Number(area).divide(1e6)

    # One round trip for every window, rather than one per window.
    values: List[float] = ee.List([built_km2(y) for y in windows]).getInfo()

    baseline = next(v for y, v in zip(windows, values) if y == MODEL_EPOCH)
    series = []
    for year, value in zip(windows, values):
        series.append(
            {
                "year": year,
                "built_up_km2": round(float(value), 1),
                "change_vs_model_epoch_pct": round((value - baseline) / baseline * 100, 1),
                "change_vs_model_epoch_km2": round(float(value - baseline), 1),
            }
        )
        LOGGER.info(
            "  %d  %8.1f km2  %+6.1f%% vs %d",
            year,
            value,
            series[-1]["change_vs_model_epoch_pct"],
            MODEL_EPOCH,
        )

    latest = series[-1]
    return {
        "source": "Google Dynamic World V1 (GOOGLE/DYNAMICWORLD/V1), modal class per year",
        "district": DISTRICT_NAME,
        "model_epoch": MODEL_EPOCH,
        "built_class": BUILT_CLASS,
        "scale_m": 30,
        "series": series,
        "drift_pct": latest["change_vs_model_epoch_pct"],
        "drift_km2": latest["change_vs_model_epoch_km2"],
        "interpretation": (
            "The model routes runoff over its training-epoch land cover, which "
            "is correct for training (the labels are the August 2018 flood "
            "inventory) but understates present-day risk on land built since. "
            "Built-up surfaces carry a much higher curve number, so the "
            "affected area sheds more runoff than the model assumes. This is a "
            "temporal-transfer problem, not a stale-file problem: refreshing "
            "the land cover without also moving the labels would misalign "
            "features from the event they describe."
        ),
    }


def load(models_dir: Optional[Path] = None) -> Optional[Dict]:
    """Read the cached measurement, or None if it has never been run."""
    path = (models_dir or MODELS_DIR) / CACHE_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build(project: str, models_dir: Optional[Path] = None) -> Dict:
    models_dir = models_dir or MODELS_DIR
    summary = measure(project)
    path = models_dir / CACHE_NAME
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", path)
    return summary


def main() -> None:  # pragma: no cover
    import os

    parser = argparse.ArgumentParser(description="Measure land-cover drift since the model epoch")
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
