"""
Flood labels from the National Database of Emergency Management (NDEM).

Why this replaces the Sentinel-1 inventory
------------------------------------------
The project trained on a single Sentinel-1 scene. Sentinel-1 revisits this area
every 12 days, and the only 2018 acquisition landed on 21 August -- four to six
days after the IMD rainfall peak of 15-17 August, once much of the water had
receded. That is the real reason the inventory held only 31 km2 for a
catastrophic flood, and why urban Kochi appeared dry in it.

NDEM is the national disaster-management inventory, compiled by NRSC from
multiple sensors and acquisitions during each event. Over the model domain:

    inventory                       in-domain   urban    urban share
    Sentinel-1, 21 Aug 2018           31.3 km2   2.0 km2      6.3%
    NDEM, 17+18 Aug 2018 union        78.7 km2  33.8 km2     42.9%

Seventeen times the urban flood signal, captured on the days it actually
flooded. The two overlap by only 8.5 km2, so this is a different observation,
not a refinement of the same one.

It also spans eight events -- 2013, 2018, 2019, 2020, 2021, 2022, 2023, 2024 --
which is what makes fitting the rainfall sensitivity possible instead of
assuming it.

What it still does not contain
------------------------------
Urban *waterlogging*. Of the 14 locations documented in public reporting as
recurrent Kochi waterlogging points, NDEM's peak-timed 2018 extent covers
**none**, exactly as the Sentinel-1 inventory covers none. NDEM maps
inundation, including urban ground near water; it does not map the
junction-scale street ponding that gets reported after every heavy shower. No
free inventory does. That needs municipal incident records.

Source: NDEM via Bhuvan/NRSC, extracted to GeoJSON/Parquet releases by
ramSeraph/india_natural_disasters, CC0 1.0.

Run:  python src/ndem_labels.py --build
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import ALIGNED_DIR, DATA_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

RELEASE = (
    "https://github.com/ramSeraph/india_natural_disasters/releases/download/floods/"
    "NDEM_KL_Floods_Inundation.parquet"
)
CACHE = DATA_DIR / "ndem" / "NDEM_KL_Floods_Inundation.parquet"

#: Acquisition timestamps grouped into events, with the IMD 3-day district
#: rainfall depth for each. The rainfall figure is what lets the rainfall
#: sensitivity be fitted across events rather than assumed.
EVENTS: Dict[str, Dict] = {
    "2018": {
        "dates": ["17-08-2018 00:00", "18-08-2018 19:00"],
        "rainfall_mm": 443.2,
        "note": "IMD peak 15-17 Aug; these are the peak-timed acquisitions",
    },
    "2019": {
        "dates": ["10-08-2019 00:00", "12-08-2019 00:00"],
        "rainfall_mm": 412.5,
        "note": "IMD peak 7-9 Aug",
    },
    "2020": {
        "dates": ["09-08-2020 00:00", "10-08-2020 00:00"],
        "rainfall_mm": None,
        "note": "rainfall not yet derived",
    },
    "2021": {
        "dates": ["16-10-2021 00:00", "19-10-2021 00:00"],
        "rainfall_mm": 173.7,
        "note": "IMD peak 17-19 Oct",
    },
}

#: The event the susceptibility model is trained on by default.
PRIMARY_EVENT = "2018"


def download(force: bool = False) -> Path:
    """Fetch the Kerala NDEM inundation dataset (42 MB), cached."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not force:
        LOGGER.info("Using cached %s (%.1f MB)", CACHE.name, CACHE.stat().st_size / 1e6)
        return CACHE
    LOGGER.info("Downloading NDEM Kerala inundation...")
    urllib.request.urlretrieve(RELEASE, CACHE)
    LOGGER.info("  %.1f MB", CACHE.stat().st_size / 1e6)
    return CACHE


def rasterize_event(event: str, aligned_dir: Optional[Path] = None) -> np.ndarray:
    """
    Burn one event's inundation polygons onto the master grid.

    Returns a boolean flooded mask. Acquisitions within an event are unioned:
    a pixel that was under water at any point during the event counts as
    flooded, which is the quantity a susceptibility model should learn.
    """
    import geopandas as gpd
    from rasterio.features import rasterize

    from feature_stack import grid_profile

    if event not in EVENTS:
        raise ValueError(f"Unknown event {event!r}. Known: {sorted(EVENTS)}")

    master = grid_profile(aligned_dir)
    gdf = gpd.read_parquet(download()).to_crs(master["crs"])

    dates = EVENTS[event]["dates"]
    sub = gdf[gdf["from_time"].astype(str).isin(dates)]
    if sub.empty:
        raise RuntimeError(f"No NDEM polygons for {event} dates {dates}")

    shapes = [(g, 1) for g in sub.geometry if g is not None and not g.is_empty]
    LOGGER.info("  %s: %d polygons across %d acquisitions", event, len(shapes), len(dates))

    mask = rasterize(
        shapes,
        out_shape=(master["height"], master["width"]),
        transform=master["transform"],
        fill=0,
        dtype="uint8",
    ).astype(bool)
    return mask


def build(events: Optional[List[str]] = None, aligned_dir: Optional[Path] = None) -> Dict:
    """
    Write an aligned label raster per event, plus a summary.

    Each raster is 1 where flooded, 0 elsewhere inside the district, and
    nodata outside -- the same convention as ground_truth_aligned.tif, so it
    is a drop-in replacement.
    """
    import rasterio

    from feature_stack import domain_mask, grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    events = events or sorted(EVENTS)
    master = grid_profile(aligned_dir)
    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")

    # District footprint: labels are only meaningful where the model is defined.
    lulc, lulc_valid = read_raster("lulc", aligned_dir=aligned_dir)
    district = lulc_valid
    domain = domain_mask(aligned_dir=aligned_dir)
    urban, urban_ok = read_raster("urban_mask", aligned_dir=aligned_dir)
    px_km2 = abs(master["transform"].a * master["transform"].e) / 1e6

    summary = {"source": "NDEM via Bhuvan/NRSC (CC0)", "events": {}}

    for event in events:
        mask = rasterize_event(event, aligned_dir)
        labels = np.where(mask, 1.0, 0.0).astype(np.float32)
        out = np.where(district, labels, NODATA).astype(np.float32)

        path = aligned_dir / f"ndem_flood_{event}_aligned.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out, 1)

        in_dom = mask & domain
        urb = in_dom & urban_ok & (urban > 0.5)
        entry = {
            "dates": EVENTS[event]["dates"],
            "rainfall_mm": EVENTS[event]["rainfall_mm"],
            "note": EVENTS[event]["note"],
            "flooded_km2_in_domain": round(float(in_dom.sum()) * px_km2, 1),
            "flooded_km2_urban": round(float(urb.sum()) * px_km2, 1),
            "urban_share": round(float(urb.sum()) / max(int(in_dom.sum()), 1), 3),
            "prevalence_in_domain": round(float(in_dom.sum()) / max(int(domain.sum()), 1), 5),
            "raster": path.name,
        }
        summary["events"][event] = entry
        LOGGER.info(
            "  %s -> %s | %.1f km2 in domain, %.1f km2 urban (%.1f%%), prevalence %.3f%%",
            event, path.name, entry["flooded_km2_in_domain"],
            entry["flooded_km2_urban"], 100 * entry["urban_share"],
            100 * entry["prevalence_in_domain"],
        )

    out_json = aligned_dir / "ndem_labels.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out_json)
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build NDEM flood labels")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--events", nargs="*", default=None, choices=sorted(EVENTS))
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.build:
        build(args.events)
    else:
        for name, cfg in sorted(EVENTS.items()):
            LOGGER.info("%s: %s  rainfall=%s", name, cfg["dates"], cfg["rainfall_mm"])


if __name__ == "__main__":  # pragma: no cover
    main()
