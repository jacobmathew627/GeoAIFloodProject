"""
Building exposure from OpenStreetMap, replacing the flat per-km2 damage guess.

Why
---
`create_alert_message()`'s "Indicative damage" was `critical_km2 *
Rs 50 Cr/km2` -- a flat rate applied to raw area, with no idea whether that
area was dense Kochi housing or an unbuilt paddy field. There is no free,
credentialless source for actual property values or a validated
flood depth-damage function for this district, so this module does not
attempt to predict damage. It replaces the area-only guess with something
narrower but real: the *replacement value of the building stock actually
sitting in the critical-risk zone*, from mapped OSM footprints and a cited
construction rate. Labelled "building value exposed", not "damage" -- it is
an exposure figure (what is there), not a damage prediction (what a given
flood would do to it), because nothing available here can honestly attempt
the second question.

Rate
----
Rs 2,500/sqft is the Kerala PWD-adopted 2025 mid-range/standard residential
construction rate for Kochi and other urban centres (Delhi Schedule of Rates
2021, applied statewide from 1 April 2025 with a location cost index; urban
centres run 15-25% above the rural base rate). That converts to
Rs 26,910/m2 (1 sqft = 0.092903 m2). This is a *replacement-cost* rate, not
a damage rate -- flood damage to a structure is normally a fraction of full
rebuild cost, not a total loss, but no citable India-specific depth-damage
percentage was found to apply here, so the honest choice is to report
exposure at full replacement value and label it as such rather than invent a
damage fraction.

Source: nobroker.in and keralahousedesigns.com construction-cost surveys,
both citing the Kerala PWD's 2025 DSoR adoption (checked 2026-08-27).

Coverage
--------
Only `way["building"]` footprints are queried -- the same scope as
src/osm_drainage.py. Buildings mapped as multipolygon relations (complex
outlines, a minority in this region) are not fetched. OSM building coverage
in Kerala is generally good (an actively mapped state) but volunteer-driven,
so this is a floor on exposure, not a ceiling.

Output
------
    building_area_aligned.tif   building footprint area (m2) per master-grid cell

Run:  python src/building_exposure.py --build
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, DATA_DIR, GEO, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

CACHE = DATA_DIR / "osm" / "ernakulam_buildings.geojson"

#: Kerala PWD-adopted 2025 mid-range urban construction rate. See module
#: docstring for the source and the "exposure, not damage" caveat.
RS_PER_SQFT = 2500.0
SQFT_PER_M2 = 1.0 / 0.092903
RS_PER_M2 = RS_PER_SQFT * SQFT_PER_M2
CR_PER_RS = 1e-7  # 1 crore = 1e7 rupees


def _bbox() -> Tuple[float, float, float, float]:
    """District bbox as (south, west, north, east) for Overpass, unbuffered --
    unlike osm_drainage.py's, buildings outside the district contribute
    nothing to a district exposure figure."""
    min_lon, min_lat, max_lon, max_lat = GEO.district_bbox
    return (min_lat, min_lon, max_lat, max_lon)


def fetch(force: bool = False, timeout: int = 300) -> Path:
    """Download building footprints from Overpass, cached as GeoJSON."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not force:
        LOGGER.info("Using cached %s", CACHE.name)
        return CACHE

    south, west, north, east = _bbox()
    query = (
        f"[out:json][timeout:{timeout}];"
        f'(way["building"]({south},{west},{north},{east}););'
        f"out geom;"
    )
    data = urllib.parse.urlencode({"data": query}).encode()

    last = None
    for endpoint in OVERPASS_ENDPOINTS:
        LOGGER.info("Querying %s ...", endpoint)
        try:
            req = urllib.request.Request(
                endpoint, data=data, headers={"User-Agent": "geoai-flood/1.0"}
            )
            with urllib.request.urlopen(req, timeout=timeout + 60) as r:
                payload = json.load(r)
            break
        except Exception as exc:  # rate limits and gateway timeouts are common
            LOGGER.warning("  %s: %s", type(exc).__name__, str(exc)[:90])
            last = exc
    else:
        raise RuntimeError(f"All Overpass endpoints failed: {last}")

    features = []
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        coords = [[p["lon"], p["lat"]] for p in geom if p.get("lon") is not None]
        # A footprint needs a closed ring of at least 4 points (3 corners +
        # closing point); Overpass repeats the first node at the end for a
        # closed way, so an open way here is not a real footprint.
        if len(coords) < 4 or coords[0] != coords[-1]:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [coords]},
                "properties": {"osm_id": el.get("id")},
            }
        )

    CACHE.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    LOGGER.info("  %d building footprints -> %s", len(features), CACHE.name)
    return CACHE


def build(aligned_dir: Optional[Path] = None, force: bool = False) -> Dict:
    """Rasterise building footprint area onto the master grid."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize

    from feature_stack import grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    path = fetch(force=force)

    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]
    transform = master["transform"]
    cell_area_m2 = abs(transform.a * transform.e)

    gdf = gpd.read_file(path).set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs(master["crs"])
    LOGGER.info("Rasterising %d building footprints at %.0f m", len(gdf), abs(transform.a))

    # A cell either has a mapped footprint touching it or it does not --
    # sub-cell coverage fraction is not attempted, matching the precision
    # osm_drainage.py already uses for its channel mask. At 10 m this
    # slightly overcounts small buildings' contribution (whole-cell area for
    # a partial footprint) and undercounts large ones split across a cell
    # boundary in roughly offsetting ways.
    footprint = rasterize(
        [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
        out_shape=(H, W),
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    LOGGER.info("  %d cells with a mapped building", int(footprint.sum()))

    area_m2 = np.where(footprint, np.float32(cell_area_m2), np.float32(0.0))

    _, district = read_raster("lulc", aligned_dir=aligned_dir)
    out = np.where(district, area_m2, NODATA).astype(np.float32)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")
    out_path = aligned_dir / "building_area_aligned.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    total_m2 = float(area_m2[district].sum())
    summary = {
        "source": "OpenStreetMap contributors (ODbL)",
        "n_buildings": int(len(gdf)),
        "footprint_cells": int(footprint.sum()),
        "total_footprint_m2": total_m2,
        "rs_per_m2": RS_PER_M2,
        "rate_source": (
            "Kerala PWD 2025 mid-range urban construction rate, "
            "Rs 2,500/sqft (Delhi SoR 2021 adopted statewide from 1 Apr 2025)"
        ),
        "total_replacement_value_cr": total_m2 * RS_PER_M2 * CR_PER_RS,
        "raster": out_path.name,
        "caveat": (
            "Replacement-cost EXPOSURE, not a damage prediction: no India- "
            "specific depth-damage function was available to discount this "
            "to expected loss. Only way-tagged footprints are counted, not "
            "multipolygon relations."
        ),
    }
    (aligned_dir / "building_exposure.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "  %s -> %.0f buildings, %.2f km2 footprint, Rs %.0f Cr replacement value",
        out_path.name,
        len(gdf),
        total_m2 / 1e6,
        summary["total_replacement_value_cr"],
    )
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build OSM building exposure raster")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.build:
        build(force=args.force)
    else:
        fetch(force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
