"""
Drainage network from OpenStreetMap.

Why
---
Urban waterlogging is governed by whether water can get away, and nothing in
this model knew where the drains are. The official network is being digitised
by Kochi Corporation under AMRUT but needs a data request. OpenStreetMap has
enough to work with, free and without credentials: over the Kochi bbox alone,
162 ways tagged `waterway=drain`, 492 `canal` and 71 `ditch` — 331 km of
mapped channel at a density of 0.36 km/km².

This is a *proxy*, and an uneven one. OSM coverage in Indian cities is patchy
and volunteer-driven, so absence of a drain in the data is weak evidence of
absence on the ground, and nothing here captures pipe diameter, invert level,
condition or blockage — which is what actually decides whether a street floods.
Treat the derived rasters as "how close is the nearest mapped channel", not as
drainage capacity.

Two rasters are produced on the master grid:

    osm_drain_dist   metres to the nearest mapped drain, ditch or canal
    osm_drain_density  km of channel per km2, in a 1 km neighbourhood

Source: OpenStreetMap contributors, ODbL. Attribution required in any output
that uses these layers.

Run:  python src/osm_drainage.py --build
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, DATA_DIR, GEO, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

CACHE = DATA_DIR / "osm" / "kochi_drainage.geojson"

#: Channel classes to collect. `drain` and `ditch` are constructed drainage;
#: `canal` in Kochi means the tidal canal system that is the city's primary
#: storm-water route.
WATERWAY_CLASSES = ("drain", "ditch", "canal")

#: Neighbourhood radius for the density raster, in metres.
DENSITY_RADIUS_M = 1000.0


def _bbox() -> Tuple[float, float, float, float]:
    """District bbox as (south, west, north, east) for Overpass."""
    min_lon, min_lat, max_lon, max_lat = GEO.district_bbox
    # Widen slightly: a drain just outside the boundary still drains the edge.
    return (min_lat - 0.15, min_lon - 0.10, max_lat + 0.20, max_lon + 0.45)


def fetch(force: bool = False, timeout: int = 300) -> Path:
    """Download the drainage network from Overpass, cached as GeoJSON."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not force:
        LOGGER.info("Using cached %s", CACHE.name)
        return CACHE

    south, west, north, east = _bbox()
    pattern = "|".join(WATERWAY_CLASSES)
    query = (
        f"[out:json][timeout:{timeout}];"
        f'(way["waterway"~"^({pattern})$"]({south},{west},{north},{east}););'
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
    counts: Dict[str, int] = {}
    for el in payload.get("elements", []):
        geom = el.get("geometry") or []
        coords = [[p["lon"], p["lat"]] for p in geom if p.get("lon") is not None]
        if len(coords) < 2:
            continue
        kind = el.get("tags", {}).get("waterway", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords},
            "properties": {"waterway": kind, "osm_id": el.get("id")},
        })

    CACHE.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    LOGGER.info("  %d ways: %s -> %s", len(features), counts, CACHE.name)
    return CACHE


def build(aligned_dir: Optional[Path] = None, force: bool = False) -> Dict:
    """Rasterise distance-to-drain and drainage density onto the master grid."""
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from scipy.ndimage import distance_transform_edt, uniform_filter

    from feature_stack import grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    path = fetch(force=force)

    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]
    transform = master["transform"]
    cell_m = abs(transform.a)

    gdf = gpd.read_file(path).set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs(master["crs"])
    LOGGER.info("Rasterising %d channel ways at %.0f m", len(gdf), cell_m)

    channel = rasterize(
        [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty],
        out_shape=(H, W), transform=transform, fill=0, dtype="uint8",
    ).astype(bool)
    LOGGER.info("  %d cells on a mapped channel", int(channel.sum()))

    if not channel.any():
        raise RuntimeError("No channels rasterised; check the bbox and the cache")

    # Distance to the nearest channel, in metres.
    dist_m = distance_transform_edt(~channel, sampling=(cell_m, cell_m)).astype(np.float32)

    # Channel length per unit area, from the fraction of channel cells in a
    # neighbourhood. One channel cell contributes about one cell-width of
    # length, so km/km2 = fraction / cell_size_km.
    win = max(3, int(round(2 * DENSITY_RADIUS_M / cell_m)) | 1)
    frac = uniform_filter(channel.astype(np.float32), size=win, mode="constant")
    density = (frac / (cell_m / 1000.0)).astype(np.float32)

    # Restrict to the district, matching every other aligned raster.
    _, district = read_raster("lulc", aligned_dir=aligned_dir)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")

    written = {}
    for name, arr in (("osm_drain_dist", dist_m), ("osm_drain_density", density)):
        out = np.where(district, arr, NODATA).astype(np.float32)
        out_path = aligned_dir / f"{name}_aligned.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)
        vals = arr[district]
        LOGGER.info(
            "  %s -> median %.1f, p90 %.1f, max %.1f",
            out_path.name, np.median(vals), np.percentile(vals, 90), vals.max(),
        )
        written[name] = out_path.name

    summary = {
        "source": "OpenStreetMap contributors (ODbL)",
        "waterway_classes": list(WATERWAY_CLASSES),
        "n_ways": int(len(gdf)),
        "channel_cells": int(channel.sum()),
        "density_radius_m": DENSITY_RADIUS_M,
        "rasters": written,
        "caveat": (
            "Volunteer-mapped proxy. Absence of a drain is weak evidence of "
            "absence on the ground, and there is no pipe diameter, invert "
            "level or condition -- which is what decides whether a street "
            "actually floods."
        ),
    }
    (aligned_dir / "osm_drainage.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build OSM drainage rasters")
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
