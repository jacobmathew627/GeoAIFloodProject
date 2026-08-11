"""
Extend the DEM beyond the district so upstream catchments are actually seen.

The problem
-----------
`Ernakulam_Clipped_DEM.tif` is clipped to the district. Water does not respect
that boundary. Measured on the shipped DEM, the largest catchment the flow
network can resolve is **248 km2** -- while the Periyar basin draining into
Ernakulam is roughly 5,398 km2, the Chalakudy 1,704 km2 and the Muvattupuzha
1,554 km2. Every river enters the district across a nodata edge and arrives
carrying nothing, because everything upstream of that edge is off-grid.

That is not a detail. The August 2018 flood in Ernakulam was driven largely by
Periyar discharge and reservoir releases from a catchment an order of
magnitude larger than anything the model can see. The learned susceptibility
absorbs the *pattern* of that event, which is why the maps look reasonable,
but nothing in the system can respond to rain falling in the Western Ghats.

What this module does
---------------------
Mosaics a DEM over the full contributing area from the AWS open terrain tiles
(Mapzen/Terrarium, public, no credentials) and reprojects it to the project
CRS. That part works and is verified: 25,085 km2 of land against the district
DEM's 2,427 km2, with spot checks at Kochi (4 m), Munnar (1,455 m) and Anamudi
(2,465 m against a true 2,695 m at the summit itself).

Terrarium encodes elevation in the RGB channels:

    elevation_m = (R * 256 + G + B / 256) - 32768

Note that it carries ocean *bathymetry*, not zeros -- an Arabian Sea tile
decodes to about -11,600 m -- so sea is masked by elevation band, not by
assuming zeros.

STATUS: not yet wired into the model
------------------------------------
Routing this DEM is unfinished. The flow network built from it does not
reproduce the real river system: probed at points with published catchment
areas, it returns 0 km2 for the Periyar at Aluva (~5,000 km2 expected), 2 km2
at Neriamangalam (~3,300) and 1 km2 for the Chalakudy at Chalakudy town
(~1,400). Its largest accumulation, 1,145 km2, sits on the southern grid edge
rather than on any channel.

The cause is the depression fill. `pluvial.fill_depressions` is a correct
priority-flood and is fine over the 2,400 km2 district, but across 25,000 km2
of Western Ghats it produces very large flats (99th-percentile fill depth
7.7 m, maximum 124.8 m) and the flat-resolution tie-break disperses flow
instead of concentrating it into channels.

Finishing this needs a proper hydrological conditioning step -- breaching
rather than pure filling, plus flat resolution that imposes a gradient toward
the outlet (Garbrecht & Martz). WhiteboxTools (`BreachDepressionsLeastCost`,
`D8FlowAccumulation`) or RichDEM do this correctly and quickly; hand-rolling
it at this scale is not the right use of effort. Until then the model routes
only within the district and cannot see water entering from upstream.

Run:  python src/upstream_dem.py --build
"""
from __future__ import annotations

import argparse
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from config import GEOAI_NEW_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

#: Bounding box covering Ernakulam plus the Periyar, Chalakudy and
#: Muvattupuzha headwaters up to the Western Ghats crest.
#: (min_lon, min_lat, max_lon, max_lat)
BASIN_BBOX = (76.05, 9.40, 77.45, 10.70)

#: Zoom 12 is ~38 m/pixel at this latitude, comparable to the 30 m district DEM.
DEFAULT_ZOOM = 12

OUTPUT_NAME = "Upstream_DEM.tif"

#: Physically valid elevation band for this basin. Anamudi, the highest point
#: in South India, is 2,695 m; the Arabian Sea floor is not terrain.
MIN_VALID_ELEV_M = -50.0
MAX_VALID_ELEV_M = 2800.0


# ──────────────────────────────────────────────
# Web-mercator tile arithmetic
# ──────────────────────────────────────────────
def deg2tile(lon: float, lat: float, z: int) -> Tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x: int, y: int, z: int) -> Tuple[float, float]:
    """North-west corner of a tile."""
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def tile_range(bbox, z: int):
    x0, y0 = deg2tile(bbox[0], bbox[3], z)   # NW
    x1, y1 = deg2tile(bbox[2], bbox[1], z)   # SE
    return range(min(x0, x1), max(x0, x1) + 1), range(min(y0, y1), max(y0, y1) + 1)


# ──────────────────────────────────────────────
# Fetch and decode
# ──────────────────────────────────────────────
def _fetch_tile(args) -> Tuple[int, int, Optional[np.ndarray]]:
    import io
    import urllib.error
    import urllib.request

    x, y, z = args
    url = TILE_URL.format(z=z, x=x, y=y)
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            from PIL import Image

            img = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.float64)
            # Terrarium: metres above sea level, offset by 32768.
            elev = (img[..., 0] * 256.0 + img[..., 1] + img[..., 2] / 256.0) - 32768.0
            return x, y, elev.astype(np.float32)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return x, y, None  # ocean / no coverage
            if attempt == 2:
                raise
        except Exception:
            if attempt == 2:
                raise
    return x, y, None


def fetch_mosaic(bbox=BASIN_BBOX, zoom: int = DEFAULT_ZOOM, workers: int = 8):
    """Download and mosaic terrain tiles. Returns (elevation, transform, crs)."""
    from rasterio.transform import from_bounds

    xs, ys = tile_range(bbox, zoom)
    jobs = [(x, y, zoom) for y in ys for x in xs]
    LOGGER.info(
        "Fetching %d tiles at zoom %d (%d x %d) over %s...",
        len(jobs), zoom, len(xs), len(ys), bbox,
    )

    tile_px = 256
    mosaic = np.full((len(ys) * tile_px, len(xs) * tile_px), np.nan, dtype=np.float32)

    x0, y0 = min(xs), min(ys)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for x, y, elev in pool.map(_fetch_tile, jobs):
            done += 1
            if done % 50 == 0:
                LOGGER.info("  %d/%d", done, len(jobs))
            if elev is None:
                continue
            r = (y - y0) * tile_px
            c = (x - x0) * tile_px
            mosaic[r:r + tile_px, c:c + tile_px] = elev

    # Geographic bounds of the assembled mosaic (web-mercator tile grid, but
    # the tiles are square in mercator so the mosaic is a mercator raster).
    west, north = tile2deg(min(xs), min(ys), zoom)
    east, south = tile2deg(max(xs) + 1, max(ys) + 1, zoom)

    # Convert the lat/lon corners to web mercator for a linear transform.
    from pyproj import Transformer

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    left, top = to_merc.transform(west, north)
    right, bottom = to_merc.transform(east, south)

    transform = from_bounds(left, bottom, right, top, mosaic.shape[1], mosaic.shape[0])
    LOGGER.info(
        "  mosaic %s, elevation range [%.0f, %.0f] m",
        mosaic.shape, np.nanmin(mosaic), np.nanmax(mosaic),
    )
    return mosaic, transform, "EPSG:3857"


# ──────────────────────────────────────────────
# Reproject to the project grid
# ──────────────────────────────────────────────
def build(
    bbox=BASIN_BBOX,
    zoom: int = DEFAULT_ZOOM,
    resolution_m: float = 30.0,
    out_dir: Optional[Path] = None,
) -> Path:
    """Fetch, mosaic and reproject the upstream DEM into the project CRS."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    out_dir = out_dir or GEOAI_NEW_DIR
    mosaic, src_transform, src_crs = fetch_mosaic(bbox, zoom)

    dst_crs = RASTER.target_crs
    left = src_transform.c
    top = src_transform.f
    right = left + src_transform.a * mosaic.shape[1]
    bottom = top + src_transform.e * mosaic.shape[0]

    dst_transform, width, height = calculate_default_transform(
        src_crs, dst_crs, mosaic.shape[1], mosaic.shape[0],
        left=left, bottom=bottom, right=right, top=top,
        resolution=resolution_m,
    )

    LOGGER.info("Reprojecting to %s at %.0f m -> %d x %d", dst_crs, resolution_m, height, width)
    dst = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=mosaic, destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan, dst_nodata=np.nan,
    )

    # Terrarium carries ocean *bathymetry*, not zeros: an Arabian Sea tile
    # decodes to about -11,600 m. Anything below -50 m here is sea, not terrain
    # to route across.
    dst = np.where(dst < MIN_VALID_ELEV_M, np.nan, dst)

    # A handful of single-pixel spikes survive tile decoding (12 cells in the
    # default bbox, up to 10,506 m). The highest ground in South India is
    # Anamudi at 2,695 m, so anything above the ceiling is an artefact, and a
    # spurious peak would divert the flow network around it.
    spikes = np.isfinite(dst) & (dst > MAX_VALID_ELEV_M)
    if spikes.any():
        LOGGER.info("  removing %d elevation spikes above %.0f m",
                    int(spikes.sum()), MAX_VALID_ELEV_M)
        dst = np.where(spikes, np.nan, dst)

    profile = {
        "driver": "GTiff", "height": height, "width": width, "count": 1,
        "dtype": "float32", "crs": dst_crs, "transform": dst_transform,
        "nodata": RASTER.nodata_value, "compress": "lzw", "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    out_path = out_dir / OUTPUT_NAME
    with rasterio.open(out_path, "w", **profile) as f:
        f.write(np.where(np.isfinite(dst), dst, RASTER.nodata_value).astype(np.float32), 1)

    valid = np.isfinite(dst)
    LOGGER.info(
        "Wrote %s: %.0f km2 of land, elevation [%.0f, %.0f] m",
        out_path, valid.sum() * resolution_m ** 2 / 1e6,
        np.nanmin(dst), np.nanmax(dst),
    )
    return out_path


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build the upstream DEM")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--zoom", type=int, default=DEFAULT_ZOOM)
    parser.add_argument("--resolution", type=float, default=30.0)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.build:
        build(zoom=args.zoom, resolution_m=args.resolution)
    else:
        xs, ys = tile_range(BASIN_BBOX, args.zoom)
        LOGGER.info(
            "Would fetch %d tiles at zoom %d (%d x %d)",
            len(xs) * len(ys), args.zoom, len(xs), len(ys),
        )


if __name__ == "__main__":  # pragma: no cover
    main()
