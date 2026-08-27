"""
Hydrologic soil group from SoilGrids, replacing the single-group assumption.

The assumption being removed
----------------------------
`HydrologyConfig.curve_numbers` carries one curve number per land cover class,
and its own comment states the basis: "AMC II, hydrologic soil group C ...
Kerala's uplands are laterite (HSG C) and the coastal strip is alluvium
(HSG B); C is the conservative single-group choice."

That single choice is not a small approximation. Curve number varies strongly
with soil group -- built-up land runs 75 on group A against 90 on group D -- and
runoff depth is nonlinear in CN, so assuming C everywhere overstates runoff on
the sandy coastal strip and understates it on the clay-rich pockets. It also
means the rainfall slider responds identically in places whose soils behave
quite differently.

Data
----
ISRIC SoilGrids v2.0, 250 m, CC-BY 4.0, no credentials. Sand and clay mass
fractions are read straight out of the published cloud-optimised VRTs over the
district window via GDAL's /vsicurl, so nothing global is downloaded.

    https://files.isric.org/soilgrids/latest/data/

Values are in g/kg (i.e. per mille), not percent -- SoilGrids stores integers to
avoid floats, so 412 means 41.2%. Getting this wrong is silent and costs two
soil groups: both fractions come through ten times too large, and because any
clay value above 40 triggers the clay classes, the result skews to the clay
corner. A loam (group B) reads as a sandy clay (group D), which overstates
runoff everywhere.

Method and its limits
---------------------
The true NRCS hydrologic soil group is defined by saturated hydraulic
conductivity, depth to a restrictive layer, and depth to the water table.
SoilGrids publishes none of those directly, so this module uses the standard
fallback: classify USDA texture from the sand/silt/clay fractions, then map
texture to group.

That mapping is a documented approximation and is stated as one. It is right
about the first-order control -- sandy soils infiltrate, clays do not -- and
wrong wherever a shallow restrictive layer or a high water table dominates.
In Ernakulam the second case is real: the coastal strip has a water table
within a metre or two of the surface for much of the monsoon, which makes a
sandy soil behave far worse than its texture suggests. `WATERLOGGED_PROMOTION`
below handles that explicitly rather than leaving it as a silent error.

Output
------
    soil_hsg_aligned.tif   1 = A, 2 = B, 3 = C, 4 = D, on the master grid

STATUS: built, tested, and deliberately NOT wired into the model
------------------------------------------------------------------
`curve_number_from_lulc` accepts an optional `hsg` argument, but nothing in
feature_stack.py, hazard.py or live_model.py passes it. Measured on the actual
build, the reason is that this raster does not deliver what its own docstring
predicted:

    HSG A:        0 px (  0.0%)
    HSG B:   46,601 px (  0.2%)
    HSG C:  240,925 px (  1.3%)
    HSG D: 18,898,421 px ( 98.5%)
    Shannon entropy: 0.12 bits (max 2.0 for four groups equally represented)

The median district pixel (sand 39.0%, clay 31.3%) classifies as clay loam,
group D, and 250 m SoilGrids resolution does not resolve enough texture
variation across Ernakulam to move most of it out of that class. That may well
be a correct reading of the district's soils rather than a modelling error --
this is a deltaic, backwater-fed landscape, and pure sand is plausibly confined
to a beach strip too narrow for a 250 m pixel -- but it means the raster is not
supplying spatial heterogeneity. It is close to a uniform curve-number bump:
switching every LULC class from the group-C column to the actual per-pixel
column raises CN by a median +2.16 (mean +2.52), which raises runoff depth by
roughly +12.7% at 100 mm and +3.6% at 443 mm district-wide.

That is not a free improvement. The prior offset (fit_prior_offset), the risk
band edges (risk_thresholds.py) and the rainfall sensitivity beta (fit_beta.py)
are all calibrated against the current group-C-uniform curve numbers. Adopting
this raster would shift expected flooded area at every rainfall depth and
require re-running all three -- a decision that changes headline numbers project
-wide, so it should be made explicitly rather than fall out of wiring one
optional argument. Until then this module stays available and tested but
inert: build it, inspect it, but do not pass its output to
curve_number_from_lulc in the production pipeline without redoing the
calibration chain.

Run:  python src/soil_hsg.py --build
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import ALIGNED_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

SOILGRIDS_ROOT = "https://files.isric.org/soilgrids/latest/data"

#: Depth intervals to average over. The curve number describes the near-surface
#: response that governs storm runoff, so the top 15 cm is the relevant zone;
#: deeper horizons matter for baseflow, which this model does not represent.
DEPTHS = ("0-5cm", "5-15cm")

#: SoilGrids stores mass fractions in g/kg. Divide by 10 for percent.
G_PER_KG_TO_PERCENT = 10.0

#: Hydrologic soil groups, as raster codes.
HSG_CODES: Dict[str, int] = {"A": 1, "B": 2, "C": 3, "D": 4}
HSG_NAMES: Dict[int, str] = {v: k for k, v in HSG_CODES.items()}

#: USDA texture class -> hydrologic soil group.
#:
#: This follows the widely used texture-to-group correspondence (NRCS NEH-630
#: ch. 7; reproduced in most SCS-CN texts). It is an approximation of a
#: definition that is really about conductivity and restrictive layers, and the
#: module docstring says so.
TEXTURE_TO_HSG: Dict[str, str] = {
    "sand": "A",
    "loamy sand": "A",
    "sandy loam": "B",
    "loam": "B",
    "silt loam": "B",
    "silt": "B",
    "sandy clay loam": "C",
    "clay loam": "D",
    "silty clay loam": "D",
    "sandy clay": "D",
    "silty clay": "D",
    "clay": "D",
}

#: Elevation below which a sandy soil is demoted one group.
#:
#: Texture alone says the coastal strip drains freely. It does not: the water
#: table sits within a metre or two of the surface through the monsoon, so the
#: profile is already saturated when the storm arrives and infiltration
#: capacity is far below what the sand fraction implies. NRCS handles this with
#: the "dual group" convention (A/D, B/D) for soils that would be well-drained
#: if drained but are not. Below this elevation, groups A and B are demoted one
#: step toward D.
WATERLOGGED_PROMOTION = {"elevation_m": 5.0, "demote": {"A": "B", "B": "C"}}


def usda_texture(sand_pct: np.ndarray, clay_pct: np.ndarray) -> np.ndarray:
    """
    USDA texture class per pixel, as an integer index into TEXTURE_ORDER.

    Boundaries follow the USDA soil texture triangle. Silt is the remainder,
    which is how SoilGrids is meant to be used: the three fractions are
    published independently and do not always sum to exactly 1000.
    """
    sand = np.asarray(sand_pct, dtype=np.float32)
    clay = np.asarray(clay_pct, dtype=np.float32)
    silt = np.clip(100.0 - sand - clay, 0.0, 100.0)

    out = np.full(sand.shape, -1, dtype=np.int8)

    def put(mask, name):
        idx = TEXTURE_ORDER.index(name)
        out[(out < 0) & mask] = idx

    # Order matters: the triangle's classes overlap in raw inequalities, so the
    # tighter classes are assigned first and `out < 0` keeps them.
    put((clay >= 40) & (silt >= 40), "silty clay")
    put((clay >= 40) & (sand <= 45) & (silt < 40), "clay")
    put((clay >= 35) & (sand > 45), "sandy clay")
    put((clay >= 27) & (clay < 40) & (sand <= 20), "silty clay loam")
    put((clay >= 27) & (clay < 40) & (sand > 20) & (sand <= 45), "clay loam")
    put((clay >= 20) & (clay < 35) & (silt < 28) & (sand > 45), "sandy clay loam")
    put((silt >= 80) & (clay < 12), "silt")
    put((silt >= 50) & (clay < 27), "silt loam")
    put((clay >= 7) & (clay < 27) & (silt >= 28) & (silt < 50) & (sand <= 52), "loam")
    put((sand >= 85) & (clay < 10), "sand")
    put((sand >= 70) & (clay < 15), "loamy sand")
    put(sand > 52, "sandy loam")
    # Anything left is loam-like.
    put(np.isfinite(sand) & np.isfinite(clay), "loam")

    return out


TEXTURE_ORDER = (
    "sand", "loamy sand", "sandy loam", "loam", "silt loam", "silt",
    "sandy clay loam", "clay loam", "silty clay loam", "sandy clay",
    "silty clay", "clay",
)


def hsg_from_texture(texture_idx: np.ndarray) -> np.ndarray:
    """Map texture indices to HSG codes (1-4); -1 stays -1."""
    lut = np.full(len(TEXTURE_ORDER), 0, dtype=np.int8)
    for i, name in enumerate(TEXTURE_ORDER):
        lut[i] = HSG_CODES[TEXTURE_TO_HSG[name]]
    out = np.full(texture_idx.shape, 0, dtype=np.int8)
    valid = texture_idx >= 0
    out[valid] = lut[texture_idx[valid]]
    return out


def apply_waterlogged_demotion(
    hsg: np.ndarray,
    elevation_m: np.ndarray,
    rule: Optional[Dict] = None,
) -> np.ndarray:
    """
    Demote well-drained groups where the water table sits near the surface.

    Texture says the coastal sand drains; the monsoon water table says it does
    not. Applied as an explicit, named adjustment so it shows up in review
    rather than hiding inside the texture table.
    """
    rule = rule or WATERLOGGED_PROMOTION
    out = hsg.copy()
    low = np.isfinite(elevation_m) & (elevation_m <= rule["elevation_m"])
    for src, dst in rule["demote"].items():
        out[low & (hsg == HSG_CODES[src])] = HSG_CODES[dst]
    return out


def _vrt_url(prop: str, depth: str) -> str:
    return f"{SOILGRIDS_ROOT}/{prop}/{prop}_{depth}_mean.vrt"


def fetch_property(
    prop: str,
    aligned_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Read one SoilGrids property over the master-grid window, in percent.

    Averaged across DEPTHS. Reprojected straight onto the master grid by GDAL
    so no intermediate file is needed.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    from feature_stack import grid_profile

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)
    H, W = master["height"], master["width"]

    stack = []
    for depth in DEPTHS:
        url = _vrt_url(prop, depth)
        LOGGER.info("  reading %s %s ...", prop, depth)
        dst = np.full((H, W), np.nan, dtype=np.float32)
        with rasterio.open(f"/vsicurl/{url}") as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=master["transform"],
                dst_crs=master["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
        stack.append(dst)

    mean = np.nanmean(np.stack(stack), axis=0)
    # g/kg -> percent. Skipping this silently drives every texture to "sand".
    pct = mean / G_PER_KG_TO_PERCENT
    valid = np.isfinite(pct)
    if valid.any():
        LOGGER.info(
            "    %s: median %.1f%%, range %.1f-%.1f%%",
            prop, np.median(pct[valid]), pct[valid].min(), pct[valid].max(),
        )
    return pct


def build(aligned_dir: Optional[Path] = None) -> Dict:
    """Fetch SoilGrids, classify, and write the HSG raster."""
    import rasterio

    from feature_stack import grid_profile, read_raster

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)

    LOGGER.info("Fetching SoilGrids sand and clay over the district window")
    sand = fetch_property("sand", aligned_dir)
    clay = fetch_property("clay", aligned_dir)

    LOGGER.info("Classifying USDA texture and mapping to hydrologic soil group")
    texture = usda_texture(sand, clay)
    hsg = hsg_from_texture(texture)

    dem, dem_valid = read_raster("dem", aligned_dir=aligned_dir)
    dem = np.where(dem_valid, dem, np.nan)
    before = hsg.copy()
    hsg = apply_waterlogged_demotion(hsg, dem)
    n_demoted = int((hsg != before).sum())
    LOGGER.info(
        "  demoted %d px below %.0f m for a near-surface water table",
        n_demoted, WATERLOGGED_PROMOTION["elevation_m"],
    )

    _, district = read_raster("lulc", aligned_dir=aligned_dir)
    out = np.where(district & (hsg > 0), hsg, NODATA).astype(np.float32)

    profile = dict(master)
    profile.update(dtype="float32", count=1, nodata=NODATA, compress="lzw")
    out_path = aligned_dir / "soil_hsg_aligned.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)

    counts = {}
    in_district = district & (hsg > 0)
    total = int(in_district.sum())
    for code, name in HSG_NAMES.items():
        n = int((in_district & (hsg == code)).sum())
        counts[name] = {"pixels": n, "share": round(n / max(total, 1), 4)}
        LOGGER.info("  HSG %s: %8d px (%5.1f%%)", name, n, 100.0 * n / max(total, 1))

    summary = {
        "source": "ISRIC SoilGrids v2.0 (CC-BY 4.0)",
        "depths": list(DEPTHS),
        "raster": out_path.name,
        "counts": counts,
        "demoted_px": n_demoted,
        "waterlogged_rule": WATERLOGGED_PROMOTION,
        "caveat": (
            "True NRCS hydrologic soil group is defined by saturated "
            "conductivity and depth to a restrictive layer, which SoilGrids "
            "does not publish. This is a texture-based approximation, plus an "
            "explicit demotion below 5 m for the monsoon water table."
        ),
    }
    (aligned_dir / "soil_hsg.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Wrote %s", out_path.name)
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build the hydrologic soil group raster")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.build:
        build()
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
