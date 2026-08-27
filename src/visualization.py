"""
Visualization Module for GeoAI Flood Risk Project.

Colormaps, legends, risk statistics, alerts and PNG generation.

Two things to know about the arrays passed in here:
  * Invalid pixels hold RASTER.nodata_value (-9999). Every function masks with
    `> -9000` rather than an equality test, because bilinear downsampling
    perturbs the sentinel.
  * Arrays are usually downsampled for display, so pixel counts are NOT
    full-resolution counts. Any area figure must therefore be derived from the
    array's own affine transform, never from an assumed cell size.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Streamlit/FastAPI are headless; must precede pyplot.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from config import RASTER, VIZ  # noqa: E402

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

# Colour stops as (value, hex). Exposed separately from the colormap object
# so callers can introspect the ramp; `FLOOD_COLORMAP` itself is a matplotlib
# colormap and has no meaningful length.
FLOOD_COLOR_STOPS: List[Tuple[float, str]] = list(VIZ.flood_colors)
FLOOD_COLORMAP = LinearSegmentedColormap.from_list("RiskRamp", FLOOD_COLOR_STOPS)


# RGB form of the same ramp, for the byte-level colormap used by the API.
def _hex_to_rgb(hex_colour: str) -> Tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


FLOOD_COLORMAP_RGB: List[Tuple[float, Tuple[int, int, int]]] = [
    (stop, _hex_to_rgb(hex_colour)) for stop, hex_colour in FLOOD_COLOR_STOPS
]


# ──────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────
def mask_nodata(data: np.ndarray) -> np.ma.MaskedArray:
    """
    Mask nodata however it is expressed.

    Rasters read from disk carry the -9999 sentinel; the live model returns
    NaN. `np.ma.masked_less_equal` silently ignores NaN, because every NaN
    comparison is False -- so masking on the sentinel alone left every NaN
    cell unmasked, coloured black by the colormap's "bad" value, and then
    painted at the layer's alpha. That is what turned the map into a flat grey
    rectangle with the risk zones hidden underneath it.

    Every renderer must go through here.
    """
    return np.ma.masked_invalid(np.ma.masked_less_equal(data, -9000))


def pixel_area_km2_from_transform(transform: Any) -> Optional[float]:
    """
    Area of one pixel in km2, taken from an affine transform.

    Returns None when no transform is available, so callers can fall back
    explicitly rather than silently using a wrong constant.
    """
    if transform is None:
        return None
    try:
        return abs(transform.a * transform.e) / 1e6
    except AttributeError:
        return None


def valid_area_km2(data: np.ndarray, transform: Any) -> Optional[float]:
    """Total mapped area of the valid pixels in `data`."""
    px = pixel_area_km2_from_transform(transform)
    if px is None:
        return None
    return float((data > -9000).sum()) * px


# ──────────────────────────────────────────────
# Colormaps
# ──────────────────────────────────────────────
def apply_colormap(prob: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W) probability array to (H, W, 4) RGBA uint8.

    Nodata pixels become fully transparent instead of being coloured as if
    they were probability zero.
    """
    rgba = np.zeros((*prob.shape, 4), dtype=np.uint8)
    valid = prob > -9000

    for i in range(len(FLOOD_COLORMAP_RGB) - 1):
        v0, c0 = FLOOD_COLORMAP_RGB[i]
        v1, c1 = FLOOD_COLORMAP_RGB[i + 1]
        # Upper bound is exclusive except on the final segment, so a pixel is
        # never coloured twice at a stop boundary.
        upper = prob <= v1 if i == len(FLOOD_COLORMAP_RGB) - 2 else prob < v1
        mask = valid & (prob >= v0) & upper
        if not mask.any():
            continue
        t = (prob[mask] - v0) / (v1 - v0 + 1e-9)
        for ch in range(3):
            rgba[mask, ch] = (c0[ch] * (1 - t) + c1[ch] * t).astype(np.uint8)
        rgba[mask, 3] = 200

    return rgba


def prob_to_png_b64(prob: np.ndarray, max_dim: int = 1024) -> str:
    """Downscale a probability array, colourise it, return a base64 PNG."""
    h, w = prob.shape
    scale = min(max_dim / max(h, w), 1.0)

    if scale < 1.0:
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        # Resize the mask with NEAREST and the values with BILINEAR, then
        # re-apply the mask. Resizing -9999 sentinels bilinearly would smear
        # them into the valid range and paint a halo around every hole.
        valid = (prob > -9000).astype(np.uint8)
        valid_small = np.array(
            Image.fromarray(valid, mode="L").resize((new_w, new_h), Image.Resampling.NEAREST)
        )
        filled = np.where(prob > -9000, prob, 0.0).astype(np.float32)
        values_small = np.array(
            Image.fromarray(filled, mode="F").resize((new_w, new_h), Image.Resampling.BILINEAR)
        )
        prob_small = np.where(valid_small > 0, values_small, NODATA).astype(np.float32)
    else:
        prob_small = prob.astype(np.float32)

    img = Image.fromarray(apply_colormap(prob_small), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────
#: Risk bands, from lowest to highest, as (name, lower-bound attribute).
#: A band's upper bound is the next band's lower bound.
#:
#: All four configured thresholds are used. The previous scheme defined four
#: thresholds but only ever applied three of them: the class it labelled
#: "critical" was everything above `RISK.high`, and `RISK.critical` was dead
#: configuration -- so the most severe band the config described was never
#: shown anywhere.
RISK_BANDS = [
    ("safe", None),
    ("moderate", "safe"),
    ("high", "moderate"),
    ("severe", "high"),
    ("critical", "critical"),
]


def risk_band_masks(valid: np.ndarray, risk_cfg) -> Dict[str, np.ndarray]:
    """Boolean mask per risk band. The masks partition `valid` exactly."""
    edges = [getattr(risk_cfg, attr) if attr else None for _, attr in RISK_BANDS]
    masks = {}
    for i, (name, _) in enumerate(RISK_BANDS):
        lower = edges[i]
        upper = edges[i + 1] if i + 1 < len(edges) else None
        m = np.ones(valid.shape, dtype=bool)
        if lower is not None:
            m &= valid >= lower
        if upper is not None:
            m &= valid < upper
        masks[name] = m
    return masks


def compute_risk_stats(data: np.ndarray, risk_cfg, transform: Any = None) -> Dict[str, float]:
    """
    Risk-class breakdown of a probability map.

    Percentages always sum to 100 across the bands. When a transform is
    supplied, absolute areas in km2 are included as well.
    """
    valid = data[data > -9000]
    names = [n for n, _ in RISK_BANDS]
    if valid.size == 0:
        return {f"{n}_pct": 0.0 for n in names} | {"mean_prob": 0.0, "max_prob": 0.0}

    masks = risk_band_masks(valid, risk_cfg)
    stats = {f"{n}_pct": round(float(masks[n].mean() * 100), 2) for n in names}
    stats["mean_prob"] = round(float(valid.mean()), 4)
    stats["max_prob"] = round(float(valid.max()), 4)

    px_km2 = pixel_area_km2_from_transform(transform)
    if px_km2 is not None:
        # Rounded to 4 dp, not 2: at 2 dp any area below 0.005 km2 collapses to
        # zero, which silently erases small critical zones -- exactly the ones
        # a planner most needs to see.
        for n in names:
            stats[f"{n}_km2"] = round(float(masks[n].sum()) * px_km2, 4)
        stats["mapped_area_km2"] = round(float(valid.size) * px_km2, 4)
        # Expected flooded area: the integral of the probability surface. This
        # is the quantity the model is calibrated against, and it does not
        # depend on where the band edges are drawn.
        stats["expected_flooded_km2"] = round(float(valid.sum()) * px_km2, 4)

    return stats


def create_legend_html(title: str, items: List[Tuple[str, str]]) -> str:
    """HTML legend block for a Folium map."""
    html = (
        '<div style="position: fixed; bottom: 50px; left: 50px; width: 230px; '
        "height: auto; border: 2px solid grey; z-index: 9999; font-size: 14px; "
        'background-color: white; opacity: 0.9; padding: 10px;">'
        f"<b>{title}</b><br>"
    )
    for label, color in items:
        html += (
            f'<i style="background:{color};width:10px;height:10px;'
            f'display:inline-block;margin-right:5px;"></i>{label}<br>'
        )
    return html + "</div>"


# ──────────────────────────────────────────────
# Alerts
# ──────────────────────────────────────────────
def create_alert_message(
    data: np.ndarray,
    rainfall: float,
    geo,
    risk_cfg,
    transform: Any = None,
    population: Optional[np.ndarray] = None,
    building_area: Optional[np.ndarray] = None,
) -> Optional[str]:
    """
    Operational alert text derived from the hazard map.

    Areas come from the raster's own geometry. Population and building value,
    when WorldPop/OSM grids matching `data`'s shape are supplied, are real
    spatial sums over the critical-risk cells rather than estimates -- see
    src/population.py and src/building_exposure.py. Without them (or on a
    shape mismatch), both fall back to the previous district-average
    planning estimate.

    The building figure is *exposure* (replacement value of what is mapped
    in the critical zone), not a damage prediction -- no India-specific
    depth-damage function was available to discount it to expected loss, so
    it is not called "damage" and should not be read as one.
    """
    valid = data[data > -9000]
    if valid.size == 0:
        return None

    critical_mask = data >= risk_cfg.critical
    crit_pct = float((valid >= risk_cfg.critical).mean() * 100)
    risk_pct = float((valid >= risk_cfg.moderate).mean() * 100)

    px_km2 = pixel_area_km2_from_transform(transform)
    if px_km2 is None:
        # No geometry: fall back to the configured district area, and say so.
        mapped_km2 = geo.district_area_km2
        area_note = " (area scaled to nominal district extent)"
    else:
        mapped_km2 = float(valid.size) * px_km2
        area_note = ""

    critical_km2 = mapped_km2 * crit_pct / 100.0
    elevated_km2 = mapped_km2 * risk_pct / 100.0

    if population is not None and population.shape == data.shape:
        exposed = population[critical_mask & np.isfinite(population)]
        est_pop = int(exposed.sum())
        pop_note = "WorldPop 2020, summed within the critical-risk area"
    else:
        est_pop = min(
            int(critical_km2 * geo.pop_density * risk_cfg.residential_fraction),
            geo.population,
        )
        pop_note = "district-average density x residential fraction, planning estimate"

    if building_area is not None and building_area.shape == data.shape:
        from building_exposure import RS_PER_M2

        exposed_m2 = float(building_area[critical_mask & np.isfinite(building_area)].sum())
        est_damage_cr = exposed_m2 * RS_PER_M2 * 1e-7
        damage_note = (
            "OSM building footprints x Kerala PWD 2025 rate, replacement value exposed, "
            "not a damage prediction"
        )
    else:
        est_damage_cr = critical_km2 * risk_cfg.damage_per_km2_crores
        damage_note = f"at Rs {risk_cfg.damage_per_km2_crores:.0f} Cr/km2, planning estimate"

    # Trigger levels are fractions of the mapped area, calibrated against the
    # reference event. The previous version compared these percentages to 15%
    # and 25%, which were carried over from the uncalibrated score; on the
    # corrected probability scale nothing ever reached them, so the 2018
    # catastrophe itself reported "monitoring active".
    if crit_pct >= risk_cfg.critical_area_fraction_alert * 100:
        return (
            f"**CRITICAL FLOOD ALERT - Ernakulam District** | Rainfall: {rainfall:.0f} mm\n\n"
            f"**Critical risk area:** {critical_km2:,.1f} km2 "
            f"({crit_pct:.2f}% of {mapped_km2:,.0f} km2 mapped){area_note}\n\n"
            f"**Elevated risk area:** {elevated_km2:,.1f} km2 ({risk_pct:.2f}%)\n\n"
            f"**Estimated population exposed:** ~{est_pop:,} ({pop_note})\n\n"
            f"**Building value exposed:** ~Rs {est_damage_cr:,.0f} Cr ({damage_note})"
        )
    if risk_pct >= risk_cfg.elevated_area_fraction_warning * 100:
        return (
            f"**FLOOD WARNING - Ernakulam** | Rainfall: {rainfall:.0f} mm\n\n"
            f"**Elevated-risk area:** {elevated_km2:,.1f} km2 "
            f"({risk_pct:.2f}% of {mapped_km2:,.0f} km2 mapped){area_note}\n\n"
            f"**Critical-class area:** {critical_km2:,.1f} km2\n\n"
            f"**Estimated population exposed:** ~{est_pop:,} ({pop_note})"
        )
    return (
        f"**MONITORING ACTIVE** | Rainfall: {rainfall:.0f} mm | "
        f"{risk_pct:.2f}% of the mapped district above the elevated-risk threshold "
        f"({elevated_km2:,.1f} km2)"
    )


# ──────────────────────────────────────────────
# Flood probability rendering
# ──────────────────────────────────────────────
def create_flood_visualization(
    data: np.ndarray, viz, risk_cfg
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """RGBA overlay for a flood probability / hazard map."""
    masked = mask_nodata(data)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    image_rgba = FLOOD_COLORMAP(norm(masked.filled(0.0)))

    hidden = np.ma.getmaskarray(masked)
    image_rgba[..., 3] = np.where(hidden, 0.0, 0.6)

    # Legend labels are generated from the configured thresholds so they can
    # never drift out of sync with the classification the stats use. The
    # thresholds are not round numbers (they come off the precision-recall
    # curve), so they are shown to one decimal place.
    legend_items = [
        (f"Critical (>={risk_cfg.critical * 100:.1f}%)", "#a50026"),
        (f"Severe ({risk_cfg.high * 100:.1f}-{risk_cfg.critical * 100:.1f}%)", "#d73027"),
        (f"High ({risk_cfg.moderate * 100:.1f}-{risk_cfg.high * 100:.1f}%)", "#fdae61"),
        (f"Moderate ({risk_cfg.safe * 100:.1f}-{risk_cfg.moderate * 100:.1f}%)", "#fee08b"),
        (f"Safe (<{risk_cfg.safe * 100:.1f}%)", "#1a9850"),
    ]
    return image_rgba, legend_items


# ──────────────────────────────────────────────
# Static layer rendering
# ──────────────────────────────────────────────
_CMAP_SPEC = {
    "DEM": ("terrain", None),
    "Slope": ("magma", None),
    "TWI": ("YlGnBu", None),
    "SPI": ("inferno", None),
    "HAND": ("GnBu_r", (0.0, 20.0)),
    "TPI": ("RdBu_r", "symmetric"),
    "Distance to Water": ("Blues_r", (0.0, 3000.0)),
    "Distance to Built-up": ("YlOrRd_r", (0.0, 2000.0)),
    "NDVI (Vegetation)": ("RdYlGn", (-0.2, 0.8)),
    "NDWI (Water)": ("RdBu", (-0.5, 0.5)),
    "Flow Accumulation": ("Blues", "log"),
}


def create_static_visualization(
    data: np.ndarray, layer_type: str, viz
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """RGBA overlay for a static conditioning-factor layer."""
    masked = mask_nodata(data)

    if layer_type == "LULC":
        return _lulc_visualization(masked)
    if layer_type == "Sentinel-1 Ground Truth":
        return _binary_visualization(
            masked, "#0000cc", "#d9f0a3", "Flooded (SAR)", "Dry / not detected"
        )
    if layer_type == "Urban Mask":
        return _binary_visualization(
            masked, "#e82e2e", "#f0f5d6", "Built-up / urban", "Rural / non-urban"
        )

    cmap_name, norm_spec = _CMAP_SPEC.get(layer_type, ("viridis", None))
    cmap = plt.get_cmap(cmap_name)
    valid = masked.compressed()

    if norm_spec == "log":
        values = np.log1p(np.clip(masked.filled(0.0), 0, None))
        vmax = float(np.percentile(np.log1p(valid[valid > 0]), 99)) if np.any(valid > 0) else 1.0
        norm = plt.Normalize(vmin=0.0, vmax=max(vmax, 1e-6))
    elif norm_spec == "symmetric":
        values = masked.filled(0.0)
        if valid.size:
            limit = float(max(abs(np.percentile(valid, 2)), abs(np.percentile(valid, 98))))
        else:
            limit = 1.0
        norm = plt.Normalize(vmin=-limit, vmax=limit)
    elif isinstance(norm_spec, tuple):
        vmin, vmax = norm_spec
        values = masked.filled(vmin)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    else:
        values = masked.filled(0.0)
        if valid.size:
            norm = plt.Normalize(
                vmin=float(np.percentile(valid, 2)), vmax=float(np.percentile(valid, 98))
            )
        else:
            norm = plt.Normalize(vmin=0.0, vmax=1.0)

    image_rgba = cmap(norm(values))
    # getmaskarray, not .mask: a masked array with nothing masked returns the
    # scalar False, which would broadcast the alpha channel to a constant.
    image_rgba[..., 3] = np.where(np.ma.getmaskarray(masked), 0.0, 0.7)

    if valid.size:
        legend_items = [
            (f"{layer_type} min {valid.min():.2f}", "#440154"),
            (f"{layer_type} max {valid.max():.2f}", "#fde725"),
        ]
    else:
        legend_items = [(layer_type, "#808080")]
    return image_rgba, legend_items


#: Colours for the conformal prediction-set codes (see conformal.SET_*).
CONFORMAL_COLORS = {
    0: ("#8c00a0", "Indeterminate - neither label admitted"),
    1: ("#1a9850", "Confidently not flood-prone"),
    2: ("#fdae61", "Ambiguous - both labels admitted"),
    3: ("#d73027", "Confidently flood-prone"),
}


def create_conformal_visualization(
    data: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Render the conformal decision raster.

    Unlike the probability map, this layer is allowed to say "I don't know":
    the ambiguous and atypical classes are the honest output where the model
    cannot support a decision at the requested confidence.
    """
    masked = mask_nodata(data)
    hidden = np.ma.getmaskarray(masked)
    codes = np.round(masked.filled(-1)).astype(np.int32)

    image_rgba = np.zeros((*codes.shape, 4), dtype=np.float32)
    known = np.zeros(codes.shape, dtype=bool)
    for code, (hex_colour, _) in CONFORMAL_COLORS.items():
        pixels = codes == code
        h = hex_colour.lstrip("#")
        image_rgba[pixels, :3] = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
        known |= pixels

    image_rgba[..., 3] = np.where(hidden | ~known, 0.0, 0.75)

    legend_items = [(label, colour) for code, (colour, label) in sorted(CONFORMAL_COLORS.items())]
    return image_rgba, legend_items


#: Ramp for the waterlogging index. Deliberately a different hue family from
#: the flood-probability ramp: the two layers mean different things and one is
#: calibrated while the other is not, so they must not read as the same scale.
PLUVIAL_COLORMAP = LinearSegmentedColormap.from_list(
    "PluvialRamp",
    [(0.00, "#f7f4f9"), (0.25, "#d0d1e6"), (0.50, "#a6bddb"), (0.75, "#3690c0"), (1.00, "#034e7b")],
)


def create_pluvial_visualization(
    data: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Render the rain-driven waterlogging index.

    Unlike the flood probability, this is a *relative* 0-1 ranking anchored to
    the reference storm, so the legend is labelled in relative terms and
    carries no percentages -- putting "%" on an unvalidated index would invite
    it to be read as a probability.
    """
    masked = mask_nodata(data)
    hidden = np.ma.getmaskarray(masked)

    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    image_rgba = PLUVIAL_COLORMAP(norm(masked.filled(0.0)))
    image_rgba[..., 3] = np.where(hidden, 0.0, 0.7)

    legend_items = [
        ("Very high pressure", "#034e7b"),
        ("High", "#3690c0"),
        ("Moderate", "#a6bddb"),
        ("Low", "#d0d1e6"),
        ("Negligible", "#f7f4f9"),
    ]
    return image_rgba, legend_items


def _lulc_visualization(masked) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Categorical rendering for the LULC layer."""
    from config import LULC_CLASS_NAMES

    hidden = np.ma.getmaskarray(masked)
    codes = np.round(masked.filled(-9999)).astype(np.int32)

    image_rgba = np.zeros((*codes.shape, 4), dtype=np.float32)
    known = np.zeros(codes.shape, dtype=bool)
    for value, colour in VIZ.lulc_colors.items():
        pixels = codes == int(value)
        image_rgba[pixels] = np.array(colour, dtype=np.float32) / 255.0
        known |= pixels

    image_rgba[..., 3] = np.where(hidden | ~known, 0.0, 0.85)

    legend_items = [
        (
            LULC_CLASS_NAMES.get(code, f"Class {code}"),
            "#{:02x}{:02x}{:02x}".format(*VIZ.lulc_colors[code][:3]),
        )
        for code in sorted(VIZ.lulc_colors)
    ]
    return image_rgba, legend_items


def _binary_visualization(
    masked, true_hex: str, false_hex: str, true_label: str, false_label: str
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Rendering for a 0/1 layer, with the negative class left transparent."""
    hidden = np.ma.getmaskarray(masked)
    values = masked.filled(0.0)
    positive = values > 0.5

    def to_rgb(hex_colour: str):
        h = hex_colour.lstrip("#")
        return [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]

    image_rgba = np.zeros((*values.shape, 4), dtype=np.float32)
    image_rgba[positive, :3] = to_rgb(true_hex)
    image_rgba[~positive, :3] = to_rgb(false_hex)
    image_rgba[..., 3] = np.where(hidden, 0.0, np.where(positive, 0.85, 0.15))

    return image_rgba, [(true_label, true_hex), (false_label, false_hex)]
