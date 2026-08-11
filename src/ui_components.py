"""
UI Components for the GeoAI Flood Risk Dashboard.

Sidebar controls, analytics tabs, place search and map-click readout.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

LOGGER = logging.getLogger("geoai_flood")

LAYER_CHOICES = [
    "Flood Probability",
    "Conformal Confidence",
    "DEM",
    "Slope",
    "LULC",
    "TWI",
    "SPI",
    "HAND",
    "TPI",
    "Distance to Water",
    "Distance to Built-up",
    "NDVI (Vegetation)",
    "NDWI (Water)",
    "Sentinel-1 Ground Truth",
    "Flow Accumulation",
    "Urban Mask",
]


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar(rainfall_cfg, risk_cfg, known_places) -> Dict[str, Any]:
    """Render sidebar controls and return the user's selections."""
    st.sidebar.header("Controls")

    advanced_mode = st.sidebar.checkbox("Advanced analytics", value=False)
    layer_type = st.sidebar.radio("Select layer", LAYER_CHOICES)

    if advanced_mode:
        st.sidebar.subheader("Model")
        st.sidebar.info(
            "Calibrated gradient-boosted susceptibility trained on the "
            "August 2018 Sentinel-1 flood inventory, combined with SCS Curve "
            "Number runoff for the rainfall response."
        )

    st.sidebar.subheader("Rainfall conditions")
    use_live = st.sidebar.checkbox("Use live weather API", value=False)

    rainfall = None
    if use_live:
        rainfall = _fetch_live_rainfall(rainfall_cfg)
    if rainfall is None:
        rainfall = float(
            st.sidebar.slider(
                "Rainfall intensity (mm, 24h)", 0, rainfall_cfg.max_slider, 150
            )
        )

    is_2018 = st.sidebar.checkbox("Simulate the 2018 flood event", value=False)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Reference event for calibration: {rainfall_cfg.reference_event_mm:.0f} mm "
        "(August 2018 Kerala flood)"
    )

    return {
        "layer_type": layer_type,
        "advanced_mode": advanced_mode,
        "rainfall": rainfall,
        "is_2018": is_2018,
    }


def _fetch_live_rainfall(rainfall_cfg) -> Optional[float]:
    """
    Fetch the next 24 h forecast precipitation total.

    Returns None on any failure so the caller falls back to the slider; the
    previous version raised the slider inside the exception handler, which
    meant the widget key changed between reruns.
    """
    try:
        import requests

        params = {
            "latitude": rainfall_cfg.weather_params["latitude"],
            "longitude": rainfall_cfg.weather_params["longitude"],
            "hourly": "precipitation",
            "forecast_days": 2,
        }
        response = requests.get(rainfall_cfg.live_weather_url, params=params, timeout=5)
        response.raise_for_status()
        hourly = response.json()["hourly"]["precipitation"][:24]
        total = float(sum(v for v in hourly if v is not None))
        st.sidebar.success(f"Live 24 h forecast: {total:.1f} mm")
        return total
    except Exception as exc:
        LOGGER.warning("Live weather fetch failed: %s", exc)
        st.sidebar.error("Live weather unavailable; using the slider.")
        return None


# ──────────────────────────────────────────────
# Advanced analytics
# ──────────────────────────────────────────────
def render_advanced_analytics(
    data: np.ndarray,
    maps: dict,
    rainfall: float,
    geo,
    risk_cfg,
    viz,
    transform: Any = None,
) -> None:
    """Analytics tabs for the current hazard map."""
    from visualization import compute_risk_stats, pixel_area_km2_from_transform

    st.markdown("---")
    st.header("Advanced analytics")

    valid = data[data > -9000]
    if valid.size == 0:
        st.warning("No valid data to analyse.")
        return

    px_km2 = pixel_area_km2_from_transform(transform)
    stats = compute_risk_stats(data, risk_cfg, transform)

    tab1, tab2, tab3 = st.tabs(
        ["Risk statistics", "Scenario comparison", "Priority zones"]
    )

    with tab1:
        st.subheader("Risk class breakdown")

        # Band edges come from RiskThresholds, which is derived from the
        # precision-recall curve, so the labels show one decimal place.
        labels = [
            ("Safe", "safe", f"<{risk_cfg.safe:.1%}"),
            ("Moderate", "moderate", f"{risk_cfg.safe:.1%}-{risk_cfg.moderate:.1%}"),
            ("High", "high", f"{risk_cfg.moderate:.1%}-{risk_cfg.high:.1%}"),
            ("Severe", "severe", f"{risk_cfg.high:.1%}-{risk_cfg.critical:.1%}"),
            ("Critical", "critical", f">={risk_cfg.critical:.1%}"),
        ]
        cols = st.columns(len(labels))
        for col, (name, key, band) in zip(cols, labels):
            pct = stats[f"{key}_pct"]
            area = stats.get(f"{key}_km2")
            with col:
                st.metric(
                    f"{name} ({band})",
                    f"{area:,.1f} km2" if area is not None else f"{pct:.2f}%",
                    f"{pct:.2f}%",
                )

        if px_km2 is None:
            st.caption(
                "Areas unavailable: the raster carried no affine transform, so only "
                "percentages are shown."
            )
        else:
            st.caption(
                f"Mapped area {stats['mapped_area_km2']:,.0f} km2 at "
                f"{np.sqrt(px_km2 * 1e6):.0f} m resolution. Permanent water bodies are "
                "excluded from the model domain. Expected flooded area (the integral "
                f"of the probability surface) is **{stats['expected_flooded_km2']:,.1f} km2** "
                "— this is the quantity the model is calibrated against, and it does "
                "not depend on where the band edges fall."
            )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(valid, bins=50, color="steelblue", edgecolor="none", alpha=0.85)
        for threshold, colour, name in [
            (risk_cfg.safe, "#91cf60", "Moderate"),
            (risk_cfg.moderate, "#fee08b", "High"),
            (risk_cfg.high, "#fdae61", "Severe"),
            (risk_cfg.critical, "#d73027", "Critical"),
        ]:
            ax.axvline(
                threshold, color=colour, linewidth=2, linestyle="--",
                label=f"{name} (>={threshold:.1%})",
            )
        ax.set_xlim(0, 1)
        ax.set_yscale("log")
        ax.set_xlabel("Flood probability")
        ax.set_ylabel("Pixels (log scale)")
        ax.set_title(f"Hazard distribution at {rainfall:.0f} mm rainfall")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.subheader("Rainfall scenario comparison")
        if not maps:
            st.info("No pre-computed scenarios available.")
        else:
            st.caption(
                "Each column is an independently computed hazard raster, not a "
                "rescaling of the current map."
            )
            depths = sorted(maps)
            cols = st.columns(len(depths))
            for col, mm in zip(cols, depths):
                scenario = maps[mm]
                sv = scenario[scenario > -9000]
                if sv.size == 0:
                    continue
                high_pct = float((sv >= risk_cfg.moderate).mean() * 100)
                crit_pct = float((sv >= risk_cfg.high).mean() * 100)
                with col:
                    st.metric(
                        f"{mm:.0f} mm",
                        f"{high_pct:.1f}% elevated",
                        f"critical {crit_pct:.1f}%",
                    )

            # Monotonicity is a property the physics guarantees; surface it so
            # a regression in the hazard model is visible in the UI.
            crit_series = [
                float((maps[mm][maps[mm] > -9000] >= risk_cfg.critical).mean())
                for mm in depths
            ]
            if all(b >= a - 1e-9 for a, b in zip(crit_series, crit_series[1:])):
                st.success("Critical-area fraction increases monotonically with rainfall.")
            else:
                st.error(
                    "Critical-area fraction is NOT monotonic in rainfall — "
                    "the hazard rasters are inconsistent."
                )

    with tab3:
        st.subheader("Priority zones")
        critical_pct = stats["critical_pct"]
        critical_km2 = stats.get("critical_km2")

        if critical_pct > 0:
            headline = (
                f"Critical zone: {critical_km2:,.1f} km2"
                if critical_km2 is not None
                else f"Critical zone: {critical_pct:.2f}% of mapped area"
            )
            st.error(headline)
            st.markdown(
                "**Suggested actions**\n"
                "1. Issue advisories for the critical-class low-lying wards\n"
                "2. Pre-position pumps at the highest-probability drainage outlets\n"
                "3. Confirm flood shelter readiness in the critical zone\n"
                "4. Monitor river gauges upstream of the affected sub-basins"
            )
        else:
            st.success("No critical zones at the current rainfall level.")


# ──────────────────────────────────────────────
# Place search and click readout
# ──────────────────────────────────────────────
def render_place_search(known_places: dict) -> None:
    """Sidebar place lookup."""
    st.sidebar.markdown("---")
    query = st.sidebar.text_input("Go to place", placeholder="e.g. Edappally")
    if not query:
        return

    matches = [
        (name, coords)
        for name, coords in known_places.items()
        if query.strip().lower() in name.lower()
    ]
    if matches:
        for name, coords in matches[:5]:
            st.sidebar.success(f"{name}: {coords[0]:.4f}, {coords[1]:.4f}")
    else:
        st.sidebar.error(
            "Not found. Try: " + ", ".join(list(known_places)[:4])
        )


def render_map_click_info(
    st_data, data: np.ndarray, crs, transf, nodata, layer_type: str
) -> None:
    """Report the raster value at the clicked map location."""
    if data is None or crs is None or transf is None:
        return
    if not st_data or not st_data.get("last_clicked"):
        return

    from pyproj import Transformer

    clicked = st_data["last_clicked"]
    lat, lon = clicked["lat"], clicked["lng"]

    to_raster = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = to_raster.transform(lon, lat)

    # `~transform * (x, y)` returns (column, row) -- x maps to column. The
    # previous version unpacked it as (row, col), so every click read the
    # transposed pixel and silently reported the wrong value.
    col, row = ~transf * (x, y)
    row, col = int(row), int(col)

    if not (0 <= row < data.shape[0] and 0 <= col < data.shape[1]):
        st.warning("Clicked outside the raster bounds.")
        return

    value = float(data[row, col])
    if value <= -9000:
        st.warning(f"No data at {lat:.4f}, {lon:.4f} (outside the mapped district).")
        return

    if layer_type == "LULC":
        from config import LULC_CLASS_NAMES

        name = LULC_CLASS_NAMES.get(int(round(value)), f"class {value:.0f}")
        st.success(f"{lat:.4f}, {lon:.4f} — {name}")
    elif layer_type == "Flood Probability":
        st.success(f"{lat:.4f}, {lon:.4f} — flood probability {value:.1%}")
    elif layer_type == "Conformal Confidence":
        from conformal import SET_LABELS

        label = SET_LABELS.get(int(round(value)), f"code {value:.0f}")
        st.success(f"{lat:.4f}, {lon:.4f} — {label}")
    else:
        st.success(f"{lat:.4f}, {lon:.4f} — {value:.3f}")
