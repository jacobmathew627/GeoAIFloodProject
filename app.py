"""
GeoAI Flood Risk Dashboard - Streamlit frontend.

The two model layers are computed live: moving the rainfall slider re-evaluates
the model at that depth (~60 ms) rather than interpolating between pre-rendered
scenario rasters.
"""
import atexit
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (  # noqa: E402
    GEO,
    GEOAI_NEW_DIR,
    KNOWN_PLACES,
    OUTPUT_DIR,
    RAINFALL,
    RISK,
    VIZ,
    setup_logging,
)
from data_loading import load_conformal_sets, load_static_layer  # noqa: E402
from ui_components import (  # noqa: E402
    render_live_analytics,
    render_map_click_info,
    render_place_search,
    render_sidebar,
)
from visualization import (  # noqa: E402
    create_alert_message,
    create_conformal_visualization,
    create_flood_visualization,
    create_legend_html,
    create_pluvial_visualization,
    create_static_visualization,
)

setup_logging(logging.INFO)
LOGGER = logging.getLogger("geoai_flood")

st.set_page_config(
    page_title="Ernakulam Flood Susceptibility",
    layout="wide",
    initial_sidebar_state="expanded",
)

LIVE_LAYERS = ("Flood Probability (live)", "Waterlogging Index (live)")


# ──────────────────────────────────────────────
# Model loading (cached across reruns)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading the model...")
def get_live_grid():
    import live_model

    return live_model.load()


# ──────────────────────────────────────────────
# Temp file handling
# ──────────────────────────────────────────────
_TEMP_FILES: list[str] = []


def _cleanup_temp_files() -> None:
    for path in _TEMP_FILES:
        try:
            os.unlink(path)
        except OSError:
            pass
    _TEMP_FILES.clear()


atexit.register(_cleanup_temp_files)


def _write_overlay_png(image_rgba: np.ndarray) -> str:
    """
    Write an RGBA float array to a PNG and return its path.

    Streamlit re-runs the script on every widget interaction, so a fresh
    NamedTemporaryFile per run would leave one file behind per slider tick.
    The path is held in session state and overwritten in place.
    """
    from PIL import Image

    path = st.session_state.get("_overlay_png_path")
    if path is None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            path = tmp.name
        st.session_state["_overlay_png_path"] = path
        _TEMP_FILES.append(path)

    Image.fromarray((np.clip(image_rgba, 0, 1) * 255).astype(np.uint8)).save(path)
    return path


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    "<h2 style='text-align: center;'>A GEOAI-BASED FRAMEWORK FOR GEOSPATIAL FLOOD RISK MAPPING "
    "AND SHORT-TERM RAINFALL PREDICTION FOR URBAN WATERLOGGING PREVENTION</h2>",
    unsafe_allow_html=True,
)

alert_placeholder = st.empty()

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
controls = render_sidebar(RAINFALL, RISK, KNOWN_PLACES)
layer_type = controls["layer_type"]
advanced_mode = controls["advanced_mode"]
rainfall = controls["rainfall"]
is_2018 = controls["is_2018"]

if is_2018:
    rainfall = RAINFALL.reference_event_mm
    st.sidebar.error(
        f"Simulating the August 2018 event: {rainfall:.0f} mm "
        "(ERA5 3-day maximum, 14-16 Aug)"
    )

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
data = None
meta = None
grid = None

if layer_type in LIVE_LAYERS:
    try:
        import live_model

        grid = get_live_grid()
        with st.spinner(f"Evaluating the model at {rainfall:.0f} mm..."):
            if layer_type == "Flood Probability (live)":
                data = live_model.fluvial_probability(grid, rainfall)
            else:
                data = live_model.pluvial_index(grid, rainfall)
        meta = {
            "bounds": grid.bounds, "crs": grid.crs,
            "transform": grid.transform, "nodata": np.nan,
        }
    except FileNotFoundError as exc:
        st.error(
            f"{exc}\n\nBuild the live model first:\n\n"
            "```\npython align_data.py\n"
            "python src/derive_features.py\n"
            "python src/susceptibility.py --train --predict\n"
            "python src/live_model.py --build\n```"
        )
elif layer_type == "Conformal Confidence":
    data, meta = load_conformal_sets(OUTPUT_DIR)
    if data is None:
        st.error(
            "No conformal raster. Generate it with:\n\n"
            "```\npython src/susceptibility.py --conformal\n```"
        )
    else:
        st.info(
            "Prediction sets with a distribution-free coverage guarantee. "
            "Rainfall-independent: it shows where the model can support a "
            "decision at 90% confidence, not how much it will rain."
        )
else:
    data, meta = load_static_layer(layer_type, GEOAI_NEW_DIR)

# ──────────────────────────────────────────────
# Map
# ──────────────────────────────────────────────
if data is not None and meta is not None:
    import folium
    from pyproj import Transformer
    from streamlit_folium import st_folium

    bounds, crs, transf = meta["bounds"], meta["crs"], meta["transform"]

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    image_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    m = folium.Map(location=GEO.map_center, zoom_start=GEO.zoom_start, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light Map", control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Map", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", control=True).add_to(m)

    if layer_type == "Flood Probability (live)":
        image_rgba, legend_items = create_flood_visualization(data, VIZ, RISK)
        title = f"Inundation probability at {rainfall:.0f} mm"
        alert_msg = create_alert_message(data, rainfall, GEO, RISK, transf)
        if alert_msg:
            if "CRITICAL" in alert_msg:
                alert_placeholder.error(alert_msg)
            elif "WARNING" in alert_msg:
                alert_placeholder.warning(alert_msg)
            else:
                alert_placeholder.info(alert_msg)
    elif layer_type == "Waterlogging Index (live)":
        image_rgba, legend_items = create_pluvial_visualization(data)
        title = f"Waterlogging pressure at {rainfall:.0f} mm"
        alert_placeholder.warning(
            "**Unvalidated layer.** Physics only: routed SCS-CN runoff over "
            "local gradient. There are no urban waterlogging records for this "
            "district, so this index has never been tested against the "
            "phenomenon it names. Use it to rank locations, not as a probability. "
            "The calibrated layer is *Flood Probability (live)*."
        )
    elif layer_type == "Conformal Confidence":
        image_rgba, legend_items = create_conformal_visualization(data)
        title = "Prediction set (90%)"
    else:
        image_rgba, legend_items = create_static_visualization(data, layer_type, VIZ)
        title = layer_type

    m.get_root().html.add_child(folium.Element(create_legend_html(title, legend_items)))
    folium.raster_layers.ImageOverlay(
        image=_write_overlay_png(image_rgba), bounds=image_bounds,
        opacity=0.7, name=layer_type,
    ).add_to(m)
    folium.LayerControl().add_to(m)

    st_data = st_folium(m, width=1000, height=600, key="map")

    # Click readout: for the live layers this runs the point query, which
    # reports the physical quantities behind the number as well.
    if layer_type in LIVE_LAYERS and grid is not None:
        render_live_analytics(grid, rainfall, st_data, data, layer_type, RISK, transf)
    else:
        render_map_click_info(st_data, data, crs, transf, meta["nodata"], layer_type)

    render_place_search(KNOWN_PLACES)

elif layer_type not in LIVE_LAYERS and layer_type != "Conformal Confidence":
    st.warning("No data loaded. Check the file paths and try again.")
    st.info(f"Looking in: {OUTPUT_DIR} and {GEOAI_NEW_DIR}")
