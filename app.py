"""
GeoAI Flood Risk Dashboard – Streamlit Frontend.

Renders the rainfall-conditioned flood hazard maps produced by
`src/predict.py`, plus the underlying conditioning factors.
"""
import atexit
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

# Add src to path before importing project modules
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
from data_loading import (  # noqa: E402
    load_conformal_sets,
    load_hazard_maps,
    load_static_layer,
)
from hazard import blend_scenarios  # noqa: E402
from ui_components import (  # noqa: E402
    render_advanced_analytics,
    render_map_click_info,
    render_place_search,
    render_sidebar,
)
from visualization import (  # noqa: E402
    create_alert_message,
    create_conformal_visualization,
    create_flood_visualization,
    create_legend_html,
    create_static_visualization,
)

setup_logging(logging.INFO)
LOGGER = logging.getLogger("geoai_flood")

st.set_page_config(
    page_title="Ernakulam Flood Susceptibility",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

    Streamlit re-runs this whole script on every widget interaction, so a
    fresh NamedTemporaryFile per run would leave one file behind per slider
    tick. The path is held in session state and overwritten in place, and
    the interpreter cleans up whatever is left at exit.
    """
    from PIL import Image

    path = st.session_state.get("_overlay_png_path")
    if path is None or not os.path.exists(os.path.dirname(path) or "."):
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

st.markdown(
    """
<div style='text-align: justify; padding: 10px; background-color: #f0f2f6;
     border-left: 5px solid #4CAF50; margin-bottom: 20px;'>
<i style='color: #4CAF50;'>
"Our system integrates rainfall forecasting with geospatial terrain analysis
and satellite-based flood detection to generate real-time and predictive
urban waterlogging risk maps."
</i>
</div>
""",
    unsafe_allow_html=True,
)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Model Type", "Calibrated susceptibility x SCS-CN runoff")
with col2:
    st.metric("Region", "Ernakulam, Kerala, India")

alert_placeholder = st.empty()

# ──────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────
controls = render_sidebar(RAINFALL, RISK, KNOWN_PLACES)
layer_type = controls["layer_type"]
advanced_mode = controls["advanced_mode"]
rainfall = controls["rainfall"]
is_2018 = controls["is_2018"]

if is_2018:
    rainfall = 400
    st.sidebar.error("SIMULATING 2018 EXTREME FLOOD EVENT (400mm+)")

# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
data = None
meta = None
maps: dict = {}

if layer_type == "Flood Probability":
    maps, meta = load_hazard_maps(OUTPUT_DIR)
    if not maps:
        st.error(
            "No hazard maps found. Generate them first:\n\n"
            "```\npython align_data.py\n"
            "python src/susceptibility.py --train --predict\n"
            "python src/hazard.py\n```"
        )
    else:
        data = blend_scenarios(maps, rainfall)
elif layer_type == "Conformal Confidence":
    data, meta = load_conformal_sets(OUTPUT_DIR)
    if data is None:
        st.error(
            "No conformal raster found. Generate it with:\n\n"
            "```\npython src/susceptibility.py --conformal\n```"
        )
    else:
        st.info(
            "Prediction sets with a distribution-free coverage guarantee. This "
            "layer is rainfall-independent: it describes where the model can "
            "support a decision at 90% confidence, not how much it will rain."
        )
else:
    data, meta = load_static_layer(layer_type, GEOAI_NEW_DIR)

# ──────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────
if data is not None and meta is not None:
    import folium
    from pyproj import Transformer
    from streamlit_folium import st_folium

    bounds = meta["bounds"]
    crs = meta["crs"]
    transf = meta["transform"]
    nodata = meta["nodata"]

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    image_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    m = folium.Map(location=GEO.map_center, zoom_start=GEO.zoom_start, tiles=None)
    folium.TileLayer("CartoDB positron", name="Light Map", control=True).add_to(m)
    folium.TileLayer("CartoDB dark_matter", name="Dark Map", control=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street Map", control=True).add_to(m)

    if layer_type == "Flood Probability":
        image_rgba, legend_items = create_flood_visualization(data, VIZ, RISK)
        m.get_root().html.add_child(
            folium.Element(create_legend_html("Flood Risk Level", legend_items))
        )

        alert_msg = create_alert_message(data, rainfall, GEO, RISK, transf)
        if alert_msg:
            if "CRITICAL" in alert_msg:
                alert_placeholder.error(alert_msg)
            elif "WARNING" in alert_msg:
                alert_placeholder.warning(alert_msg)
            else:
                alert_placeholder.info(alert_msg)
    elif layer_type == "Conformal Confidence":
        image_rgba, legend_items = create_conformal_visualization(data)
        m.get_root().html.add_child(
            folium.Element(create_legend_html("Prediction set (90%)", legend_items))
        )
    else:
        image_rgba, legend_items = create_static_visualization(data, layer_type, VIZ)
        m.get_root().html.add_child(
            folium.Element(create_legend_html(layer_type, legend_items))
        )

    folium.raster_layers.ImageOverlay(
        image=_write_overlay_png(image_rgba),
        bounds=image_bounds,
        opacity=0.7,
        name=layer_type,
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_data = st_folium(m, width=1000, height=600)

    render_map_click_info(st_data, data, crs, transf, nodata, layer_type)
    render_place_search(KNOWN_PLACES)

    if advanced_mode and layer_type == "Flood Probability":
        render_advanced_analytics(data, maps, rainfall, GEO, RISK, VIZ, transf)

elif layer_type not in ("Flood Probability", "Conformal Confidence"):
    # The two model layers print their own, more specific, generation
    # instructions above; this is the fallback for a missing source raster.
    st.warning("No data loaded. Please check file paths and try again.")
    st.info(f"Looking for data in: {OUTPUT_DIR} and {GEOAI_NEW_DIR}")
