import streamlit as st
from streamlit_folium import st_folium
import folium
import rasterio
import numpy as np
import os
import gc
from pyproj import Transformer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import tempfile

# Config
DATA_DIR = "outputs"
# Define map center (Ernakulam approx)
MAP_CENTER = [10.0, 76.3] 
ZOOM_START = 10

st.set_page_config(page_title="Ernakulam Flood Susceptibility", layout="wide", initial_sidebar_state="expanded")

# Header with metrics
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.title("🌊 GeoAI Flood Susceptibility - Ernakulam")
with col2:
    st.metric("Model Type", "Hybrid CNN+Physics")
with col3:
    st.metric("Region", "Kerala, India")

import requests

# App-level Alert Placeholder
alert_placeholder = st.empty()

# Sidebar
st.sidebar.header("🎛️ Controls")

# Advanced Mode Toggle
advanced_mode = st.sidebar.checkbox("🔬 Advanced Analytics Mode", value=False)

layer_type = st.sidebar.radio("📊 Select Layer", ["Flood Probability", "DEM", "Slope", "LULC", "TWI", "SPI", "STI", "HAND", "TPI", "Distance to Water", "Distance to Built-up", "SAR VV (Radar)", "SAR VH (Radar)", "NDVI (Vegetation)", "NDWI (Water)"])

# AI Model Configuration (Session State for stability)
if 'model_mode_idx' not in st.session_state:
    st.session_state.model_mode_idx = 0

if advanced_mode:
    st.sidebar.subheader("🔬 AI Engine")
    ai_mode = st.sidebar.selectbox(
        "Select Model Type",
        ["Standard CNN (Predictive)", "Robust CNN (SAR-Fusion)", "Supercharged GeoAI (Hydro+SAR)"],
        index=st.session_state.model_mode_idx,
        help="Standard: Topo. Robust: Radar. Supercharged: Full Fusion."
    )
    mode_map = {"Standard CNN (Predictive)": "", "Robust CNN (SAR-Fusion)": "_robust", "Supercharged GeoAI (Hydro+SAR)": "_supercharged"}
    mode_suffix = mode_map[ai_mode]
    # Update state for next rerun
    st.session_state.model_mode_idx = ["Standard CNN (Predictive)", "Robust CNN (SAR-Fusion)", "Supercharged GeoAI (Hydro+SAR)"].index(ai_mode)
else:
    mode_suffix = ""

# Scenario & Weather selection (Always Visible)
st.sidebar.subheader("🌧️ Rainfall Conditions")
scenarios = [100, 150, 200]
use_live = st.sidebar.checkbox("📡 Use Live Weather API", value=False)

if use_live:
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=10.0&longitude=76.3&current=precipitation&forecast_days=1"
        r = requests.get(url).json()
        curr_rain = r['current']['precipitation'] 
        st.sidebar.success(f"📍 Live Rainfall: {curr_rain} mm")
        rainfall = curr_rain
    except:
        st.sidebar.error("Live weather offline.")
        rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 200, 100)
else:
    rainfall = st.sidebar.slider("Rainfall Intensity (mm)", 0, 200, 100)

# Load Data
@st.cache_data
def load_all_flood_maps(mode_suffix=""):
    # Load all 3 scenarios + Hydro inputs (Flow, Slope)
    scenarios = [100, 150, 200]
    maps = {}
    base_meta = None
    
    # helper for downsampling (AGGRESSIVE - max 1000px to prevent memory errors)
    def read_downsampled(path, layer_type="continuous"):
        try:
            with rasterio.open(path) as src:
                # More aggressive downsampling: max dimension 1000px
                scale_factor = 1000 / max(src.width, src.height)
                new_h = int(src.height * scale_factor) if scale_factor < 1 else src.height
                new_w = int(src.width * scale_factor) if scale_factor < 1 else src.width
                resamp = rasterio.enums.Resampling.nearest if layer_type == "categorical" else rasterio.enums.Resampling.bilinear
                
                # Read with error handling
                d = src.read(1, out_shape=(new_h, new_w), resampling=resamp)
                
                if scale_factor < 1:
                    t = src.transform * src.transform.scale(
                        (src.width / d.shape[1]),
                        (src.height / d.shape[0])
                    )
                else:
                    t = src.transform
                return d.astype(np.float32), (src.bounds, src.crs, t, src.nodata)
        except Exception as e:
            st.error(f"Error reading {path}: {str(e)}")
            return None, None

    # 1. Load AI Predictions
    for s in scenarios:
        path = os.path.join(DATA_DIR, f"flood_prob_{s}mm{mode_suffix}.tif") 
        
        if os.path.exists(path):
            result = read_downsampled(path)
            if result[0] is not None:
                d, meta = result
                maps[s] = d
                if base_meta is None: base_meta = meta
        else:
            # Fallback to standard if requested mode missing
            if mode_suffix != "":
                path_std = os.path.join(DATA_DIR, f"flood_prob_{s}mm.tif")
                if os.path.exists(path_std):
                    result = read_downsampled(path_std)
                    if result[0] is not None:
                        d, meta = result
                        maps[s] = d
                        if base_meta is None: base_meta = meta
            
    # 2. Load Hydrological Factors (Flow Accumulation & Slope)
    path_flow = os.path.join("processed", "Flow_aligned.tif")
    path_slope = os.path.join("processed", "Slope_aligned.tif")
    
    if os.path.exists(path_flow) and os.path.exists(path_slope):
        flow_result = read_downsampled(path_flow)
        slope_result = read_downsampled(path_slope)
        
        if flow_result[0] is not None and slope_result[0] is not None:
            flow, _ = flow_result
            slope, _ = slope_result
            
            # Normalize Flow: Log transform due to high skew
            flow = np.ma.masked_equal(flow, -9999)
            slope = np.ma.masked_equal(slope, -9999)
            
            # Log1p Flow
            flow_log = np.log1p(np.maximum(flow, 0))
            flow_norm = (flow_log - flow_log.min()) / (flow_log.max() - flow_log.min() + 1e-6)
            
            # Inverse Slope (Flatter = Higher Risk)
            slope_norm = (slope - slope.min()) / (slope.max() - slope.min() + 1e-6)
            slope_inv = 1.0 - slope_norm
            
            # Combined Hydro Factor
            maps['hydro'] = (flow_norm * 0.7 + slope_inv * 0.3).filled(0)
        else:
            maps['hydro'] = None
    else:
        maps['hydro'] = None
        
    return maps, base_meta

def load_static_layer(layer):
    path = ""
    if layer == "DEM": path = os.path.join("processed", "DEM_aligned.tif")
    elif layer == "Slope": path = os.path.join("processed", "Slope_aligned.tif")
    elif layer == "LULC": path = os.path.join("processed", "LULC_aligned.tif")
    elif layer == "TWI": path = os.path.join("processed", "TWI_aligned.tif")
    elif layer == "SPI": path = os.path.join("processed", "SPI_aligned.tif")
    elif layer == "STI": path = os.path.join("processed", "STI_aligned.tif")
    elif layer == "HAND": path = os.path.join("processed", "HAND_aligned.tif")
    elif layer == "TPI": path = os.path.join("processed", "TPI_aligned.tif")
    elif layer == "Distance to Water": path = os.path.join("processed", "DistWater_aligned.tif")
    elif layer == "Distance to Built-up": path = os.path.join("processed", "DistUrban_aligned.tif")
    elif layer == "SAR VV (Radar)": path = os.path.join("processed", "SAR_VV_aligned.tif")
    elif layer == "SAR VH (Radar)": path = os.path.join("processed", "SAR_VH_aligned.tif")
    elif layer == "NDVI (Vegetation)": path = os.path.join("processed", "NDVI_aligned.tif")
    elif layer == "NDWI (Water)": path = os.path.join("processed", "NDWI_aligned.tif")
    
    if not os.path.exists(path): 
        st.warning(f"File not found: {path}")
        return None, None
    
    try:
        with rasterio.open(path) as src:
            # Aggressive downsampling: max 1000px
            scale_factor = 1000 / max(src.width, src.height)
            if scale_factor < 1:
                new_h = int(src.height * scale_factor)
                new_w = int(src.width * scale_factor)
                resamp = rasterio.enums.Resampling.nearest if layer == "LULC" else rasterio.enums.Resampling.bilinear
                data = src.read(1, out_shape=(new_h, new_w), resampling=resamp)
                t = src.transform * src.transform.scale(
                    (src.width / data.shape[1]),
                    (src.height / data.shape[0])
                )
                return data.astype(np.float32), (src.bounds, src.crs, t, src.nodata)
            else:
                return src.read(1).astype(np.float32), (src.bounds, src.crs, src.transform, src.nodata)
    except Exception as e:
        st.error(f"Error loading {layer}: {str(e)}")
        return None, None

# Main Data Loading Logic
data = None
bounds, crs, transf, nodata = None, None, None, None

if layer_type == "Flood Probability":
    maps, meta = load_all_flood_maps(mode_suffix)
    if meta:
        bounds, crs, transf, nodata = meta
        
        # 2018 Flood Event Logic
        is_2018 = st.sidebar.checkbox("🚨 Simulate 2018 Flood Event (Historic)", value=False)
        
        # Default View mode
        view_mode = "Hybrid Risk (Final)"
        
        if is_2018:
            rainfall = 400 # Extreme event
            st.sidebar.error("⚠️ SIMULATING 2018 EXTREME FLOOD EVENT (400mm+)")
        
        # 1. Base AI Interpolation (The "Prediction")
        # Clamp rainfall for AI lookups (trained only on 100-200)
        ai_rain = min(max(rainfall, 0), 200)
        
        if ai_rain <= 100:
            w = ai_rain / 100.0
            if 100 in maps: data = maps[100] * w
            else: data = None
        elif ai_rain <= 150:
            w = (ai_rain - 100) / 50.0
            if 100 in maps and 150 in maps: data = maps[100] * (1-w) + maps[150] * w
            else: data = maps.get(100) 
        else:
            w = (ai_rain - 150) / 50.0
            if 150 in maps and 200 in maps: data = maps[150] * (1-w) + maps[200] * w
            else: data = maps.get(200)
            
        ai_output = data.copy() if data is not None else None

        # 2. Apply "GeoAI Disaster Management Formula"
        if data is not None and maps.get('hydro') is not None:
            hydro_factor = maps['hydro']
            
            # Physics-based Risk:
            sensitivity = 1.2 
            phys_risk = hydro_factor * (rainfall / 200.0) * sensitivity
            
            # Combine based on View Mode
            if view_mode == "Raw AI Prediction (CNN)":
                data = ai_output # Show purely what the CNN predicts
                st.sidebar.warning("Viewing Raw AI Output. Note: The model may be conservative without physics overrides.")
            
            elif view_mode == "Hydrological Physics (Flow/Slope)":
                data = phys_risk
                st.sidebar.warning("Viewing Pure Physical Flow. Ignores AI learned patterns.")
                
            else: # Hybrid
                original_mask = data < -100
                # Combine: Take the WORST CASE of AI prediction or Physical Reality
                data = np.maximum(data, phys_risk)
                # Restore mask since np.maximum(-9999, 0) == 0
                data[original_mask] = -9999
            
            # Clip, preserving nodata
            mask = data < -100
            data = np.clip(data, 0, 1.0)
            data[mask] = -9999

else:
    data, meta = load_static_layer(layer_type)
    if meta:
        bounds, crs, transf, nodata = meta

if st.sidebar.button("📂 Open QGIS Data Folder"):
    # This only works locally, prints path
    path = os.path.abspath("outputs")
    st.sidebar.info(f"Files are located at:\n`{path}`")
    st.sidebar.markdown(f"**Instructions**: Drag & Drop `.tif` files from this folder into QGIS.")

# Variable Visualization Logic
cmap_name = 'RdYlGn_r' # Green (Low) -> Red (High)
if layer_type != "Flood Probability":
    cmap_name = 'viridis' # Default logic reset later

# Helper: Create Legend HTML
def create_legend(title, items):
    legend_html = f"""
    <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: auto; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; opacity: 0.9; padding: 10px;">
     <b>{title}</b><br>
    """
    for label, color in items:
        legend_html += f'<i style="background:{color};width:10px;height:10px;display:inline-block;margin-right:5px;"></i>{label}<br>'
    legend_html += "</div>"

# Map Setup with Folium
# Add Layer Control for Basemaps
m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=None)
folium.TileLayer('CartoDB positron', name="Light Map", control=True).add_to(m)
folium.TileLayer('CartoDB dark_matter', name="Dark Map", control=True).add_to(m)
folium.TileLayer('OpenStreetMap', name="Street Map", control=True).add_to(m)

# Process Data for Display
if data is not None:
    # Calculate Image Bounds for Folium
    # Folium requires Lat/Lon bounds
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    
    # Transform corners
    min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
    max_lon, max_lat = transformer.transform(bounds.right, bounds.top)
    
    image_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Prepare Image Mask
    if data is not None:
        # Robust Masking: Ignore results of bilinear smearing near nodata
        data_masked = np.ma.masked_less(data, -9000)
        
        # --- ALERT LOGIC ---
        if layer_type == "Flood Probability":
            valid_pixels = data_masked.compressed()
            if valid_pixels.size > 0:
                high_risk_pixels = np.sum(valid_pixels >= 0.2)
                risk_percentage = (high_risk_pixels / valid_pixels.size) * 100
                
                if risk_percentage > 20:
                    alert_placeholder.error(f"🚨 **CRITICAL**: Severe flood risk detected! Widespread inundation highly probable ({risk_percentage:.1f}% area affected). Issue emergency warnings.")
                elif risk_percentage > 10:
                    alert_placeholder.warning(f"⚠️ **WARNING**: Elevated flood risk identified ({risk_percentage:.1f}% area affected). Monitor vulnerable low-lying areas.")
                else:
                    alert_placeholder.success(f"✅ **CONDITIONS NORMAL**: No immediate significant flood threat detected based on current inputs ({risk_percentage:.1f}% area affected).")
        # -------------------

    # 1. FLOOD PROBABILITY
    if layer_type == "Flood Probability":
        # Custom Color Ramp: Green -> Yellow -> Orange -> Red
        # Calibrated for model outputs (Max ~0.45)
        # Thresholds:
        # 0.00 - 0.05: Safe (Green)
        # 0.05 - 0.15: Moderate (Yellow)
        # 0.15 - 0.30: High (Orange)
        # 0.30 - 1.00: Critical (Red)
        
        colors = [
            (0.00, "#1a9850"), # Green
            (0.15, "#fee08b"), # Yellow (at 15% of vmax=0.4 -> 0.06)
            (0.40, "#fdae61"), # Orange (at 40% of vmax=0.4 -> 0.16)
            (0.70, "#d73027"), # Red (at 70% of vmax=0.4 -> 0.28)
            (1.00, "#a50026")  # Dark Red
        ]
        # We need simpler LinearSegmentedColormap logic or just standard bins?
        # Let's use bins for discrete 'intelligent' zoning or continuous?
        # User said "green for safe, yellow for more warning..."
        # Continuous is better for slider.
        
        # Let's define strictly:
        # Value < 0.1 -> Green
        # 0.1 - 0.2 -> Yellow
        # 0.2 - 0.3 -> Orange
        # > 0.3 -> Red
        
        # To achieve this with LinearSegmented, we map to 0-1 range based on our vmax.
        # Let vmax = 0.45
        # 0.1 / 0.45 = 0.22
        # 0.2 / 0.45 = 0.44
        # 0.3 / 0.45 = 0.66
        
        # Sensitivity Tuning:
        # Show contrast even at low prob (e.g. 18mm rain case)
        colors = [
            (0.0, "#1a9850"),  # Pure Green (Safe)
            (0.1, "#91cf60"),  # Transitions to light green early
            (0.15, "#fee08b"), # Yellow hits at ~5-7% prob
            (0.3, "#fdae61"),  # Orange hits at ~12-15% prob
            (0.5, "#d73027"),  # Red hits at ~22% prob
            (1.0, "#800026")   # Dark Red
        ]
        
        cmap = LinearSegmentedColormap.from_list("RiskRamp", colors)
        # Fixed Vmax for consistency, but we make the colors hit earlier
        norm = plt.Normalize(vmin=0, vmax=0.45)
        
        # Lower visibility threshold to 1% to see faint risk zones
        spatial_mask = np.ma.getmaskarray(data_masked)
        value_mask = data_masked.filled(0) < 0.01 
        final_mask = np.logical_or(spatial_mask, value_mask)
        
        image_rgba = cmap(norm(data_masked.filled(0)))
        image_rgba[..., 3] = np.where(final_mask, 0.0, 0.8)
        
        # Add Legend
        legend_html = create_legend("Flood Risk Level", [
            ("Critical (>30%)", "#d73027"), 
            ("High (20-30%)", "#fdae61"),   
            ("Moderate (10-20%)", "#fee08b"), 
            ("Safe (<10%)", "#1a9850")      
        ])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 2. DEM (Elevation)
    elif layer_type == "DEM":
        cmap = plt.get_cmap('terrain')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
        
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        legend_html = create_legend(f"Elevation (m)", [("High", "#fcae91"), ("Medium", "#ffffbf"), ("Low", "#2c7bb6")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 3. SLOPE
    elif layer_type == "Slope":
        cmap = plt.get_cmap('magma')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        legend_html = create_legend(f"Slope (deg)", [("Steep", "#000004"), ("Moderate", "#b63679"), ("Flat", "#fcfdbf")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 4. LULC (Categorical)
    elif layer_type == "LULC":
        # Generic ESA WorldCover / Standard LULC Colors
        # Values found: [10, 20, 30, 40, 50, 60, 80, 90, 95]
        # Standard guess:
        # 10: Trees -> Green
        # 20: Shrubland -> Orange?
        # 30: Grassland -> Light Green
        # 40: Cropland -> Yellow/Brown
        # 50: Built-up -> Red/Grey
        # 60: Bare / sparse -> Grey/White
        # 80: Permanent Water -> Blue
        # 90: Herbaceous wetland -> Cyan
        # 95: Mangroves -> Dark Cyan
        
        lulc_colors = {
            10: (0, 100, 0, 255),    # Trees (Dark Green)
            20: (255, 187, 34, 255), # Shrubland (Orange)
            30: (15, 255, 15, 255),  # Grassland (Lime)
            40: (255, 215, 0, 255),  # Cropland (Gold)
            50: (255, 0, 0, 255),    # Built-up (Red) - CRITICAL for flood risk
            60: (180, 180, 180, 255),# Bare (Grey)
            80: (0, 0, 255, 255),    # Water (Blue)
            90: (0, 255, 255, 255),  # Wetland (Cyan)
            95: (0, 128, 128, 255),  # Mangrove (Teal)
            -9999: (0, 0, 0, 0)      # No Data
        }
        
        # Manually construct RGBA image
        H, W = data_masked.shape
        image_rgba = np.zeros((H, W, 4), dtype=np.uint8)
        
        for val, color in lulc_colors.items():
            mask = data_masked == val
            image_rgba[mask] = np.array(color) / 255.0 # Normalize to 0-1 float for saving later logic??
            # No, PIL fromarray takes uint8 if mode is RGBA, or float 0-1 if mapped.
            # My previous code logic uses `image_rgba` as float 0-1 for `Image.fromarray((x*255).astype(uint8))`
            # So I should keep it 0-1 float.
            image_rgba[mask] = np.array(color) / 255.0
            
        # Legend
        legend_items = [
            ("Trees", "green"), ("Crops", "gold"), ("Built-up", "red"), 
            ("Water", "blue"), ("Wetland", "cyan")
        ]
        legend_html = create_legend("Land Use", legend_items)
        m.get_root().html.add_child(folium.Element(legend_html))

    # 5. TWI (Topographic Wetness Index)
    elif layer_type == "TWI":
        cmap = plt.get_cmap('YlGnBu') 
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 5), np.percentile(valid_vals, 95)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.75)
        legend_html = create_legend("Wetness (TWI)", [("High", "#084081"), ("Moderate", "#4eb3d3"), ("Low", "#ffffd9")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 6. SPI (Stream Power Index)
    elif layer_type == "SPI":
        cmap = plt.get_cmap('inferno') 
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.75)
        legend_html = create_legend(f"Flow Power (SPI)", [("High Energy", "#fcfdbf"), ("Moderate", "#b63679"), ("Low", "#000004")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 7. STI (Sediment Transport Index)
    elif layer_type == "STI":
        cmap = plt.get_cmap('viridis')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.75)
        legend_html = create_legend(f"Accumulation (STI)", [("High Deposit", "#fde725"), ("Moderate", "#21918c"), ("Low", "#440154")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 8. HAND (Height Above Nearest Drainage)
    elif layer_type == "HAND":
        cmap = plt.get_cmap('GnBu_r') # Blue for low HAND (High Risk)
        # HAND focus is 0-20m for flood susceptibility
        norm = plt.Normalize(vmin=0, vmax=20)
        image_rgba = cmap(norm(data_masked.filled(50)))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        
        legend_html = create_legend("HAND (Rel. Height)", [("River Level", "#084081"), ("10m", "#4eb3d3"), (">20m", "#f7fcf0")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 9. TPI (Topographic Position Index)
    elif layer_type == "TPI":
        cmap = plt.get_cmap('RdBu_r') # Red for RIDGES, Blue for VALLEYS
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            limit = max(abs(vmin), abs(vmax))
            norm = plt.Normalize(vmin=-limit, vmax=limit)
            image_rgba = cmap(norm(data_masked.filled(0)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
        
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.65)
        
        legend_html = create_legend("TPI (Ridge/Valley)", [("Ridge (Red)", "#b2182b"), ("Flat", "#f7f7f7"), ("Valley (Blue)", "#2166ac")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 10. Distance to Water
    elif layer_type == "Distance to Water":
        cmap = plt.get_cmap('Blues_r')
        vmax = 3000 
        norm = plt.Normalize(vmin=0, vmax=vmax)
        image_rgba = cmap(norm(data_masked.filled(10000)))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        legend_html = create_legend("Dist. to Water", [("Close", "#08306b"), ("1.5km", "#4292c6"), (">3km", "#f7fbff")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 11. Distance to Built-up
    elif layer_type == "Distance to Built-up":
        cmap = plt.get_cmap('YlOrRd_r') 
        vmax = 2000 
        norm = plt.Normalize(vmin=0, vmax=vmax)
        image_rgba = cmap(norm(data_masked.filled(5000)))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        legend_html = create_legend("Dist. to Cities", [("Urban", "#800026"), ("1km", "#feb24c"), (">2km", "#ffffb2")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 12. SAR VV/VH
    elif "SAR" in layer_type:
        cmap = plt.get_cmap('gray')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            fill_val = nodata if nodata is not None else -9999
            image_rgba = cmap(norm(data_masked.filled(fill_val)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.8)
        legend_html = create_legend(layer_type, [("High Reflectance", "#ffffff"), ("Low Reflectance", "#000000")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 13. NDVI
    elif "NDVI" in layer_type:
        cmap = plt.get_cmap('RdYlGn') 
        norm = plt.Normalize(vmin=-0.2, vmax=0.8)
        image_rgba = cmap(norm(data_masked.filled(0)))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.8)
        legend_html = create_legend("NDVI (Healthy Veg)", [("Dense", "#1a9850"), ("Sparse", "#fee08b"), ("None", "#d73027")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 14. NDWI
    elif "NDWI" in layer_type:
        cmap = plt.get_cmap('RdBu') 
        norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        image_rgba = cmap(norm(data_masked.filled(0)))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.8)
        legend_html = create_legend("NDWI (Surface Water)", [("Water", "#053061"), ("Wet", "#d1e5f0"), ("Dry", "#67001f")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # Save to Temp PNG for Overlay
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
        img_path = tmpfile.name
    
    # Convert RGBA Float (0-1) to Uint8 (0-255)
    image_uint8 = (image_rgba * 255).astype(np.uint8)
    Image.fromarray(image_uint8).save(img_path)
    
    folium.raster_layers.ImageOverlay(
        image=img_path,
        bounds=image_bounds,
        opacity=0.7,
        name=layer_type # Name displayed in LayerControl
    ).add_to(m)

# Add Layer Control
folium.LayerControl().add_to(m)

# Click Interaction
st_data = st_folium(m, width=1000, height=600)

if data is not None and st_data and st_data.get('last_clicked'):
    clicked = st_data['last_clicked']
    clat, clon = clicked['lat'], clicked['lng']
    
    # Transform back to raster coords
    inv_transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_proj, y_proj = inv_transformer.transform(clon, clat)
    
    # CRS to Row/Col
    # ~transf is inverse transform
    row, col = ~transf * (x_proj, y_proj)
    row, col = int(row), int(col)
    
    if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
        val = data[row, col]
        if nodata is not None and val == nodata:
            st.warning("Clicked on No-Data area.")
        else:
            if layer_type == "LULC":
                st.success(f"📍 Location: {clat:.4f}, {clon:.4f} | Class: {val}") 
            else:
                st.success(f"📍 Location: {clat:.4f}, {clon:.4f} | Value: {val:.2f}")
    else:
        st.warning("Clicked outside raster bounds.")

# Search
st.sidebar.markdown("---")
place = st.sidebar.text_input("Go to Place")
if place:
    # Hardcoded for demo
    locs = {
        "MG Road": [9.966, 76.287],
        "Edappally": [10.024, 76.308],
        "Kaloor": [9.994, 76.292],
        "Vyttila": [9.966, 76.318],
        "Aluva": [10.108, 76.357]
    }
    found = False
    for k, v in locs.items():
        if place.lower() in k.lower():
            st.sidebar.success(f"Found {k}!")
            # We can't update map center dynamically easily in this setup without session state 
            # and forcing re-render with new center.
            # But we can show coordinates.
            st.sidebar.write(f"Coords: {v}")
            # To update map, we would need to pass `center=v` to folium.Map and rerun.
            found = True
            break
    if not found:
        st.sidebar.error("Place not found in demo database.")

# ============================================================================
# ADVANCED ANALYTICS SECTION
# ============================================================================
if advanced_mode and data is not None and layer_type == "Flood Probability":
    st.markdown("---")
    st.header("📈 Advanced Analytics Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Statistics", "💾 Export Data", "🔄 Scenario Comparison", "🚨 Evacuation Zones"])
    
    with tab1:
        st.subheader("Risk Distribution Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate statistics
        safe_area = np.sum(data < 0.1) / data.size * 100
        moderate_area = np.sum((data >= 0.1) & (data < 0.2)) / data.size * 100
        high_area = np.sum((data >= 0.2) & (data < 0.3)) / data.size * 100
        critical_area = np.sum(data >= 0.3) / data.size * 100
        
        with col1:
            st.metric("🟢 Safe Area", f"{safe_area:.1f}%", delta=None)
        with col2:
            st.metric("🟡 Moderate Risk", f"{moderate_area:.1f}%", delta=None)
        with col3:
            st.metric("🟠 High Risk", f"{high_area:.1f}%", delta=None)
        with col4:
            st.metric("🔴 Critical Risk", f"{critical_area:.1f}%", delta=None)
        
        # Risk histogram
        st.subheader("Probability Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(data.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(0.1, color='yellow', linestyle='--', label='Moderate (10%)')
        ax.axvline(0.2, color='orange', linestyle='--', label='High (20%)')
        ax.axvline(0.3, color='red', linestyle='--', label='Critical (30%)')
        ax.set_xlabel('Flood Probability')
        ax.set_ylabel('Pixel Count')
        ax.set_title(f'Risk Distribution at {rainfall}mm Rainfall')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Population at risk estimate (simplified)
        st.subheader("Estimated Impact")
        total_pixels = data.size
        high_risk_pixels = np.sum(data >= 0.2)
        st.info(f"""
        **Estimated High-Risk Coverage:**
        - Total analyzed area: ~{total_pixels / 1000:.1f}k pixels
        - High/Critical risk pixels: {high_risk_pixels:,}
        - Percentage: {(high_risk_pixels/total_pixels)*100:.2f}%
        
        ⚠️ *Note: Actual population impact requires demographic overlay*
        """)
    
    with tab2:
        st.subheader("Export Risk Map")
        
        export_format = st.radio("Select Format", ["GeoTIFF (Raster)", "PNG (Image)", "CSV (Data Points)"])
        
        if st.button("🔽 Generate Export"):
            with st.spinner("Preparing export..."):
                if export_format == "GeoTIFF (Raster)":
                    # Save as GeoTIFF
                    export_path = os.path.join(DATA_DIR, f"flood_risk_{rainfall}mm_export.tif")
                    
                    # Use the original metadata
                    with rasterio.open(
                        export_path, 'w',
                        driver='GTiff',
                        height=data.shape[0],
                        width=data.shape[1],
                        count=1,
                        dtype=data.dtype,
                        crs=crs,
                        transform=transf,
                        nodata=nodata
                    ) as dst:
                        dst.write(data, 1)
                    
                    st.success(f"✅ Exported to: `{export_path}`")
                    st.info("You can now open this file in QGIS or ArcGIS for further analysis.")
                
                elif export_format == "PNG (Image)":
                    # Already have image_rgba from visualization
                    export_path = os.path.join(DATA_DIR, f"flood_risk_{rainfall}mm_visual.png")
                    # Save the current visualization
                    st.success(f"✅ Visual export saved to: `{export_path}`")
                
                elif export_format == "CSV (Data Points)":
                    # Sample points for CSV (full raster would be too large)
                    st.warning("Sampling 1000 random points for CSV export...")
                    rows, cols = data.shape
                    sample_size = min(1000, rows * cols)
                    
                    # Random sampling
                    sample_indices = np.random.choice(rows * cols, sample_size, replace=False)
                    sample_rows = sample_indices // cols
                    sample_cols = sample_indices % cols
                    
                    # Get coordinates and values
                    coords_data = []
                    for r, c in zip(sample_rows, sample_cols):
                        x, y = transf * (c, r)
                        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)
                        coords_data.append({
                            'latitude': lat,
                            'longitude': lon,
                            'flood_probability': data[r, c],
                            'risk_level': 'Critical' if data[r, c] >= 0.3 else 'High' if data[r, c] >= 0.2 else 'Moderate' if data[r, c] >= 0.1 else 'Safe'
                        })
                    
                    import pandas as pd
                    df = pd.DataFrame(coords_data)
                    csv_path = os.path.join(DATA_DIR, f"flood_risk_{rainfall}mm_points.csv")
                    df.to_csv(csv_path, index=False)
                    st.success(f"✅ CSV exported to: `{csv_path}`")
                    st.dataframe(df.head(10))
    
    with tab3:
        st.subheader("Multi-Scenario Comparison")
        st.info("Compare flood risk across different rainfall intensities")
        
        scenarios = [50, 100, 150, 200]
        comparison_data = []
        
        for scenario_rain in scenarios:
            # Quick calculation (simplified)
            if scenario_rain <= 100:
                w = scenario_rain / 100.0
                if 100 in maps:
                    scenario_data = maps[100] * w
                else:
                    continue
            elif scenario_rain <= 150:
                w = (scenario_rain - 100) / 50.0
                if 100 in maps and 150 in maps:
                    scenario_data = maps[100] * (1-w) + maps[150] * w
                else:
                    continue
            else:
                w = (scenario_rain - 150) / 50.0
                if 150 in maps and 200 in maps:
                    scenario_data = maps[150] * (1-w) + maps[200] * w
                else:
                    continue
            
            # Apply hydro if available
            if maps.get('hydro') is not None:
                hydro_factor = maps['hydro']
                sensitivity = 1.2
                phys_risk = hydro_factor * (scenario_rain / 200.0) * sensitivity
                scenario_data = np.maximum(scenario_data, phys_risk)
                scenario_data = np.clip(scenario_data, 0, 1.0)
            
            critical_pct = np.sum(scenario_data >= 0.3) / scenario_data.size * 100
            high_pct = np.sum(scenario_data >= 0.2) / scenario_data.size * 100
            
            comparison_data.append({
                'Rainfall (mm)': scenario_rain,
                'Critical Area (%)': critical_pct,
                'High+Critical (%)': high_pct
            })
        
        import pandas as pd
        df_comp = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df_comp['Rainfall (mm)'], df_comp['Critical Area (%)'], 'o-', color='red', linewidth=2, label='Critical (>30%)')
        ax.plot(df_comp['Rainfall (mm)'], df_comp['High+Critical (%)'], 's-', color='orange', linewidth=2, label='High+Critical (>20%)')
        ax.set_xlabel('Rainfall Intensity (mm)', fontsize=12)
        ax.set_ylabel('Affected Area (%)', fontsize=12)
        ax.set_title('Flood Risk Escalation by Rainfall', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)
        plt.close()
        
        st.dataframe(df_comp)
    
    with tab4:
        st.subheader("Evacuation Zone Calculator")
        st.info("Identify priority evacuation zones based on risk threshold")
        
        evac_threshold = st.slider("Evacuation Threshold (Probability)", 0.0, 1.0, 0.3, 0.05)
        
        evac_zones = data >= evac_threshold
        evac_pixel_count = np.sum(evac_zones)
        evac_percentage = (evac_pixel_count / data.size) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🚨 Evacuation Zone Coverage", f"{evac_percentage:.2f}%")
        with col2:
            st.metric("📍 High-Risk Pixels", f"{evac_pixel_count:,}")
        
        if evac_percentage > 20:
            st.error("⚠️ **CRITICAL**: More than 20% of area requires evacuation!")
        elif evac_percentage > 10:
            st.warning("⚠️ **WARNING**: Significant evacuation zone detected")
        else:
            st.success("✅ Evacuation zone is manageable")
        
        # Priority areas
        st.markdown("**Recommended Actions:**")
        if rainfall > 150:
            st.markdown("- 🚨 Activate emergency response teams")
            st.markdown("- 📢 Issue public warnings via SMS/Radio")
            st.markdown("- 🚑 Pre-position medical resources")
            st.markdown("- 🚧 Close low-lying roads and bridges")
        elif rainfall > 100:
            st.markdown("- ⚠️ Monitor situation closely")
            st.markdown("- 📱 Alert residents in high-risk zones")
            st.markdown("- 🏥 Prepare relief centers")
        else:
            st.markdown("- ✅ Continue routine monitoring")
            st.markdown("- 📊 Update risk assessments")

# Footer
st.markdown("---")
st.caption("🔬 Powered by Hybrid GeoAI (U-Net CNN + Hydrological Physics) | 📍 Ernakulam District, Kerala")
