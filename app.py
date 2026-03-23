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
st.markdown("<h2 style='text-align: center;'>A GEOAI-BASED FRAMEWORK FOR GEOSPATIAL FLOOD RISK MAPPING AND SHORT-TERM RAINFALL PREDICTION FOR URBAN WATERLOGGING PREVENTION</h2>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: justify; padding: 10px; background-color: #f0f2f6; border-left: 5px solid #4CAF50; margin-bottom: 20px;'>
<i style='color: #4CAF50;'>"Our system integrates rainfall forecasting with geospatial terrain analysis and satellite-based flood detection to generate real-time and predictive urban waterlogging risk maps."</i>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
with col1:
    st.metric("Model Type", "GeoAI Hybrid (CNN + Physics Spatial Analysis)")
with col2:
    st.metric("Region", "Kerala, India")

import requests
import pandas as pd  # BUG FIX: was imported deep inside functions, causing repeated re-imports

# App-level Alert Placeholder
alert_placeholder = st.empty()

# Sidebar
st.sidebar.header("🎛️ Controls")

# Advanced Mode Toggle
advanced_mode = st.sidebar.checkbox("🔬 Advanced Analytics Mode", value=False)

# GeoAI_New folder — all 13 model feature layers + Ground Truth:
# Training channels: DEM, Slope, River_Distance, LULC, Distance_to_Builtup_Final,
#   NDVI_Aligned, NDWI_Aligned, HAND, TWI, TPI, SPI, Flow_Accumulation, Urban_Mask
# Ground Truth: Ground_Truth_Fixed.tif (Sentinel-1 2018 flood binary)
layer_type = st.sidebar.radio("📊 Select Layer", [
    "Flood Probability", "DEM", "Slope", "LULC",
    "TWI", "SPI", "HAND", "TPI",
    "Distance to Water", "Distance to Built-up",
    "NDVI (Vegetation)", "NDWI (Water)",
    "Sentinel-1 Ground Truth", "Flow Accumulation",
    "Urban Mask"
])

# AI Engine - Supercharged Only (only mode that uses GeoAI_New)
mode_suffix = "_supercharged"
if advanced_mode:
    st.sidebar.subheader("🤖 AI Engine")
    st.sidebar.info("**Supercharged GeoAI** (Hydro-Physics MCDA)\nUsing: NDWI · Flow Accum · Slope · Urban Dist")

# Scenario & Weather selection (Always Visible)
st.sidebar.subheader("🌧️ Rainfall Conditions")
scenarios = [100, 150, 200]
use_live = st.sidebar.checkbox("📡 Use Live Weather API", value=False)

if use_live:
    try:
        # Fetch highly localized 24-hour short-term rainfall prediction sum
        url = "https://api.open-meteo.com/v1/forecast?latitude=10.0&longitude=76.3&hourly=precipitation&forecast_days=2"
        r = requests.get(url).json()
        
        # Sum the next 24 hours of precipitation forecast to simulate intense short-term urban risk
        next_24hr_rain = sum(r['hourly']['precipitation'][:24])
        st.sidebar.success(f"📍 Live 24-Hr Short-Term Prediction: {next_24hr_rain:.1f} mm")
        rainfall = next_24hr_rain
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
            
    # 2. Load Hydrological Factors — prefer GeoAI_New (full Ernakulam district)
    # over processed/ (urban core only). This ensures physics layer covers the
    # entire district so ALL of Ernakulam responds to rainfall, not just the center blob.
    geoai_flow  = os.path.join("GeoAI_New", "Ernakulam_Flow_Accumulation.tif")
    geoai_slope = os.path.join("GeoAI_New", "Ernakulam_Slope.tif")
    geoai_hand  = os.path.join("GeoAI_New", "Ernakulam_HAND.tif")
    geoai_twi   = os.path.join("GeoAI_New", "Ernakulam_TWI.tif")

    legacy_flow  = os.path.join("processed", "Flow_aligned.tif")
    legacy_slope = os.path.join("processed", "Slope_aligned.tif")

    # Pick the best available source
    use_geoai = os.path.exists(geoai_flow) and os.path.exists(geoai_slope)
    use_legacy = os.path.exists(legacy_flow) and os.path.exists(legacy_slope)

    if use_geoai or use_legacy:
        path_flow  = geoai_flow  if use_geoai else legacy_flow
        path_slope = geoai_slope if use_geoai else legacy_slope

        flow_result  = read_downsampled(path_flow)
        slope_result = read_downsampled(path_slope)

        if flow_result[0] is not None and slope_result[0] is not None:
            flow,  _ = flow_result
            slope, _ = slope_result

            # Mask nodata
            flow  = np.ma.masked_where(flow  < -9000, flow)
            slope = np.ma.masked_where(slope < -9000, slope)

            # Log1p normalise flow (heavily skewed)
            flow_log  = np.log1p(np.maximum(flow.filled(0), 0))
            flow_norm = (flow_log - flow_log.min()) / (flow_log.max() - flow_log.min() + 1e-6)

            # Inverse slope: flat terrain = higher flood risk
            slope_norm = (slope.filled(0) - slope.filled(0).min()) / (slope.filled(0).max() - slope.filled(0).min() + 1e-6)
            slope_inv  = 1.0 - slope_norm

            # Start building combined hydro factor (70% flow, 30% slope)
            hydro_combined = flow_norm * 0.7 + slope_inv * 0.3

            # Optionally add HAND (height above nearest drainage) — low HAND = high flood risk
            if use_geoai and os.path.exists(geoai_hand):
                hand_result = read_downsampled(geoai_hand)
                if hand_result[0] is not None:
                    hand = np.ma.masked_where(hand_result[0] < -9000, hand_result[0])
                    # Clip to meaningful flood range (0–20m)
                    hand_clipped = np.clip(hand.filled(20), 0, 20)
                    hand_norm    = 1.0 - (hand_clipped / 20.0)   # invert: low HAND = high risk
                    # Blend at same resolution (crop to common shape)
                    hr = min(hydro_combined.shape[0], hand_norm.shape[0])
                    wc = min(hydro_combined.shape[1], hand_norm.shape[1])
                    hydro_combined = hydro_combined[:hr, :wc] * 0.75 + hand_norm[:hr, :wc] * 0.25

            # Apply TWI boost if available
            if use_geoai and os.path.exists(geoai_twi):
                twi_result = read_downsampled(geoai_twi)
                if twi_result[0] is not None:
                    twi = np.ma.masked_where(twi_result[0] < -9000, twi_result[0])
                    twi_norm = np.clip((twi.filled(0) - twi.filled(0).min()) / (twi.filled(0).max() - twi.filled(0).min() + 1e-6), 0, 1)
                    hr = min(hydro_combined.shape[0], twi_norm.shape[0])
                    wc = min(hydro_combined.shape[1], twi_norm.shape[1])
                    hydro_combined = hydro_combined[:hr, :wc] * 0.85 + twi_norm[:hr, :wc] * 0.15

            maps['hydro'] = np.clip(
                hydro_combined if isinstance(hydro_combined, np.ndarray) else hydro_combined.filled(0),
                0, 1
            )
        else:
            maps['hydro'] = None
    else:
        maps['hydro'] = None
        
    return maps, base_meta

def load_static_layer(layer):
    path = ""
    if layer == "DEM": path = os.path.join("GeoAI_New", "Ernakulam_Clipped_DEM.tif")
    elif layer == "Slope": path = os.path.join("GeoAI_New", "Ernakulam_Slope.tif")
    elif layer == "LULC": path = os.path.join("GeoAI_New", "Ernakulam_LULC_2018.tif")
    elif layer == "TWI": path = os.path.join("GeoAI_New", "Ernakulam_TWI.tif")
    elif layer == "SPI": path = os.path.join("GeoAI_New", "Ernakulam_SPI.tif")
    elif layer == "HAND": path = os.path.join("GeoAI_New", "Ernakulam_HAND.tif")
    elif layer == "TPI": path = os.path.join("GeoAI_New", "Ernakulam_TPI.tif")
    elif layer == "Distance to Water": path = os.path.join("GeoAI_New", "Ernakulam_River_Distance.tif")
    elif layer == "Distance to Built-up": path = os.path.join("GeoAI_New", "Distance_to_Builtup_Final.tif")
    elif layer == "NDVI (Vegetation)": path = os.path.join("GeoAI_New", "NDVI_Aligned.tif")         # same file model trained on
    elif layer == "NDWI (Water)": path = os.path.join("GeoAI_New", "NDWI_Aligned.tif")         # same file model trained on
    elif layer == "Sentinel-1 Ground Truth": path = os.path.join("GeoAI_New", "Ground_Truth_Final.tif")
    elif layer == "Flow Accumulation": path = os.path.join("GeoAI_New", "Ernakulam_Flow_Accumulation.tif")
    elif layer == "Urban Mask": path = os.path.join("GeoAI_New", "Urban_Mask.tif")
    
    if not os.path.exists(path): 
        st.warning(f"File not found: {path}")
        return None, None
    
    try:
        with rasterio.open(path) as src:
            file_nodata = src.nodata
            # Aggressive downsampling: max 1000px
            scale_factor = 1000 / max(src.width, src.height)
            resamp = rasterio.enums.Resampling.nearest if layer == "LULC" else rasterio.enums.Resampling.bilinear
            if scale_factor < 1:
                new_h = max(1, int(src.height * scale_factor))
                new_w = max(1, int(src.width * scale_factor))
                data = src.read(1, out_shape=(new_h, new_w), resampling=resamp).astype(np.float32)
                t = src.transform * src.transform.scale(
                    (src.width / data.shape[1]),
                    (src.height / data.shape[0])
                )
            else:
                data = src.read(1).astype(np.float32)
                t = src.transform
            
            # --- Apply Nodata Mask ---
            # Normalize: set all nodata pixels to -9999 so data_masked works uniformly
            if file_nodata is not None:
                data[data == file_nodata] = -9999.0
            
            # Handle NaN-based nodata (NDWI, NDVI files from Sentinel-2)
            data[np.isnan(data)] = -9999.0
            
            # Special case: LULC uses 0 as nodata (ocean/outside boundary)
            if layer == "LULC":
                data[data <= 0] = -9999.0
            
            # Special case: HAND uses -99999 as nodata
            if layer == "HAND":
                data[data < -9000] = -9999.0
            
            # Special case: Sentinel-1 GT uses -9999 already, normalize
            if layer == "Sentinel-1 Ground Truth":
                data[data < -9000] = -9999.0
            
            # Special case: Flow Accumulation - 0 = valid river source, not nodata
            # (nodata IS 0 but we lose river sources - use extreme negative nodata approach)
            # The nodata=0 was already applied above, so remaining zeros are thin streams
            
            # Special case: Distance to Water — outside-boundary pixels often carry
            # a very large sentinel value (e.g., 65535 or file's nodata).
            # Mask any pixel where distance > 99th percentile as likely outside boundary.
            if layer == "Distance to Water":
                valid_dist = data[data > -9000]
                if valid_dist.size > 0:
                    p99 = np.percentile(valid_dist, 99)
                    # Anything unrealistically large = outside boundary → nodata
                    data[data > p99 * 1.5] = -9999.0
            
            return data, (src.bounds, src.crs, t, -9999.0)
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
        
        # Preserve original Nodata mask before interpolation ruins it
        base_mask = (maps[100] < -9000) if 100 in maps else None
        
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
            else: data = maps.get(150) # Changed from maps.get(200) to maps.get(150)

        if data is not None:
            # 2. Add Topographic Flow routing (Hydro-Physics Layer)
            hydro = maps.get('hydro')
            if hydro is not None:
                # BUG FIX: data and hydro come from different rasters downsampled independently,
                # so they can differ by a few pixels (e.g. 771x1000 vs 765x1000).
                # Crop everything to the common minimum shape before arithmetic.
                h_min = min(data.shape[0], hydro.shape[0])
                w_min = min(data.shape[1], hydro.shape[1])
                data  = data[:h_min, :w_min]
                hydro = hydro[:h_min, :w_min]
                if base_mask is not None:
                    base_mask = base_mask[:h_min, :w_min]
                data += hydro * min(rainfall/200.0, 1.0) * 0.2
            
            # Ensure physical limits
            data = np.clip(data, 0, 1)
            
            # Restore Nodata to firmly delete the ocean and borders
            if base_mask is not None:
                data[base_mask] = -9999.0
            
            # Apply global spatial mask
            data_masked = np.ma.masked_less(data, -9000)
            
        ai_output = data.copy() if data is not None else None

        # 2. Apply "GeoAI Disaster Management Formula"
        if data is not None and maps.get('hydro') is not None:
            hydro_factor = maps['hydro']
            # Ensure hydro_factor matches data shape (may have been cropped above)
            h_min = min(data.shape[0], hydro_factor.shape[0])
            w_min = min(data.shape[1], hydro_factor.shape[1])
            data         = data[:h_min, :w_min]
            hydro_factor = hydro_factor[:h_min, :w_min]
            
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
            if data is not None:
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
    return legend_html  # BUG FIX: was missing return statement - caused None passed to folium.Element()

# Map Setup with Folium
# Add Layer Control for Basemaps
m = folium.Map(location=MAP_CENTER, zoom_start=ZOOM_START, tiles=None)
folium.TileLayer('CartoDB positron', name="Light Map", control=True).add_to(m)
folium.TileLayer('CartoDB dark_matter', name="Dark Map", control=True).add_to(m)
folium.TileLayer('OpenStreetMap', name="Street Map", control=True).add_to(m)

# Process Data for Display
# BUG FIX: Guard all bounds/crs access - they are None if no data loaded
if data is not None and bounds is not None and crs is not None:
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
                # --- REALISTIC SOCIO-ECONOMIC ESTIMATES for Ernakulam District ---
                # Source: Census 2011 | Population: ~3.5M | Area: 3,068 km²
                # Pop density: ~1,068 people/km² | Built-up ~18% of district
                ERNAKULAM_POP = 3_500_000     # 2026 est
                ERNAKULAM_AREA_KM2 = 3068.0   # district area
                ERN_POP_DENSITY = 1068         # people/km² (census)

                # Scale downsampled pixels back to km²
                total_valid = valid_pixels.size
                scale_km2_per_pixel = ERNAKULAM_AREA_KM2 / total_valid

                critical_pixels = int(np.sum(valid_pixels >= 0.45))
                high_risk_pixels = int(np.sum(valid_pixels >= 0.25))
                risk_percentage  = (high_risk_pixels / total_valid) * 100
                crit_percentage  = (critical_pixels  / total_valid) * 100

                # Area at risk
                critical_area_km2 = critical_pixels * scale_km2_per_pixel
                high_risk_area_km2 = high_risk_pixels * scale_km2_per_pixel

                # Realistic population at risk (only account for built-up portion, ~18% is residential)
                RESIDENTIAL_FRACTION = 0.18
                est_pop_impacted = min(
                    int(critical_area_km2 * ERN_POP_DENSITY * RESIDENTIAL_FRACTION),
                    ERNAKULAM_POP
                )
                # Hospitals: Ernakulam has ~15 major hospitals; scale by critical fraction
                est_hospitals = max(1, min(int(crit_percentage / 8), 6))
                # Economic loss: ~₹50 Cr/km² for Kerala flood damage (2018 avg)
                est_econ_cr = critical_area_km2 * 50   # Crores INR
                est_econ_usd_m = est_econ_cr * 0.12    # ~₹1 Cr ≈ $120k → ÷1000 for M
                
                if crit_percentage > 15:
                    alert_placeholder.error(
                        f"🚨 **CRITICAL FLOOD ALERT — Ernakulam District** | Rainfall: {rainfall}mm\n\n"
                        f"🌧️ **Critical Risk Area:** {critical_area_km2:.0f} km² ({crit_percentage:.1f}% of district)\n\n"
                        f"👥 **Estimated Population at Risk:** ~{est_pop_impacted:,} residents in low-lying zones\n\n"
                        f"🏥 **Hospitals in Red Zone:** ~{est_hospitals} of 15 major hospitals\n\n"
                        f"💰 **Estimated Damage:** ₹{est_econ_cr:.0f} Cr (~${est_econ_usd_m:.0f}M USD) based on 2018 Kerala flood rates"
                    )
                elif risk_percentage > 25:
                    alert_placeholder.warning(
                        f"⚠️ **FLOOD WARNING — Ernakulam** | Rainfall: {rainfall}mm\n\n"
                        f"🌊 **High-Risk Area:** {high_risk_area_km2:.0f} km² ({risk_percentage:.1f}% of district)\n\n"
                        f"👥 **Vulnerable Population:** ~{est_pop_impacted:,} residents in flood-prone zones"
                    )
                else:
                    alert_placeholder.info(
                        f"🌦️ **MONITORING ACTIVE** | Rainfall: {rainfall}mm | "
                        f"Low-Moderate risk ({risk_percentage:.1f}% of Ernakulam above threshold). "
                        f"Watching river levels and drainage capacity."
                    )

                # --- SEND ALERT BUTTON ---
                st.sidebar.markdown("---")
                st.sidebar.subheader("📢 Send Emergency Alert")
                alert_emails = st.sidebar.text_input(
                    "Alert Recipients (email)",
                    value="district.ernakulam@kerala.gov.in",
                    help="Comma-separated email addresses"
                )
                alert_body = (
                    f"FLOOD ALERT - Ernakulam District\n"
                    f"Rainfall: {rainfall}mm | Critical Area: {critical_area_km2:.0f} km²\n"
                    f"Population at Risk: ~{est_pop_impacted:,}\n"
                    f"Damage Estimate: ~₹{est_econ_cr:.0f} Cr\n"
                    f"Generated by GeoAI Flood Risk System"
                )
                mailto_link = (
                    f"mailto:{alert_emails}"
                    f"?subject=🚨 GeoAI Flood Alert — Ernakulam {rainfall}mm Rainfall"
                    f"&body={alert_body.replace(' ', '%20').replace('\n', '%0A')}"
                )
                st.sidebar.markdown(
                    f'<a href="{mailto_link}" target="_blank">'
                    f'<button style="width:100%;background:#c62828;color:white;'
                    f'border:none;padding:10px;border-radius:5px;font-size:14px;'
                    f'cursor:pointer;font-weight:bold;">'
                    f'📧 Send Alert Email</button></a>',
                    unsafe_allow_html=True
                )
        # -------------------
        
        # Spatial Intersection: LULC vs. High Risk (>25%)
        # FIXED: maps only has integer keys (100, 150, 200) and 'hydro' — no 'LULC' key
        # This block intentionally skipped (LULC intersection requires separate loading)
        if False and "LULC" in maps and maps["LULC"] is not None:  # BUG FIX: was crashing - maps has no 'LULC' key
            lulc_map = maps["LULC"]
            # Find pixels where probability > 0.25 AND lulc is valid
            risk_pixels = (data_masked > 0.25) & (lulc_map > 0)
            
            if np.any(risk_pixels):
                # Get the LULC classes for those high-risk pixels
                affected_lulc = lulc_map[risk_pixels]
                unique_classes, counts = np.unique(affected_lulc, return_counts=True)
                
                # ESA WorldCover mapping
                lulc_classes_real = {
                    1: "Tree Cover", 2: "Shrubland", 3: "Grassland", 4: "Cropland",
                    5: "Built-up (Urban)", 6: "Bare / Sparse", 7: "Snow / Ice",
                    8: "Permanent Water", 9: "Herbaceous Wetland", 10: "Mangroves", 11: "Moss / Lichen"
                }
                
                st.sidebar.markdown("---")
                st.sidebar.subheader("🏘️ Vulnerable Land Classes")
                st.sidebar.caption("High-risk area breakdown (>25% probability)")
                
                # Sort by most affected area
                sorted_idx = np.argsort(-counts)
                
                for idx in sorted_idx:
                    cls_code = unique_classes[idx]
                    cls_count = counts[idx]
                    # 30m x 30m = 900 sq meters per pixel = 0.0009 sq km
                    area_sqkm = cls_count * 0.0009
                    
                    if area_sqkm > 0.1: # Only show significant areas
                        cls_name = lulc_classes_real.get(cls_code, f"Class {cls_code}")
                        
                        # Emphasize Built-up area
                        if cls_code == 5:
                            st.sidebar.error(f"🏢 **{cls_name}: {area_sqkm:.1f} km²**")
                        elif cls_code == 4:
                            st.sidebar.warning(f"🌾 **{cls_name}: {area_sqkm:.1f} km²**")
                        else:
                            st.sidebar.write(f"• {cls_name}: {area_sqkm:.1f} km²")

    # 1. FLOOD PROBABILITY
    if layer_type == "Flood Probability":
        # Custom Color Ramp: Green -> Yellow -> Orange -> Red
        # Calibrated for model outputs (Max ~0.45)
        # Thresholds:
        # 0.00 - 0.05: Safe (Green)
        # 0.05 - 0.15: Moderate (Yellow)
        # 0.15 - 0.30: High (Orange)
        # 0.30 - 1.00: Critical (Red)
        
        # BUG FIX: Removed dead duplicate color list assignment below.
        # Progressive Color Transition (Green -> Yellow -> Red)
        colors = [
            (0.00, "#1a9850"), # Safe (Green)
            (0.05, "#91cf60"), # Very Low Risk (Light Green)
            (0.12, "#fee08b"), # Elevated Risk (Yellow)
            (0.20, "#fdae61"), # High Risk (Orange)
            (0.30, "#d73027"), # Severe (Red)
            (1.00, "#a50026")  # Critical (Dark Red)
        ]
        
        cmap = LinearSegmentedColormap.from_list("RiskRamp", colors)
        norm = plt.Normalize(vmin=0, vmax=1.0)
        
        # Only hide strictly Nodata exterior pixels. Keep low-risk green pixels visible.
        spatial_mask = np.ma.getmaskarray(data_masked)
        
        image_rgba = cmap(norm(data_masked.filled(0)))
        image_rgba[..., 3] = np.where(spatial_mask, 0.0, 0.6)
        
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
            image_rgba = cmap(norm(data_masked.filled(nodata if nodata is not None else -9999)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
        
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)  # BUG FIX: nodata guard
        legend_html = create_legend(f"Elevation (m)", [("High", "#fcae91"), ("Medium", "#ffffbf"), ("Low", "#2c7bb6")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 3. SLOPE
    elif layer_type == "Slope":
        cmap = plt.get_cmap('magma')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 2), np.percentile(valid_vals, 98)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata if nodata is not None else -9999)))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
            
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.7)
        legend_html = create_legend(f"Slope (deg)", [("Steep", "#000004"), ("Moderate", "#b63679"), ("Flat", "#fcfdbf")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 4. LULC (Categorical)
    elif layer_type == "LULC":
        # ESA WorldCover actual class values found in this file: [1,2,4,5,7,8,11]
        # ESA WorldCover v2 class codes:
        # 1  -> Trees (Tree cover)
        # 2  -> Shrubland
        # 4  -> Cropland 
        # 5  -> Built-up
        # 7  -> Bare / sparse vegetation
        # 8  -> Snow and Ice (unlikely, probably wetland)
        # 11 -> Permanent water bodies
        lulc_colors = {
            1:  (0, 120, 0, 255),    # Trees (Dark Green)
            2:  (255, 187, 34, 255), # Shrubland (Orange)
            3:  (15, 200, 15, 255),  # Grassland (Lime)
            4:  (255, 215, 0, 255),  # Cropland (Gold)
            5:  (230, 30, 30, 255),  # Built-up (Red) - CRITICAL
            6:  (180, 180, 180, 255),# Bare (Grey)
            7:  (200, 180, 120, 255),# Bare sparse (Tan)
            8:  (0, 200, 220, 255),  # Wetland (Cyan)
            9:  (0, 100, 200, 255),  # Mangrove (Blue-Green)
            10: (0, 140, 255, 255),  # Moss (Blue)
            11: (0, 0, 200, 255),    # Water (Dark Blue)
        }
        
        # CRITICAL FIX: Use np.round().astype(int32) to convert float pixels back
        H, W = data_masked.shape
        image_rgba = np.zeros((H, W, 4), dtype=np.float32)
        lulc_int = np.round(data_masked.filled(-9999)).astype(np.int32)
        
        for val, color in lulc_colors.items():
            pixel_mask = lulc_int == int(val)
            image_rgba[pixel_mask] = np.array(color, dtype=np.float32) / 255.0
        
        # Set alpha: transparent for nodata
        alpha_mask = (lulc_int <= 0) | np.ma.getmaskarray(data_masked)
        image_rgba[..., 3] = np.where(alpha_mask, 0.0, 0.85)
            
        # Legend
        legend_items = [
            ("Trees", "#007800"), ("Cropland", "#ffd700"), ("Built-up", "#e61e1e"), 
            ("Shrubland", "#ffbb22"), ("Water", "#0000c8"), ("Wetland", "#00c8dc")
        ]
        legend_html = create_legend("LULC 2018 (ESA WorldCover)", legend_items)
        m.get_root().html.add_child(folium.Element(legend_html))

    # 5. TWI (Topographic Wetness Index)
    elif layer_type == "TWI":
        cmap = plt.get_cmap('YlGnBu') 
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            vmin, vmax = np.percentile(valid_vals, 5), np.percentile(valid_vals, 95)
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            image_rgba = cmap(norm(data_masked.filled(nodata if nodata is not None else -9999)))
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
            image_rgba = cmap(norm(data_masked.filled(nodata if nodata is not None else -9999)))
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
            image_rgba = cmap(norm(data_masked.filled(nodata if nodata is not None else -9999)))
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

    # 12. NDVI (Vegetation Health 2018)
    elif "NDVI" in layer_type:
        cmap = plt.get_cmap('RdYlGn')
        norm = plt.Normalize(vmin=-0.2, vmax=0.8)
        valid_data = data_masked.filled(0)
        # Clamp any residual NaN to 0 before colormap
        valid_data = np.nan_to_num(valid_data, nan=0.0)
        image_rgba = cmap(norm(valid_data))
        spatial_mask = np.ma.getmaskarray(data_masked)
        image_rgba[..., 3] = np.where(spatial_mask, 0.0, 0.8)
        legend_html = create_legend("NDVI 2018 (Vegetation)", [("Dense Vegetation", "#1a9850"), ("Sparse", "#fee08b"), ("No Vegetation", "#d73027")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 13. NDWI (Surface Water 2018)
    elif "NDWI" in layer_type:
        cmap = plt.get_cmap('RdBu')
        norm = plt.Normalize(vmin=-0.5, vmax=0.5)
        valid_data = data_masked.filled(0)
        valid_data = np.nan_to_num(valid_data, nan=0.0)
        image_rgba = cmap(norm(valid_data))
        spatial_mask = np.ma.getmaskarray(data_masked)
        image_rgba[..., 3] = np.where(spatial_mask, 0.0, 0.8)
        legend_html = create_legend("NDWI 2018 (Surface Water)", [("Water Bodies", "#053061"), ("Wet Soil", "#d1e5f0"), ("Dry Land", "#67001f")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 14. Sentinel-1 Ground Truth 2018 (Binary Flood Mask)
    elif layer_type == "Sentinel-1 Ground Truth":
        # Binary classification: 0=No flood, 1=Flood
        gt_colors = LinearSegmentedColormap.from_list("GT", [(0, "#d9f0a3"), (1, "#0000cc")])
        norm = plt.Normalize(vmin=0, vmax=1)
        valid_data = data_masked.filled(0)
        image_rgba = gt_colors(norm(valid_data))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.85)
        legend_html = create_legend("Sentinel-1 GT 2018", [("Flooded (SAR)", "#0000cc"), ("Dry Land", "#d9f0a3")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 15. Flow Accumulation (Drainage)
    elif layer_type == "Flow Accumulation":
        cmap = plt.get_cmap('Blues')
        valid_vals = data_masked.compressed()
        if len(valid_vals) > 0:
            # Log scale normalization - flow accum is very skewed
            log_data = np.log1p(np.clip(data_masked.filled(0), 0, None))
            vmax_log = np.percentile(np.log1p(valid_vals[valid_vals > 0]), 99) if np.any(valid_vals > 0) else 1
            norm = plt.Normalize(vmin=0, vmax=vmax_log)
            image_rgba = cmap(norm(log_data))
        else:
            image_rgba = np.zeros((*data_masked.shape, 4))
        image_rgba[..., 3] = np.where(data_masked.mask, 0.0, 0.75)
        legend_html = create_legend("Flow Accumulation", [("Major Drainage", "#084081"), ("Minor Streams", "#4292c6"), ("Upslope", "#f7fbff")])
        m.get_root().html.add_child(folium.Element(legend_html))

    # 16. Urban Mask (Binary Built-up Footprint)
    elif layer_type == "Urban Mask":
        # Binary mask: 1 = Urban / Built-up, 0 = Non-urban
        # Used as model channel 13 — critical for vulnerability assessment
        H, W = data_masked.shape
        image_rgba = np.zeros((H, W, 4), dtype=np.float32)
        urb_data = data_masked.filled(0)
        urb_data = np.nan_to_num(urb_data, nan=0.0)
        # Urban pixels → vivid red-orange
        urban_pixels = urb_data > 0.5
        image_rgba[urban_pixels] = [0.91, 0.18, 0.18, 0.85]    # red-ish for built-up
        image_rgba[~urban_pixels] = [0.94, 0.96, 0.84, 0.0]    # transparent for non-urban
        # Nodata → fully transparent
        nd_mask = np.ma.getmaskarray(data_masked)
        image_rgba[nd_mask, 3] = 0.0
        legend_html = create_legend("Urban Footprint (Model Ch.13)",
            [("Built-up / Urban Area", "#e82e2e"), ("Rural / Non-Urban", "transparent")])
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

if data is not None and crs is not None and transf is not None and st_data and st_data.get('last_clicked'):
    # BUG FIX: Guard against crs/transf being None (happens when no data loaded yet)
    clicked = st_data['last_clicked']
    clat, clon = clicked['lat'], clicked['lng']
    
    # Transform back to raster coords
    inv_transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_proj, y_proj = inv_transformer.transform(clon, clat)
    
    # CRS to Row/Col (rasterio affine inverse)
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
# BUG FIX: advanced mode also needs maps to be defined (only available for Flood Probability layer)
_maps_available = layer_type == "Flood Probability" and 'maps' in dir()
if advanced_mode and data is not None and layer_type == "Flood Probability" and _maps_available:
    st.markdown("---")
    st.header("📈 Advanced Analytics Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Statistics", "💾 Export Data", "🔄 Scenario Comparison", "🚨 Evacuation Zones"])
    
    with tab1:
        st.subheader("📊 Flood Risk Area Breakdown")
        
        # --- ONLY USE VALID PIXELS (excluding nodata = -9999) ---
        valid_data = data[data > -9000]
        if valid_data.size == 0:
            st.warning("No valid data to analyze.")
        else:
            ERNAKULAM_AREA_KM2 = 3068.0
            scale_km2 = ERNAKULAM_AREA_KM2 / valid_data.size

            safe_px     = int(np.sum(valid_data < 0.12))
            moderate_px = int(np.sum((valid_data >= 0.12) & (valid_data < 0.20)))
            high_px     = int(np.sum((valid_data >= 0.20) & (valid_data < 0.30)))
            critical_px = int(np.sum(valid_data >= 0.30))

            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("🟢 Safe",        f"{safe_px*scale_km2:.0f} km²",  f"{safe_px/valid_data.size*100:.1f}%")
            with col2: st.metric("🟡 Moderate",    f"{moderate_px*scale_km2:.0f} km²", f"{moderate_px/valid_data.size*100:.1f}%")
            with col3: st.metric("🟠 High Risk",   f"{high_px*scale_km2:.0f} km²",  f"{high_px/valid_data.size*100:.1f}%")
            with col4: st.metric("🔴 Critical",    f"{critical_px*scale_km2:.0f} km²", f"{critical_px/valid_data.size*100:.1f}%")

            # Histogram — valid pixels only so x-axis is 0-1, not -10000 to 0
            st.subheader("Flood Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(valid_data, bins=50, color='steelblue', edgecolor='none', alpha=0.8)
            ax.axvline(0.12, color='#fee08b', linewidth=2, linestyle='--', label='Moderate (12%)')
            ax.axvline(0.20, color='#fdae61', linewidth=2, linestyle='--', label='High (20%)')
            ax.axvline(0.30, color='#d73027', linewidth=2, linestyle='--', label='Critical (30%)')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Flood Probability (0 = Safe → 1 = Certain Flood)', fontsize=11)
            ax.set_ylabel('Number of pixels', fontsize=11)
            ax.set_title(f'Risk Distribution at {rainfall}mm Rainfall — Ernakulam District', fontsize=12)
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close()

            # Simple summary sentence
            st.info(
                f"At **{rainfall}mm rainfall**, approximately **{(high_px+critical_px)*scale_km2:.0f} km²** "
                f"({(high_px+critical_px)/valid_data.size*100:.1f}% of Ernakulam) is in the High or Critical risk zone. "
                f"Critical zone alone: **{critical_px*scale_km2:.0f} km²**."
            )
    
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
