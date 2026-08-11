"""
Centralized Configuration Module for GeoAI Flood Risk Project
All paths, parameters, and settings in one place for easy maintenance.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

# ──────────────────────────────────────────────
# Project Root Detection
# ──────────────────────────────────────────────
def _find_project_root() -> Path:
    """Find project root by looking for marker files."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "requirements.txt").exists():
            return parent
    return current.parent

PROJECT_ROOT = _find_project_root()
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = PROJECT_ROOT / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
STATIC_DIR = PROJECT_ROOT / "static"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
GEOAI_NEW_DIR = PROJECT_ROOT / "GeoAI_New"
GEOAI_DATA_DIR = PROJECT_ROOT / "GeoAI_Data"

# Ensure directories exist
for d in [DATA_DIR, PROCESSED_DIR, OUTPUT_DIR, MODELS_DIR, STATIC_DIR, EVALUATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Logging Configuration
# ──────────────────────────────────────────────
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("geoai_flood")
    return logger

LOGGER = setup_logging()

# ──────────────────────────────────────────────
# Data Classes for Configuration
# ──────────────────────────────────────────────
@dataclass(frozen=True)
class RasterConfig:
    """Raster processing configuration."""
    nodata_value: float = -9999.0
    target_crs: str = "EPSG:32643"  # UTM Zone 43N
    # Master grid is the ESA-derived LULC raster, which is 10 m — not 30 m.
    # Read from the raster at runtime via `pixel_area_km2()`; this is the
    # documented fallback only.
    cell_size: float = 10.0
    max_dimension: int = 1000  # For downsampling in visualization
    block_size: int = 1024  # For windowed processing

    @property
    def pixel_area_km2(self) -> float:
        return (self.cell_size ** 2) / 1e6

@dataclass(frozen=True)
class ModelConfig:
    """Model configuration."""
    # PyTorch UNet
    unet_channels: int = 6
    unet_classes: int = 1
    
    # TensorFlow Attention UNet
    attention_unet_channels: int = 13
    patch_size: int = 256
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    
    # Inference
    tile_size: int = 512
    tile_overlap: int = 64
    device: str = "auto"  # auto, cuda, cpu

@dataclass(frozen=True)
class GeoConfig:
    """Geographic configuration for Ernakulam."""
    map_center: tuple = (10.0, 76.3)
    zoom_start: int = 10
    district_bbox: tuple = (76.16, 9.85, 76.45, 10.15)  # min_lon, min_lat, max_lon, max_lat
    district_area_km2: float = 3068.0
    population: int = 3_500_000
    pop_density: float = 1068.0  # people/km²

@dataclass(frozen=True)
class RainfallConfig:
    """Rainfall scenario configuration."""
    scenarios: tuple = (50, 100, 150, 200, 250, 300, 400, 443, 500)
    max_slider: int = 600
    live_weather_url: str = "https://api.open-meteo.com/v1/forecast"
    weather_params: dict = field(default_factory=lambda: {
        "latitude": 10.0,
        "longitude": 76.3,
        "hourly": "precipitation",
        "forecast_days": 2
    })

    # Reference event: the storm depth the observed 2018 flood extent is taken
    # to represent. The hazard model reduces exactly to the fitted
    # susceptibility here, so this constant is the single anchor of the whole
    # rainfall response.
    #
    # DERIVED from the IMD 0.25 deg gauge-based gridded analysis -- the
    # official Indian rainfall product (src/reference_rainfall.py, cached in
    # models/reference_rainfall.json). For August 2018 over the district:
    #
    #     max 1-day  191.1 mm
    #     max 2-day  334.9 mm
    #     max 3-day  443.2 mm   (15-17 Aug)   <- used
    #     max 5-day  526.3 mm
    #     month      919.9 mm
    #
    # The 3-day window is chosen deliberately: HYDRO.amc is III, which already
    # encodes a wet antecedent 5 days, so the storm depth must be the burst
    # itself or antecedent wetness is counted twice. The 15-17 Aug window IMD
    # identifies matches the documented severe spell.
    #
    # History: 400 mm was originally a guess. ERA5 reanalysis put the 3-day
    # maximum at 331.6 mm, so it was set to 332. IMD, being gauge-based, reads
    # 1.34x higher than ERA5 for this event (2.09x for 2019, 1.49x for 2021) --
    # a reanalysis smooths orographic extremes, and the Western Ghats flank is
    # where that hurts most. IMD is the authority, so 443 it is.
    #
    # Raising the reference makes every *other* scenario less severe, because
    # the observed 2018 extent is now attributed to a larger storm. The
    # calibration itself is untouched: hazard equals susceptibility at the
    # reference depth whatever that depth is.
    #
    # Caveat that remains: no rainfall product captures the Periyar reservoir
    # releases that drove much of the 2018 inundation, so this is still a proxy
    # for total forcing rather than the whole story.
    reference_event_mm: float = 443.0


@dataclass(frozen=True)
class HydrologyConfig:
    """SCS Curve Number runoff configuration."""
    # Curve numbers, AMC II, hydrologic soil group C. Kerala's uplands are
    # laterite (HSG C) and the coastal strip is alluvium (HSG B); C is the
    # conservative single-group choice. Values follow USDA NEH-630 Table 2-2.
    #
    # LULC class identities were derived empirically from this dataset by
    # cross-tabulating each class against NDVI, NDWI, elevation, slope and the
    # urban mask -- the raster does not use the standard ESA WorldCover codes.
    curve_numbers: dict = field(default_factory=lambda: {
        1: 100.0,  # permanent water (backwaters) - NDVI 0.05, DEM 5.5 m
        2: 70.0,   # tree cover        - NDVI 0.48, DEM 93 m, slope 6.3 deg
        4: 90.0,   # wetland / paddy   - DEM 4.4 m, 28% flooded in 2018
        5: 80.0,   # cropland / grass  - NDVI 0.41, DEM 21 m
        7: 88.0,   # built-up          - coincides exactly with urban mask
        8: 89.0,   # bare / sparse     - marginal class, 5.9k px
        11: 74.0,  # shrubland         - DEM 114 m, slope 9.6 deg
    })
    default_curve_number: float = 80.0

    # Initial abstraction ratio. The classic SCS value is 0.20; re-analysis of
    # the USDA rainfall-runoff database supports ~0.05, which is now widely
    # preferred. 0.05 requires S to be rescaled (S005 = 1.33 * S020^1.15).
    initial_abstraction_ratio: float = 0.05

    # Antecedent moisture condition. Monsoon-season Kerala is wet, so AMC III
    # (saturated) is the right default for a flood-forecasting product.
    amc: str = "III"  # "I" (dry), "II" (average), "III" (wet)

    # Sensitivity of flood odds to runoff, in logit units per natural-log unit
    # of the runoff ratio Q(P) / Q(P_reference). beta = 1.8 means halving the
    # runoff depth relative to the reference storm multiplies the flood odds
    # by exp(-1.8 * ln 2) = 0.29. See hazard.combine.
    runoff_logit_beta: float = 1.8

@dataclass(frozen=True)
class RiskThresholds:
    """
    Flood risk classification thresholds, on the calibrated probability scale.

    These are NOT round numbers, and they should not be. They were read off
    the precision-recall curve of the reference-event hazard map against the
    2018 Sentinel-1 inventory, so each boundary has an operational meaning:

        safe / moderate  0.022  captures 95% of the observed flood extent
        moderate / high  0.070  captures 81%, precision 0.22
        high / critical  0.133  the maximum-F1 point; captures 54%, precision 0.33
        critical         0.271  precision 0.53 -- a coin-flip or better

    The district base rate is 1.4%, so even the lowest band is ~1.6x the
    no-skill rate and the critical band is ~38x it.

    The previous values (0.10 / 0.20 / 0.30 / 0.50) were inherited from the
    uncalibrated score, which was inflated by roughly 11x. Carried onto the
    corrected scale they classified the actual 2018 catastrophe as
    "monitoring active".

    Re-derive with the precision-recall curve whenever the model is retrained;
    the thresholds are properties of the fitted probabilities, not constants.
    """
    safe: float = 0.022
    moderate: float = 0.070
    high: float = 0.133
    critical: float = 0.271

    # Alert triggers, as a fraction of the mapped district area. Calibrated so
    # that the reference event (400 mm, 0.65% of the district in the critical
    # class) trips CRITICAL.
    critical_area_fraction_alert: float = 0.005
    elevated_area_fraction_warning: float = 0.02

    # For socio-economic estimates (planning figures, not model outputs)
    residential_fraction: float = 0.18
    damage_per_km2_crores: float = 50.0
    hospitals_total: int = 15

@dataclass(frozen=True)
class VisualizationConfig:
    """Visualization configuration."""
    # Flood probability colormap. The stops sit exactly on the RiskThresholds
    # band edges, so a colour change on the map means a class change in the
    # statistics. They are deliberately bunched at the low end: the calibrated
    # probabilities are genuine, and at a 1.4% district base rate almost all
    # of the mass sits below 0.10. An evenly spaced ramp renders the whole
    # district flat green.
    flood_colors: list = field(default_factory=lambda: [
        (0.000, "#1a9850"),  # Safe
        (0.022, "#91cf60"),  # Moderate (95% of observed flooding is above this)
        (0.070, "#fee08b"),  # High
        (0.133, "#fdae61"),  # Severe (max-F1 operating point)
        (0.271, "#d73027"),  # Critical (precision 0.53)
        (1.000, "#a50026"),  # Extreme
    ])
    
    # LULC colours. Class identities were derived empirically from this
    # dataset (see LULC_CLASS_NAMES); the raster does not use standard ESA
    # WorldCover codes, so the previous mapping mislabelled every class.
    lulc_colors: dict = field(default_factory=lambda: {
        1:  (0, 80, 200, 255),    # Permanent water
        2:  (0, 110, 0, 255),     # Tree cover
        4:  (0, 190, 200, 255),   # Wetland / paddy
        5:  (200, 200, 60, 255),  # Cropland / grassland
        7:  (220, 40, 40, 255),   # Built-up
        8:  (180, 170, 150, 255), # Bare / sparse
        11: (170, 190, 90, 255),  # Shrubland
    })

@dataclass(frozen=True)
class APIConfig:
    """FastAPI configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass(frozen=True)
class StreamlitConfig:
    """Streamlit configuration."""
    port: int = 8501
    address: str = "0.0.0.0"
    theme_base: str = "light"

# ──────────────────────────────────────────────
# Global Config Instances
# ──────────────────────────────────────────────
RASTER = RasterConfig()
MODEL = ModelConfig()
GEO = GeoConfig()
RAINFALL = RainfallConfig()
RISK = RiskThresholds()
VIZ = VisualizationConfig()
API = APIConfig()
STREAMLIT = StreamlitConfig()
HYDRO = HydrologyConfig()

# ──────────────────────────────────────────────
# Aligned data (produced by align_data.py)
# ──────────────────────────────────────────────
ALIGNED_DIR = PROJECT_ROOT / "data_aligned"

# Class identities established empirically from this dataset by
# cross-tabulating each class against NDVI, NDWI, DEM, slope, the urban mask
# and the 2018 flood inventory. Documented so the mapping can be audited.
LULC_CLASS_NAMES = {
    1: "Permanent water",
    2: "Tree cover",
    4: "Wetland / paddy",
    5: "Cropland / grassland",
    7: "Built-up",
    8: "Bare / sparse",
    11: "Shrubland",
}

# LULC class treated as standing water year-round. Sentinel-1 cannot tell
# permanent water from flood water, so these pixels are removed from the
# label set -- 80.3% of the raw "flood" inventory falls here, and leaving
# them in trains a lake detector rather than a flood model.
PERMANENT_WATER_CLASS = 1

# ──────────────────────────────────────────────
# File Path Helpers
# ──────────────────────────────────────────────
def get_model_path(name: str) -> Path:
    """Get full path to model file."""
    return MODELS_DIR / name

def get_output_path(name: str) -> Path:
    """Get full path to output file."""
    return OUTPUT_DIR / name

def get_processed_path(name: str) -> Path:
    """Get full path to processed file."""
    return PROCESSED_DIR / name

def get_geoai_new_path(name: str) -> Path:
    """Get full path to GeoAI_New file."""
    return GEOAI_NEW_DIR / name

def get_static_path(name: str) -> Path:
    """Get full path to static file."""
    return STATIC_DIR / name

# ──────────────────────────────────────────────
# Feature Channel Definitions
# ──────────────────────────────────────────────
# Conditioning factors for the susceptibility model, as produced by
# align_data.py. Order is fixed: it defines the model's input vector and is
# persisted alongside the trained model so the two cannot drift apart.
SUSCEPTIBILITY_FEATURES = [
    "dem",         # elevation (m)
    "slope",       # degrees
    "hand",        # height above nearest drainage (m) - dominant control
    "twi",         # topographic wetness index
    "tpi",         # topographic position index
    "spi",         # stream power index (signed log)
    "flow",        # log1p flow accumulation
    "river_dist",  # distance to drainage (m)
    "urban_dist",  # distance to built-up (m)
    "ndvi",        # vegetation
    "ndwi",        # surface water / moisture
    "urban_mask",  # impervious fraction proxy
    "curve_number",  # derived in hydrology.py from LULC
    # Context features (src/derive_features.py). Everything above describes
    # the pixel itself; these describe what surrounds it, which is what a
    # pixel-independent model structurally cannot see.
    "upstream_cn",    # catchment-average curve number, routed on the D8 network
    "dem_rel_1km",    # elevation relative to the ~1 km neighbourhood mean
]

# Legacy PyTorch UNet channel orders, kept so the archived .pth models remain
# loadable. These reference the lowercase aligned rasters in data_aligned/.
PYTORCH_STANDARD_FEATURES = ["dem", "slope", "flow", "lulc"]
PYTORCH_ROBUST_FEATURES = PYTORCH_STANDARD_FEATURES + ["sar_vv", "sar_vh"]
PYTORCH_SUPERCHARGED_FEATURES = PYTORCH_ROBUST_FEATURES + [
    "twi", "river_dist", "urban_dist",
]

# ──────────────────────────────────────────────
# Model File Names
# ──────────────────────────────────────────────
# NOTE: geoai_flood_final.pth is a different architecture (64 base channels,
# `inc.double_conv.*` keys) from the UNet in inference_final.py (32 base
# channels, `inc.*` keys) and cannot be loaded by it. It is deliberately
# absent from this map rather than listed and broken.
MODEL_FILES = {
    "pytorch_standard": "flood_model_real2018.pth",       # 4 channels
    "pytorch_robust": "flood_model_robust_sar.pth",       # 6 channels
    "pytorch_supercharged": "flood_model_supercharged.pth",  # 9 channels
}

# Susceptibility model produced by src/susceptibility.py
SUSCEPTIBILITY_MODEL = "susceptibility_model.joblib"

# ──────────────────────────────────────────────
# Known Locations for Search
# ──────────────────────────────────────────────
KNOWN_PLACES = {
    "Ernakulam": [9.980, 76.280],
    "MG Road": [9.966, 76.287],
    "Edappally": [10.024, 76.308],
    "Kaloor": [9.994, 76.292],
    "Vyttila": [9.966, 76.318],
    "Aluva": [10.108, 76.357],
    "Kakkanad": [10.011, 76.340],
    "Perumbavoor": [10.109, 76.475],
    "Muvattupuzha": [9.982, 76.582],
    "North Paravur": [10.158, 76.214],
}

# ──────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────
def validate_environment() -> list[str]:
    """Validate that required directories and files exist. Returns list of warnings."""
    warnings = []
    
    for name, path in [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("SRC_DIR", SRC_DIR),
        ("PROCESSED_DIR", PROCESSED_DIR),
        ("OUTPUT_DIR", OUTPUT_DIR),
        ("MODELS_DIR", MODELS_DIR),
        ("STATIC_DIR", STATIC_DIR),
        ("GEOAI_NEW_DIR", GEOAI_NEW_DIR),
    ]:
        if not path.exists():
            warnings.append(f"{name} does not exist: {path}")
    
    # Check for at least one model
    model_files = list(MODELS_DIR.glob("*.pth")) + list(MODELS_DIR.glob("*.h5"))
    if not model_files:
        warnings.append(f"No model files found in {MODELS_DIR}")
    
    return warnings

# Run validation on import
_ENV_WARNINGS = validate_environment()
for w in _ENV_WARNINGS:
    LOGGER.warning(w)