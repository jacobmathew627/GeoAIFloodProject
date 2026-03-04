import rasterio
import numpy as np
import os
from pyproj import Transformer

# Known waterlogging spots in Ernakulam/Kochi
SPOTS = {
    "M.G. Road (Urban Center)": (9.9723, 76.2821),
    "Edappally (High Traffic/Canal)": (10.0252, 76.3101),
    "Kalamassery (Low Lying Industrial)": (10.0381, 76.3262),
    "Kochi InfoPark (Marshland Fringe)": (10.0094, 76.3639)
}

OUTPUTS_DIR = "outputs"

def validate_spots():
    print("Validating Supercharged GeoAI against Kochi Hotspots (200mm Scenario)...")
    
    # Models to compare
    modes = ["standard", "supercharged"]
    
    for spot_name, (lat, lon) in SPOTS.items():
        print(f"Spot: {spot_name} ({lat}, {lon})")
        
        for mode in modes:
            suffix = "" if mode == "standard" else f"_{mode}"
            path = os.path.join(OUTPUTS_DIR, f"flood_prob_200mm{suffix}.tif")
            
            if not os.path.exists(path):
                print(f"  [{mode.upper()}] File missing: {path}")
                continue
                
            with rasterio.open(path) as src:
                # Coordinate transformation to UTM (the CRS of our aligned rasters)
                # Assuming CRS is UTM 43N (EPSG:32643) based on previous context
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                utx, uty = transformer.transform(lon, lat)
                
                # Sample the pixel
                row, col = src.index(utx, uty)
                
                # Bounds check
                if 0 <= row < src.height and 0 <= col < src.width:
                    val = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0,0]
                    risk = "High" if val > 0.7 else ("Medium" if val > 0.4 else "Low")
                    print(f"  -> {mode.upper()}: Prob {val:.2f} | Risk: {risk}")
                else:
                    print(f"  -> {mode.upper()}: Outside raster bounds.")

if __name__ == "__main__":
    validate_spots()
