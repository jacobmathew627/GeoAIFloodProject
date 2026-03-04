import rasterio
import os

data_dir = "."
files = [f for f in os.listdir(data_dir) if f.endswith('.tif')]

print(f"{'File':<30} {'Shape':<15} {'CRS':<10} {'Dtype':<10} {'Bounds'}")
print("-" * 100)

for f in files:
    path = os.path.join(data_dir, f)
    try:
        with rasterio.open(path) as src:
            print(f"{f:<30} {str(src.shape):<15} {str(src.crs):<10} {str(src.dtypes[0]):<10} {src.bounds}")
    except Exception as e:
        print(f"{f:<30} Error: {e}")
