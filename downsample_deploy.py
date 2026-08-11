import os
import rasterio
from rasterio.enums import Resampling

deploy_dir = '../hf_deploy'

for root, _, files in os.walk(deploy_dir):
    for f in files:
        if f.endswith('.tif'):
            path = os.path.join(root, f)
            print(f'Processing {path}...')
            with rasterio.open(path) as src:
                scale_factor = 1000 / max(src.width, src.height)
                if scale_factor >= 1.0: 
                    print(f'Skipping {f}, already small enough.')
                    continue
                new_h, new_w = int(src.height * scale_factor), int(src.width * scale_factor)
                resamp = Resampling.nearest if ('LULC' in f or 'Mask' in f) else Resampling.bilinear
                data = src.read(out_shape=(src.count, new_h, new_w), resampling=resamp)
                t = src.transform * src.transform.scale((src.width / data.shape[-1]), (src.height / data.shape[-2]))
                profile = src.profile
                profile.update(transform=t, width=new_w, height=new_h)
            
            # Write back out
            with rasterio.open(path, 'w', **profile) as dst:
                dst.write(data)
            print(f'Successfully downsampled {f} to {new_h}x{new_w}')
