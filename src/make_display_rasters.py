"""
Pre-downsample the static display layers, so a container image does not have to
carry 3.7 GB of rasters to draw a 1000 px map.

Why
---
Every static layer the dashboard shows is read through
`data_loading.read_downsampled()`, which resamples to `RASTER.max_dimension`
(1000 px on the long edge) before the colormap is applied. Nothing in the UI --
not the overlay, not the map-click readout, which samples the already-reduced
array -- ever touches full resolution. The full-resolution files exist because
they are the *model's* inputs, not the display's.

That made the Docker image about 8 GB, which is over the image-size limit of
most hosts. This script writes the same 14 layers, already reduced, into
`display/`. `data_loading.get_layer_path()` prefers that directory when it
exists and falls back to `GeoAI_New/`, so:

  * a local checkout is unchanged (no `display/` -> reads GeoAI_New/ as before)
  * an image ships ~50 MB of display rasters instead of ~1 GB, and can leave
    GeoAI_New/ out entirely

Fidelity
--------
The reduction uses `read_downsampled()` itself rather than a reimplementation,
so the pixels written here are byte-identical to what the app would have
computed on the fly. Reading the result back is then a no-op resample
(`scale == 1.0`, so `read_downsampled` returns the file as-is) plus a second
application of the per-layer nodata rules, which is idempotent: those rules only
ever move an in-range value to the nodata sentinel, and already-sentinel pixels
are excluded by the `data > -9000` guard at the top of
`_apply_layer_nodata_rules`.

This is a *packaging* step. It does not touch data_aligned/, which is what the
susceptibility model is actually trained and predicted from.

Run:  python src/make_display_rasters.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import DISPLAY_DIR, GEOAI_NEW_DIR, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")


def build(
    source_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    max_dim: Optional[int] = None,
) -> Dict[str, object]:
    """Write display-resolution copies of every layer in LAYER_REGISTRY."""
    import rasterio

    from data_loading import LAYER_REGISTRY, read_downsampled

    source_dir = source_dir or GEOAI_NEW_DIR
    out_dir = out_dir or DISPLAY_DIR
    max_dim = max_dim or RASTER.max_dimension
    out_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Dict[str, object]] = {}
    skipped: Dict[str, str] = {}
    bytes_in = 0
    bytes_out = 0

    for layer_name, (filename, layer_kind) in LAYER_REGISTRY.items():
        src_path = source_dir / filename
        if not src_path.exists():
            LOGGER.warning("  %-26s SKIP (missing %s)", layer_name, filename)
            skipped[layer_name] = f"missing {filename}"
            continue

        data, meta = read_downsampled(src_path, layer_kind=layer_kind, max_dim=max_dim)
        if data is None or meta is None:
            LOGGER.warning("  %-26s SKIP (unreadable)", layer_name)
            skipped[layer_name] = "unreadable"
            continue

        # Keep the source filename so get_layer_path() can resolve either
        # directory with the same LAYER_REGISTRY entry.
        out_path = out_dir / filename
        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": meta["crs"],
            "transform": meta["transform"],
            "nodata": RASTER.nodata_value,
            "compress": "lzw",
            # Predictor 3 is the floating-point predictor: these are smooth
            # continuous surfaces, so it compresses them substantially better
            # than LZW alone.
            "predictor": 3,
        }
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data.astype(np.float32), 1)

        size_in = src_path.stat().st_size
        size_out = out_path.stat().st_size
        bytes_in += size_in
        bytes_out += size_out
        written[layer_name] = {
            "file": filename,
            "shape": list(data.shape),
            "bytes": size_out,
        }
        LOGGER.info(
            "  %-26s %s  %.1f MB -> %.2f MB  (%.0fx smaller)",
            layer_name,
            f"{data.shape[0]}x{data.shape[1]}",
            size_in / 1e6,
            size_out / 1e6,
            size_in / max(size_out, 1),
        )

    LOGGER.info(
        "Wrote %d layers to %s: %.0f MB -> %.1f MB (%.0fx smaller)",
        len(written),
        out_dir,
        bytes_in / 1e6,
        bytes_out / 1e6,
        bytes_in / max(bytes_out, 1),
    )
    if skipped:
        LOGGER.warning("Skipped %d layers: %s", len(skipped), skipped)

    return {
        "max_dim": max_dim,
        "source_dir": str(source_dir),
        "out_dir": str(out_dir),
        "layers_written": written,
        "layers_skipped": skipped,
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Build display-resolution rasters")
    parser.add_argument("--max-dim", type=int, default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    LOGGER.info("Building display rasters...")
    build(max_dim=args.max_dim)


if __name__ == "__main__":  # pragma: no cover
    main()
