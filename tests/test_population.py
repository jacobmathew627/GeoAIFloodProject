"""
Tests for src/population.py -- the WorldPop align() nodata regression.

GEE's WorldPop export leaves `nodata` unset in the GeoTIFF header while
genuinely using -99999 as its out-of-ROI sentinel. align() must recognise
that sentinel even when `src.nodata is None`, both in the reproject() call
and in the diagnostic sum -- otherwise -99999 blends into the bilinear
resampling like a real population value, or dominates a naive sum.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RASTER  # noqa: E402
from population import align  # noqa: E402


def _write_raster(path, array, transform, crs="EPSG:32643", nodata=None):
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
        crs=crs,
        transform=transform,
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array, 1)


@pytest.fixture
def aligned_dir(tmp_path):
    d = tmp_path / "data_aligned"
    d.mkdir()
    # 20x20 master grid at 10 m, north-up, 200x200 m extent -- the "district"
    # is the whole grid here so conservation can be checked directly.
    transform = Affine(10.0, 0.0, 0.0, 0.0, -10.0, 200.0)
    lulc = np.ones((20, 20), dtype=np.float32)
    _write_raster(d / "lulc_aligned.tif", lulc, transform, nodata=-9999.0)
    return d


class TestAlignSentinelHandling:
    def test_gee_sentinel_does_not_leak_into_output(self, tmp_path, aligned_dir):
        """The exact bug: nodata tag unset, real sentinel is -99999."""
        src_transform = Affine(50.0, 0.0, 0.0, 0.0, -50.0, 200.0)
        src = np.full((4, 4), 100.0, dtype=np.float32)
        src[0, 2] = -99999.0
        src[2, 1] = -99999.0
        src_path = tmp_path / "worldpop_2020.tif"
        _write_raster(src_path, src, src_transform, nodata=None)

        out_path = align(src_path, aligned_dir=aligned_dir)

        with rasterio.open(out_path) as f:
            out = f.read(1)
            out_nodata = f.nodata

        valid = out[out != out_nodata]
        assert valid.size > 0
        # A leaked sentinel would show up as a large negative value (roughly
        # -99999 scaled by the destination/source cell-area ratio, i.e. in
        # the thousands). Every real population count is >= 0.
        assert valid.min() > -1.0

    def test_output_nodata_tag_matches_project_convention(self, tmp_path, aligned_dir):
        src_transform = Affine(50.0, 0.0, 0.0, 0.0, -50.0, 200.0)
        src = np.full((4, 4), 50.0, dtype=np.float32)
        src_path = tmp_path / "worldpop_2020.tif"
        _write_raster(src_path, src, src_transform, nodata=None)

        out_path = align(src_path, aligned_dir=aligned_dir)

        with rasterio.open(out_path) as f:
            assert f.nodata == RASTER.nodata_value

    def test_conserves_total_when_no_sentinel_present(self, tmp_path, aligned_dir):
        """Sanity check independent of the sentinel bug: plain resampling
        should conserve the source total (within bilinear-boundary tolerance),
        not silently inflate or shrink it by the ~100x disaggregation trap
        documented in the module docstring."""
        src_transform = Affine(50.0, 0.0, 0.0, 0.0, -50.0, 200.0)
        src = np.full((4, 4), 100.0, dtype=np.float32)
        src_path = tmp_path / "worldpop_2020.tif"
        _write_raster(src_path, src, src_transform, nodata=None)
        true_total = float(src.sum())

        out_path = align(src_path, aligned_dir=aligned_dir)

        with rasterio.open(out_path) as f:
            out = f.read(1)
            out_nodata = f.nodata
        total_after = float(out[out != out_nodata].sum())

        assert total_after == pytest.approx(true_total, rel=0.1)

    def test_sentinel_reduces_conserved_total_proportionally(self, tmp_path, aligned_dir):
        """With 2 of 16 source cells masked out, the conserved total should
        drop by roughly their share, not swing wildly positive or negative."""
        src_transform = Affine(50.0, 0.0, 0.0, 0.0, -50.0, 200.0)
        src = np.full((4, 4), 100.0, dtype=np.float32)
        src[0, 2] = -99999.0
        src[2, 1] = -99999.0
        src_path = tmp_path / "worldpop_2020.tif"
        _write_raster(src_path, src, src_transform, nodata=None)
        true_total = 14 * 100.0

        out_path = align(src_path, aligned_dir=aligned_dir)

        with rasterio.open(out_path) as f:
            out = f.read(1)
            out_nodata = f.nodata
        total_after = float(out[out != out_nodata].sum())

        assert total_after == pytest.approx(true_total, rel=0.2)

    def test_explicit_source_nodata_tag_is_still_respected(self, tmp_path, aligned_dir):
        """When the source file DOES set nodata properly (unlike the real
        GEE export), align() must not override it with the -99999 fallback."""
        src_transform = Affine(50.0, 0.0, 0.0, 0.0, -50.0, 200.0)
        src = np.full((4, 4), 100.0, dtype=np.float32)
        src[0, 0] = -1.0  # an explicit, different sentinel
        src_path = tmp_path / "worldpop_2020.tif"
        _write_raster(src_path, src, src_transform, nodata=-1.0)

        out_path = align(src_path, aligned_dir=aligned_dir)

        with rasterio.open(out_path) as f:
            out = f.read(1)
            out_nodata = f.nodata
        valid = out[out != out_nodata]
        assert valid.min() > -1.0
