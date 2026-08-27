"""
Tests for the display-raster packaging path.

`display/` holds pre-downsampled copies of the static layers so a container
image carries 21 MB instead of the 3.7 GB of full-resolution originals. The
risk this introduces is a silent one: if `get_layer_path()` resolved to the
wrong directory, or the reduction changed the values, the app would keep
rendering -- just from the wrong pixels. These pin the resolution order and
the fidelity of the reduction.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import RASTER  # noqa: E402
from data_loading import LAYER_REGISTRY, get_layer_path  # noqa: E402


def _write(path, array, nodata=RASTER.nodata_value):
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 1100000.0),
        nodata=nodata,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(np.float32), 1)


LAYER = "DEM"
FILENAME = LAYER_REGISTRY[LAYER][0]


class TestLayerPathResolution:
    def test_prefers_display_over_full_resolution(self, tmp_path, monkeypatch):
        import data_loading

        display = tmp_path / "display"
        source = tmp_path / "GeoAI_New"
        _write(display / FILENAME, np.ones((10, 10), dtype=np.float32))
        _write(source / FILENAME, np.ones((100, 100), dtype=np.float32))

        monkeypatch.setattr(data_loading, "DISPLAY_DIR", display)
        monkeypatch.setattr(data_loading, "GEOAI_NEW_DIR", source)

        assert get_layer_path(LAYER) == display / FILENAME

    def test_falls_back_to_full_resolution_when_no_display_dir(self, tmp_path, monkeypatch):
        """A local checkout has no display/ and must behave exactly as before."""
        import data_loading

        source = tmp_path / "GeoAI_New"
        _write(source / FILENAME, np.ones((100, 100), dtype=np.float32))

        monkeypatch.setattr(data_loading, "DISPLAY_DIR", tmp_path / "does_not_exist")
        monkeypatch.setattr(data_loading, "GEOAI_NEW_DIR", source)

        assert get_layer_path(LAYER) == source / FILENAME

    def test_explicit_directory_is_not_redirected_to_display(self, tmp_path, monkeypatch):
        """
        An explicit geoai_dir is a caller override. Silently redirecting it to
        display/ would make it impossible to read the originals on purpose --
        which src/make_display_rasters.py itself must do to build them.
        """
        import data_loading

        display = tmp_path / "display"
        source = tmp_path / "GeoAI_New"
        _write(display / FILENAME, np.ones((10, 10), dtype=np.float32))
        _write(source / FILENAME, np.ones((100, 100), dtype=np.float32))

        monkeypatch.setattr(data_loading, "DISPLAY_DIR", display)

        assert get_layer_path(LAYER, geoai_dir=source) == source / FILENAME

    def test_unknown_layer_returns_none(self):
        assert get_layer_path("Not A Real Layer") is None

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        import data_loading

        monkeypatch.setattr(data_loading, "DISPLAY_DIR", tmp_path / "nope")
        monkeypatch.setattr(data_loading, "GEOAI_NEW_DIR", tmp_path / "also_nope")
        assert get_layer_path(LAYER) is None


class TestReductionFidelity:
    def test_reduction_preserves_values_within_display_size(self, tmp_path):
        """
        A raster already at or below max_dim must survive the round trip
        unchanged: read_downsampled short-circuits resampling at scale >= 1,
        so building a display copy of it is a value-preserving operation.
        """
        from make_display_rasters import build

        source = tmp_path / "GeoAI_New"
        out = tmp_path / "display"
        rng = np.random.default_rng(0)
        original = rng.uniform(1.0, 500.0, size=(40, 50)).astype(np.float32)
        _write(source / FILENAME, original)

        summary = build(source_dir=source, out_dir=out, max_dim=1000)

        assert LAYER in summary["layers_written"]
        with rasterio.open(out / FILENAME) as src:
            written = src.read(1)
        assert np.allclose(written, original, atol=1e-3)

    def test_downsamples_when_source_exceeds_max_dim(self, tmp_path):
        from make_display_rasters import build

        source = tmp_path / "GeoAI_New"
        out = tmp_path / "display"
        _write(source / FILENAME, np.full((400, 400), 12.0, dtype=np.float32))

        build(source_dir=source, out_dir=out, max_dim=100)

        with rasterio.open(out / FILENAME) as src:
            assert max(src.height, src.width) <= 100
            # A constant field must stay that constant through resampling --
            # if it does not, nodata is bleeding into the average.
            assert src.read(1)[0, 0] == pytest.approx(12.0, abs=1e-3)

    def test_missing_source_is_skipped_not_fatal(self, tmp_path):
        """
        A partial GeoAI_New must not abort the packaging step: the layers that
        do exist should still be written.
        """
        from make_display_rasters import build

        source = tmp_path / "GeoAI_New"
        out = tmp_path / "display"
        _write(source / FILENAME, np.ones((20, 20), dtype=np.float32))

        summary = build(source_dir=source, out_dir=out, max_dim=1000)

        assert LAYER in summary["layers_written"]
        assert len(summary["layers_skipped"]) == len(LAYER_REGISTRY) - 1

    def test_nodata_sentinel_is_carried_through(self, tmp_path):
        from make_display_rasters import build

        source = tmp_path / "GeoAI_New"
        out = tmp_path / "display"
        arr = np.full((30, 30), 50.0, dtype=np.float32)
        arr[0, :] = RASTER.nodata_value
        _write(source / FILENAME, arr)

        build(source_dir=source, out_dir=out, max_dim=1000)

        with rasterio.open(out / FILENAME) as src:
            assert src.nodata == RASTER.nodata_value
            written = src.read(1)
        # The nodata band must still read as nodata, not as an interpolated
        # value somewhere between -9999 and 50.
        assert (written[0, :] < -9000).all()
