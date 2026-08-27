"""
Tests for routing the upstream DEM.

The routing itself needs WhiteboxTools and a 25,000 km2 DEM, so what is tested
here is the validation logic -- which is the part that matters. The previous
routing attempt produced a flow network that looked like rivers and was wrong by
two orders of magnitude, and it was the catchment-area probe that caught it.
These tests pin that probe's behaviour, including that it fails loudly rather
than quietly aligning a bad raster.
"""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from upstream_routing import (
    BREACH_DIST_CELLS,
    PROBES,
    RATIO_OK,
    SNAP_RADIUS_M,
    validate,
)


@pytest.fixture
def accum(tmp_path):
    """
    A synthetic accumulation raster covering the probe points.

    The grid is derived from the probe coordinates rather than hardcoded, so it
    cannot drift out from under them if a probe is added or moved. Values are in
    m2, matching WhiteboxTools' "catchment area" output.
    """
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    xs, ys = zip(*(to_grid.transform(p["lon"], p["lat"]) for p in PROBES))

    cell = 30.0
    margin = 3000.0  # enough room for the 1 km snap tests plus overshoot
    origin_x = min(xs) - margin
    origin_y = max(ys) + margin
    width = int((max(xs) - min(xs) + 2 * margin) / cell) + 1
    height = int((max(ys) - min(ys) + 2 * margin) / cell) + 1
    transform = from_origin(origin_x, origin_y, cell, cell)

    data = np.full((height, width), 1e5, dtype=np.float32)  # off-channel: 0.1 km2
    path = tmp_path / "accum_area.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return path, transform, (height, width)


def _stamp(path, rowcols_values):
    """Write specific cells into an existing raster."""
    with rasterio.open(path, "r+") as src:
        a = src.read(1)
        for (r, c), v in rowcols_values:
            a[r, c] = v
        src.write(a, 1)


def _rowcol(transform, lon, lat):
    from pyproj import Transformer

    to_grid = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
    x, y = to_grid.transform(lon, lat)
    col, row = ~transform * (x, y)
    return int(row), int(col)


class TestProbeTable:
    def test_every_probe_has_a_published_area_and_a_source(self):
        for p in PROBES:
            assert p["expected_km2"] > 0
            assert p["source"], f"{p['name']} needs a provenance note"

    def test_probes_are_inside_kerala(self):
        for p in PROBES:
            assert 8.0 < p["lat"] < 11.5, p["name"]
            assert 76.0 < p["lon"] < 77.5, p["name"]

    def test_areas_are_ordered_like_the_basins(self):
        """Periyar at Aluva must dominate; it is the largest contributing area."""
        by_name = {p["name"]: p["expected_km2"] for p in PROBES}
        assert by_name["Periyar at Aluva"] == max(by_name.values())

    def test_acceptance_band_is_wide_enough_to_be_honest(self):
        """
        The failure this guards against was off by 100x, not 2x. A tight band
        would fail on DEM resolution and approximate gauge coordinates alone.
        """
        lo, hi = RATIO_OK
        assert lo < 1.0 < hi
        assert hi / lo >= 3.0


class TestValidate:
    def test_reports_ratio_near_one_for_a_correct_network(self, accum):
        path, transform, _ = accum
        stamps = []
        for p in PROBES:
            r, c = _rowcol(transform, p["lon"], p["lat"])
            stamps.append(((r, c), p["expected_km2"] * 1e6))
        _stamp(path, stamps)

        results = validate(path)
        assert len(results) == len(PROBES)
        for r in results:
            assert r["ratio"] == pytest.approx(1.0, abs=0.01), r["name"]
            assert r["snapped_m"] == 0.0

    def test_snaps_to_a_nearby_channel(self, accum):
        """
        A hand-typed coordinate lands off-channel, where accumulation is tiny.
        Without snapping every probe reads near zero and correct routing looks
        broken -- which is a way to reject a good network.
        """
        path, transform, _ = accum
        p = PROBES[0]
        r, c = _rowcol(transform, p["lon"], p["lat"])
        offset = 10  # 300 m away, inside the 1 km snap radius
        _stamp(path, [((r, c + offset), p["expected_km2"] * 1e6)])

        results = validate(path, probes=(p,))
        assert results[0]["ratio"] == pytest.approx(1.0, abs=0.01)
        assert results[0]["snapped_m"] == pytest.approx(offset * 30.0, abs=1.0)

    def test_does_not_snap_beyond_the_radius(self, accum):
        path, transform, _ = accum
        p = PROBES[0]
        r, c = _rowcol(transform, p["lon"], p["lat"])
        far = int(SNAP_RADIUS_M / 30.0) + 20
        _stamp(path, [((r, c + far), p["expected_km2"] * 1e6)])

        results = validate(path, probes=(p,))
        # Only the 0.1 km2 background is reachable, so the ratio collapses.
        assert results[0]["ratio"] < 0.01

    def test_catches_the_old_failure_mode(self, accum):
        """
        The previous routing returned near-zero at every gauge. That must show
        up as a ratio far below the acceptance band, not as a pass.
        """
        path, _, _ = accum  # left at the 0.1 km2 background everywhere
        results = validate(path)
        for r in results:
            assert r["ratio"] < RATIO_OK[0], r["name"]

    def test_handles_probes_outside_the_grid(self, tmp_path):
        transform = from_origin(0.0, 100000.0, 30.0, 30.0)
        path = tmp_path / "tiny.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=10,
            width=10,
            count=1,
            dtype="float32",
            crs="EPSG:32643",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype="float32"), 1)

        results = validate(path)
        assert all(r["found_km2"] is None for r in results)
        assert all("note" in r for r in results)

    def test_negative_sentinels_are_not_read_as_area(self, accum):
        """
        WhiteboxTools writes large negative values for nodata. Treating one as
        an area would be silent nonsense; treating it as the maximum in a snap
        window would be worse.
        """
        path, transform, _ = accum
        p = PROBES[0]
        r, c = _rowcol(transform, p["lon"], p["lat"])
        _stamp(path, [((r, c), -32768.0), ((r, c + 1), p["expected_km2"] * 1e6)])

        results = validate(path, probes=(p,))
        assert results[0]["found_km2"] > 0
        assert results[0]["ratio"] == pytest.approx(1.0, abs=0.01)

    def test_missing_file_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            validate(tmp_path / "nope.tif")


class TestConfiguration:
    def test_breach_distance_spans_a_ghats_valley(self):
        """At 30 m, the search distance must be kilometres, not metres."""
        assert BREACH_DIST_CELLS * 30.0 >= 3000.0

    def test_snap_radius_cannot_cross_basins(self):
        assert 100.0 <= SNAP_RADIUS_M <= 2000.0
