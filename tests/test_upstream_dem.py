"""
Tests for the upstream DEM acquisition.

Only the offline parts: tile arithmetic, the terrarium decode and the validity
band. Fetching needs the network. Routing on this DEM is not tested because it
is not finished -- see the module docstring.
"""
import numpy as np
import pytest

from upstream_dem import (
    BASIN_BBOX,
    MAX_VALID_ELEV_M,
    MIN_VALID_ELEV_M,
    deg2tile,
    tile2deg,
    tile_range,
)


def decode_terrarium(r, g, b):
    """The encoding the module relies on."""
    return (r * 256.0 + g + b / 256.0) - 32768.0


class TestTileArithmetic:
    def test_round_trips_to_the_containing_tile(self):
        lon, lat = 76.28, 9.97  # Kochi
        z = 12
        x, y = deg2tile(lon, lat, z)
        west, north = tile2deg(x, y, z)
        east, south = tile2deg(x + 1, y + 1, z)
        assert west <= lon <= east
        assert south <= lat <= north

    def test_zoom_increases_tile_count_fourfold(self):
        xs_a, ys_a = tile_range(BASIN_BBOX, 10)
        xs_b, ys_b = tile_range(BASIN_BBOX, 11)
        ratio = (len(xs_b) * len(ys_b)) / (len(xs_a) * len(ys_a))
        assert 3.0 <= ratio <= 6.0

    def test_range_covers_the_bbox_corners(self):
        z = 12
        xs, ys = tile_range(BASIN_BBOX, z)
        x0, y0 = deg2tile(BASIN_BBOX[0], BASIN_BBOX[3], z)
        x1, y1 = deg2tile(BASIN_BBOX[2], BASIN_BBOX[1], z)
        assert x0 in xs and x1 in xs
        assert y0 in ys and y1 in ys

    def test_y_increases_southward(self):
        z = 12
        _, y_north = deg2tile(76.3, 10.6, z)
        _, y_south = deg2tile(76.3, 9.5, z)
        assert y_south > y_north


class TestBasinBbox:
    def test_contains_ernakulam(self):
        lon_min, lat_min, lon_max, lat_max = BASIN_BBOX
        assert lon_min < 76.30 < lon_max
        assert lat_min < 10.00 < lat_max

    def test_reaches_the_western_ghats_crest(self):
        """The Periyar headwaters sit near 77.2 E; the box must pass them."""
        assert BASIN_BBOX[2] >= 77.2

    def test_is_well_formed(self):
        assert BASIN_BBOX[0] < BASIN_BBOX[2]
        assert BASIN_BBOX[1] < BASIN_BBOX[3]


class TestTerrariumDecode:
    def test_sea_level_encoding(self):
        # 32768 = 128 * 256 -> exactly 0 m
        assert decode_terrarium(128, 0, 0) == pytest.approx(0.0)

    def test_decodes_a_known_summit_band(self):
        """Anamudi is 2,695 m; that must land inside the valid band."""
        elev = decode_terrarium(138, 135, 0)  # 138*256+135-32768 = 2695
        assert elev == pytest.approx(2695.0)
        assert MIN_VALID_ELEV_M < elev < MAX_VALID_ELEV_M

    def test_ocean_bathymetry_is_rejected(self):
        """
        Terrarium carries bathymetry, so sea decodes to large negatives -- an
        Arabian Sea tile reads about -11,600 m. Treating it as terrain would
        route the whole district into the ocean floor.
        """
        deep = decode_terrarium(83, 0, 0)
        assert deep < -11_000
        assert deep < MIN_VALID_ELEV_M

    def test_spikes_are_rejected(self):
        """Single-pixel artefacts reached 10,506 m in the real mosaic."""
        assert 10_506 > MAX_VALID_ELEV_M

    def test_validity_band_is_physical_for_this_basin(self):
        assert MIN_VALID_ELEV_M < 0
        # Anamudi 2,695 m must fit; Himalayan values must not.
        assert 2695 < MAX_VALID_ELEV_M < 4000

    def test_decode_is_monotone_in_the_red_channel(self):
        vals = [decode_terrarium(r, 0, 0) for r in (0, 64, 128, 192, 255)]
        assert all(b > a for a, b in zip(vals, vals[1:]))
