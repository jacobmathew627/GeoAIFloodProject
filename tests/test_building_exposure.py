"""
Tests for the OSM building exposure configuration.

Fetching and rasterising need the network and the aligned grid, so what is
pinned here is the configuration, the bbox arithmetic, the rate conversion,
and the labelling discipline: this module reports exposure (what is there),
never a damage prediction (what a flood would do to it), because no citable
India-specific depth-damage function was available to ground the latter.
"""

from config import GEO
from building_exposure import RS_PER_M2, RS_PER_SQFT, _bbox


class TestBbox:
    def test_returns_south_west_north_east(self):
        south, west, north, east = _bbox()
        assert south < north
        assert west < east

    def test_matches_the_district_exactly(self):
        """
        Unlike osm_drainage.py's widened bbox (a channel just outside the
        boundary still matters), a building outside the district contributes
        nothing to a district exposure figure -- so this one is unbuffered.
        """
        min_lon, min_lat, max_lon, max_lat = GEO.district_bbox
        south, west, north, east = _bbox()
        assert south == min_lat
        assert north == max_lat
        assert west == min_lon
        assert east == max_lon


class TestConstructionRate:
    def test_rate_is_a_realistic_kerala_urban_rate(self):
        # Kerala PWD 2025 mid-range/standard urban rate is documented at
        # roughly Rs 2,000-3,500/sqft (see module docstring); this pins the
        # module to a value inside that band rather than an arbitrary one.
        assert 1800.0 <= RS_PER_SQFT <= 3500.0

    def test_per_m2_conversion_is_correct(self):
        # 1 sqft = 0.092903 m2, so Rs/m2 should be noticeably larger than Rs/sqft.
        assert RS_PER_M2 == RS_PER_SQFT / 0.092903
        assert 20000.0 <= RS_PER_M2 <= 40000.0


class TestLabellingDiscipline:
    def test_module_documents_exposure_vs_damage(self):
        import building_exposure

        doc = building_exposure.__doc__ or ""
        for token in ("exposure", "not a damage", "replacement"):
            assert token in doc, f"module docstring should mention {token!r}"

    def test_summary_caveat_names_the_limitation(self):
        import inspect

        import building_exposure

        source = inspect.getsource(building_exposure.build)
        assert "depth-damage" in source
