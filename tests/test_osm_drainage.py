"""
Tests for the OSM drainage configuration.

Fetching and rasterising need the network and the aligned grid. What is pinned
here is the configuration and the bbox arithmetic, plus the one finding that
must not be quietly reversed: proximity to a mapped channel is associated with
*more* waterlogging in this city, not less.
"""

from config import GEO
from osm_drainage import DENSITY_RADIUS_M, WATERWAY_CLASSES, _bbox


class TestConfiguration:
    def test_collects_constructed_drainage_and_canals(self):
        assert set(WATERWAY_CLASSES) == {"drain", "ditch", "canal"}

    def test_canal_is_included(self):
        """
        Kochi's storm-water system is its tidal canal network, so excluding
        `canal` would drop the primary drainage route.
        """
        assert "canal" in WATERWAY_CLASSES

    def test_density_radius_is_neighbourhood_scale(self):
        # Large enough to describe a locality, small enough to vary within a city.
        assert 200.0 <= DENSITY_RADIUS_M <= 2000.0


class TestBbox:
    def test_returns_south_west_north_east(self):
        south, west, north, east = _bbox()
        assert south < north
        assert west < east

    def test_contains_the_district(self):
        min_lon, min_lat, max_lon, max_lat = GEO.district_bbox
        south, west, north, east = _bbox()
        assert south <= min_lat and north >= max_lat
        assert west <= min_lon and east >= max_lon

    def test_is_widened_beyond_the_district(self):
        """
        A channel just outside the boundary still drains the edge of the
        district, so the query area is deliberately larger.
        """
        min_lon, min_lat, max_lon, max_lat = GEO.district_bbox
        south, west, north, east = _bbox()
        assert south < min_lat
        assert north > max_lat
        assert east > max_lon

    def test_stays_within_kerala(self):
        south, west, north, east = _bbox()
        assert 8.0 < south < 11.5
        assert 10.0 < north < 12.0
        assert 74.5 < west < 77.5
        assert 76.0 < east < 78.5


class TestDrainageDirectionOfEffect:
    """
    Measured against the 14 documented hotspots with an elevation-matched
    urban background. Originally recorded as:

        far from a drain     AUC 0.287
        sparse drainage      AUC 0.328
        NEAR a canal         AUC 0.713  (95% CI 0.566-0.855)

    Re-derived later with the committed helpers in waterlogging_validation
    (bootstrap_auc over the same elevation-matched background):

        NEAR a canal         AUC 0.698  (95% CI 0.555-0.840)
        drainage density     AUC 0.652

    The original run is not reproducible from anything committed --
    waterlogging_validation.evaluate() scores only the fluvial and pluvial
    surfaces, so the drainage numbers came from an ad-hoc script that was not
    kept. Treat ~0.70 as the figure and the *direction* as the finding.

    Note also that the "far from a drain" row is not simply one minus the
    "near" row: sample_at takes the maximum within 150 m, so negating the
    surface changes which pixel is sampled and the two are not symmetric.

    Hotspots sit a median 304 m from a mapped channel against 701 m for
    background, with roughly twice the drainage density. Kochi's canals are
    tidal: when the tide is high or the channel silted they back up into the
    streets, so proximity is a risk factor rather than a mitigation.

    This test is a guard against someone "fixing" the sign on the assumption
    that drains must reduce flooding. If the physics index is ever given a
    drainage term, it must not be signed from that assumption.
    """

    def test_module_documents_the_proxy_limits(self):
        import osm_drainage

        doc = osm_drainage.__doc__ or ""
        for token in ("proxy", "ODbL"):
            assert token in doc, f"module docstring should mention {token!r}"

    def test_drainage_is_not_signed_in_the_physics_index(self):
        """
        The pluvial index must not acquire a drainage term whose sign came from
        the assumption that drains reduce flooding. The measurement says the
        opposite here, and the only evidence for the sign is the 14 hotspots
        that are also the validation set -- so signing it from them would be
        circular. If drainage enters the model it goes in as a *feature* of the
        learned susceptibility model, which is fitted on independent NDEM flood
        labels.
        """
        import inspect

        import pluvial

        source = inspect.getsource(pluvial)
        for token in ("osm_drain", "drain_dist", "drain_density"):
            assert token not in source, (
                f"pluvial.py references {token!r}: drainage must be learned "
                "from NDEM labels, not hand-signed against the hotspot set"
            )
