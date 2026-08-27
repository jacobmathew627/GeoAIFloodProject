"""
Tests for the documented-hotspot validation.

Offline parts only: geocoding needs the network. The statistics and the
hotspot table are what carry the claim, so those are pinned.
"""

import numpy as np
import pytest

from waterlogging_validation import (
    DOCUMENTED_HOTSPOTS,
    SEARCH_RADIUS_M,
    bootstrap_auc,
)


class TestHotspotTable:
    def test_every_entry_is_attributable(self):
        """A label without a source cannot be audited, so it cannot be used."""
        for h in DOCUMENTED_HOTSPOTS:
            assert h["name"]
            assert h["source"], f"{h['name']} has no source"
            assert h["queries"], f"{h['name']} has no query"

    def test_queries_are_variants_not_duplicates(self):
        for h in DOCUMENTED_HOTSPOTS:
            assert len(set(h["queries"])) == len(h["queries"]), h["name"]

    def test_names_unique(self):
        names = [h["name"] for h in DOCUMENTED_HOTSPOTS]
        assert len(set(names)) == len(names)

    def test_enough_points_to_test_something(self):
        """Below about ten points the bootstrap interval is uninformative."""
        assert len(DOCUMENTED_HOTSPOTS) >= 10

    def test_queries_are_geographically_scoped(self):
        """An unscoped query resolves to the wrong continent."""
        for h in DOCUMENTED_HOTSPOTS:
            for q in h["queries"]:
                assert any(
                    token in q for token in ("Kochi", "Ernakulam", "Kerala")
                ), f"{h['name']}: {q!r} is not scoped to the study area"

    def test_search_radius_absorbs_geocoding_error_but_stays_local(self):
        # Big enough to cover a neighbourhood centroid offset, small enough
        # not to swallow a whole catchment.
        assert 50.0 <= SEARCH_RADIUS_M <= 500.0


class TestBootstrapAUC:
    def test_perfect_separation(self):
        pos = np.array([0.9, 0.95, 0.99, 0.92])
        neg = np.linspace(0.0, 0.4, 200)
        auc, lo, hi = bootstrap_auc(pos, neg, n_boot=200)
        assert auc == pytest.approx(1.0)
        assert lo > 0.9

    def test_no_separation_straddles_half(self):
        rng = np.random.default_rng(0)
        pos = rng.random(20)
        neg = rng.random(500)
        auc, lo, hi = bootstrap_auc(pos, neg, n_boot=300)
        assert 0.3 < auc < 0.7
        assert lo < 0.5 < hi, "an unskilled model must not exclude chance"

    def test_reversed_separation_scores_below_half(self):
        pos = np.array([0.01, 0.02, 0.03, 0.04])
        neg = np.linspace(0.5, 1.0, 200)
        auc, _, hi = bootstrap_auc(pos, neg, n_boot=200)
        assert auc == pytest.approx(0.0)
        assert hi < 0.5

    def test_interval_brackets_the_point_estimate(self):
        rng = np.random.default_rng(1)
        pos = rng.normal(0.7, 0.1, 15)
        neg = rng.normal(0.4, 0.1, 400)
        auc, lo, hi = bootstrap_auc(pos, neg, n_boot=400)
        assert lo <= auc <= hi

    def test_smaller_sample_widens_the_interval(self):
        """The honesty check: fewer points must not look more certain."""
        rng = np.random.default_rng(2)
        neg = rng.normal(0.4, 0.15, 400)
        big = rng.normal(0.7, 0.15, 60)
        small = big[:6]

        _, lo_b, hi_b = bootstrap_auc(big, neg, n_boot=400)
        _, lo_s, hi_s = bootstrap_auc(small, neg, n_boot=400)
        assert (hi_s - lo_s) > (hi_b - lo_b)

    def test_is_deterministic_for_a_given_seed(self):
        rng = np.random.default_rng(3)
        pos, neg = rng.random(12), rng.random(300)
        a = bootstrap_auc(pos, neg, n_boot=200, seed=7)
        b = bootstrap_auc(pos, neg, n_boot=200, seed=7)
        assert a == b
