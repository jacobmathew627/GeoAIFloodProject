"""
Tests for fitting the rainfall sensitivity beta.

The real fit needs the susceptibility raster and the NDEM labels. What is
tested here is the estimator itself, on synthetic surfaces where the right
answer is known by construction, plus the identifiability argument that
motivates the whole module.
"""
import numpy as np
import pytest

from config import RAINFALL, RASTER
from fit_beta import BETA_BOUNDS, expected_area_km2, fit


@pytest.fixture
def surface():
    """A small synthetic domain: mixed susceptibility over mixed land cover."""
    rng = np.random.default_rng(0)
    n = 5000
    s = rng.uniform(0.02, 0.6, size=n)
    cn = rng.choice([61.0, 74.0, 89.0, 98.0], size=n)
    return s, cn


class TestExpectedArea:
    def test_reference_depth_is_independent_of_beta(self, surface):
        """
        The identifiability claim the module rests on: at P_ref every shift term
        is zero, so the expected area is sum(S) whatever beta is. An event at
        the reference depth therefore carries no information about beta.
        """
        s, cn = surface
        ref = RAINFALL.reference_event_mm
        areas = [expected_area_km2(s, cn, ref, b) for b in (0.0, 1.8, 5.0)]
        assert areas[0] == pytest.approx(areas[1], rel=1e-9)
        assert areas[1] == pytest.approx(areas[2], rel=1e-9)

    def test_reference_area_equals_summed_susceptibility(self, surface):
        s, cn = surface
        px_km2 = (RASTER.cell_size / 1000.0) ** 2
        got = expected_area_km2(s, cn, RAINFALL.reference_event_mm, 1.8)
        assert got == pytest.approx(float(s.sum()) * px_km2, rel=1e-9)

    def test_more_rain_means_more_area(self, surface):
        s, cn = surface
        areas = [expected_area_km2(s, cn, p, 1.8) for p in (50, 150, 300, 450, 600)]
        assert areas == sorted(areas)

    def test_beta_zero_freezes_the_response(self, surface):
        """With beta = 0 the slider does nothing -- the degenerate case."""
        s, cn = surface
        a_low = expected_area_km2(s, cn, 50.0, 0.0)
        a_high = expected_area_km2(s, cn, 600.0, 0.0)
        assert a_low == pytest.approx(a_high, rel=1e-9)

    def test_larger_beta_amplifies_below_and_above_reference(self, surface):
        s, cn = surface
        ref = RAINFALL.reference_event_mm
        below_soft = expected_area_km2(s, cn, 100.0, 1.0)
        below_hard = expected_area_km2(s, cn, 100.0, 4.0)
        above_soft = expected_area_km2(s, cn, ref + 150.0, 1.0)
        above_hard = expected_area_km2(s, cn, ref + 150.0, 4.0)
        assert below_hard < below_soft, "a larger beta must shrink extent below P_ref"
        assert above_hard > above_soft, "a larger beta must grow extent above P_ref"

    def test_area_is_bounded_by_the_domain(self, surface):
        s, cn = surface
        px_km2 = (RASTER.cell_size / 1000.0) ** 2
        assert expected_area_km2(s, cn, 2000.0, 8.0) <= s.size * px_km2 + 1e-9

    def test_stays_positive_at_trivial_rainfall(self, surface):
        """The ratio floor must keep the logit finite rather than -inf."""
        s, cn = surface
        area = expected_area_km2(s, cn, 0.0, 1.8)
        assert np.isfinite(area)
        assert area >= 0.0


class TestFit:
    def test_recovers_a_planted_beta(self, surface):
        """
        Generate extents *from* a known beta, then fit it back. Two off-reference
        events, matching what the NDEM inventory actually supplies.
        """
        s, cn = surface
        truth = 2.4
        events = [
            {"event": "a", "rainfall_mm": 412.5,
             "observed_km2": expected_area_km2(s, cn, 412.5, truth)},
            {"event": "b", "rainfall_mm": 173.7,
             "observed_km2": expected_area_km2(s, cn, 173.7, truth)},
        ]
        assert fit((s, cn), events) == pytest.approx(truth, abs=0.02)

    @pytest.mark.parametrize("truth", [0.5, 1.8, 3.6, 6.0])
    def test_recovers_across_the_bracket(self, surface, truth):
        s, cn = surface
        events = [{"event": "x", "rainfall_mm": 173.7,
                   "observed_km2": expected_area_km2(s, cn, 173.7, truth)}]
        assert fit((s, cn), events) == pytest.approx(truth, abs=0.05)

    def test_result_is_inside_the_bracket(self, surface):
        s, cn = surface
        # An extent far larger than any beta can produce: the fit must clamp,
        # not diverge.
        events = [{"event": "x", "rainfall_mm": 100.0, "observed_km2": 1e6}]
        beta = fit((s, cn), events)
        assert BETA_BOUNDS[0] <= beta <= BETA_BOUNDS[1]

    def test_a_reference_only_event_leaves_beta_unidentified(self, surface):
        """
        Every beta gives the same loss, so the search returns the bracket
        midpoint rather than anything meaningful. This is why run() filters
        the reference event out instead of fitting on it.
        """
        s, cn = surface
        ref = RAINFALL.reference_event_mm
        events = [{"event": "ref", "rainfall_mm": ref,
                   "observed_km2": expected_area_km2(s, cn, ref, 1.8)}]
        beta = fit((s, cn), events)
        losses = [
            abs(expected_area_km2(s, cn, ref, b) - events[0]["observed_km2"])
            for b in (0.0, beta, 8.0)
        ]
        assert max(losses) < 1e-6, "loss is flat in beta, so the fit is arbitrary"

    def test_lower_rainfall_events_carry_more_leverage(self, surface):
        """
        2021 at 173.7 mm sits far from the 443 mm reference and 2019 at 412.5 mm
        sits close to it, so the low event dominates the fit. Worth pinning:
        it explains why the leave-one-out spread is wide.
        """
        s, cn = surface
        spread_near = abs(
            expected_area_km2(s, cn, 412.5, 1.0) - expected_area_km2(s, cn, 412.5, 4.0)
        )
        spread_far = abs(
            expected_area_km2(s, cn, 173.7, 1.0) - expected_area_km2(s, cn, 173.7, 4.0)
        )
        assert spread_far > spread_near
