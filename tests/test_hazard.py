"""
Tests for the hazard combination and scenario blending.

The central guarantees: hazard reduces to susceptibility at the reference
event, increases monotonically with rainfall, stays a probability, and never
turns a nodata pixel into a valid one.
"""
import numpy as np
import pytest

from config import RAINFALL
from hazard import NODATA, blend_scenarios, combine, logit, sigmoid


@pytest.fixture
def susceptibility():
    return np.array(
        [
            [0.02, 0.20, 0.55],
            [0.80, 0.99, 0.35],
            [np.nan, 0.10, 0.45],
        ],
        dtype=np.float32,
    )


class TestLogitSigmoid:
    def test_round_trip(self):
        p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
        np.testing.assert_allclose(sigmoid(logit(p)), p, atol=1e-6)

    def test_sigmoid_stable_at_extremes(self):
        z = np.array([-800.0, 800.0])
        out = sigmoid(z)
        assert np.isfinite(out).all()
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(1.0)

    def test_logit_clamps_endpoints(self):
        assert np.isfinite(logit(np.array([0.0, 1.0]))).all()


class TestCombine:
    def test_reduces_to_susceptibility_at_reference_event(
        self, susceptibility, curve_number_grid
    ):
        """At P = P_ref the model must reproduce the calibration event."""
        hazard = combine(
            susceptibility, curve_number_grid, RAINFALL.reference_event_mm
        )
        both = np.isfinite(hazard) & np.isfinite(susceptibility)
        np.testing.assert_allclose(hazard[both], susceptibility[both], atol=1e-5)

    def test_monotonic_in_rainfall(self, susceptibility, curve_number_grid):
        previous = None
        for depth in (0, 50, 100, 150, 200, 300, 400, 500):
            hazard = combine(susceptibility, curve_number_grid, depth)
            finite = np.isfinite(hazard)
            if previous is not None:
                assert (hazard[finite] >= previous[finite] - 1e-6).all(), (
                    f"hazard decreased going to {depth} mm"
                )
            previous = hazard

    def test_stays_a_probability(self, susceptibility, curve_number_grid):
        for depth in (0, 200, 2000):
            hazard = combine(susceptibility, curve_number_grid, depth)
            finite = hazard[np.isfinite(hazard)]
            assert (finite >= 0.0).all()
            assert (finite <= 1.0).all()

    def test_less_rain_than_reference_lowers_hazard(
        self, susceptibility, curve_number_grid
    ):
        low = combine(susceptibility, curve_number_grid, 50.0)
        ref = combine(susceptibility, curve_number_grid, RAINFALL.reference_event_mm)
        both = np.isfinite(low) & np.isfinite(ref)
        assert (low[both] <= ref[both] + 1e-6).all()

    def test_nodata_never_becomes_valid(self, susceptibility, curve_number_grid):
        hazard = combine(susceptibility, curve_number_grid, 150.0)
        assert np.isnan(hazard[2, 0])  # NaN susceptibility
        assert np.isnan(hazard[2, 0])  # NaN curve number at the same pixel

    def test_land_cover_changes_the_response(self, susceptibility):
        """
        At equal susceptibility, the impervious surface must stay the more
        hazardous one below the reference event. A scalar rainfall multiplier
        cannot express this, and an absolute runoff difference inverts it.
        """
        flat = np.full(susceptibility.shape, 0.5, dtype=np.float32)
        built_up = np.full(susceptibility.shape, 94.0, dtype=np.float32)
        forest = np.full(susceptibility.shape, 84.0, dtype=np.float32)

        assert (combine(flat, built_up, 150.0) > combine(flat, forest, 150.0)).all()
        assert (combine(flat, built_up, 50.0) > combine(flat, forest, 50.0)).all()

    def test_no_runoff_means_no_hazard(self):
        """Rainfall below the initial abstraction produces no runoff at all."""
        flat = np.full((2, 2), 0.9, dtype=np.float32)
        forest = np.full((2, 2), 60.0, dtype=np.float32)
        assert (combine(flat, forest, 0.0) < 1e-3).all()


class TestCombineWithRoutedRatio:
    """
    combine()'s runoff_ratio parameter is what lets hazard.py, live_model.py
    and fit_beta.py all feed in pluvial.routed_runoff_ratio() instead of the
    pointwise ratio computed from curve_number -- the actual "route the
    runoff" change. curve_number is still required even in this path, since
    it also defines the model domain, so every test still passes it.
    """

    def test_explicit_ratio_of_one_reduces_to_susceptibility(
        self, susceptibility, curve_number_grid
    ):
        """The routed path must satisfy the same reference-event identity as
        the pointwise one: ratio == 1 everywhere gives hazard == susceptibility,
        regardless of what rainfall_mm or curve_number say."""
        ratio = np.ones_like(susceptibility)
        hazard = combine(susceptibility, curve_number_grid, 999.0, runoff_ratio=ratio)
        both = np.isfinite(hazard) & np.isfinite(susceptibility)
        np.testing.assert_allclose(hazard[both], susceptibility[both], atol=1e-5)

    def test_explicit_ratio_overrides_the_pointwise_one(
        self, susceptibility, curve_number_grid
    ):
        """A routed ratio that differs from the pointwise one at this
        rainfall must actually change the result -- otherwise the parameter
        is being silently ignored."""
        pointwise = combine(susceptibility, curve_number_grid, 200.0)
        routed = combine(
            susceptibility, curve_number_grid, 200.0,
            runoff_ratio=np.full_like(susceptibility, 5.0),
        )
        both = np.isfinite(pointwise) & np.isfinite(routed)
        assert not np.allclose(pointwise[both], routed[both])

    def test_routed_ratio_still_stays_a_probability(
        self, susceptibility, curve_number_grid
    ):
        ratio = np.full_like(susceptibility, 50.0)  # a large, catchment-inflated ratio
        hazard = combine(susceptibility, curve_number_grid, 300.0, runoff_ratio=ratio)
        finite = hazard[np.isfinite(hazard)]
        assert (finite >= 0.0).all() and (finite <= 1.0).all()

    def test_routed_ratio_respects_nodata_the_same_way(
        self, susceptibility, curve_number_grid
    ):
        ratio = np.full_like(susceptibility, 2.0)
        hazard = combine(susceptibility, curve_number_grid, 200.0, runoff_ratio=ratio)
        assert np.isnan(hazard[2, 0])  # NaN susceptibility, same pixel as the other test

    def test_nan_in_the_routed_ratio_propagates_to_nan_hazard(
        self, susceptibility, curve_number_grid
    ):
        """A reprojection edge effect that leaves a hole in the routed ratio
        must not silently read as zero hazard -- it must stay unknown."""
        ratio = np.full_like(susceptibility, 2.0)
        ratio[0, 0] = np.nan
        hazard = combine(susceptibility, curve_number_grid, 200.0, runoff_ratio=ratio)
        assert np.isnan(hazard[0, 0])


class TestBlendScenarios:
    @pytest.fixture
    def scenarios(self):
        base = np.array([[0.1, 0.4], [0.7, NODATA]], dtype=np.float32)
        return {
            100.0: base,
            200.0: np.array([[0.3, 0.6], [0.9, NODATA]], dtype=np.float32),
        }

    def test_empty_returns_none(self):
        assert blend_scenarios({}, 150.0) is None

    def test_below_range_returns_lowest(self, scenarios):
        out = blend_scenarios(scenarios, 10.0)
        np.testing.assert_array_equal(out, scenarios[100.0])

    def test_above_range_returns_highest(self, scenarios):
        out = blend_scenarios(scenarios, 900.0)
        np.testing.assert_array_equal(out, scenarios[200.0])

    def test_interpolates_between(self, scenarios):
        out = blend_scenarios(scenarios, 150.0)
        assert 0.1 < out[0, 0] < 0.3
        assert 0.4 < out[0, 1] < 0.6

    def test_preserves_nodata(self, scenarios):
        out = blend_scenarios(scenarios, 150.0)
        assert out[1, 1] == pytest.approx(NODATA)

    def test_monotonic_across_the_bracket(self, scenarios):
        values = [blend_scenarios(scenarios, mm)[0, 0] for mm in range(100, 201, 10)]
        assert all(b >= a - 1e-6 for a, b in zip(values, values[1:]))

    def test_endpoints_are_exact(self, scenarios):
        np.testing.assert_allclose(
            blend_scenarios(scenarios, 100.0)[0, 0], 0.1, atol=1e-6
        )
        np.testing.assert_allclose(
            blend_scenarios(scenarios, 200.0)[0, 0], 0.3, atol=1e-6
        )
