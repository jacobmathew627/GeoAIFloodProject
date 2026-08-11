"""
Tests for the SCS Curve Number runoff model.

These encode the physical properties the old hard-coded rainfall multiplier
table violated: monotonicity in rainfall, boundedness by the rainfall depth,
and a genuine initial abstraction threshold.
"""
import numpy as np
import pytest

from config import HYDRO, LULC_CLASS_NAMES
from hydrology import (
    adjust_cn_for_amc,
    curve_number_from_lulc,
    potential_retention,
    runoff_coefficient,
    runoff_depth,
)


class TestCurveNumberGrid:
    def test_maps_every_known_lulc_class(self):
        codes = sorted(HYDRO.curve_numbers)
        lulc = np.array([codes], dtype=np.float32)
        valid = np.ones_like(lulc, dtype=bool)

        cn = curve_number_from_lulc(lulc, valid, amc="II")

        assert np.isfinite(cn).all()
        for i, code in enumerate(codes):
            assert cn[0, i] == pytest.approx(HYDRO.curve_numbers[code])

    def test_unknown_class_falls_back_to_default(self):
        lulc = np.array([[99.0]], dtype=np.float32)
        cn = curve_number_from_lulc(lulc, np.ones_like(lulc, dtype=bool), amc="II")
        assert cn[0, 0] == pytest.approx(HYDRO.default_curve_number)

    def test_invalid_pixels_stay_nan(self):
        lulc = np.array([[7.0, 7.0]], dtype=np.float32)
        valid = np.array([[True, False]])
        cn = curve_number_from_lulc(lulc, valid, amc="II")
        assert np.isfinite(cn[0, 0])
        assert np.isnan(cn[0, 1])

    def test_every_configured_class_has_a_name(self):
        assert set(HYDRO.curve_numbers) <= set(LULC_CLASS_NAMES)


class TestAMCAdjustment:
    def test_amc_ii_is_identity(self):
        cn = np.array([[70.0, 88.0]], dtype=np.float32)
        np.testing.assert_allclose(adjust_cn_for_amc(cn, "II"), cn)

    def test_wet_raises_and_dry_lowers(self):
        cn = np.array([[70.0]], dtype=np.float32)
        assert adjust_cn_for_amc(cn, "III")[0, 0] > 70.0
        assert adjust_cn_for_amc(cn, "I")[0, 0] < 70.0

    def test_stays_within_bounds(self):
        cn = np.array([[30.0, 100.0]], dtype=np.float32)
        for amc in ("I", "II", "III"):
            out = adjust_cn_for_amc(cn, amc)
            assert (out[np.isfinite(out)] <= 100.0).all()
            assert (out[np.isfinite(out)] >= 30.0).all()

    def test_rejects_unknown_amc(self):
        with pytest.raises(ValueError, match="amc must be"):
            adjust_cn_for_amc(np.array([[70.0]], dtype=np.float32), "IV")


class TestRunoff:
    def test_monotonic_in_rainfall(self, curve_number_grid):
        """The property the old multiplier table failed to satisfy."""
        depths = [0, 25, 50, 100, 150, 200, 300, 400, 600]
        previous = None
        for depth in depths:
            q = runoff_depth(depth, curve_number_grid)
            finite = np.isfinite(q)
            if previous is not None:
                assert (q[finite] >= previous[finite] - 1e-5).all(), (
                    f"runoff decreased going to {depth} mm"
                )
            previous = q

    def test_never_exceeds_rainfall(self, curve_number_grid):
        for depth in (10, 100, 400, 1000):
            q = runoff_depth(depth, curve_number_grid)
            assert (q[np.isfinite(q)] <= depth + 1e-4).all()

    def test_non_negative(self, curve_number_grid):
        q = runoff_depth(5.0, curve_number_grid)
        assert (q[np.isfinite(q)] >= 0.0).all()

    def test_zero_below_initial_abstraction(self):
        # Forest under dry conditions retains a lot; a light shower runs off nowhere.
        cn = adjust_cn_for_amc(np.array([[55.0]], dtype=np.float32), "I")
        s = float(potential_retention(cn)[0, 0])
        ia = HYDRO.initial_abstraction_ratio * s
        assert runoff_depth(max(ia - 1.0, 0.0), cn)[0, 0] == pytest.approx(0.0)

    def test_open_water_converts_all_rainfall(self):
        cn = np.array([[100.0]], dtype=np.float32)
        assert runoff_depth(120.0, cn)[0, 0] == pytest.approx(120.0, rel=1e-3)

    def test_impervious_runs_off_more_than_forest(self):
        built_up = np.array([[HYDRO.curve_numbers[7]]], dtype=np.float32)
        forest = np.array([[HYDRO.curve_numbers[2]]], dtype=np.float32)
        assert runoff_depth(150.0, built_up)[0, 0] > runoff_depth(150.0, forest)[0, 0]

    def test_nan_propagates(self, curve_number_grid):
        q = runoff_depth(100.0, curve_number_grid)
        assert np.isnan(q[2, 0])

    def test_rejects_negative_rainfall(self, curve_number_grid):
        with pytest.raises(ValueError, match="must be >= 0"):
            runoff_depth(-1.0, curve_number_grid)

    def test_runoff_coefficient_in_unit_interval(self, curve_number_grid):
        c = runoff_coefficient(200.0, curve_number_grid)
        finite = np.isfinite(c)
        assert (c[finite] >= 0.0).all()
        assert (c[finite] <= 1.0 + 1e-6).all()


class TestPotentialRetention:
    def test_decreases_with_curve_number(self):
        cn = np.array([[50.0, 70.0, 90.0, 100.0]], dtype=np.float32)
        s = potential_retention(cn)[0]
        assert (np.diff(s) < 0).all()

    def test_zero_at_cn_100(self):
        assert potential_retention(np.array([[100.0]], dtype=np.float32))[0, 0] == pytest.approx(
            0.0, abs=1e-6
        )
