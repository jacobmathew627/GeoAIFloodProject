"""
Tests for the short-term rainfall forecast.

The one that matters most is leakage: a feature row for time t must contain
nothing from after t, and the target must be strictly the following days. A
time series shuffled or mis-indexed produces spectacular scores and no skill,
the temporal twin of the spatial-autocorrelation problem in the flood model.
"""
import numpy as np
import pytest

from rainfall_forecast import (
    FEATURE_NAMES,
    HORIZON_DAYS,
    WARN_THRESHOLDS,
    build_features,
    climatology_baseline,
    persistence_baseline,
)


@pytest.fixture
def series():
    """Four years of synthetic daily rainfall with a monsoon cycle."""
    n = 4 * 365
    dates = np.datetime64("2000-01-01", "D") + np.arange(n)
    doy = np.arange(n) % 365
    seasonal = 12.0 * np.exp(-((doy - 190) ** 2) / (2 * 45.0 ** 2))
    rng = np.random.default_rng(0)
    rain = np.clip(seasonal * rng.gamma(1.2, 1.0, size=n), 0, None).astype(float)
    return dates, rain


class TestFeatureConstruction:
    def test_shapes_line_up(self, series):
        dates, rain = series
        X, y, stamps = build_features(dates, rain)
        assert X.shape[0] == y.shape[0] == stamps.shape[0]
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_target_is_the_next_three_days(self, series):
        dates, rain = series
        X, y, stamps = build_features(dates, rain)
        # Locate a sample by its timestamp and recompute the target by hand.
        i = 100
        t = int(np.flatnonzero(dates == stamps[i])[0])
        expected = rain[t + 1: t + 1 + HORIZON_DAYS].sum()
        assert y[i] == pytest.approx(expected, rel=1e-5)

    def test_no_future_leakage_in_features(self, series):
        """
        Perturbing the future must not change any feature for time t. If it
        does, the model can see the answer.
        """
        dates, rain = series
        X_before, _, stamps = build_features(dates, rain)

        i = 200
        t = int(np.flatnonzero(dates == stamps[i])[0])
        tampered = rain.copy()
        tampered[t + 1:] *= 7.5          # rewrite everything after t

        X_after, _, _ = build_features(dates, tampered)
        np.testing.assert_allclose(X_before[i], X_after[i], rtol=1e-6)

    def test_rain_t_is_the_current_day(self, series):
        dates, rain = series
        X, _, stamps = build_features(dates, rain)
        i = 150
        t = int(np.flatnonzero(dates == stamps[i])[0])
        assert X[i, FEATURE_NAMES.index("rain_t")] == pytest.approx(rain[t], rel=1e-5)

    def test_trailing_sums_are_ordered(self, series):
        dates, rain = series
        X, _, _ = build_features(dates, rain)
        s3 = X[:, FEATURE_NAMES.index("sum_3d")]
        s7 = X[:, FEATURE_NAMES.index("sum_7d")]
        s30 = X[:, FEATURE_NAMES.index("sum_30d")]
        assert (s7 >= s3 - 1e-4).all()
        assert (s30 >= s7 - 1e-4).all()

    def test_seasonality_is_bounded(self, series):
        dates, rain = series
        X, _, _ = build_features(dates, rain)
        for name in ("doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2"):
            col = X[:, FEATURE_NAMES.index(name)]
            assert col.min() >= -1.0001 and col.max() <= 1.0001

    def test_features_are_finite(self, series):
        dates, rain = series
        X, y, _ = build_features(dates, rain)
        assert np.isfinite(X).all()
        assert np.isfinite(y).all()

    def test_wet_days_within_range(self, series):
        dates, rain = series
        X, _, _ = build_features(dates, rain)
        col = X[:, FEATURE_NAMES.index("wet_days_7d")]
        assert col.min() >= 0 and col.max() <= 7


class TestBaselines:
    def test_persistence_is_the_trailing_three_day_sum(self, series):
        dates, rain = series
        X, _, _ = build_features(dates, rain)
        np.testing.assert_allclose(
            persistence_baseline(X), X[:, FEATURE_NAMES.index("sum_3d")]
        )

    def test_climatology_is_seasonal(self, series):
        """It must be wetter in the monsoon peak than in the dry season."""
        dates, rain = series
        _, y, stamps = build_features(dates, rain)
        clim = climatology_baseline(stamps, y, stamps)

        doy = (
            stamps.astype("datetime64[D]")
            - stamps.astype("datetime64[Y]").astype("datetime64[D]")
        ).astype(int) + 1
        wet = clim[(doy > 170) & (doy < 210)].mean()
        dry = clim[(doy > 1) & (doy < 40)].mean()
        assert wet > dry * 3

    def test_climatology_is_non_negative(self, series):
        dates, rain = series
        _, y, stamps = build_features(dates, rain)
        assert (climatology_baseline(stamps, y, stamps) >= 0).all()

    def test_climatology_uses_only_training_data(self, series):
        """
        The test-period climatology must be computed from training years only.
        Passing disjoint stamps must still produce a value for every test day.
        """
        dates, rain = series
        _, y, stamps = build_features(dates, rain)
        half = stamps.size // 2
        clim = climatology_baseline(stamps[:half], y[:half], stamps[half:])
        assert clim.shape[0] == stamps[half:].shape[0]
        assert np.isfinite(clim).all()


class TestConfiguration:
    def test_horizon_matches_the_hazard_model(self):
        """
        The forecast target must be the same quantity the flood model consumes,
        or the chain from prediction to hazard map does not close.
        """
        from reference_rainfall import STORM_WINDOW_DAYS

        assert HORIZON_DAYS == STORM_WINDOW_DAYS

    def test_warning_thresholds_are_ordered(self):
        assert list(WARN_THRESHOLDS) == sorted(WARN_THRESHOLDS)

    def test_feature_names_unique(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
