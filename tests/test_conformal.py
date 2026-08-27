"""
Tests for split conformal prediction.

The central property is the finite-sample coverage guarantee: on exchangeable
data the prediction set contains the truth at least (1 - alpha) of the time.
"""

import numpy as np
import pytest

import conformal
from conformal import (
    SET_AMBIGUOUS,
    SET_DRY,
    SET_EMPTY,
    SET_FLOOD,
    SET_LABELS,
    average_set_size,
    classify,
    conditional_coverage,
    coverage,
)


def _well_calibrated(n, seed=0):
    """Probabilities that are honestly calibrated, with matching labels."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.0, 1.0, size=n)
    y = (rng.uniform(size=n) < p).astype(int)
    return p, y


class TestFit:
    def test_returns_usable_thresholds(self):
        p, y = _well_calibrated(5000)
        t = conformal.fit(p, y, alpha=0.10)
        assert 0.0 <= t.q <= 1.0
        assert t.include_positive_above == pytest.approx(1.0 - t.q)
        assert t.include_negative_below == pytest.approx(t.q)
        assert t.n_calibration == 5000

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            conformal.fit(np.zeros(5), np.zeros(4))

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="empty"):
            conformal.fit(np.array([]), np.array([]))

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_bad_alpha(self, alpha):
        p, y = _well_calibrated(100)
        with pytest.raises(ValueError, match="alpha"):
            conformal.fit(p, y, alpha=alpha)

    def test_smaller_alpha_gives_wider_sets(self):
        p, y = _well_calibrated(5000)
        strict = conformal.fit(p, y, alpha=0.01)
        loose = conformal.fit(p, y, alpha=0.20)
        assert strict.q >= loose.q
        assert average_set_size(p, strict) >= average_set_size(p, loose)


class TestCoverageGuarantee:
    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
    def test_marginal_coverage_met_on_fresh_data(self, alpha):
        """The guarantee: coverage on exchangeable held-out data >= 1 - alpha."""
        p_cal, y_cal = _well_calibrated(20_000, seed=1)
        p_test, y_test = _well_calibrated(20_000, seed=2)

        t = conformal.fit(p_cal, y_cal, alpha=alpha)
        achieved = coverage(p_test, y_test, t)
        # Finite-sample slack; the guarantee is on expectation.
        assert achieved >= (1 - alpha) - 0.02

    def test_holds_for_a_rare_positive_class(self):
        """Flood pixels are ~1.4% of the district."""
        rng = np.random.default_rng(3)
        p_cal = rng.beta(1, 60, size=40_000)
        y_cal = (rng.uniform(size=40_000) < p_cal).astype(int)
        p_test = rng.beta(1, 60, size=40_000)
        y_test = (rng.uniform(size=40_000) < p_test).astype(int)

        t = conformal.fit(p_cal, y_cal, alpha=0.10)
        assert coverage(p_test, y_test, t) >= 0.88

    def test_holds_even_for_a_badly_miscalibrated_model(self):
        """Distribution-free: the guarantee does not assume a good model."""
        rng = np.random.default_rng(4)
        y_cal = (rng.uniform(size=20_000) < 0.3).astype(int)
        p_cal = rng.uniform(0.0, 1.0, size=20_000)  # pure noise
        y_test = (rng.uniform(size=20_000) < 0.3).astype(int)
        p_test = rng.uniform(0.0, 1.0, size=20_000)

        t = conformal.fit(p_cal, y_cal, alpha=0.10)
        assert coverage(p_test, y_test, t) >= 0.88


class TestPredictionSets:
    def test_four_codes_are_distinct(self):
        assert len({SET_EMPTY, SET_DRY, SET_AMBIGUOUS, SET_FLOOD}) == 4
        assert set(SET_LABELS) == {SET_EMPTY, SET_DRY, SET_AMBIGUOUS, SET_FLOOD}

    def test_classify_covers_every_pixel(self):
        p, y = _well_calibrated(2000)
        t = conformal.fit(p, y, alpha=0.10)
        codes = classify(p, t)
        assert set(np.unique(codes)) <= set(SET_LABELS)
        assert codes.shape == p.shape

    def test_confident_extremes(self):
        p, y = _well_calibrated(5000)
        t = conformal.fit(p, y, alpha=0.10)
        # q < 1 for any sensible calibration, so 1.0 is flood-only and 0.0 dry-only.
        assert classify(np.array([1.0]), t)[0] == SET_FLOOD
        assert classify(np.array([0.0]), t)[0] == SET_DRY

    def test_set_size_between_zero_and_two(self):
        p, y = _well_calibrated(2000)
        t = conformal.fit(p, y, alpha=0.10)
        assert 0.0 <= average_set_size(p, t) <= 2.0

    def test_shape_preserved_for_2d(self):
        p, y = _well_calibrated(400)
        t = conformal.fit(p, y, alpha=0.10)
        grid = p.reshape(20, 20)
        assert classify(grid, t).shape == (20, 20)


class TestMondrian:
    """
    Class-conditional calibration. The motivating failure: with a 1.4%
    positive class, marginal coverage is dominated by dry land and the flood
    class can be almost uncovered while the headline number looks fine.
    """

    def _rare_positive(self, n, seed):
        rng = np.random.default_rng(seed)
        p = rng.beta(1, 60, size=n)
        y = (rng.uniform(size=n) < p).astype(int)
        return p, y

    def test_flags_itself_as_mondrian(self):
        p, y = self._rare_positive(40_000, 10)
        t = conformal.fit_mondrian(p, y, alpha=0.10)
        assert t.mondrian is True
        assert "Mondrian" in str(t)

    def test_covers_the_rare_class_where_marginal_does_not(self):
        p_cal, y_cal = self._rare_positive(60_000, 11)
        p_test, y_test = self._rare_positive(60_000, 12)

        marginal = conformal.fit(p_cal, y_cal, alpha=0.10)
        mondrian = conformal.fit_mondrian(p_cal, y_cal, alpha=0.10)

        cov_marg = conformal.class_conditional_coverage(p_test, y_test, marginal)
        cov_mond = conformal.class_conditional_coverage(p_test, y_test, mondrian)

        # The whole point: the positive class is better covered.
        assert cov_mond["flood"] > cov_marg["flood"]
        assert cov_mond["flood"] >= 0.85

    def test_both_classes_meet_the_target(self):
        p_cal, y_cal = self._rare_positive(60_000, 13)
        p_test, y_test = self._rare_positive(60_000, 14)
        t = conformal.fit_mondrian(p_cal, y_cal, alpha=0.10)
        cov = conformal.class_conditional_coverage(p_test, y_test, t)
        assert cov["flood"] >= 0.85
        assert cov["dry"] >= 0.85

    def test_rejects_a_class_with_no_calibration_points(self):
        p = np.linspace(0, 1, 100)
        y = np.zeros(100, dtype=int)
        with pytest.raises(ValueError, match="no calibration points"):
            conformal.fit_mondrian(p, y, alpha=0.10)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            conformal.fit_mondrian(np.zeros(5), np.zeros(4))

    def test_report_includes_class_conditional(self):
        p, y = self._rare_positive(40_000, 15)
        t = conformal.fit_mondrian(p, y, alpha=0.10)
        summary = conformal.report(p, y, t)
        assert "class_conditional_coverage" in summary
        assert set(summary["class_conditional_coverage"]) == {"flood", "dry"}


class TestConditionalCoverage:
    def test_reports_every_stratum(self):
        p, y = _well_calibrated(20_000)
        t = conformal.fit(p, y, alpha=0.10)
        rows = conditional_coverage(p, y, t, n_bins=5)
        assert rows
        assert sum(r["n"] for r in rows) == p.size
        for r in rows:
            assert 0.0 <= r["coverage"] <= 1.0

    def test_detects_a_stratum_that_fails(self):
        """
        Marginal coverage can pass while a stratum fails badly -- the exact
        failure the Himachal Pradesh study reported (82.9% overall, 45-59% in
        high-risk zones).
        """
        rng = np.random.default_rng(5)
        # Model is right at the low end and inverted at the high end.
        p = np.concatenate([rng.uniform(0.0, 0.2, 19_000), rng.uniform(0.8, 1.0, 1_000)])
        y = np.concatenate(
            [
                (rng.uniform(size=19_000) < 0.1).astype(int),
                np.zeros(1_000, dtype=int),  # predicted ~0.9, actually never floods
            ]
        )
        t = conformal.fit(p, y, alpha=0.10)
        rows = conditional_coverage(p, y, t, n_bins=5)
        assert min(r["coverage"] for r in rows) < max(r["coverage"] for r in rows)

    def test_report_returns_serialisable_summary(self):
        import json

        p, y = _well_calibrated(5000)
        t = conformal.fit(p, y, alpha=0.10)
        summary = conformal.report(p, y, t)
        json.dumps(summary)  # must not raise
        assert summary["target_coverage"] == pytest.approx(0.90)
        assert "conditional_coverage" in summary
