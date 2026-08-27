"""
Tests for training-set sampling and cross-validation grouping.

These cover the parts of the modelling pipeline that can be exercised without
the 2.6 GB aligned raster stack.
"""

import numpy as np
import pytest

from feature_stack import _allocate_stratified
from hazard import logit, sigmoid
from susceptibility import SpatialEnsemble, calibration_report, spatial_blocks


class TestStratifiedAllocation:
    def test_meets_quota_when_capacity_allows(self):
        rng = np.random.default_rng(0)
        strata = [np.arange(1000), np.arange(1000, 2000), np.arange(2000, 3000)]
        out = _allocate_stratified(strata, 300, rng)
        assert out.size == 300

    def test_spreads_across_strata(self):
        rng = np.random.default_rng(0)
        strata = [np.arange(0, 1000), np.arange(1000, 2000), np.arange(2000, 3000)]
        out = _allocate_stratified(strata, 300, rng)
        per = [np.sum((out >= lo) & (out < lo + 1000)) for lo in (0, 1000, 2000)]
        assert min(per) > 0
        # Roughly even, not all from one stratum.
        assert max(per) - min(per) <= 2

    def test_redistributes_from_thin_strata(self):
        """
        Regression: a thin stratum used to cap the draw at its own size and
        the shortfall was silently dropped, leaving the classes unbalanced.
        """
        rng = np.random.default_rng(0)
        strata = [np.arange(2), np.arange(100, 1100), np.arange(2000, 3000)]
        out = _allocate_stratified(strata, 300, rng)
        assert out.size == 300

    def test_capped_by_total_capacity(self):
        rng = np.random.default_rng(0)
        strata = [np.arange(5), np.arange(100, 105)]
        out = _allocate_stratified(strata, 500, rng)
        assert out.size == 10

    def test_no_duplicates(self):
        rng = np.random.default_rng(1)
        strata = [np.arange(0, 50), np.arange(50, 100)]
        out = _allocate_stratified(strata, 80, rng)
        assert len(np.unique(out)) == out.size

    def test_empty_strata_are_skipped(self):
        rng = np.random.default_rng(0)
        strata = [np.empty(0, dtype=np.int64), np.arange(100)]
        out = _allocate_stratified(strata, 50, rng)
        assert out.size == 50

    def test_all_empty_returns_empty(self):
        rng = np.random.default_rng(0)
        out = _allocate_stratified([np.empty(0, dtype=np.int64)], 50, rng)
        assert out.size == 0


class TestSpatialBlocks:
    def test_nearby_pixels_share_a_block(self):
        row = np.array([0, 10, 20])
        col = np.array([0, 10, 20])
        assert len(np.unique(spatial_blocks(row, col, block_px=500))) == 1

    def test_distant_pixels_differ(self):
        row = np.array([0, 5000])
        col = np.array([0, 5000])
        assert len(np.unique(spatial_blocks(row, col, block_px=500))) == 2

    def test_block_ids_are_unique_per_cell(self):
        row = np.array([0, 0, 600, 600])
        col = np.array([0, 600, 0, 600])
        assert len(np.unique(spatial_blocks(row, col, block_px=500))) == 4


class TestFittedPriorOffset:
    """
    The offset is solved so the expected flooded area matches the observed
    extent, because the closed form assumes randomly drawn absences and ours
    are elevation-stratified.
    """

    class _FakeCalibrator:
        @staticmethod
        def predict(p):
            return p

    def _model(self, raw):
        model = SpatialEnsemble()
        model.calibrator_ = self._FakeCalibrator()
        model.models_ = [None]
        model._raw = lambda X: raw.reshape(1, -1)  # noqa: SLF001
        return model

    def test_hits_the_target_prevalence(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.01, 0.99, size=20_000)
        model = self._model(raw)

        model.fit_prior_offset(np.zeros((raw.size, 1)), 0.014)
        achieved = model.predict_proba(np.zeros((raw.size, 1)))[:, 1].mean()
        assert achieved == pytest.approx(0.014, rel=1e-3)

    def test_offset_is_negative_when_over_predicting(self):
        raw = np.full(1000, 0.5)
        model = self._model(raw)
        assert model.fit_prior_offset(np.zeros((1000, 1)), 0.01) < 0

    def test_offset_is_positive_when_under_predicting(self):
        raw = np.full(1000, 0.01)
        model = self._model(raw)
        assert model.fit_prior_offset(np.zeros((1000, 1)), 0.5) > 0

    def test_no_shift_when_already_on_target(self):
        raw = np.full(1000, 0.2)
        model = self._model(raw)
        assert model.fit_prior_offset(np.zeros((1000, 1)), 0.2) == pytest.approx(0.0, abs=1e-6)

    def test_saturating_target_is_clamped_not_infinite(self):
        raw = np.full(1000, 0.5)
        model = self._model(raw)
        offset = model.fit_prior_offset(np.zeros((1000, 1)), 1e-12)
        assert np.isfinite(offset)


class TestClosedFormPriorCorrection:
    """
    Regression: training is balanced 1:1 but the district is ~1.4% flooded,
    so uncorrected probabilities overstated the expected flooded area by a
    factor of ~11 at the reference event.
    """

    def test_balanced_sample_needs_no_shift(self):
        model = SpatialEnsemble()
        model.set_prior_offset(0.5, 0.5)
        assert model.prior_offset_ == pytest.approx(0.0)

    def test_rare_event_shifts_probabilities_down(self):
        model = SpatialEnsemble()
        model.set_prior_offset(domain_prevalence=0.014, sample_prevalence=0.5)
        assert model.prior_offset_ < 0
        assert model.prior_offset_ == pytest.approx(np.log(0.014 / 0.986), abs=1e-9)

    def test_offset_recovers_the_population_base_rate(self):
        """A sample-scale 0.5 must map to the population prevalence."""
        prevalence = 0.014
        model = SpatialEnsemble()
        model.set_prior_offset(prevalence, 0.5)
        corrected = sigmoid(logit(np.array([0.5])) + model.prior_offset_)
        assert corrected[0] == pytest.approx(prevalence, rel=1e-6)

    def test_correction_is_monotone(self):
        """Ranking must be unchanged, so AUC is unaffected."""
        model = SpatialEnsemble()
        model.set_prior_offset(0.014, 0.5)
        p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        corrected = sigmoid(logit(p) + model.prior_offset_)
        assert (np.diff(corrected) > 0).all()

    def test_more_common_event_shifts_up(self):
        model = SpatialEnsemble()
        model.set_prior_offset(domain_prevalence=0.9, sample_prevalence=0.5)
        assert model.prior_offset_ > 0

    def test_clamps_degenerate_prevalence(self):
        model = SpatialEnsemble()
        model.set_prior_offset(0.0, 0.5)
        assert np.isfinite(model.prior_offset_)
        model.set_prior_offset(1.0, 0.5)
        assert np.isfinite(model.prior_offset_)


class TestCalibrationReport:
    def test_perfect_calibration_has_no_gap(self):
        rng = np.random.default_rng(0)
        p = rng.random(20_000)
        y = (rng.random(20_000) < p).astype(int)
        rows = calibration_report(y, p)
        assert rows
        for _, _, predicted, observed, _ in rows:
            assert abs(predicted - observed) < 0.05

    def test_detects_miscalibration(self):
        p = np.full(10_000, 0.9)
        y = np.zeros(10_000, dtype=int)  # predicted 0.9, observed 0.0
        rows = calibration_report(y, p)
        assert any(abs(predicted - observed) > 0.5 for _, _, predicted, observed, _ in rows)

    def test_empty_bins_are_omitted(self):
        p = np.full(100, 0.05)
        y = np.zeros(100, dtype=int)
        rows = calibration_report(y, p)
        assert len(rows) == 1

    def test_includes_probability_one(self):
        p = np.ones(50)
        y = np.ones(50, dtype=int)
        rows = calibration_report(y, p)
        assert sum(n for *_, n in rows) == 50
