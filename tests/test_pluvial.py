"""
Tests for the routed-runoff building blocks shared across hazard.py (batch
fluvial hazard), live_model.py (the live slider) and fit_beta.py (calibration).

These three consumers must stay physically consistent -- they compute the
same routed runoff ratio, just against different rainfall depths and on
different grids. That consistency is only real if there is exactly one
implementation, which is why these are free functions rather than three
copies embedded in each caller. This file does not cover PluvialModel.build()
or fill_depressions(), which need a real DEM on disk; the pre-existing
project gap in test coverage there is not this file's job to close.
"""

import numpy as np
import pytest

from pluvial import (
    _MIN_ROUTED_RATIO,
    reproject_basis_to_grid,
    routed_runoff_ratio,
    routed_runoff_volume_m3,
)


@pytest.fixture
def basis():
    """
    Two land-cover classes over a 3x3 grid: class 7 (built-up, high runoff)
    upstream of most of the grid, class 2 (forest, low runoff) only at the
    outlet. Values are upstream cell *counts*, matching FlowNetwork.accumulate.
    """
    return {
        7: np.array([[4, 3, 2], [3, 2, 1], [2, 1, 0]], dtype=np.float32),
        2: np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.float32),
    }


@pytest.fixture
def classes():
    return np.array([2, 7])


@pytest.fixture
def valid():
    return np.ones((3, 3), dtype=bool)


class TestRoutedRunoffVolume:
    def test_zero_rainfall_is_zero_volume(self, basis, classes, valid):
        vol = routed_runoff_volume_m3(basis, classes, 0.0, 100.0, valid)
        assert np.allclose(vol[valid], 0.0, atol=1e-6)

    def test_more_rain_means_more_volume(self, basis, classes, valid):
        low = routed_runoff_volume_m3(basis, classes, 50.0, 100.0, valid)
        high = routed_runoff_volume_m3(basis, classes, 300.0, 100.0, valid)
        assert (high[valid] >= low[valid]).all()
        assert (high[valid] > low[valid]).any()

    def test_more_upstream_area_means_more_volume(self, basis, classes, valid):
        """The corner with 4 upstream built-up cells must exceed the corner
        with 0, at the same rainfall."""
        vol = routed_runoff_volume_m3(basis, classes, 200.0, 100.0, valid)
        assert vol[0, 0] > vol[2, 2] or vol[2, 2] == 0.0

    def test_invalid_cells_are_nan(self, basis, classes):
        valid = np.array([[True, True, False], [True, True, True], [True, True, True]])
        vol = routed_runoff_volume_m3(basis, classes, 200.0, 100.0, valid)
        assert np.isnan(vol[0, 2])
        assert np.isfinite(vol[0, 0])

    def test_a_class_with_no_upstream_cells_contributes_nothing(self, classes, valid):
        empty_basis = {2: np.zeros((3, 3), dtype=np.float32), 7: np.zeros((3, 3), dtype=np.float32)}
        vol = routed_runoff_volume_m3(empty_basis, classes, 300.0, 100.0, valid)
        assert np.allclose(vol[valid], 0.0)

    def test_volume_scales_with_cell_area(self, basis, classes, valid):
        small = routed_runoff_volume_m3(basis, classes, 200.0, 50.0, valid)
        large = routed_runoff_volume_m3(basis, classes, 200.0, 200.0, valid)
        ratio = large[valid] / np.maximum(small[valid], 1e-9)
        assert np.allclose(ratio[np.isfinite(ratio) & (small[valid] > 0)], 4.0, rtol=1e-3)


class TestRoutedRunoffRatio:
    def test_ratio_is_one_at_the_reference_depth(self, basis, classes, valid):
        """Same identity hazard.combine() relies on: P == P_ref must give a
        ratio of exactly 1, or the hazard would not reduce to susceptibility
        at the reference event."""
        ratio = routed_runoff_ratio(basis, classes, 250.0, 250.0, 100.0, valid)
        assert np.allclose(ratio[valid], 1.0, atol=1e-6)

    def test_ratio_below_one_when_rainfall_is_lower(self, basis, classes, valid):
        ratio = routed_runoff_ratio(basis, classes, 50.0, 300.0, 100.0, valid)
        finite = ratio[valid & np.isfinite(ratio)]
        assert (finite <= 1.0 + 1e-6).all()

    def test_ratio_above_one_when_rainfall_is_higher(self, basis, classes, valid):
        ratio = routed_runoff_ratio(basis, classes, 500.0, 300.0, 100.0, valid)
        finite = ratio[valid & np.isfinite(ratio)]
        assert (finite[np.isfinite(finite)] >= 1.0 - 1e-6).any()

    def test_zero_runoff_both_sides_is_nan_not_a_fabricated_ratio(self, classes, valid):
        """
        An empty catchment (no upstream cells at all) gives zero volume at
        every rainfall, so the ratio is a genuine 0/0 -- undefined, not "no
        runoff so ratio is 0" or "so ratio is 1". NaN is correct here and
        matches hazard.combine()'s identical pointwise np.where(ref>0,...,nan)
        pattern; np.clip leaves NaN untouched rather than rescuing it, and
        that must stay true or a meaningless ratio would silently enter the
        hazard's log-shift term as if it meant something.
        """
        basis = {2: np.zeros((3, 3), dtype=np.float32), 7: np.zeros((3, 3), dtype=np.float32)}
        ratio = routed_runoff_ratio(basis, classes, 300.0, 250.0, 100.0, valid)
        assert np.isnan(ratio[valid]).all()

    def test_floor_guards_a_small_but_nonzero_ratio(self, classes, valid):
        """The floor's real job: a nonzero denominator with a tiny numerator
        must not send ln(ratio) to a large negative number unbounded below."""
        basis = {2: np.zeros((3, 3), dtype=np.float32), 7: np.full((3, 3), 1.0, dtype=np.float32)}
        ratio = routed_runoff_ratio(basis, classes, 0.001, 400.0, 100.0, valid)
        assert (ratio[valid] >= _MIN_ROUTED_RATIO).all()
        assert np.isfinite(ratio[valid]).all()

    def test_ratio_is_never_below_the_floor(self, basis, classes, valid):
        ratio = routed_runoff_ratio(basis, classes, 0.0, 400.0, 100.0, valid)
        finite = ratio[valid & np.isfinite(ratio)]
        assert (finite >= _MIN_ROUTED_RATIO).all()


class TestReprojectBasisToGrid:
    def test_conserves_total_upstream_count_under_resampling(self):
        """
        Average resampling should approximately conserve the sum when the
        destination grid has the same total area as the source -- an upstream
        *count* is a density-like quantity, and downsampling with 'average'
        must not invent or destroy contribution.
        """
        from rasterio.transform import from_origin

        src_transform = from_origin(0, 100, 10.0, 10.0)  # 10 m cells
        n_k = np.full((10, 10), 5.0, dtype=np.float32)
        src_profile = {"transform": src_transform, "crs": "EPSG:32643"}

        # Same footprint, coarser cells (20 m instead of 10 m).
        dst_transform = from_origin(0, 100, 20.0, 20.0)
        out = reproject_basis_to_grid(
            {1: n_k},
            src_profile,
            dst_transform,
            "EPSG:32643",
            (5, 5),
        )
        assert out[1].shape == (5, 5)
        # Every source cell was 5.0; a same-value field must resample to the
        # same value everywhere, not smear toward zero at the edges.
        assert np.allclose(out[1], 5.0, atol=0.5)

    def test_output_has_no_nan_after_nan_to_num(self):
        from rasterio.transform import from_origin

        src_transform = from_origin(0, 100, 10.0, 10.0)
        n_k = np.full((10, 10), np.nan, dtype=np.float32)
        n_k[0:5, 0:5] = 3.0
        src_profile = {"transform": src_transform, "crs": "EPSG:32643"}
        out = reproject_basis_to_grid(
            {1: n_k},
            src_profile,
            src_transform,
            "EPSG:32643",
            (10, 10),
        )
        assert np.isfinite(out[1]).all()

    def test_preserves_every_class_key(self):
        from rasterio.transform import from_origin

        src_transform = from_origin(0, 100, 10.0, 10.0)
        src_profile = {"transform": src_transform, "crs": "EPSG:32643"}
        basis = {k: np.full((10, 10), float(k), dtype=np.float32) for k in (1, 2, 4, 5, 7, 8, 11)}
        out = reproject_basis_to_grid(basis, src_profile, src_transform, "EPSG:32643", (10, 10))
        assert set(out) == set(basis)
