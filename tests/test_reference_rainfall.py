"""
Tests for the reference-event derivation.

Only the offline logic is covered: `fetch_daily` needs the network, so the
accumulation arithmetic and the grid construction are tested directly. The
accumulation window is what decides the constant that anchors every hazard
map, so it is worth pinning.
"""
import numpy as np
import pytest

from config import HYDRO, RAINFALL
from reference_rainfall import (
    DISTRICT_BOX,
    EVENTS,
    STORM_WINDOW_DAYS,
    max_accumulation,
    sample_grid,
)


class TestSampleGrid:
    def test_returns_n_squared_points(self):
        assert len(sample_grid(DISTRICT_BOX, 3)) == 9
        assert len(sample_grid(DISTRICT_BOX, 4)) == 16

    def test_points_lie_inside_the_box(self):
        for lat, lon in sample_grid(DISTRICT_BOX, 4):
            assert DISTRICT_BOX["lat"][0] <= lat <= DISTRICT_BOX["lat"][1]
            assert DISTRICT_BOX["lon"][0] <= lon <= DISTRICT_BOX["lon"][1]

    def test_box_covers_the_mapped_district(self):
        """The footprint of the master grid, in WGS84."""
        assert DISTRICT_BOX["lat"][0] < 9.80 < DISTRICT_BOX["lat"][1]
        assert DISTRICT_BOX["lon"][0] < 76.30 < DISTRICT_BOX["lon"][1]

    def test_corners_are_included(self):
        pts = sample_grid(DISTRICT_BOX, 2)
        assert (DISTRICT_BOX["lat"][0], DISTRICT_BOX["lon"][0]) in pts
        assert (DISTRICT_BOX["lat"][1], DISTRICT_BOX["lon"][1]) in pts


class TestMaxAccumulation:
    def test_single_day_window_is_the_maximum(self):
        daily = np.array([1.0, 9.0, 3.0, 2.0])
        total, i = max_accumulation(daily, 1)
        assert total == pytest.approx(9.0)
        assert i == 1

    def test_finds_the_wettest_run(self):
        daily = np.array([1.0, 2.0, 10.0, 10.0, 1.0])
        total, i = max_accumulation(daily, 2)
        assert total == pytest.approx(20.0)
        assert i == 2

    def test_longer_window_never_totals_less(self):
        rng = np.random.default_rng(0)
        daily = rng.uniform(0, 50, size=31)
        totals = [max_accumulation(daily, w)[0] for w in (1, 2, 3, 5, 7)]
        assert all(b >= a - 1e-9 for a, b in zip(totals, totals[1:]))

    def test_window_longer_than_series_is_nan(self):
        total, i = max_accumulation(np.array([1.0, 2.0]), 5)
        assert np.isnan(total)
        assert i is None

    def test_full_window_equals_the_sum(self):
        daily = np.array([1.0, 2.0, 3.0])
        total, i = max_accumulation(daily, 3)
        assert total == pytest.approx(6.0)
        assert i == 0

    def test_index_points_at_the_window_start(self):
        daily = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 0.0])
        total, i = max_accumulation(daily, 3)
        assert i == 2
        assert daily[i:i + 3].sum() == pytest.approx(total)


class TestConfigConsistency:
    def test_reference_matches_the_derived_2018_depth(self):
        """
        332 mm is the ERA5 3-day maximum for August 2018 (331.6, 14-16 Aug).
        If this drifts, either the constant or the derivation changed and the
        two must be reconciled.
        """
        assert RAINFALL.reference_event_mm == pytest.approx(332.0, abs=1.0)

    def test_reference_is_a_scenario(self):
        assert RAINFALL.reference_event_mm in RAINFALL.scenarios

    def test_storm_window_pairs_with_amc_three(self):
        """
        AMC III already encodes a wet antecedent 5 days, so the storm window
        must be shorter than that or antecedent wetness is counted twice.
        """
        assert HYDRO.amc == "III"
        assert STORM_WINDOW_DAYS < 5

    def test_known_events_are_sentinel_1_era(self):
        """Sentinel-1A launched in 2014; anything earlier has no SAR inventory."""
        for event in EVENTS:
            assert int(event) >= 2014
