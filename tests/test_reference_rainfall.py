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
    merge_results,
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
        assert daily[i : i + 3].sum() == pytest.approx(total)


class TestConfigConsistency:
    def test_reference_matches_the_derived_2018_depth(self):
        """
        443 mm is the IMD gauge-based 3-day maximum for August 2018 (443.2,
        15-17 Aug). If this drifts, either the constant or the derivation
        changed and the two must be reconciled.
        """
        assert RAINFALL.reference_event_mm == pytest.approx(443.0, abs=1.0)

    def test_reference_exceeds_the_era5_estimate(self):
        """
        Regression: the reference was briefly taken from ERA5 (331.6 mm).
        ERA5 smooths orographic extremes and under-reads the gauge analysis by
        1.34x for this event, so the authoritative figure must be higher.
        """
        assert RAINFALL.reference_event_mm > 331.6

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


class TestMergeResults:
    """
    Regression coverage for a real bug: running the CLI for a single event
    (`--event 2020`, no `--all`) used to overwrite the whole
    models/reference_rainfall.json, destroying the cached 2018/2019/2021
    results that fit_beta.py and the README both depend on. Found by doing
    exactly that while deriving 2020's rainfall.
    """

    def test_new_event_is_added_without_losing_existing_ones(self):
        existing = {"2018": {"reference_event_mm": 443.2}, "2019": {"reference_event_mm": 412.5}}
        new = {"2020": {"reference_event_mm": 305.5}}
        merged = merge_results(existing, new)
        assert set(merged) == {"2018", "2019", "2020"}
        assert merged["2018"] == existing["2018"]
        assert merged["2019"] == existing["2019"]
        assert merged["2020"] == new["2020"]

    def test_rerunning_an_event_updates_it_in_place(self):
        """A rerun of an existing event should overwrite that event only."""
        existing = {"2018": {"reference_event_mm": 400.0}, "2021": {"reference_event_mm": 173.7}}
        new = {"2018": {"reference_event_mm": 443.2}}
        merged = merge_results(existing, new)
        assert merged["2018"]["reference_event_mm"] == 443.2
        assert merged["2021"] == existing["2021"]

    def test_empty_existing_file_just_adopts_new_results(self):
        assert merge_results({}, {"2018": {"reference_event_mm": 443.2}}) == {
            "2018": {"reference_event_mm": 443.2}
        }

    def test_a_full_all_run_overwrites_everything_it_covers(self):
        """`--all` computes every known event, so the merge should end up
        identical to `new` whenever `new` already covers every key."""
        existing = {"2018": {"reference_event_mm": 331.6}}  # stale ERA5-era value
        new = {"2018": {"reference_event_mm": 443.2}, "2019": {"reference_event_mm": 412.5}}
        assert merge_results(existing, new) == new
