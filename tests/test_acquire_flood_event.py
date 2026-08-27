"""
Tests for the Sentinel-1 change-detection flood acquisition.

build_flood_image()/acquire() need a live, authenticated Earth Engine session
and align() needs a real master grid on disk -- neither is unit tested here,
matching the project's existing precedent for src/upstream_routing.py's own
align(). What is tested is the event table (every window must be internally
consistent -- baseline before event, not absurdly long or short) and the
change-detection thresholds, which is where a transcription mistake would
actually cost something (a wrong date silently changes which storm gets
measured; nothing would visibly break).
"""
from datetime import date

import pytest

from acquire_flood_event import (
    EVENTS,
    MAX_HAND_M,
    MAX_SLOPE_DEG,
    VV_DROP_MIN_DB,
    VV_WATER_MAX_DB,
)


def _parse(d: str) -> date:
    y, m, day = (int(x) for x in d.split("-"))
    return date(y, m, day)


class TestEventWindows:
    def test_every_event_has_a_baseline_and_an_event_window(self):
        for name, windows in EVENTS.items():
            assert set(windows) == {"event", "baseline"}, name

    def test_baseline_precedes_the_event_window(self):
        """
        A baseline sampled *after* the flood would include recovery/receding
        water in the "normal" reference and understate the backscatter drop.
        """
        for name, w in EVENTS.items():
            baseline_end = _parse(w["baseline"][1])
            event_start = _parse(w["event"][0])
            assert baseline_end <= event_start, name

    def test_event_windows_are_short(self):
        """
        The event window should bracket one storm, not a whole season -- a
        multi-week window would let min() pick up an unrelated later dip.
        """
        for name, w in EVENTS.items():
            start, end = _parse(w["event"][0]), _parse(w["event"][1])
            assert 0 < (end - start).days <= 21, name

    def test_baseline_windows_are_long_enough_to_median_over(self):
        """A single-scene baseline is noisy; acquire() only warns below 2."""
        for name, w in EVENTS.items():
            start, end = _parse(w["baseline"][0]), _parse(w["baseline"][1])
            assert (end - start).days >= 30, name

    def test_2026_event_window_starts_before_the_month(self):
        """
        Regression: the 2026 Kerala floods began in late July, peaking
        ~1 Aug. A window starting on 2026-08-01 (the pattern every other
        event uses) would clip the onset. This one deliberately starts
        2026-07-15 instead.
        """
        start = _parse(EVENTS["2026"]["event"][0])
        assert start.month == 7

    def test_2026_event_window_covers_the_confirmed_peak(self):
        """News reporting and an ERA5 cross-check both place the peak on
        2026-08-01; the acquisition window must contain that date."""
        start = _parse(EVENTS["2026"]["event"][0])
        end = _parse(EVENTS["2026"]["event"][1])
        peak = date(2026, 8, 1)
        assert start <= peak <= end

    def test_no_two_events_share_a_year_and_overlap(self):
        """Two flood events the same year would need distinct windows, or
        the wrong one's baseline could sample the other's floodwater."""
        by_year = {}
        for name, w in EVENTS.items():
            by_year.setdefault(name[:4], []).append(w)
        for year, windows in by_year.items():
            if len(windows) < 2:
                continue
            spans = sorted(
                (_parse(w["baseline"][0]), _parse(w["event"][1])) for w in windows
            )
            for (s1, e1), (s2, e2) in zip(spans, spans[1:]):
                assert e1 < s2, year


class TestChangeDetectionThresholds:
    def test_water_ceiling_is_a_plausible_vv_value(self):
        """Open water in VV typically reads below about -16 dB."""
        assert -25.0 < VV_WATER_MAX_DB < -10.0

    def test_drop_threshold_is_positive(self):
        """A drop is baseline-minus-event; requiring it positive is what
        rules out permanently dark surfaces that never changed."""
        assert VV_DROP_MIN_DB > 0

    def test_slope_exclusion_is_gentle(self):
        """Water does not pond on a hillside; the cutoff should be a few
        degrees, not tens of degrees."""
        assert 0 < MAX_SLOPE_DEG <= 10.0

    def test_hand_exclusion_is_a_plausible_fluvial_reach(self):
        assert 0 < MAX_HAND_M <= 30.0


class TestModuleDocumentation:
    def test_status_note_does_not_overclaim_test_coverage(self):
        """
        The module previously claimed 'everything below the GEE calls is
        exercised by tests' when no test file existed at all. Whatever the
        docstring says now must not repeat a claim that isn't true.
        """
        import acquire_flood_event

        doc = acquire_flood_event.__doc__ or ""
        assert "exercised by tests" not in doc
