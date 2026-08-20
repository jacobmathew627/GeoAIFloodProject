"""
Tests for the NDEM flood-label configuration.

Offline only: building rasters needs the 42 MB download and the aligned grid.
What matters here is that the event table stays coherent, because it is what
lets the rainfall sensitivity be fitted across events.
"""
import pytest

from ndem_labels import EVENTS, PRIMARY_EVENT


class TestEventTable:
    def test_primary_event_exists(self):
        assert PRIMARY_EVENT in EVENTS

    def test_primary_event_is_the_peak_timed_one(self):
        """
        2018 is the event with peak-timed acquisitions and the largest urban
        signal, which is why it is the default training label.
        """
        assert PRIMARY_EVENT == "2018"
        assert any("17-08-2018" in d for d in EVENTS["2018"]["dates"])

    def test_every_event_has_dates_and_a_note(self):
        for name, cfg in EVENTS.items():
            assert cfg["dates"], name
            assert cfg["note"], name

    def test_dates_are_ddmmyyyy_with_time(self):
        for name, cfg in EVENTS.items():
            for d in cfg["dates"]:
                # NDEM stores "DD-MM-YYYY HH:MM"; parsing it as ISO silently
                # yields the wrong year, so the format is pinned.
                head, _, tail = d.partition(" ")
                parts = head.split("-")
                assert len(parts) == 3, d
                assert len(parts[0]) == 2 and len(parts[1]) == 2 and len(parts[2]) == 4, d
                assert ":" in tail, d

    def test_event_year_matches_its_dates(self):
        for name, cfg in EVENTS.items():
            for d in cfg["dates"]:
                assert d.split(" ")[0].endswith(name), f"{name} has date {d}"

    def test_rainfall_matches_the_imd_derivation(self):
        """
        These depths come from src/reference_rainfall.py against IMD gridded
        rainfall. If one drifts the two must be reconciled, because beta is
        fitted from the pairing of rainfall to extent.
        """
        assert EVENTS["2018"]["rainfall_mm"] == pytest.approx(443.2, abs=0.5)
        assert EVENTS["2019"]["rainfall_mm"] == pytest.approx(412.5, abs=0.5)
        assert EVENTS["2021"]["rainfall_mm"] == pytest.approx(173.7, abs=0.5)

    def test_reference_event_matches_config(self):
        from config import RAINFALL

        assert EVENTS[PRIMARY_EVENT]["rainfall_mm"] == pytest.approx(
            RAINFALL.reference_event_mm, abs=1.0
        )

    def test_at_least_three_events_have_rainfall(self):
        """Fitting a sensitivity needs more than two points."""
        with_rain = [e for e, c in EVENTS.items() if c["rainfall_mm"] is not None]
        assert len(with_rain) >= 3

    def test_rainfall_ordering_is_plausible(self):
        """2018 was the most extreme of the three; 2021 the least."""
        assert EVENTS["2018"]["rainfall_mm"] > EVENTS["2019"]["rainfall_mm"]
        assert EVENTS["2019"]["rainfall_mm"] > EVENTS["2021"]["rainfall_mm"]
