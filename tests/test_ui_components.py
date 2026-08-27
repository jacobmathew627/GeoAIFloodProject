"""
Tests for the sidebar rainfall inputs.

The bug this pins: the hazard model's rainfall input is a 3-day cumulative
depth (RainfallConfig.reference_event_mm, and _fetch_model_forecast's own
docstring), but _fetch_live_rainfall used to sum only the first 24 hours of
forecast and feed that straight into the model -- understating an
equivalent-intensity storm's risk by roughly 3x relative to the other two
rainfall sources on the same sidebar. And the manual slider was labelled
"mm, 24h" while actually driving the same 3-day-total quantity.
"""

import inspect
from unittest.mock import MagicMock, patch

import pytest

import ui_components


def _hourly_response(n_hours, mm_per_hour=2.0):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"hourly": {"precipitation": [mm_per_hour] * n_hours}}
    return resp


class TestFetchLiveRainfall:
    def test_sums_72_hours_not_24(self):
        """
        A regression test for the exact bug: summing only hourly[:24] out of
        a longer forecast silently truncated a 3-day storm to its first day.
        """
        cfg = MagicMock()
        cfg.weather_params = {"latitude": 10.0, "longitude": 76.3}
        cfg.live_weather_url = "https://api.open-meteo.com/v1/forecast"

        with patch("requests.get", return_value=_hourly_response(72, mm_per_hour=2.0)):
            total = ui_components._fetch_live_rainfall(cfg)

        assert total == pytest.approx(144.0)

    def test_requests_three_forecast_days(self):
        cfg = MagicMock()
        cfg.weather_params = {"latitude": 10.0, "longitude": 76.3}
        cfg.live_weather_url = "https://api.open-meteo.com/v1/forecast"

        with patch("requests.get", return_value=_hourly_response(72)) as mock_get:
            ui_components._fetch_live_rainfall(cfg)

        _, kwargs = mock_get.call_args
        assert kwargs["params"]["forecast_days"] == 3

    def test_ignores_hours_beyond_72(self):
        """A longer forecast window than needed should still only contribute
        its first 3 days -- the model's input is a 3-day total, not more."""
        cfg = MagicMock()
        cfg.weather_params = {"latitude": 10.0, "longitude": 76.3}
        cfg.live_weather_url = "https://api.open-meteo.com/v1/forecast"

        with patch("requests.get", return_value=_hourly_response(96, mm_per_hour=1.0)):
            total = ui_components._fetch_live_rainfall(cfg)

        assert total == pytest.approx(72.0)

    def test_returns_none_on_failure(self):
        cfg = MagicMock()
        cfg.weather_params = {"latitude": 10.0, "longitude": 76.3}
        cfg.live_weather_url = "https://api.open-meteo.com/v1/forecast"

        with patch("requests.get", side_effect=Exception("network down")):
            assert ui_components._fetch_live_rainfall(cfg) is None

    def test_treats_none_hourly_values_as_zero(self):
        cfg = MagicMock()
        cfg.weather_params = {"latitude": 10.0, "longitude": 76.3}
        cfg.live_weather_url = "https://api.open-meteo.com/v1/forecast"
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"hourly": {"precipitation": [1.0, None, 1.0] + [0.0] * 69}}

        with patch("requests.get", return_value=resp):
            total = ui_components._fetch_live_rainfall(cfg)

        assert total == pytest.approx(2.0)


class TestSliderLabelling:
    def test_slider_is_labelled_as_a_3_day_total_not_24h(self):
        """
        Pins the label text directly against the source rather than driving
        a full Streamlit AppTest harness -- the thing that must never
        regress is that "24h" does not reappear next to this slider.
        """
        source = inspect.getsource(ui_components.render_sidebar)
        assert "24h" not in source
        assert "3-day" in source

    def test_reference_event_caption_still_present(self):
        source = inspect.getsource(ui_components.render_sidebar)
        assert "reference_event_mm" in source
