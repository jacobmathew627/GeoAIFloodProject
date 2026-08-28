"""
Tests for the land-cover drift measurement.

The measurement itself needs Earth Engine, so what is pinned here is the
configuration, the cache contract, and the framing that must not be quietly
softened: the 2018 land cover is correct for *training* and wrong for
*inference*, and the app must say so rather than presenting a present-day
number as if the surface underneath it were current.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import landcover_drift  # noqa: E402
from landcover_drift import BUILT_CLASS, MODEL_EPOCH, WINDOWS, load  # noqa: E402


class TestConfiguration:
    def test_model_epoch_matches_the_training_inventory(self):
        """
        The epoch must be the year of the flood inventory the model is trained
        on. If these ever diverge the drift figure is measuring the wrong gap.
        """
        from ndem_labels import EVENTS

        assert str(MODEL_EPOCH) in EVENTS

    def test_built_class_is_dynamic_worlds_built_code(self):
        # 0 water, 1 trees, 2 grass, 3 flooded_veg, 4 crops, 5 shrub, 6 built,
        # 7 bare, 8 snow. Getting this wrong would silently measure grassland.
        assert BUILT_CLASS == 6

    def test_windows_span_epoch_to_recent_and_are_ordered(self):
        assert WINDOWS[0] == MODEL_EPOCH
        assert list(WINDOWS) == sorted(WINDOWS)
        assert WINDOWS[-1] >= 2024


class TestCacheContract:
    def test_load_returns_none_when_never_measured(self, tmp_path):
        assert load(models_dir=tmp_path) is None

    def test_load_reads_a_cached_measurement(self, tmp_path):
        payload = {"drift_pct": 23.3, "drift_km2": 161.1, "model_epoch": 2018}
        (tmp_path / "landcover_drift.json").write_text(json.dumps(payload), encoding="utf-8")
        assert load(models_dir=tmp_path)["drift_pct"] == 23.3


@pytest.mark.requires_model
class TestRecordedMeasurement:
    """Guards on the checked-in measurement, when one is present."""

    def test_recorded_drift_is_material_and_positive(self):
        drift = load()
        if drift is None:
            pytest.skip("drift never measured")
        # Kochi has not shrunk. A negative or ~zero figure means the reducer or
        # the class code is wrong, not that the city stopped growing.
        assert drift["drift_pct"] > 5.0
        assert drift["drift_km2"] > 0

    def test_series_is_anchored_on_the_model_epoch(self):
        drift = load()
        if drift is None:
            pytest.skip("drift never measured")
        epoch_rows = [r for r in drift["series"] if r["year"] == drift["model_epoch"]]
        assert len(epoch_rows) == 1
        assert epoch_rows[0]["change_vs_model_epoch_pct"] == 0.0


class TestFramingIsNotSoftened:
    def test_module_states_this_is_temporal_transfer_not_a_stale_file(self):
        doc = landcover_drift.__doc__ or ""
        assert "training" in doc.lower()
        assert "temporal-transfer" in doc.lower() or "temporal transfer" in doc.lower()

    def test_direction_of_the_error_is_stated(self):
        """
        Understating risk is the dangerous direction. It must be named
        explicitly, in the module and in the UI, not left for a reader to infer.
        """
        import inspect

        import ui_components

        doc = landcover_drift.__doc__ or ""
        assert "understate" in doc.lower()
        ui_src = inspect.getsource(ui_components._render_landcover_drift)
        assert "understate" in ui_src.lower()

    def test_ui_is_silent_rather_than_inventing_a_number(self, monkeypatch):
        """
        With no cached measurement the panel must render nothing at all -- a
        fabricated or placeholder drift figure would be worse than silence.
        """
        import inspect

        import ui_components

        src = inspect.getsource(ui_components._render_landcover_drift)
        assert "if not drift" in src and "return" in src
