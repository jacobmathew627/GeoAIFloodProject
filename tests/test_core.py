"""
Tests for configuration, data loading and visualisation.

Several of these are regression tests for specific defects; those are named
and commented so the behaviour is not "simplified" back out later.
"""
import numpy as np
import pytest

from config import (
    GEO,
    HYDRO,
    LULC_CLASS_NAMES,
    MODEL_FILES,
    PERMANENT_WATER_CLASS,
    RAINFALL,
    RASTER,
    RISK,
    SUSCEPTIBILITY_FEATURES,
    VIZ,
    get_model_path,
    get_output_path,
    validate_environment,
)
from data_loading import LAYER_REGISTRY, _apply_layer_nodata_rules
from visualization import (
    FLOOD_COLOR_STOPS,
    FLOOD_COLORMAP,
    apply_colormap,
    compute_risk_stats,
    create_alert_message,
    create_flood_visualization,
    create_legend_html,
    pixel_area_km2_from_transform,
    prob_to_png_b64,
    risk_band_masks,
    valid_area_km2,
)

NODATA = RASTER.nodata_value


class TestConfig:
    def test_raster_config(self):
        assert RASTER.nodata_value == -9999.0
        assert RASTER.target_crs == "EPSG:32643"

    def test_cell_size_matches_the_master_grid(self):
        """
        Regression: cell_size was 30.0 while the LULC master grid is 10 m,
        so every km2 figure in the dashboard was 9x too large.
        """
        assert RASTER.cell_size == 10.0
        assert RASTER.pixel_area_km2 == pytest.approx(1e-4)

    def test_risk_thresholds_are_ordered(self):
        assert 0 < RISK.safe < RISK.moderate < RISK.high < RISK.critical < 1

    def test_risk_thresholds_suit_the_calibrated_scale(self):
        """
        Regression: the thresholds were 0.10/0.20/0.30/0.50, inherited from a
        score inflated ~11x. On the corrected scale (1.4% base rate) they
        classified the 2018 catastrophe as "monitoring active".
        """
        assert RISK.safe < 0.05, "lowest band must sit near the district base rate"
        assert RISK.critical < 0.5, "critical band must be reachable on the calibrated scale"

    def test_colormap_stops_match_the_risk_bands(self):
        """A colour change on the map must mean a class change in the stats."""
        stops = {round(v, 3) for v, _ in FLOOD_COLOR_STOPS}
        for edge in (RISK.safe, RISK.moderate, RISK.high, RISK.critical):
            assert round(edge, 3) in stops, f"no colour stop at band edge {edge}"

    def test_alert_triggers_are_fractions(self):
        assert 0 < RISK.critical_area_fraction_alert < 1
        assert 0 < RISK.elevated_area_fraction_warning < 1

    def test_rainfall_scenarios_sorted_and_cover_reference(self):
        assert list(RAINFALL.scenarios) == sorted(RAINFALL.scenarios)
        assert RAINFALL.reference_event_mm in RAINFALL.scenarios

    def test_path_helpers(self):
        assert "models" in str(get_model_path("test.pth"))
        assert "outputs" in str(get_output_path("test.tif"))

    def test_validate_environment_returns_list(self):
        assert isinstance(validate_environment(), list)

    def test_model_files_all_loadable_architectures(self):
        """
        Regression: geoai_flood_final.pth is a different architecture from the
        UNet in inference_final.py and could never be loaded, but was listed
        as a supported model.
        """
        assert "pytorch_final" not in MODEL_FILES

    def test_lulc_colors_match_documented_classes(self):
        assert set(VIZ.lulc_colors) == set(LULC_CLASS_NAMES)

    def test_permanent_water_class_is_named(self):
        assert LULC_CLASS_NAMES[PERMANENT_WATER_CLASS] == "Permanent water"

    def test_curve_number_present_in_feature_list(self):
        assert "curve_number" in SUSCEPTIBILITY_FEATURES
        assert len(set(SUSCEPTIBILITY_FEATURES)) == len(SUSCEPTIBILITY_FEATURES)


class TestLayerRegistry:
    def test_every_layer_has_a_kind(self):
        for name, (filename, kind) in LAYER_REGISTRY.items():
            assert filename.endswith(".tif"), name
            assert kind and kind.islower(), name

    def test_nodata_rules_actually_fire(self):
        """
        Regression: rules were keyed on layer names but the caller passed
        "continuous"/"categorical", so no rule ever matched and impossible
        values were rendered as if they were real.
        """
        data = np.array([[5.0, 20.0, NODATA]], dtype=np.float32)
        out = _apply_layer_nodata_rules(data.copy(), "lulc")
        assert out[0, 0] == 5.0          # valid class
        assert out[0, 1] == NODATA       # class 20 does not exist
        assert out[0, 2] == NODATA       # already nodata

    def test_rules_never_resurrect_nodata(self):
        data = np.full((3, 3), NODATA, dtype=np.float32)
        for kind in {k for _, k in LAYER_REGISTRY.values()}:
            out = _apply_layer_nodata_rules(data.copy(), kind)
            assert (out == NODATA).all(), kind

    def test_ndvi_range_rule(self):
        data = np.array([[-9999.0, -0.5, 0.5, 3.0]], dtype=np.float32)
        out = _apply_layer_nodata_rules(data.copy(), "ndvi")
        assert out[0, 1] == pytest.approx(-0.5)
        assert out[0, 2] == pytest.approx(0.5)
        assert out[0, 3] == NODATA


class TestColormap:
    def test_flood_color_stops_span_unit_interval(self):
        assert FLOOD_COLOR_STOPS[0][0] == 0.0
        assert FLOOD_COLOR_STOPS[-1][0] == 1.0
        assert len(FLOOD_COLOR_STOPS) == 6

    def test_colormap_object_is_a_matplotlib_colormap(self):
        """
        Regression: a test asserted len(FLOOD_COLORMAP) == 6 against a
        LinearSegmentedColormap, which has no length.
        """
        assert hasattr(FLOOD_COLORMAP, "__call__")
        assert FLOOD_COLORMAP(0.5).__len__() == 4

    def test_apply_colormap_shape_and_dtype(self):
        prob = np.array([[0.0, 0.5], [0.8, 1.0]], dtype=np.float32)
        rgba = apply_colormap(prob)
        assert rgba.shape == (2, 2, 4)
        assert rgba.dtype == np.uint8
        assert (rgba[..., 3] == 200).all()

    def test_nodata_is_transparent(self):
        """
        Regression: nodata fell outside every colour segment and so kept
        alpha 0 only by accident; an explicit test pins the behaviour.
        """
        prob = np.array([[NODATA, 0.5]], dtype=np.float32)
        rgba = apply_colormap(prob)
        assert rgba[0, 0, 3] == 0
        assert rgba[0, 1, 3] == 200

    def test_png_encoding_round_trips(self, probability_map):
        import base64
        import io

        from PIL import Image

        encoded = prob_to_png_b64(probability_map, max_dim=16)
        img = Image.open(io.BytesIO(base64.b64decode(encoded)))
        assert img.mode == "RGBA"
        assert max(img.size) <= 16

    def test_png_does_not_smear_nodata(self):
        """
        Regression: downscaling with a bilinear filter over -9999 sentinels
        smeared them into the valid range, painting a halo around every hole.
        """
        prob = np.full((64, 64), 0.9, dtype=np.float32)
        prob[:32, :] = NODATA
        encoded = prob_to_png_b64(prob, max_dim=32)

        import base64
        import io

        from PIL import Image

        rgba = np.array(Image.open(io.BytesIO(base64.b64decode(encoded))))
        alpha = rgba[..., 3]
        assert (alpha[:14, :] == 0).all()      # solidly nodata
        assert (alpha[18:, :] == 200).all()    # solidly valid


class TestNodataMasking:
    """
    Regression: the live model returns NaN for nodata while rasters read from
    disk use -9999. The renderers masked only on the sentinel, and every NaN
    comparison is False, so NaN cells went unmasked, were coloured black by the
    colormap's "bad" value, and then painted at the layer's alpha -- turning
    the whole map into a flat grey rectangle with the risk zones underneath it.
    """

    def test_masks_the_sentinel(self):
        from visualization import mask_nodata

        m = mask_nodata(np.array([[0.5, NODATA]], dtype=np.float32))
        assert not m.mask[0, 0]
        assert m.mask[0, 1]

    def test_masks_nan(self):
        from visualization import mask_nodata

        m = mask_nodata(np.array([[0.5, np.nan]], dtype=np.float32))
        assert not m.mask[0, 0]
        assert m.mask[0, 1]

    def test_masks_both_at_once(self):
        from visualization import mask_nodata

        m = mask_nodata(np.array([[0.5, np.nan, NODATA, 0.9]], dtype=np.float32))
        assert list(m.mask[0]) == [False, True, True, False]

    @pytest.mark.parametrize("bad", [np.nan, NODATA])
    def test_flood_overlay_leaves_nodata_transparent(self, bad):
        data = np.full((8, 8), 0.5, dtype=np.float32)
        data[0, :] = bad
        rgba, _ = create_flood_visualization(data, VIZ, RISK)
        assert (rgba[0, :, 3] == 0).all(), "nodata must not be painted"
        assert (rgba[1:, :, 3] > 0).all(), "valid data must be painted"

    @pytest.mark.parametrize("bad", [np.nan, NODATA])
    def test_pluvial_overlay_leaves_nodata_transparent(self, bad):
        from visualization import create_pluvial_visualization

        data = np.full((8, 8), 0.5, dtype=np.float32)
        data[0, :] = bad
        rgba, _ = create_pluvial_visualization(data)
        assert (rgba[0, :, 3] == 0).all()
        assert (rgba[1:, :, 3] > 0).all()

    def test_all_nan_renders_fully_transparent(self):
        data = np.full((4, 4), np.nan, dtype=np.float32)
        rgba, _ = create_flood_visualization(data, VIZ, RISK)
        assert (rgba[..., 3] == 0).all()


class TestRiskStats:
    BANDS = ("safe_pct", "moderate_pct", "high_pct", "severe_pct", "critical_pct")

    def test_percentages_sum_to_100(self):
        data = np.array(
            [[0.005, 0.05, 0.10], [0.20, 0.60, NODATA]], dtype=np.float32
        )
        stats = compute_risk_stats(data, RISK)
        assert sum(stats[k] for k in self.BANDS) == pytest.approx(100.0, abs=0.5)

    def test_bands_partition_the_data(self):
        """Every valid pixel lands in exactly one band."""
        rng = np.random.default_rng(0)
        data = rng.random((40, 40)).astype(np.float32)
        masks = risk_band_masks(data.ravel(), RISK)
        stacked = np.vstack([masks[n] for n in ("safe", "moderate", "high", "severe", "critical")])
        assert (stacked.sum(axis=0) == 1).all()

    def test_all_four_thresholds_are_used(self):
        """
        Regression: the config defined four thresholds but the classification
        applied only three, leaving RISK.critical as dead configuration.
        """
        just_below = np.array([[RISK.critical - 1e-4]], dtype=np.float32)
        just_above = np.array([[RISK.critical + 1e-4]], dtype=np.float32)
        assert compute_risk_stats(just_below, RISK)["severe_pct"] == pytest.approx(100.0)
        assert compute_risk_stats(just_above, RISK)["critical_pct"] == pytest.approx(100.0)

    def test_nodata_excluded_from_mean(self):
        data = np.array([[0.5, NODATA]], dtype=np.float32)
        stats = compute_risk_stats(data, RISK)
        assert stats["mean_prob"] == pytest.approx(0.5)

    def test_all_nodata_returns_zeros(self):
        data = np.full((4, 4), NODATA, dtype=np.float32)
        stats = compute_risk_stats(data, RISK)
        assert all(v == 0.0 for v in stats.values())

    def test_areas_derived_from_transform(self, identity_transform):
        data = np.full((10, 10), 0.5, dtype=np.float32)
        stats = compute_risk_stats(data, RISK, identity_transform)
        # 100 pixels at 10 m = 100 * 100 m2 = 0.01 km2
        assert stats["mapped_area_km2"] == pytest.approx(0.01)
        assert stats["critical_km2"] == pytest.approx(0.01)

    def test_expected_flooded_area_is_the_integral(self, identity_transform):
        """Independent of where the band edges fall."""
        data = np.full((10, 10), 0.25, dtype=np.float32)
        stats = compute_risk_stats(data, RISK, identity_transform)
        assert stats["expected_flooded_km2"] == pytest.approx(0.25 * 0.01)

    def test_areas_absent_without_transform(self):
        stats = compute_risk_stats(np.full((4, 4), 0.5, dtype=np.float32), RISK)
        assert "mapped_area_km2" not in stats


class TestGeometry:
    def test_pixel_area_from_transform(self, identity_transform):
        assert pixel_area_km2_from_transform(identity_transform) == pytest.approx(1e-4)

    def test_none_transform_returns_none(self):
        assert pixel_area_km2_from_transform(None) is None

    def test_valid_area_excludes_nodata(self, identity_transform):
        data = np.array([[0.5, NODATA], [0.5, 0.5]], dtype=np.float32)
        assert valid_area_km2(data, identity_transform) == pytest.approx(3e-4)


class TestAlerts:
    def test_none_when_no_valid_data(self):
        data = np.full((4, 4), NODATA, dtype=np.float32)
        assert create_alert_message(data, 150.0, GEO, RISK) is None

    def test_critical_alert_triggers(self, identity_transform):
        data = np.full((10, 10), 0.9, dtype=np.float32)
        message = create_alert_message(data, 400.0, GEO, RISK, identity_transform)
        assert "CRITICAL" in message

    def test_monitoring_when_quiet(self, identity_transform):
        data = np.full((10, 10), 0.01, dtype=np.float32)
        message = create_alert_message(data, 20.0, GEO, RISK, identity_transform)
        assert "MONITORING" in message

    def test_labels_estimates_as_estimates(self, identity_transform):
        data = np.full((10, 10), 0.9, dtype=np.float32)
        message = create_alert_message(data, 400.0, GEO, RISK, identity_transform)
        assert "planning estimate" in message


class TestLegend:
    def test_contains_labels_and_colours(self):
        html = create_legend_html("Test", [("A", "#ff0000"), ("B", "#00ff00")])
        for token in ("Test", "A", "B", "#ff0000", "#00ff00"):
            assert token in html


class TestHydrologyConfigIntegration:
    def test_permanent_water_has_maximum_curve_number(self):
        assert HYDRO.curve_numbers[PERMANENT_WATER_CLASS] == 100.0

    def test_built_up_sheds_more_than_forest(self):
        assert HYDRO.curve_numbers[7] > HYDRO.curve_numbers[2]
