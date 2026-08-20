"""
Tests for the hydrologic soil group derivation.

Fetching needs the network, so what is tested here is the classification: the
USDA texture triangle against canonical points, the texture-to-group mapping,
the near-surface water table demotion, and the curve number lookup that
consumes the result.
"""
import numpy as np
import pytest

from config import HYDRO
from hydrology import curve_number_from_lulc, runoff_depth
from soil_hsg import (
    G_PER_KG_TO_PERCENT,
    HSG_CODES,
    TEXTURE_ORDER,
    TEXTURE_TO_HSG,
    WATERLOGGED_PROMOTION,
    apply_waterlogged_demotion,
    hsg_from_texture,
    usda_texture,
)


def classify(sand_pct, clay_pct):
    """Texture name for a single sand/clay pair."""
    idx = usda_texture(
        np.array([[sand_pct]], dtype=np.float32),
        np.array([[clay_pct]], dtype=np.float32),
    )[0, 0]
    return TEXTURE_ORDER[idx] if idx >= 0 else None


class TestTextureTriangle:
    @pytest.mark.parametrize("sand,clay,expected", [
        (90, 5, "sand"),
        (80, 5, "loamy sand"),
        (65, 10, "sandy loam"),
        (40, 20, "loam"),
        (20, 15, "silt loam"),
        (10, 5, "silt"),
        (60, 25, "sandy clay loam"),
        (30, 33, "clay loam"),
        (10, 33, "silty clay loam"),
        (50, 40, "sandy clay"),
        (10, 45, "silty clay"),
        (20, 50, "clay"),
    ])
    def test_canonical_points(self, sand, clay, expected):
        assert classify(sand, clay) == expected

    def test_every_class_is_reachable(self):
        """A class in TEXTURE_ORDER that nothing maps to would be dead code."""
        seen = set()
        for sand in range(0, 101, 2):
            for clay in range(0, 101 - sand, 2):
                seen.add(classify(sand, clay))
        missing = set(TEXTURE_ORDER) - seen
        assert not missing, f"unreachable texture classes: {missing}"

    def test_never_returns_unclassified_for_valid_fractions(self):
        sand, clay = np.meshgrid(np.arange(0, 101, 5.0), np.arange(0, 101, 5.0))
        ok = (sand + clay) <= 100
        idx = usda_texture(sand.astype(np.float32), clay.astype(np.float32))
        assert (idx[ok] >= 0).all()

    def test_nan_input_stays_unclassified(self):
        idx = usda_texture(
            np.array([[np.nan]], dtype=np.float32),
            np.array([[np.nan]], dtype=np.float32),
        )
        assert idx[0, 0] == -1


class TestHsgMapping:
    def test_every_texture_has_a_group(self):
        for name in TEXTURE_ORDER:
            assert name in TEXTURE_TO_HSG

    def test_sand_is_the_best_group_and_clay_the_worst(self):
        assert TEXTURE_TO_HSG["sand"] == "A"
        assert TEXTURE_TO_HSG["clay"] == "D"

    def test_codes_are_ordered_a_to_d(self):
        assert HSG_CODES["A"] < HSG_CODES["B"] < HSG_CODES["C"] < HSG_CODES["D"]

    def test_maps_indices_to_codes(self):
        idx = np.array([[TEXTURE_ORDER.index("sand"),
                         TEXTURE_ORDER.index("clay")]], dtype=np.int8)
        got = hsg_from_texture(idx)
        assert got[0, 0] == HSG_CODES["A"]
        assert got[0, 1] == HSG_CODES["D"]

    def test_unclassified_becomes_zero_not_a_group(self):
        got = hsg_from_texture(np.array([[-1]], dtype=np.int8))
        assert got[0, 0] == 0


class TestUnitConversion:
    def test_soilgrids_is_g_per_kg_not_percent(self):
        """SoilGrids stores 412 for 41.2%."""
        assert G_PER_KG_TO_PERCENT == 10.0

    def test_raw_values_misclassify_toward_clay(self):
        """
        Skipping the conversion is a silent, plausible-looking error. Both
        fractions come through ten times too large, and since any clay value
        above 40 triggers the clay classes, the result skews to the clay corner
        rather than the sand corner -- so it *overstates* runoff.
        """
        raw_sand, raw_clay = 412.0, 250.0  # g/kg, i.e. 41.2% and 25.0%
        correct = classify(raw_sand / G_PER_KG_TO_PERCENT,
                           raw_clay / G_PER_KG_TO_PERCENT)
        wrong = classify(raw_sand, raw_clay)
        assert correct == "loam"
        assert wrong == "sandy clay"
        # And that error costs two whole soil groups.
        assert TEXTURE_TO_HSG[correct] == "B"
        assert TEXTURE_TO_HSG[wrong] == "D"


class TestWaterloggedDemotion:
    def test_demotes_well_drained_soils_on_the_low_coast(self):
        hsg = np.array([[HSG_CODES["A"], HSG_CODES["B"]]], dtype=np.int8)
        dem = np.array([[1.0, 1.0]], dtype=np.float32)
        got = apply_waterlogged_demotion(hsg, dem)
        assert got[0, 0] == HSG_CODES["B"]
        assert got[0, 1] == HSG_CODES["C"]

    def test_leaves_higher_ground_alone(self):
        hsg = np.array([[HSG_CODES["A"]]], dtype=np.int8)
        dem = np.array([[50.0]], dtype=np.float32)
        assert apply_waterlogged_demotion(hsg, dem)[0, 0] == HSG_CODES["A"]

    def test_never_promotes(self):
        """The rule may only make soils worse, never better."""
        hsg = np.array([[1, 2, 3, 4]], dtype=np.int8)
        dem = np.zeros((1, 4), dtype=np.float32)
        got = apply_waterlogged_demotion(hsg, dem)
        assert (got >= hsg).all()

    def test_already_poor_groups_are_untouched(self):
        hsg = np.array([[HSG_CODES["C"], HSG_CODES["D"]]], dtype=np.int8)
        dem = np.zeros((1, 2), dtype=np.float32)
        got = apply_waterlogged_demotion(hsg, dem)
        assert got[0, 0] == HSG_CODES["C"]
        assert got[0, 1] == HSG_CODES["D"]

    def test_nan_elevation_is_not_treated_as_low(self):
        hsg = np.array([[HSG_CODES["A"]]], dtype=np.int8)
        dem = np.array([[np.nan]], dtype=np.float32)
        assert apply_waterlogged_demotion(hsg, dem)[0, 0] == HSG_CODES["A"]

    def test_threshold_is_coastal_scale(self):
        assert 0.0 < WATERLOGGED_PROMOTION["elevation_m"] <= 10.0


class TestCurveNumberWithSoil:
    """
    The lookup that consumes the soil raster. The group-C column must reproduce
    the old single-group table exactly, so any change in model output is
    attributable to the soil data rather than to a different CN table.
    """

    def test_group_c_reproduces_the_single_group_table(self):
        for cls, old in HYDRO.curve_numbers.items():
            lulc = np.array([[float(cls)]], dtype=np.float32)
            valid = np.array([[True]])
            hsg = np.array([[float(HSG_CODES["C"])]], dtype=np.float32)
            with_soil = curve_number_from_lulc(lulc, valid, hsg=hsg)
            without = curve_number_from_lulc(lulc, valid)
            assert with_soil[0, 0] == pytest.approx(without[0, 0], abs=1e-4), cls
            assert HYDRO.curve_numbers_by_hsg[cls][2] == old

    def test_group_a_runs_off_less_than_group_d(self):
        lulc = np.full((1, 2), 7.0, dtype=np.float32)  # built-up
        valid = np.ones((1, 2), dtype=bool)
        hsg = np.array([[1.0, 4.0]], dtype=np.float32)
        cn = curve_number_from_lulc(lulc, valid, hsg=hsg)
        assert cn[0, 0] < cn[0, 1]
        q = runoff_depth(100.0, cn)
        assert q[0, 0] < q[0, 1]

    def test_missing_soil_falls_back_to_group_c(self):
        """
        A partial soil raster must degrade to the previous behaviour, not punch
        NaN holes into the curve number grid.
        """
        lulc = np.full((1, 3), 7.0, dtype=np.float32)
        valid = np.ones((1, 3), dtype=bool)
        hsg = np.array([[np.nan, 0.0, 9.0]], dtype=np.float32)
        cn = curve_number_from_lulc(lulc, valid, hsg=hsg)
        expected = curve_number_from_lulc(lulc, valid)
        assert np.allclose(cn, expected, atol=1e-4)
        assert np.isfinite(cn).all()

    def test_invalid_pixels_stay_nan(self):
        lulc = np.full((1, 2), 7.0, dtype=np.float32)
        valid = np.array([[True, False]])
        hsg = np.array([[2.0, 2.0]], dtype=np.float32)
        cn = curve_number_from_lulc(lulc, valid, hsg=hsg)
        assert np.isfinite(cn[0, 0])
        assert np.isnan(cn[0, 1])

    def test_water_runs_off_completely_on_every_group(self):
        lulc = np.full((1, 4), 1.0, dtype=np.float32)
        valid = np.ones((1, 4), dtype=bool)
        hsg = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        cn = curve_number_from_lulc(lulc, valid, hsg=hsg)
        assert np.allclose(cn, 100.0)

    def test_table_is_monotonic_across_groups(self):
        """A to D must never improve; that would invert the physics."""
        for cls, row in HYDRO.curve_numbers_by_hsg.items():
            assert list(row) == sorted(row), f"class {cls} is not monotonic: {row}"

    def test_every_lulc_class_has_a_soil_row(self):
        assert set(HYDRO.curve_numbers_by_hsg) == set(HYDRO.curve_numbers)

    def test_default_row_has_four_groups(self):
        assert len(HYDRO.default_curve_numbers_by_hsg) == 4
        for row in HYDRO.curve_numbers_by_hsg.values():
            assert len(row) == 4
