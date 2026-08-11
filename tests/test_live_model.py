"""
Tests for the live rainfall engine and the pluvial index.

The properties that matter for a slider: evaluating a new rainfall must be
cheap, monotonic, bounded, and must not resurrect nodata.
"""
import numpy as np
import pytest

from pluvial import (
    class_runoff_depths,
    fill_depressions,
    label_depressions,
)


class TestClassRunoff:
    def test_one_depth_per_class(self):
        classes = np.array([1, 2, 7])
        q = class_runoff_depths(150.0, classes)
        assert set(q) == {1, 2, 7}

    def test_monotonic_in_rainfall(self):
        classes = np.array([1, 2, 4, 5, 7, 11])
        previous = None
        for p in (0, 25, 50, 100, 200, 400):
            q = class_runoff_depths(p, classes)
            if previous is not None:
                for k in classes:
                    assert q[int(k)] >= previous[int(k)] - 1e-6
            previous = q

    def test_never_exceeds_rainfall(self):
        classes = np.array([1, 2, 7])
        for p in (10, 100, 400):
            for v in class_runoff_depths(p, classes).values():
                assert v <= p + 1e-4

    def test_open_water_sheds_everything(self):
        # class 1 is permanent water, CN 100
        assert class_runoff_depths(120.0, np.array([1]))[1] == pytest.approx(120.0, rel=1e-3)

    def test_impervious_sheds_more_than_forest(self):
        q = class_runoff_depths(150.0, np.array([2, 7]))
        assert q[7] > q[2]  # built-up vs tree cover

    def test_unknown_class_falls_back(self):
        q = class_runoff_depths(150.0, np.array([99]))
        assert q[99] > 0


class TestDepressionFill:
    """
    Kept and tested even though the model does not use it: the fill is correct,
    and the reason it is unused (1 m vertical DEM quantisation) is a property
    of the data, not of the code.
    """

    def test_fills_a_pit_to_its_rim(self):
        elev = np.array([
            [5.0, 5.0, 5.0],
            [5.0, 1.0, 5.0],
            [5.0, 5.0, 5.0],
        ])
        valid = np.ones_like(elev, dtype=bool)
        filled = fill_depressions(elev, valid)
        assert filled[1, 1] == pytest.approx(5.0)

    def test_never_lowers_the_surface(self):
        rng = np.random.default_rng(0)
        elev = rng.uniform(0, 10, size=(20, 20))
        valid = np.ones_like(elev, dtype=bool)
        filled = fill_depressions(elev, valid)
        assert (filled >= elev - 1e-9).all()

    def test_leaves_a_monotone_slope_untouched(self):
        elev = np.tile(np.arange(6, 0, -1, dtype=float), (4, 1))
        valid = np.ones_like(elev, dtype=bool)
        filled = fill_depressions(elev, valid)
        assert np.allclose(filled, elev)

    def test_invalid_cells_are_nan(self):
        elev = np.array([[1.0, 2.0], [3.0, 4.0]])
        valid = np.array([[True, True], [True, False]])
        filled = fill_depressions(elev, valid)
        assert np.isnan(filled[1, 1])

    def test_labels_only_real_depressions(self):
        depth = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.0]])
        labels, n = label_depressions(depth, min_depth_m=0.05)
        assert n == 1
        assert labels[1, 1] == 1
        assert labels[0, 0] == 0

    def test_shallow_noise_is_not_a_depression(self):
        depth = np.full((4, 4), 0.01)
        _, n = label_depressions(depth, min_depth_m=0.05)
        assert n == 0


# ──────────────────────────────────────────────
# The live grid. Built synthetically so the tests need no rasters.
# ──────────────────────────────────────────────
@pytest.fixture
def grid():
    from affine import Affine
    from rasterio.coords import BoundingBox

    from live_model import LiveGrid

    h, w = 12, 16
    rng = np.random.default_rng(0)
    susc = rng.uniform(0.001, 0.9, size=(h, w)).astype(np.float32)
    susc[0, :] = np.nan  # a nodata band

    classes = np.array([2, 7])
    basis = {
        2: rng.uniform(1, 500, size=(h, w)).astype(np.float32),
        7: rng.uniform(1, 500, size=(h, w)).astype(np.float32),
    }
    cn = np.where(np.isfinite(susc), 88.0, np.nan).astype(np.float32)

    return LiveGrid(
        susceptibility=susc,
        curve_number=cn,
        basis=basis,
        classes=classes,
        tan_slope=np.full((h, w), 0.02, dtype=np.float32),
        cell_area_m2=100.0,
        cell_width_m=10.0,
        transform=Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 1100000.0),
        crs="EPSG:32643",
        bounds=BoundingBox(600000.0, 1099880.0, 600160.0, 1100000.0),
        shape=(h, w),
        pluvial_lo=0.0,
        pluvial_hi=20.0,
    )


class TestFluvial:
    def test_monotonic_in_rainfall(self, grid):
        from live_model import fluvial_probability

        previous = None
        for mm in (0, 50, 100, 200, 400):
            p = fluvial_probability(grid, mm)
            finite = np.isfinite(p)
            if previous is not None:
                assert (p[finite] >= previous[finite] - 1e-6).all()
            previous = p

    def test_stays_a_probability(self, grid):
        from live_model import fluvial_probability

        for mm in (0, 150, 2000):
            p = fluvial_probability(grid, mm)
            v = p[np.isfinite(p)]
            assert (v >= 0).all() and (v <= 1).all()

    def test_nodata_stays_nodata(self, grid):
        from live_model import fluvial_probability

        p = fluvial_probability(grid, 150)
        assert np.isnan(p[0, :]).all()

    def test_reduces_to_susceptibility_at_reference(self, grid):
        from config import RAINFALL
        from live_model import fluvial_probability

        p = fluvial_probability(grid, RAINFALL.reference_event_mm)
        m = np.isfinite(p) & np.isfinite(grid.susceptibility)
        np.testing.assert_allclose(p[m], grid.susceptibility[m], atol=1e-5)


class TestPluvial:
    def test_bounded_zero_to_one(self, grid):
        from live_model import pluvial_index

        for mm in (10, 150, 500):
            v = pluvial_index(grid, mm)
            f = v[np.isfinite(v)]
            assert (f >= 0).all() and (f <= 1).all()

    def test_monotonic_in_rainfall(self, grid):
        from live_model import pluvial_index

        previous = None
        for mm in (25, 50, 100, 200, 400):
            v = pluvial_index(grid, mm)
            finite = np.isfinite(v)
            if previous is not None:
                assert (v[finite] >= previous[finite] - 1e-6).all()
            previous = v

    def test_nodata_stays_nodata(self, grid):
        from live_model import pluvial_index

        assert np.isnan(pluvial_index(grid, 150)[0, :]).all()

    def test_steeper_ground_sheds_more(self, grid):
        """Same runoff, steeper slope -> lower wetness."""
        from live_model import _pluvial_raw

        flat = _pluvial_raw(grid, 150)
        grid.tan_slope = np.full(grid.shape, 0.5, dtype=np.float32)
        steep = _pluvial_raw(grid, 150)
        m = np.isfinite(flat) & np.isfinite(steep)
        assert (steep[m] < flat[m]).all()


class TestQuery:
    def test_outside_the_grid_returns_none(self, grid):
        from live_model import query

        assert query(grid, 0.0, 0.0, 150) is None

    def test_reports_the_physical_quantities(self, grid):
        from live_model import query
        from pyproj import Transformer

        # Centre of a valid row
        x, y = grid.transform * (8.5, 6.5)
        lon, lat = Transformer.from_crs(
            grid.crs, "EPSG:4326", always_xy=True
        ).transform(x, y)

        r = query(grid, lat, lon, 150.0)
        assert r is not None
        assert 0 <= r["fluvial_probability"] <= 1
        assert 0 <= r["pluvial_index"] <= 1
        assert 0 < r["runoff_mm"] <= 150.0
        assert 0 <= r["runoff_coefficient"] <= 1


class TestPersistence:
    def test_round_trips_without_pickling_a_class(self, grid, tmp_path):
        """
        Regression: the cache used to be a joblib pickle of the LiveGrid
        dataclass. Building it via `python src/live_model.py` defined the class
        in __main__, so the app could not unpickle it.
        """
        from live_model import load, save

        save(grid, tmp_path)
        restored = load(tmp_path)

        np.testing.assert_allclose(
            restored.susceptibility, grid.susceptibility, equal_nan=True
        )
        assert restored.shape == grid.shape
        assert restored.crs == grid.crs
        assert sorted(restored.basis) == sorted(grid.basis)
        assert restored.pluvial_hi == pytest.approx(grid.pluvial_hi)

    def test_cache_contains_no_pickled_objects(self, grid, tmp_path):
        from live_model import save

        path = save(grid, tmp_path)
        # allow_pickle=False is the actual guarantee we want.
        with np.load(path, allow_pickle=False) as z:
            assert "susceptibility" in z
            assert "header_json" in z

    def test_missing_cache_raises_actionably(self, tmp_path):
        from live_model import load

        with pytest.raises(FileNotFoundError, match="--build"):
            load(tmp_path)
