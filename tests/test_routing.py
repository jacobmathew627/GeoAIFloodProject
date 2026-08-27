"""
Tests for D8 flow routing on synthetic terrain.

The properties that matter downstream: the network must be acyclic (a cycle
double-counts mass in the single-pass accumulation), accumulation must
conserve total weight, and water must never be routed uphill.
"""

import numpy as np
import pytest

from routing import (
    _topological_order,
    accumulate,
    d8_receivers,
    upstream_mean,
)


def _ranks(elev, valid):
    order = _topological_order(elev, valid)
    rank_flat = np.empty(order.size, dtype=np.int64)
    rank_flat[order] = np.arange(order.size, dtype=np.int64)
    return order, rank_flat.reshape(elev.shape)


def _network(elev, valid=None, tiebreak=None):
    elev = np.asarray(elev, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(elev)
    order, rank = _ranks(elev, valid)
    receiver = d8_receivers(elev, valid, tiebreak, rank)
    return elev, valid, receiver, order


@pytest.fixture
def ramp():
    """A simple east-facing slope: every cell drains one column right."""
    return np.tile(np.arange(6, 0, -1, dtype=np.float32), (4, 1))


@pytest.fixture
def bowl():
    """A single low point at the centre of a 5x5 grid."""
    r, c = np.mgrid[0:5, 0:5]
    return (np.abs(r - 2) + np.abs(c - 2)).astype(np.float32)


class TestReceivers:
    def test_never_routes_uphill(self, bowl):
        elev, valid, receiver, _ = _network(bowl)
        flat_elev = elev.ravel()
        for i in np.flatnonzero(valid.ravel()):
            j = receiver[i]
            assert flat_elev[j] <= flat_elev[i] + 1e-6

    def test_network_is_acyclic(self, bowl):
        """Following receivers must always terminate."""
        elev, valid, receiver, _ = _network(bowl)
        for i in np.flatnonzero(valid.ravel()):
            seen = set()
            cur = i
            for _ in range(elev.size + 1):
                if cur in seen:
                    pytest.fail(f"cycle reached from cell {i}")
                seen.add(cur)
                nxt = receiver[cur]
                if nxt == cur:
                    break
                cur = nxt
            else:
                pytest.fail(f"no terminal cell reached from {i}")

    def test_ramp_drains_downslope(self, ramp):
        _, _, receiver, _ = _network(ramp)
        h, w = ramp.shape
        # A cell not in the last column should move strictly right.
        idx = 1 * w + 2
        assert receiver[idx] == 1 * w + 3

    def test_bowl_centre_is_terminal(self, bowl):
        _, _, receiver, _ = _network(bowl)
        centre = 2 * 5 + 2
        assert receiver[centre] == centre

    def test_invalid_cells_are_self_receiving(self):
        elev = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
        valid = np.isfinite(elev)
        _, _, receiver, _ = _network(elev, valid)
        assert receiver[3] == 3


class TestAccumulation:
    def test_conserves_total_mass(self, bowl):
        """Every unit of weight must arrive at some terminal cell."""
        elev, valid, receiver, order = _network(bowl)
        w = np.ones_like(elev)
        acc = accumulate(w, receiver, order, valid)

        terminal = np.array([receiver[i] == i for i in range(elev.size)]).reshape(elev.shape)
        assert np.nansum(acc[terminal & valid]) == pytest.approx(valid.sum())

    def test_downstream_is_at_least_upstream(self, ramp):
        elev, valid, receiver, order = _network(ramp)
        acc = accumulate(np.ones_like(elev), receiver, order, valid)
        flat = acc.ravel()
        for i in np.flatnonzero(valid.ravel()):
            j = receiver[i]
            if j != i:
                assert flat[j] >= flat[i] - 1e-5

    def test_every_cell_counts_itself(self, bowl):
        elev, valid, receiver, order = _network(bowl)
        acc = accumulate(np.ones_like(elev), receiver, order, valid)
        assert np.nanmin(acc[valid]) >= 1.0

    def test_bowl_centre_collects_everything(self, bowl):
        elev, valid, receiver, order = _network(bowl)
        acc = accumulate(np.ones_like(elev), receiver, order, valid)
        assert acc[2, 2] == pytest.approx(float(valid.sum()))

    def test_invalid_cells_are_nan(self):
        elev = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
        valid = np.isfinite(elev)
        _, _, receiver, order = _network(elev, valid)
        acc = accumulate(np.ones_like(elev), receiver, order, valid)
        assert np.isnan(acc[1, 1])

    def test_zero_weights_accumulate_to_zero(self, bowl):
        elev, valid, receiver, order = _network(bowl)
        acc = accumulate(np.zeros_like(elev), receiver, order, valid)
        assert np.nansum(acc) == pytest.approx(0.0)


class TestUpstreamMean:
    def test_constant_field_is_preserved(self, bowl):
        """The catchment average of a constant must be that constant."""
        elev, valid, receiver, order = _network(bowl)
        values = np.full_like(elev, 7.5)
        out = upstream_mean(values, receiver, order, valid)
        assert np.allclose(out[valid], 7.5, atol=1e-4)

    def test_bounded_by_the_input_range(self, bowl):
        rng = np.random.default_rng(0)
        elev, valid, receiver, order = _network(bowl)
        values = rng.uniform(60, 100, size=elev.shape).astype(np.float32)
        out = upstream_mean(values, receiver, order, valid)
        assert np.nanmin(out[valid]) >= values[valid].min() - 1e-3
        assert np.nanmax(out[valid]) <= values[valid].max() + 1e-3

    def test_outlet_averages_the_whole_grid(self, bowl):
        elev, valid, receiver, order = _network(bowl)
        rng = np.random.default_rng(1)
        values = rng.uniform(0, 1, size=elev.shape).astype(np.float32)
        out = upstream_mean(values, receiver, order, valid)
        assert out[2, 2] == pytest.approx(float(values[valid].mean()), abs=1e-4)


class TestFlatResolution:
    def test_flats_stall_without_a_tiebreak(self):
        """A perfectly flat plateau has no steepest descent."""
        elev = np.zeros((4, 4), dtype=np.float32)
        elev[3, 3] = -1.0
        valid = np.ones_like(elev, dtype=bool)
        order, rank = _ranks(elev, valid)

        no_tb = d8_receivers(elev, valid, None, rank)
        stalled = sum(1 for i in range(elev.size) if no_tb[i] == i)
        assert stalled > 1, "expected the flat cells to have no receiver"

    def test_tiebreak_drains_the_flat(self):
        elev = np.zeros((4, 4), dtype=np.float32)
        elev[3, 3] = -1.0
        valid = np.ones_like(elev, dtype=bool)
        order, rank = _ranks(elev, valid)

        # Accumulation increasing toward the outlet corner.
        r, c = np.mgrid[0:4, 0:4]
        tb = (r + c).astype(np.float64)

        with_tb = d8_receivers(elev, valid, tb, rank)
        stalled = sum(1 for i in range(elev.size) if with_tb[i] == i)
        assert stalled < 4

    def test_tiebreak_still_acyclic(self):
        elev = np.zeros((5, 5), dtype=np.float32)
        elev[4, 4] = -1.0
        valid = np.ones_like(elev, dtype=bool)
        order, rank = _ranks(elev, valid)
        rng = np.random.default_rng(2)
        tb = rng.uniform(size=elev.shape)

        receiver = d8_receivers(elev, valid, tb, rank)
        for i in range(elev.size):
            seen, cur = set(), i
            for _ in range(elev.size + 1):
                assert cur not in seen, "cycle introduced by the flat tie-break"
                seen.add(cur)
                if receiver[cur] == cur:
                    break
                cur = receiver[cur]
