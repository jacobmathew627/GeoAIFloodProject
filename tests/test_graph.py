"""
Tests for sub-catchment graph construction and directed message passing.

The graph experiment concluded *against* adopting the GNN, but the harness is
kept and tested: a negative result is only trustworthy if the machinery that
produced it is correct.
"""

import numpy as np
import pytest
import torch

import catchment_graph
from graph_model import DirectedSAGELayer, FlowGNN, GraphDataset, _standardise
from routing import _topological_order, d8_receivers


class _FakeNet:
    """A minimal stand-in for routing.FlowNetwork over a synthetic DEM."""

    def __init__(self, elev, cell_area=900.0):
        self.elev = np.asarray(elev, dtype=np.float32)
        self.valid = np.isfinite(self.elev)
        self.order = _topological_order(self.elev, self.valid)
        rank_flat = np.empty(self.order.size, dtype=np.int64)
        rank_flat[self.order] = np.arange(self.order.size, dtype=np.int64)
        self.receiver = d8_receivers(
            self.elev, self.valid, None, rank_flat.reshape(self.elev.shape)
        )
        self._cell_area = cell_area

    @property
    def cell_area_m2(self):
        return self._cell_area

    def contributing_area_m2(self):
        from routing import accumulate

        return accumulate(
            np.full(self.elev.shape, self._cell_area, dtype=np.float32),
            self.receiver,
            self.order,
            self.valid,
        )


@pytest.fixture
def y_network():
    """
    Two headwater branches joining into one trunk:

        9 . 9
        8 . 8
        . 7 .
        . 6 .
        . 5 .
    """
    n = np.nan
    return np.array(
        [
            [9, n, 9],
            [8, n, 8],
            [n, 7, n],
            [n, 6, n],
            [n, 5, n],
        ],
        dtype=np.float32,
    )


class TestDelineation:
    def test_labels_every_valid_cell(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        assert (g.labels[net.valid] >= 0).all()
        assert (g.labels[~net.valid] == -1).all()

    def test_node_cell_counts_sum_to_valid(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        assert g.node_cells.sum() == int(net.valid.sum())

    def test_edges_are_within_range(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        if len(g.edges):
            assert g.edges.min() >= 0
            assert g.edges.max() < g.n_nodes

    def test_no_self_edges(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        if len(g.edges):
            assert (g.edges[:, 0] != g.edges[:, 1]).all()

    def test_graph_is_acyclic(self, y_network):
        """Inherited from the D8 DAG; a cycle would break message passing."""
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        adj = {}
        for u, v in g.edges:
            adj.setdefault(int(u), []).append(int(v))

        state = {}

        def visit(n):
            if state.get(n) == 1:
                pytest.fail("cycle in the sub-catchment graph")
            if state.get(n) == 2:
                return
            state[n] = 1
            for m in adj.get(n, []):
                visit(m)
            state[n] = 2

        for n in range(g.n_nodes):
            visit(n)

    def test_finer_threshold_gives_more_nodes(self, y_network):
        net = _FakeNet(y_network, cell_area=1e6)
        coarse = catchment_graph.build(net, min_area_km2=5.0)
        fine = catchment_graph.build(net, min_area_km2=1.0)
        assert fine.n_nodes >= coarse.n_nodes

    def test_raises_when_no_outlets(self):
        elev = np.full((3, 3), np.nan, dtype=np.float32)
        net = _FakeNet(elev)
        with pytest.raises(ValueError, match="No sub-catchment outlets"):
            catchment_graph.build(net, min_area_km2=1.0)


class TestAggregation:
    def test_constant_field_is_preserved(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        values = np.where(net.valid, 4.25, np.nan).astype(np.float32)
        out = catchment_graph.aggregate_to_nodes(values, g)
        assert np.allclose(out[np.isfinite(out)], 4.25, atol=1e-5)

    def test_bounded_by_input_range(self, y_network):
        rng = np.random.default_rng(0)
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        values = np.where(net.valid, rng.uniform(2, 9, size=y_network.shape), np.nan)
        out = catchment_graph.aggregate_to_nodes(values.astype(np.float32), g)
        finite = out[np.isfinite(out)]
        assert finite.min() >= np.nanmin(values) - 1e-4
        assert finite.max() <= np.nanmax(values) + 1e-4

    def test_adjacency_directions_are_mirrored(self, y_network):
        net = _FakeNet(y_network)
        g = catchment_graph.build(net, min_area_km2=0.0)
        up, down = catchment_graph.adjacency(g)
        assert up.shape == down.shape == (len(g.edges), 2)
        if len(g.edges):
            # up is (receiver, contributor); down is (contributor, receiver)
            assert (up[:, 0] == down[:, 1]).all()
            assert (up[:, 1] == down[:, 0]).all()


class TestMessagePassing:
    def test_aggregate_means_over_neighbours(self):
        h = torch.tensor([[1.0], [3.0], [5.0]])
        # node 0 receives from 1 and 2
        edges = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        out = DirectedSAGELayer._aggregate(h, edges, 3)
        assert out[0, 0] == pytest.approx(4.0)
        assert out[1, 0] == pytest.approx(0.0)

    def test_aggregate_handles_no_edges(self):
        h = torch.randn(5, 3)
        out = DirectedSAGELayer._aggregate(h, torch.zeros((0, 2), dtype=torch.long), 5)
        assert out.shape == (5, 3)
        assert torch.all(out == 0)

    def test_forward_shape(self):
        model = FlowGNN(in_dim=6, hidden=8, layers=2)
        x = torch.randn(10, 6)
        up = torch.tensor([[1, 0], [2, 1]], dtype=torch.long)
        down = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        assert model(x, up, down).shape == (10,)

    def test_edges_change_the_output(self):
        """If removing edges changed nothing, message passing would be inert."""
        torch.manual_seed(0)
        model = FlowGNN(in_dim=4, hidden=8, layers=2)
        x = torch.randn(6, 4)
        up = torch.tensor([[1, 0], [2, 1], [3, 2]], dtype=torch.long)
        down = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
        empty = torch.zeros((0, 2), dtype=torch.long)
        assert not torch.allclose(model(x, up, down), model(x, empty, empty))

    def test_upstream_and_downstream_are_not_symmetric(self):
        """
        The whole justification for the architecture: swapping the direction
        of every edge must change the answer, or the model is treating the
        river as undirected.
        """
        torch.manual_seed(1)
        model = FlowGNN(in_dim=4, hidden=8, layers=1)
        x = torch.randn(6, 4)
        up = torch.tensor([[1, 0], [2, 1]], dtype=torch.long)
        down = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        assert not torch.allclose(model(x, up, down), model(x, down, up))


class TestStandardisation:
    def test_uses_training_statistics_only(self):
        X = np.array([[0.0], [1.0], [100.0]], dtype=np.float32)
        train = np.array([0, 1])
        out = _standardise(X, train)
        # mean 0.5, sd 0.5 over the training rows
        assert out[0, 0] == pytest.approx(-1.0, abs=1e-4)
        assert out[1, 0] == pytest.approx(1.0, abs=1e-4)
        assert out[2, 0] > 100

    def test_constant_column_does_not_divide_by_zero(self):
        X = np.full((4, 2), 3.0, dtype=np.float32)
        out = _standardise(X, np.arange(4))
        assert np.isfinite(out).all()

    def test_nan_becomes_zero(self):
        X = np.array([[np.nan], [1.0], [2.0]], dtype=np.float32)
        out = _standardise(X, np.array([1, 2]))
        assert np.isfinite(out).all()


class TestGraphDataset:
    def test_holds_consistent_shapes(self):
        d = GraphDataset(
            X=np.zeros((5, 3), dtype=np.float32),
            y=np.zeros(5, dtype=np.float32),
            up=np.zeros((2, 2), dtype=np.int64),
            down=np.zeros((2, 2), dtype=np.int64),
            row=np.zeros(5, dtype=np.float32),
            col=np.zeros(5, dtype=np.float32),
            weight=np.ones(5),
            features=["a", "b", "c"],
        )
        assert d.X.shape[0] == d.y.size == d.row.size == d.col.size
        assert d.X.shape[1] == len(d.features)
