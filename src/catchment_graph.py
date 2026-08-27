"""
Sub-catchment graph construction.

Turns the D8 flow network into a directed graph whose nodes are real drainage
units and whose edges are the flow paths between them. This is the structure
a message-passing model needs, and it is not derivable from a raster: two
cells 50 m apart can sit in different sub-catchments and never exchange a
drop of water, while two cells 8 km apart on the same channel are directly
coupled.

Delineation follows standard practice:

  1. Cells whose upslope contributing area exceeds `min_area_km2` are
     "channel" cells.
  2. A channel cell with two or more channel donors is a *junction*. Junctions
     and terminal cells (pits, outlets) become sub-catchment outlets.
  3. Every cell is assigned to the outlet it eventually drains through, by
     walking the network from downstream to upstream.

The result is one node per inter-junction reach and its local hillslopes,
which is the same object the Himachal Pradesh study used (460 sub-watersheds,
1,700 directed edges) -- see arXiv:2603.15681.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOGGER = logging.getLogger("geoai_flood")


@dataclass
class CatchmentGraph:
    """A directed sub-catchment graph over the routing grid."""

    labels: np.ndarray  # (H, W) int32, -1 outside the network
    n_nodes: int
    edges: np.ndarray  # (E, 2) int32, edge u -> v means u drains into v
    node_cells: np.ndarray  # (n_nodes,) int64, cell count per node
    node_row: np.ndarray  # (n_nodes,) float32, mean row of each node
    node_col: np.ndarray  # (n_nodes,) float32, mean col of each node

    def summary(self) -> str:
        return (
            f"{self.n_nodes} sub-catchments, {len(self.edges)} directed edges, "
            f"median size {np.median(self.node_cells):.0f} cells "
            f"(min {self.node_cells.min()}, max {self.node_cells.max()})"
        )


def build(
    net,
    min_area_km2: float = 0.5,
    aligned_dir: Optional[str] = None,
) -> CatchmentGraph:
    """
    Delineate sub-catchments over a `routing.FlowNetwork`.

    Args:
        net: A built FlowNetwork.
        min_area_km2: Contributing area above which a cell counts as channel.
            Smaller values give more, finer sub-catchments.
    """
    valid = net.valid
    receiver = net.receiver
    order = net.order  # descending elevation, i.e. upstream before downstream
    h, w = valid.shape
    n_cells = h * w

    area = net.contributing_area_m2()
    min_area_m2 = min_area_km2 * 1e6

    valid_flat = valid.ravel()
    area_flat = np.where(np.isfinite(area), area, 0.0).ravel()
    is_channel = valid_flat & (area_flat >= min_area_m2)

    # Count channel donors per cell: how many channel cells drain directly in.
    donors = np.zeros(n_cells, dtype=np.int32)
    src = np.flatnonzero(is_channel)
    dst = receiver[src]
    moved = dst != src
    np.add.at(donors, dst[moved], 1)

    terminal = valid_flat & (receiver == np.arange(n_cells))
    junction = is_channel & (donors >= 2)
    outlets = np.flatnonzero(terminal | junction)

    LOGGER.info(
        "  %d channel cells (>= %.2f km2), %d junctions, %d terminals -> %d outlets",
        int(is_channel.sum()),
        min_area_km2,
        int(junction.sum()),
        int(terminal.sum()),
        outlets.size,
    )
    if outlets.size == 0:
        raise ValueError("No sub-catchment outlets found; lower min_area_km2")

    # ── Assign every cell to the outlet it drains through ──
    #
    # Walk downstream-to-upstream: a cell that is not itself an outlet inherits
    # its receiver's label. Reversing the topological order guarantees the
    # receiver is already labelled.
    label = np.full(n_cells, -1, dtype=np.int32)
    label[outlets] = np.arange(outlets.size, dtype=np.int32)

    for i in order[::-1]:
        if not valid_flat[i] or label[i] >= 0:
            continue
        r = receiver[i]
        if r != i:
            label[i] = label[r]

    labelled = label >= 0
    LOGGER.info(
        "  labelled %.2fM of %.2fM valid cells",
        labelled.sum() / 1e6,
        valid_flat.sum() / 1e6,
    )

    # ── Edges between adjacent sub-catchments ──
    #
    # An edge exists where a cell in node u drains into a cell in node v.
    # Direction is inherited from the flow network, so the graph is a DAG.
    cells = np.flatnonzero(labelled)
    src_label = label[cells]
    dst_label = label[receiver[cells]]
    keep = (dst_label >= 0) & (src_label != dst_label)
    edges = np.unique(np.stack([src_label[keep], dst_label[keep]], axis=1), axis=0).astype(np.int32)

    counts = np.bincount(label[labelled], minlength=outlets.size).astype(np.int64)
    rows, cols = np.divmod(cells, w)
    node_row = np.bincount(src_label, weights=rows, minlength=outlets.size)
    node_col = np.bincount(src_label, weights=cols, minlength=outlets.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        node_row = np.where(counts > 0, node_row / np.maximum(counts, 1), np.nan)
        node_col = np.where(counts > 0, node_col / np.maximum(counts, 1), np.nan)

    graph = CatchmentGraph(
        labels=label.reshape(h, w),
        n_nodes=int(outlets.size),
        edges=edges,
        node_cells=counts,
        node_row=node_row.astype(np.float32),
        node_col=node_col.astype(np.float32),
    )
    LOGGER.info("  %s", graph.summary())
    return graph


def aggregate_to_nodes(
    values: np.ndarray,
    graph: CatchmentGraph,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Area-weighted mean of a routing-grid raster within each sub-catchment."""
    lab = graph.labels.ravel()
    v = values.ravel()
    ok = (lab >= 0) & np.isfinite(v)

    w = np.ones_like(v, dtype=np.float64) if weights is None else weights.ravel()
    w = np.where(np.isfinite(w), w, 0.0)

    num = np.bincount(lab[ok], weights=(v[ok] * w[ok]), minlength=graph.n_nodes)
    den = np.bincount(lab[ok], weights=w[ok], minlength=graph.n_nodes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan).astype(np.float32)


def adjacency(graph: CatchmentGraph) -> Tuple[np.ndarray, np.ndarray]:
    """
    Row-normalised upstream and downstream neighbour indices.

    Returns (upstream_edges, downstream_edges) as (E, 2) arrays where column 0
    is the receiving node and column 1 the contributing neighbour. Keeping the
    two directions separate is the whole point: a sub-catchment is affected by
    what is above it in a completely different way from what is below it, and
    an undirected graph throws that away.
    """
    e = graph.edges
    # u -> v : v's upstream neighbour is u; u's downstream neighbour is v.
    upstream = np.stack([e[:, 1], e[:, 0]], axis=1)
    downstream = np.stack([e[:, 0], e[:, 1]], axis=1)
    return upstream.astype(np.int64), downstream.astype(np.int64)
