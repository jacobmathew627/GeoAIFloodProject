"""
Does the drainage graph actually carry information the pixel model cannot use?

Builds the sub-catchment graph, aggregates every conditioning factor and the
flood inventory onto it, then compares a directed message-passing model
against a gradient-boosted model on identical node features under identical
spatial-block folds. The only difference between the two is whether flow
connectivity is available as structure.

Run:  python src/run_graph_experiment.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from config import ALIGNED_DIR, MODELS_DIR, SUSCEPTIBILITY_FEATURES, setup_logging

LOGGER = logging.getLogger("geoai_flood")


def _to_routing_grid(values, valid, master_profile, net) -> np.ndarray:
    """Resample a master-grid raster onto the coarser routing grid."""
    dst = np.full(net.elev.shape, np.nan, dtype=np.float32)
    reproject(
        source=np.where(valid, values, np.nan).astype(np.float32),
        destination=dst,
        src_transform=master_profile["transform"],
        src_crs=master_profile["crs"],
        dst_transform=net.profile["transform"],
        dst_crs=net.profile["crs"],
        resampling=Resampling.average,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def build_dataset(aligned_dir: Optional[Path] = None, min_area_km2: float = 0.5):
    """Assemble the node-level dataset for the graph experiment."""
    import catchment_graph
    from feature_stack import (
        domain_mask,
        flood_labels,
        grid_profile,
        read_feature,
    )
    from graph_model import GraphDataset
    from routing import FlowNetwork

    aligned_dir = aligned_dir or ALIGNED_DIR
    master = grid_profile(aligned_dir)

    LOGGER.info("Building flow network...")
    net = FlowNetwork()

    LOGGER.info("Delineating sub-catchments (min area %.2f km2)...", min_area_km2)
    graph = catchment_graph.build(net, min_area_km2=min_area_km2)

    # Restrict to the model domain: permanent water is excluded there and must
    # be excluded here too, or the graph learns the backwaters again.
    LOGGER.info("Projecting the model domain onto the routing grid...")
    domain = domain_mask(aligned_dir=aligned_dir)
    domain_route = _to_routing_grid(
        domain.astype(np.float32), np.ones_like(domain, dtype=bool), master, net
    )
    domain_weight = np.where(np.isfinite(domain_route), domain_route, 0.0)

    LOGGER.info("Aggregating %d features onto nodes...", len(SUSCEPTIBILITY_FEATURES))
    columns = []
    for name in SUSCEPTIBILITY_FEATURES:
        values, valid = read_feature(name, aligned_dir=aligned_dir)
        route = _to_routing_grid(values, valid, master, net)
        columns.append(catchment_graph.aggregate_to_nodes(route, graph, domain_weight))
        del values, valid, route

    X = np.stack(columns, axis=1).astype(np.float32)

    LOGGER.info("Aggregating the flood inventory onto nodes...")
    flood, gt_valid = flood_labels(aligned_dir=aligned_dir)
    flood_route = _to_routing_grid(
        (flood & gt_valid).astype(np.float32), np.ones_like(flood, dtype=bool), master, net
    )
    y = catchment_graph.aggregate_to_nodes(flood_route, graph, domain_weight)

    # Keep nodes that have both features and a meaningful amount of domain.
    node_domain = catchment_graph.aggregate_to_nodes(domain_weight, graph)
    keep = (
        np.isfinite(X).all(axis=1)
        & np.isfinite(y)
        & np.isfinite(graph.node_row)
        & (node_domain > 0.25)
        & (graph.node_cells >= 5)
    )
    LOGGER.info("  keeping %d of %d nodes", int(keep.sum()), graph.n_nodes)

    remap = np.full(graph.n_nodes, -1, dtype=np.int64)
    remap[np.flatnonzero(keep)] = np.arange(int(keep.sum()))

    up, down = catchment_graph.adjacency(graph)

    def _remap_edges(e):
        a, b = remap[e[:, 0]], remap[e[:, 1]]
        ok = (a >= 0) & (b >= 0)
        return np.stack([a[ok], b[ok]], axis=1).astype(np.int64)

    data = GraphDataset(
        X=X[keep],
        y=np.clip(y[keep], 0.0, 1.0),
        up=_remap_edges(up),
        down=_remap_edges(down),
        row=graph.node_row[keep],
        col=graph.node_col[keep],
        weight=graph.node_cells[keep].astype(np.float64),
        features=list(SUSCEPTIBILITY_FEATURES),
    )
    LOGGER.info(
        "  dataset: %d nodes, %d upstream edges, %d downstream edges",
        data.X.shape[0], len(data.up), len(data.down),
    )
    LOGGER.info(
        "  node flooded-fraction: mean %.4f, p95 %.4f, max %.4f",
        data.y.mean(), np.percentile(data.y, 95), data.y.max(),
    )
    return data, graph, net


def main() -> None:  # pragma: no cover
    import argparse

    import graph_model

    parser = argparse.ArgumentParser(description="Graph vs tabular experiment")
    parser.add_argument("--min-area-km2", type=float, default=0.5)
    parser.add_argument("--positive-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    LOGGER.info("=" * 66)
    LOGGER.info("Does drainage connectivity carry information a pixel model cannot?")
    LOGGER.info("=" * 66)

    data, graph, _ = build_dataset(min_area_km2=args.min_area_km2)
    results, oof_gnn, oof_tab = graph_model.compare(
        data, positive_fraction=args.positive_fraction, seed=args.seed
    )
    results["min_area_km2"] = args.min_area_km2
    results["n_edges"] = int(len(data.up))

    out = MODELS_DIR / "graph_experiment.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":  # pragma: no cover
    main()
