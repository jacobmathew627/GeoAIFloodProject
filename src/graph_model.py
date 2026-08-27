"""
Directed message passing over the sub-catchment graph.

This is an architectural change, not another feature. Every model in this
project so far scores locations independently: the gradient-boosted
susceptibility model looks at one pixel's feature vector at a time, and the
`upstream_cn` feature added earlier is a hand-crafted summary of the network
rather than something the model can reason over. Here the network itself is
the computation -- each sub-catchment's representation is built from its own
attributes *and* from its neighbours', iterated, so information propagates
along real flow paths.

The architecture is a directed GraphSAGE variant:

    h_v^{k+1} = ReLU( W_self h_v^k
                    + W_up   mean_{u in upstream(v)}   h_u^k
                    + W_down mean_{w in downstream(v)} h_w^k )

Upstream and downstream neighbours get separate weight matrices. That
separation is the point. A sub-catchment is affected by what lies above it
(runoff arriving) in a completely different way from what lies below it
(whether there is anywhere for the water to go). Collapsing them into one
undirected aggregation discards the asymmetry that makes a river a river.

Whether this actually helps is an empirical question, and `compare()` answers
it honestly: the same nodes, the same spatial-block folds, and a
gradient-boosted model on identical node features as the baseline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

LOGGER = logging.getLogger("geoai_flood")


# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
class DirectedSAGELayer(nn.Module):
    """One round of directed neighbourhood aggregation."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.up_lin = nn.Linear(in_dim, out_dim, bias=False)
        self.down_lin = nn.Linear(in_dim, out_dim, bias=False)

    @staticmethod
    def _aggregate(h: torch.Tensor, edges: torch.Tensor, n_nodes: int) -> torch.Tensor:
        """Mean of h over neighbours, scattered onto the receiving node."""
        out = torch.zeros(n_nodes, h.shape[1], dtype=h.dtype, device=h.device)
        if edges.numel() == 0:
            return out
        target, source = edges[:, 0], edges[:, 1]
        out.index_add_(0, target, h[source])
        count = torch.zeros(n_nodes, dtype=h.dtype, device=h.device)
        count.index_add_(0, target, torch.ones_like(target, dtype=h.dtype))
        return out / count.clamp(min=1.0).unsqueeze(1)

    def forward(self, h: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        n = h.shape[0]
        return (
            self.self_lin(h)
            + self.up_lin(self._aggregate(h, up, n))
            + self.down_lin(self._aggregate(h, down, n))
        )


class FlowGNN(nn.Module):
    """Directed GraphSAGE over the sub-catchment graph."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        dims = [in_dim] + [hidden] * layers
        self.layers = nn.ModuleList(DirectedSAGELayer(dims[i], dims[i + 1]) for i in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = self.dropout(torch.relu(norm(layer(h, up, down))))
        return self.head(h).squeeze(-1)


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
@dataclass
class GraphDataset:
    """Node features, soft labels and connectivity for the whole district."""

    X: np.ndarray  # (n_nodes, n_features)
    y: np.ndarray  # (n_nodes,) flooded area fraction in [0, 1]
    up: np.ndarray  # (E, 2)
    down: np.ndarray  # (E, 2)
    row: np.ndarray  # (n_nodes,) mean row, for spatial blocking
    col: np.ndarray  # (n_nodes,) mean col
    weight: np.ndarray  # (n_nodes,) node area in cells
    features: List[str]


def _standardise(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X[train], axis=0)
    sd = np.nanstd(X[train], axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    out = (X - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def train_gnn(
    data: GraphDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    hidden: int = 64,
    layers: int = 2,
    epochs: int = 400,
    lr: float = 5e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    device: str = "cpu",
) -> Tuple[FlowGNN, np.ndarray]:
    """
    Fit the GNN and return (model, predictions for every node).

    Message passing runs over the *whole* graph every step -- that is
    unavoidable and correct, since a node's representation depends on its
    neighbours regardless of which fold they belong to. Only the loss is
    restricted to training nodes, which is standard transductive practice. It
    does mean the spatial-block guarantee is weaker here than for the tabular
    model: neighbouring blocks still exchange information through the graph.
    That is stated rather than papered over.
    """
    torch.manual_seed(seed)

    X = _standardise(data.X, train_idx)
    x = torch.from_numpy(X).to(device)
    y = torch.from_numpy(data.y.astype(np.float32)).to(device)
    up = torch.from_numpy(data.up).to(device)
    down = torch.from_numpy(data.down).to(device)
    w = torch.from_numpy(np.log1p(data.weight).astype(np.float32)).to(device)

    tr = torch.from_numpy(train_idx.astype(np.int64)).to(device)
    va = torch.from_numpy(val_idx.astype(np.int64)).to(device)

    model = FlowGNN(x.shape[1], hidden=hidden, layers=layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, up, down)
        # Weight by log area: a 10,000-cell sub-catchment should not count the
        # same as a 30-cell sliver, but linear area weighting lets a handful of
        # huge nodes dominate.
        loss = (loss_fn(logits[tr], y[tr]) * w[tr]).sum() / w[tr].sum()
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x, up, down)
            val_loss = float((loss_fn(val_logits[va], y[va]) * w[va]).sum() / w[va].sum())

        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 60:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(x, up, down)).cpu().numpy()
    return model, preds


# ──────────────────────────────────────────────
# Honest comparison
# ──────────────────────────────────────────────
def compare(
    data: GraphDataset,
    positive_fraction: float = 0.10,
    n_splits: int = 5,
    block_px: int = 160,
    seed: int = 0,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Spatial-block comparison of the GNN against a tabular baseline.

    Both models see exactly the same node features and the same folds. The
    only difference is whether the flow network is available as structure.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    y_bin = (data.y >= positive_fraction).astype(int)
    if len(np.unique(y_bin)) < 2:
        raise ValueError("Binarised node labels are single-class; adjust positive_fraction")

    blocks = (data.row // block_px).astype(np.int64) * 10_000 + (data.col // block_px).astype(
        np.int64
    )
    LOGGER.info(
        "Node-level comparison: %d nodes, %d positive (%.1f%%), %d spatial blocks",
        data.X.shape[0],
        int(y_bin.sum()),
        100 * y_bin.mean(),
        len(np.unique(blocks)),
    )

    oof_gnn = np.full(y_bin.shape, np.nan)
    oof_tab = np.full(y_bin.shape, np.nan)
    # Control: the identical network with every edge removed. Comparing the
    # GNN against HistGradientBoosting alone confounds two things -- the graph
    # structure and the model family (neural net vs boosted trees). This
    # isolates the graph's contribution, which is the actual question.
    oof_mlp = np.full(y_bin.shape, np.nan)
    no_edges = np.zeros((0, 2), dtype=np.int64)

    splitter = GroupKFold(n_splits=n_splits)
    for k, (tr, te) in enumerate(splitter.split(data.X, y_bin, groups=blocks)):
        if len(np.unique(y_bin[tr])) < 2 or len(np.unique(y_bin[te])) < 2:
            LOGGER.warning("  fold %d single-class; skipped", k)
            continue

        # Carve a validation slice out of training for early stopping, by
        # block, so it is not adjacent to the training nodes.
        tr_blocks = np.unique(blocks[tr])
        rng = np.random.default_rng(seed + k)
        val_blocks = rng.choice(tr_blocks, size=max(1, len(tr_blocks) // 5), replace=False)
        is_val = np.isin(blocks, val_blocks)
        fit_idx = tr[~is_val[tr]]
        val_idx = tr[is_val[tr]]
        if val_idx.size == 0 or fit_idx.size == 0:
            fit_idx, val_idx = tr, tr

        _, preds = train_gnn(data, fit_idx, val_idx, seed=seed + k)
        oof_gnn[te] = preds[te]

        edgeless = GraphDataset(
            X=data.X,
            y=data.y,
            up=no_edges,
            down=no_edges,
            row=data.row,
            col=data.col,
            weight=data.weight,
            features=data.features,
        )
        _, mlp_preds = train_gnn(edgeless, fit_idx, val_idx, seed=seed + k)
        oof_mlp[te] = mlp_preds[te]

        tab = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            min_samples_leaf=10,
            l2_regularization=1.0,
            early_stopping=True,
            random_state=seed + k,
        )
        tab.fit(data.X[fit_idx], y_bin[fit_idx])
        oof_tab[te] = tab.predict_proba(data.X[te])[:, 1]

    scored = np.isfinite(oof_gnn) & np.isfinite(oof_tab) & np.isfinite(oof_mlp)

    def _score(pred):
        return {
            "auc_roc": float(roc_auc_score(y_bin[scored], pred[scored])),
            "auc_pr": float(average_precision_score(y_bin[scored], pred[scored])),
        }

    results = {
        "n_nodes": int(data.X.shape[0]),
        "n_scored": int(scored.sum()),
        "n_edges": int(len(data.up)),
        "positive_fraction_threshold": positive_fraction,
        "node_base_rate": float(y_bin.mean()),
        "graph_gnn": _score(oof_gnn),
        "mlp_no_edges": _score(oof_mlp),
        "tabular_gbt": _score(oof_tab),
    }
    # The graph's own contribution: same architecture, edges on vs off.
    results["graph_contribution_auc"] = (
        results["graph_gnn"]["auc_roc"] - results["mlp_no_edges"]["auc_roc"]
    )
    results["graph_contribution_auc_pr"] = (
        results["graph_gnn"]["auc_pr"] - results["mlp_no_edges"]["auc_pr"]
    )
    results["gnn_vs_best_tabular_auc"] = (
        results["graph_gnn"]["auc_roc"] - results["tabular_gbt"]["auc_roc"]
    )

    for name, key in [
        ("boosted trees (no graph)", "tabular_gbt"),
        ("same net, edges OFF     ", "mlp_no_edges"),
        ("same net, edges ON      ", "graph_gnn"),
    ]:
        LOGGER.info(
            "  %s AUC-ROC=%.4f AUC-PR=%.4f",
            name,
            results[key]["auc_roc"],
            results[key]["auc_pr"],
        )
    LOGGER.info(
        "  graph contribution (edges ON - OFF): %+.4f AUC-ROC, %+.4f AUC-PR",
        results["graph_contribution_auc"],
        results["graph_contribution_auc_pr"],
    )
    LOGGER.info(
        "  GNN vs best tabular model:          %+.4f AUC-ROC",
        results["gnn_vs_best_tabular_auc"],
    )
    return results, oof_gnn, oof_tab
