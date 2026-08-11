"""
Generate the results figures from measured numbers.

Replaces `generate_all_charts.py`, which fabricated its inputs: its ROC curves
came from `synthetic_roc(auc)` -- a function that *invents* a curve to match a
target AUC -- and its confusion matrix was back-derived from a target precision
and recall. Its baseline rows had no corresponding run anywhere in the repo.

Every panel here is computed from a file on disk:

    models/benchmark.json               real baselines, real ROC curves
    models/susceptibility_metrics.json  spatial CV, reliability, importance
    models/graph_experiment.json        the graph ablation
    models/reference_rainfall.json      ERA5 storm accumulations
    outputs/flood_hazard_332mm.tif      for the precision-recall curve
    data_aligned/ground_truth_aligned.tif

Anything missing is skipped with a warning rather than faked.

Run:  python evaluation/generate_figures.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

LOGGER = logging.getLogger("figures")

# ──────────────────────────────────────────────
# Palette (validated: node scripts/validate_palette.js, light mode)
# Categorical slots are used in fixed order and never cycled.
# ──────────────────────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATUS_GOOD = "#0ca30c"
STATUS_CRIT = "#d03b3b"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",          # solid hairlines; dashed grids read as thresholds
    "legend.frameon": False,
    "legend.fontsize": 9,
})


def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        LOGGER.warning("missing %s -- skipping the figures that need it", path.name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _grid(ax, axis: str = "y") -> None:
    ax.grid(True, axis=axis, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, name: str) -> None:
    out = HERE / name
    fig.savefig(out)
    plt.close(fig)
    LOGGER.info("wrote %s", out.name)


# ──────────────────────────────────────────────
# 1. ROC curves, computed from predictions
# ──────────────────────────────────────────────
def fig_roc(bench: Dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    models = bench["models"]
    order = sorted(models, key=lambda k: -models[k]["spatial_auc_roc"])

    if len(order) > len(SERIES):
        LOGGER.warning(
            "%d models but %d categorical slots; the tail is drawn in muted grey "
            "rather than cycling hues", len(order), len(SERIES),
        )

    for i, name in enumerate(order):
        m = models[name]
        curve = m.get("roc_curve")
        if not curve:
            continue
        # Never cycle categorical hues: past the last slot, fall back to a
        # neutral so two different models can never share a colour.
        colour = SERIES[i] if i < len(SERIES) else MUTED
        ax.plot(
            curve["fpr"], curve["tpr"], linewidth=2, color=colour,
            label=f"{name} — AUC {m['spatial_auc_roc']:.3f}", zorder=3,
        )

    ax.plot([0, 1], [0, 1], linewidth=1, color=AXIS, zorder=1)
    ax.annotate(
        "no skill", xy=(0.62, 0.58), color=MUTED, fontsize=8, rotation=36,
        rotation_mode="anchor",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC under spatial-block cross-validation")
    # AUC in the legend is the direct label that satisfies the contrast relief
    # rule for the lower-contrast slots.
    ax.legend(loc="lower right")
    _grid(ax, "both")
    _save(fig, "fig1_roc_spatial_cv.png")


# ──────────────────────────────────────────────
# 2. The methodological headline: CV inflation
# ──────────────────────────────────────────────
def fig_cv_inflation(bench: Dict) -> None:
    models = bench["models"]
    order = sorted(models, key=lambda k: models[k]["spatial_auc_roc"])
    y = np.arange(len(order))
    spatial = np.array([models[k]["spatial_auc_roc"] for k in order])
    random = np.array([models[k]["random_auc_roc"] for k in order])

    fig, ax = plt.subplots(figsize=(7.2, 0.62 * len(order) + 2.0))

    # Dumbbell: the gap between the two dots *is* the inflation, which a pair
    # of bars starting at zero would bury.
    for i in y:
        ax.plot(
            [spatial[i], random[i]], [i, i], linewidth=2, color=GRID, zorder=2,
            solid_capstyle="round",
        )
    ax.scatter(spatial, y, s=90, color=SERIES[0], zorder=4, label="Spatial-block CV (honest)")
    ax.scatter(random, y, s=90, color=SERIES[1], zorder=4, label="Random k-fold (inflated)")

    for i in y:
        ax.annotate(
            f"+{random[i] - spatial[i]:.3f}",
            xy=(max(spatial[i], random[i]) + 0.004, i),
            va="center", fontsize=8, color=INK_2,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("AUC-ROC")
    ax.set_title(
        "Random k-fold overstates every model\n"
        "Neighbouring 10 m pixels are near-duplicates, so a random split leaks",
        loc="left", pad=28,
    )
    # Headroom for the inflation labels, which sit to the right of the rightmost
    # dot and would otherwise be clipped by the axis.
    ax.set_xlim(min(spatial.min(), random.min()) - 0.02, min(1.06, random.max() + 0.05))
    ax.set_ylim(-0.6, len(order) - 0.4)
    # Above the plot, not inside it: at "lower left" the legend landed on top of
    # the bottom row's dots and label.
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.12),
        mode="expand", ncol=2, borderaxespad=0,
    )
    _grid(ax, "x")
    _save(fig, "fig2_cv_inflation.png")


# ──────────────────────────────────────────────
# 3. Reliability of the calibrated probabilities
# ──────────────────────────────────────────────
def fig_reliability(metrics: Dict) -> None:
    rows = metrics.get("reliability")
    if not rows:
        LOGGER.warning("no reliability table -- skipping fig3")
        return

    predicted = [r["predicted"] for r in rows]
    observed = [r["observed"] for r in rows]
    counts = np.array([r["n"] for r in rows], dtype=float)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 6.2), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12},
    )

    ax.plot([0, 1], [0, 1], linewidth=1, color=AXIS, zorder=1)
    ax.plot(
        predicted, observed, linewidth=2, color=SERIES[0], marker="o",
        markersize=7, markerfacecolor=SERIES[0], markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=3, label="Held-out calibration split",
    )
    gap = metrics.get("worst_calibration_gap_balanced_scale")
    if gap is not None:
        ax.annotate(
            f"worst deviation {gap:.3f}", xy=(0.05, 0.90), color=INK_2, fontsize=9,
        )
    ax.annotate("perfect calibration", xy=(0.55, 0.49), color=MUTED, fontsize=8,
                rotation=36, rotation_mode="anchor")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Probabilities mean what they say", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    _grid(ax, "both")

    ax2.bar(predicted, counts, width=0.055, color=MUTED, zorder=3)
    ax2.set_ylabel("Samples")
    ax2.set_xlabel("Predicted probability (balanced scale)")
    _grid(ax2, "y")
    _save(fig, "fig3_reliability.png")


# ──────────────────────────────────────────────
# 4. Where the risk thresholds come from
# ──────────────────────────────────────────────
def fig_threshold_derivation() -> None:
    try:
        import rasterio
        from sklearn.metrics import precision_recall_curve

        from config import RISK
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("cannot build fig4 (%s)", exc)
        return

    hazard_path = ROOT / "outputs" / "flood_hazard_332mm.tif"
    gt_path = ROOT / "data_aligned" / "ground_truth_aligned.tif"
    lulc_path = ROOT / "data_aligned" / "lulc_aligned.tif"
    if not all(p.exists() for p in (hazard_path, gt_path, lulc_path)):
        LOGGER.warning("rasters missing -- skipping fig4")
        return

    def read(p):
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float32)
            nd = s.nodata
        a[a == np.float32(nd)] = np.nan
        return a

    hazard, gt, lulc = read(hazard_path), read(gt_path), read(lulc_path)
    m = np.isfinite(hazard) & np.isfinite(gt) & np.isfinite(lulc) & (np.round(lulc) != 1)

    rng = np.random.default_rng(0)
    idx = np.flatnonzero(m.ravel())
    if idx.size > 3_000_000:
        idx = rng.choice(idx, size=3_000_000, replace=False)
    y = (gt.ravel()[idx] > 0.5).astype(np.int8)
    p = hazard.ravel()[idx]

    precision, recall, thresholds = precision_recall_curve(y, p)
    precision, recall = precision[:-1], recall[:-1]

    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.plot(recall, precision, linewidth=2, color=SERIES[0], zorder=3,
            label="Hazard map at the reference event")
    ax.axhline(float(y.mean()), linewidth=1, color=AXIS, zorder=1)
    ax.annotate(f"no skill ({y.mean():.3f})", xy=(0.55, y.mean() + 0.012),
                color=MUTED, fontsize=8)

    # Only the band name sits beside its dot; the numbers go in one block in
    # the empty region under the curve. Four two-line labels on the curve
    # collided with each other and with the no-skill line however they were
    # offset -- the points are simply too close at the high-recall end.
    bands = [
        ("Critical", RISK.critical, (12, 8), "left"),
        ("Severe", RISK.high, (12, 8), "left"),
        ("High", RISK.moderate, (10, -16), "left"),
        ("Moderate", RISK.safe, (-12, 10), "right"),
    ]
    rows = []
    for label, t, offset, ha in bands:
        j = int(np.argmin(np.abs(thresholds - t)))
        ax.scatter([recall[j]], [precision[j]], s=95, color=SERIES[1], zorder=5,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(
            label, xy=(recall[j], precision[j]), xytext=offset,
            textcoords="offset points", fontsize=8.5, color=INK_2, ha=ha,
            fontweight="semibold",
        )
        rows.append(f"{label:<9s} p≥{t:.3f}   recall {recall[j]:.2f}   precision {precision[j]:.2f}")

    ax.annotate(
        "\n".join(rows),
        xy=(0.035, 0.30), xycoords="axes fraction", va="top", ha="left",
        fontsize=8, color=INK_2, family="DejaVu Sans Mono", linespacing=1.6,
    )

    ax.set_xlabel("Recall — share of the observed 2018 flood captured")
    ax.set_ylabel("Precision")
    ax.set_title("Risk band edges are read off the precision-recall curve", loc="left")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, min(1.0, precision.max() * 1.15))
    ax.legend(loc="upper right")
    _grid(ax, "both")
    _save(fig, "fig4_threshold_derivation.png")


# ──────────────────────────────────────────────
# 5. Conformal coverage: the marginal trap
# ──────────────────────────────────────────────
def fig_conformal(metrics: Dict) -> None:
    cf = metrics.get("conformal")
    if not cf:
        LOGGER.warning("no conformal block -- skipping fig5")
        return
    marginal = cf.get("marginal_variant")
    if not marginal:
        LOGGER.warning("no marginal variant -- skipping fig5")
        return

    target = cf["target_coverage"]
    groups = ["Overall\n(marginal)", "On truly\nflooded pixels", "On truly\ndry pixels"]
    marg = [
        marginal["marginal_coverage"],
        marginal["class_conditional_coverage"]["flood"],
        marginal["class_conditional_coverage"]["dry"],
    ]
    mond = [
        cf["marginal_coverage"],
        cf["class_conditional_coverage"]["flood"],
        cf["class_conditional_coverage"]["dry"],
    ]

    x = np.arange(len(groups))
    w = 0.30
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # Offset either side of centre leaves a surface gap between the bars.
    ax.bar(x - w / 2 - 0.012, marg, w, color=SERIES[1], zorder=3,
           label="Marginal split conformal")
    ax.bar(x + w / 2 + 0.012, mond, w, color=SERIES[0], zorder=3,
           label="Class-conditional (Mondrian)")

    ax.axhline(target, linewidth=1.4, color=STATUS_CRIT, zorder=4)
    # Above the line at the far left, the one region clear of both the bars and
    # their value labels.
    ax.annotate(f"{target:.0%} target", xy=(-0.52, target + 0.022),
                color=STATUS_CRIT, fontsize=8.5, ha="left")

    for xi, (a, b) in enumerate(zip(marg, mond)):
        ax.annotate(f"{a:.3f}", xy=(xi - w / 2 - 0.012, a + 0.022), ha="center",
                    fontsize=8.5, color=INK_2)
        ax.annotate(f"{b:.3f}", xy=(xi + w / 2 + 0.012, b + 0.022), ha="center",
                    fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Coverage")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_title(
        "A marginal guarantee can be met while the flood class is uncovered\n"
        "98.6% of the district is dry, so the average hides the failure",
        loc="left", pad=28,
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand", ncol=2, borderaxespad=0,
    )
    _grid(ax, "y")
    _save(fig, "fig5_conformal_coverage.png")


# ──────────────────────────────────────────────
# 6. The graph ablation (a negative result)
# ──────────────────────────────────────────────
def fig_graph_ablation(graph: Dict) -> None:
    keys = [("tabular_gbt", "Boosted trees\n(no graph)"),
            ("mlp_no_edges", "Same network\nedges OFF"),
            ("graph_gnn", "Same network\nedges ON")]
    if not all(k in graph for k, _ in keys):
        LOGGER.warning("graph_experiment.json lacks the ablation -- skipping fig6")
        return

    x = np.arange(len(keys))
    w = 0.30
    roc = [graph[k]["auc_roc"] for k, _ in keys]
    pr = [graph[k]["auc_pr"] for k, _ in keys]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.bar(x - w / 2 - 0.012, roc, w, color=SERIES[0], zorder=3, label="AUC-ROC")
    ax.bar(x + w / 2 + 0.012, pr, w, color=SERIES[1], zorder=3, label="AUC-PR")

    for xi, (a, b) in enumerate(zip(roc, pr)):
        ax.annotate(f"{a:.3f}", xy=(xi - w / 2 - 0.012, a + 0.014), ha="center",
                    fontsize=8.5, color=INK_2)
        ax.annotate(f"{b:.3f}", xy=(xi + w / 2 + 0.012, b + 0.014), ha="center",
                    fontsize=8.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in keys])
    ax.set_ylabel("Out-of-fold score")
    ax.set_ylim(0, 1.06)
    ax.set_xlim(-0.55, len(keys) - 0.45)
    n_nodes = graph.get("n_nodes", "?")
    n_edges = graph.get("n_edges", "?")
    ax.set_title(
        f"Drainage connectivity did not help ({n_nodes} sub-catchments, {n_edges} edges)\n"
        "The edges-off control rules out model family as the explanation",
        loc="left", pad=28,
    )
    ax.legend(
        loc="lower left", bbox_to_anchor=(0, 1.0, 1, 0.1),
        mode="expand", ncol=2, borderaxespad=0,
    )
    _grid(ax, "y")
    _save(fig, "fig6_graph_ablation.png")


# ──────────────────────────────────────────────
# 7. Permutation importance
# ──────────────────────────────────────────────
def fig_importance(metrics: Dict) -> None:
    imp = metrics.get("permutation_importance")
    if not imp:
        LOGGER.warning("no permutation importance -- skipping fig7")
        return

    context = {"upstream_cn", "dem_rel_1km"}
    order = sorted(imp, key=lambda k: imp[k])
    values = [imp[k] for k in order]
    # One series, one colour; emphasis marks the two context features rather
    # than ramping colour by magnitude (which the bar length already shows).
    colors = [SERIES[1] if k in context else SERIES[0] for k in order]

    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(order) + 1.9))
    ax.barh(np.arange(len(order)), values, height=0.62, color=colors, zorder=3)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("AUC drop when the feature is shuffled")
    ax.set_title(
        "Drainage-network context earns its place\n"
        "Orange: features describing surroundings rather than the pixel itself",
        loc="left",
    )
    for i, v in enumerate(values):
        ax.annotate(f"{v:+.4f}", xy=(v + 0.0015, i), va="center", fontsize=8,
                    color=INK_2)
    ax.set_xlim(min(0, min(values) * 1.2), max(values) * 1.22)
    _grid(ax, "x")
    _save(fig, "fig7_feature_importance.png")


# ──────────────────────────────────────────────
# 8. Where the reference depth came from
# ──────────────────────────────────────────────
def fig_reference_rainfall(rain: Dict) -> None:
    windows = [1, 2, 3, 5, 7]
    events = [e for e in ("2018", "2019", "2021") if e in rain]
    if not events:
        LOGGER.warning("no rainfall events -- skipping fig8")
        return

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for i, event in enumerate(events):
        depths = [rain[event][f"max_{w}day_mm"] for w in windows]
        ax.plot(windows, depths, linewidth=2, color=SERIES[i], marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                zorder=3, label=f"{event} event")
        ax.annotate(f"{depths[-1]:.0f} mm", xy=(windows[-1] + 0.12, depths[-1]),
                    fontsize=8.5, color=INK_2, va="center")

    ref = rain[events[0]]["reference_event_mm"]
    ax.scatter([3], [ref], s=150, facecolor="none", edgecolor=STATUS_CRIT,
               linewidth=2, zorder=5)
    # In the gap between the 2018 and 2019 curves, just right of the circled
    # point: directly below it the label ran across the 2018 line.
    ax.annotate(
        f"reference depth {ref:.0f} mm\n3-day storm, pairs with AMC III",
        xy=(0.38, 0.56), xycoords="axes fraction", va="top", ha="left",
        fontsize=8.5, color=STATUS_CRIT,
    )
    ax.set_xticks(windows)
    ax.set_xlabel("Accumulation window (days)")
    ax.set_ylabel("District-mean depth (mm)")
    ax.set_title(
        "The reference event is derived, not assumed\n"
        "ERA5 reanalysis, 3x3 grid over the mapped district",
        loc="left",
    )
    ax.set_xlim(0.7, 7.9)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    _grid(ax, "both")
    _save(fig, "fig8_reference_rainfall.png")


# ──────────────────────────────────────────────
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    bench = _load(ROOT / "models" / "benchmark.json")
    metrics = _load(ROOT / "models" / "susceptibility_metrics.json")
    graph = _load(ROOT / "models" / "graph_experiment.json")
    rain = _load(ROOT / "models" / "reference_rainfall.json")

    if bench:
        fig_roc(bench)
        fig_cv_inflation(bench)
    if metrics:
        fig_reliability(metrics)
        fig_conformal(metrics)
        fig_importance(metrics)
    if graph:
        fig_graph_ablation(graph)
    if rain:
        fig_reference_rainfall(rain)
    fig_threshold_derivation()

    LOGGER.info("done")


if __name__ == "__main__":
    main()
