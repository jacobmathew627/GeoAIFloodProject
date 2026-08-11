"""
Benchmark real baseline models under identical spatial-block cross-validation.

The figures in `evaluation/` were generated from hardcoded numbers: the ROC
curves came from `synthetic_roc(auc)`, which *invents* a curve to match a
target AUC, and the confusion matrix was back-derived from a target precision
and recall. The baseline rows (logistic regression, SVM, random forest, CNN,
U-Net) had no corresponding run anywhere in the repository.

This module produces the same comparison honestly: every model is fitted on
the same samples, evaluated out-of-fold on the same spatial blocks, and its
ROC and PR curves are computed from its own predictions.

Run:  python src/benchmark_models.py
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from config import MODELS_DIR, setup_logging
from feature_stack import sample_training_points
from susceptibility import spatial_blocks

LOGGER = logging.getLogger("geoai_flood")


def build_models(seed: int = 0) -> Dict[str, object]:
    """The baseline family, all on identical features."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    return {
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        # LinearSVC rather than an RBF kernel: kernel SVM is O(n^2) and will
        # not finish on 100k+ samples. Reported as what it is.
        "Linear SVM": make_pipeline(
            StandardScaler(), LinearSVC(C=1.0, dual="auto", random_state=seed)
        ),
        "MLP (2x64)": make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True,
                random_state=seed,
            ),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed
        ),
        "Gradient Boosting (ours)": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=25, random_state=seed,
        ),
    }


def _scores(model, X: np.ndarray) -> np.ndarray:
    """Positive-class score, whether or not the model has predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raw = model.decision_function(X)
    # Map to (0, 1) monotonically; ranking metrics are unaffected.
    return 1.0 / (1.0 + np.exp(-raw))


def run(n_per_class: int = 60_000, seed: int = 42, n_splits: int = 5) -> Dict:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        f1_score,
        jaccard_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    sample = sample_training_points(n_per_class=n_per_class, seed=seed)
    X, y = sample["X"], sample["y"]
    groups = spatial_blocks(sample["row"], sample["col"])
    LOGGER.info(
        "Benchmark set: %d samples, %d features, %.1f%% positive, %d blocks",
        X.shape[0], X.shape[1], 100 * y.mean(), len(np.unique(groups)),
    )

    results = {}
    for name, factory in build_models(seed).items():
        LOGGER.info("Fitting %s...", name)
        oof_spatial = np.full(y.shape, np.nan)
        oof_random = np.full(y.shape, np.nan)

        for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=groups):
            m = build_models(seed)[name]
            m.fit(X[tr], y[tr])
            oof_spatial[te] = _scores(m, X[te])

        for tr, te in StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ).split(X, y):
            m = build_models(seed)[name]
            m.fit(X[tr], y[tr])
            oof_random[te] = _scores(m, X[te])

        ok = np.isfinite(oof_spatial)
        pred = (oof_spatial[ok] >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y[ok], oof_spatial[ok])
        # Thin the curve for storage; 200 points is plenty for a figure.
        idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)

        results[name] = {
            "spatial_auc_roc": float(roc_auc_score(y[ok], oof_spatial[ok])),
            "spatial_auc_pr": float(average_precision_score(y[ok], oof_spatial[ok])),
            "random_auc_roc": float(roc_auc_score(y, oof_random)),
            "precision": float(precision_score(y[ok], pred, zero_division=0)),
            "recall": float(recall_score(y[ok], pred, zero_division=0)),
            "f1": float(f1_score(y[ok], pred, zero_division=0)),
            "iou": float(jaccard_score(y[ok], pred, zero_division=0)),
            "brier": float(brier_score_loss(y[ok], np.clip(oof_spatial[ok], 0, 1))),
            "roc_curve": {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()},
            "confusion": {
                "tn": int(((pred == 0) & (y[ok] == 0)).sum()),
                "fp": int(((pred == 1) & (y[ok] == 0)).sum()),
                "fn": int(((pred == 0) & (y[ok] == 1)).sum()),
                "tp": int(((pred == 1) & (y[ok] == 1)).sum()),
            },
        }
        r = results[name]
        LOGGER.info(
            "  %-24s spatial AUC=%.4f  random AUC=%.4f  (inflation %+.4f)  F1=%.3f",
            name, r["spatial_auc_roc"], r["random_auc_roc"],
            r["random_auc_roc"] - r["spatial_auc_roc"], r["f1"],
        )

    payload = {
        "note": (
            "All models fitted on identical samples and evaluated out-of-fold. "
            "Spatial-block CV is the honest number; random k-fold is reported "
            "alongside it to show the inflation from spatial autocorrelation. "
            "ROC curves are computed from predictions, not synthesised."
        ),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": list(sample["features"]),
        "positive_rate": float(y.mean()),
        "n_spatial_blocks": int(len(np.unique(groups))),
        "n_splits": n_splits,
        "models": results,
    }

    out = MODELS_DIR / "benchmark.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return payload


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Benchmark baselines honestly")
    parser.add_argument("--samples", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    setup_logging(logging.INFO)
    run(n_per_class=args.samples, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover
    main()
