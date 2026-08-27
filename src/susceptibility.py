"""
Flood susceptibility model.

Learns P(flood | terrain, land cover) from the August 2018 Sentinel-1 flood
inventory. The output is a *rainfall-independent* susceptibility surface; the
rainfall response lives in hydrology.py and the two are combined in hazard.py.

Three things here differ from the previous pipeline and matter:

1. Permanent water is excluded from the label set. 80.3% of the raw SAR
   "flood" inventory is the Vembanad backwater system, so a model trained on
   it is a lake detector.

2. Performance is reported under *spatial block* cross-validation, not random
   k-fold. Neighbouring pixels of a 10 m raster are near-duplicates, so a
   random split leaks the test set into training and inflates AUC. Both
   numbers are reported so the gap is visible.

3. Probabilities are calibrated (isotonic, fitted out-of-fold) so that a
   pixel reported at 0.30 actually floods about 30% of the time in the
   inventory. The old maps were raw scores multiplied by a rainfall constant
   and then clipped, which is not a probability of anything.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (
    MODELS_DIR,
    OUTPUT_DIR,
    SUSCEPTIBILITY_MODEL,
    setup_logging,
)
from feature_stack import (
    build_matrix,
    grid_profile,
    iter_stripes,
    sample_domain_points,
    sample_training_points,
)

LOGGER = logging.getLogger("geoai_flood")

# Side of a cross-validation block, in pixels. At 10 m this is 5 km, which is
# comfortably beyond the correlation range of the terrain predictors.
BLOCK_SIZE_PX = 500


@dataclass
class CVMetrics:
    """Held-out performance for one cross-validation scheme."""

    scheme: str
    auc_roc: float
    auc_pr: float
    brier: float
    n_folds: int

    def __str__(self) -> str:
        return (
            f"{self.scheme:<18s} AUC-ROC={self.auc_roc:.4f}  "
            f"AUC-PR={self.auc_pr:.4f}  Brier={self.brier:.4f}  "
            f"({self.n_folds} folds)"
        )


# ──────────────────────────────────────────────
# Model factory
# ──────────────────────────────────────────────
def make_estimator(random_state: int = 0):
    """Gradient-boosted trees: handles the strongly non-linear HAND/TWI response."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.06,
        max_leaf_nodes=31,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=random_state,
    )


def spatial_blocks(row: np.ndarray, col: np.ndarray, block_px: int = BLOCK_SIZE_PX) -> np.ndarray:
    """Assign each sample to a square spatial block, used as the CV group."""
    br = row // block_px
    bc = col // block_px
    return (br * (bc.max() + 1) + bc).astype(np.int64)


# ──────────────────────────────────────────────
# Spatially calibrated ensemble
# ──────────────────────────────────────────────
class SpatialEnsemble:
    """
    An ensemble of gradient-boosted models, one per spatial fold, wrapped in
    an isotonic calibrator fitted on out-of-fold predictions.

    sklearn's CalibratedClassifierCV cannot be handed a GroupKFold without
    turning on metadata routing, and even then it calibrates on folds that
    are not the ones used for evaluation. Doing it explicitly gives three
    things that matter here:

      * calibration fitted strictly out-of-fold on spatially disjoint blocks,
        so the isotonic curve is not fitted to predictions the base model has
        already memorised;
      * an honest ensemble mean at prediction time;
      * a per-pixel spread across folds, which is the model's own uncertainty
        and is written out alongside the susceptibility surface.
    """

    def __init__(self, n_folds: int = 5, random_state: int = 0):
        self.n_folds = n_folds
        self.random_state = random_state
        self.models_: List = []
        self.calibrator_ = None
        # Case-control prior correction, in logit units. See set_prior_offset.
        self.prior_offset_: float = 0.0

    def set_prior_offset(self, domain_prevalence: float, sample_prevalence: float) -> None:
        """
        Closed-form case-control correction, used as the starting point.

            ln( pi_pop / (1 - pi_pop) ) - ln( pi_sample / (1 - pi_sample) )

        This is exact only when the absences are a *random* sample of the
        population. Ours are not: they are stratified across elevation to
        match the presence distribution, which deliberately enriches them in
        flood-like terrain. Use `fit_prior_offset` where a domain sample is
        available; this remains as a fallback.
        """
        pop = float(np.clip(domain_prevalence, 1e-9, 1 - 1e-9))
        smp = float(np.clip(sample_prevalence, 1e-9, 1 - 1e-9))
        self.prior_offset_ = float(np.log(pop / (1 - pop)) - np.log(smp / (1 - smp)))

    def fit_prior_offset(self, X_domain: np.ndarray, target_prevalence: float) -> float:
        """
        Solve for the logit offset that reproduces the observed flood extent.

        The offset c is chosen so that, over a uniform sample of the district,

            mean_i sigmoid( logit(p_i) + c )  ==  target_prevalence

        i.e. the expected flooded area equals the area actually observed in
        the 2018 inventory. mean(sigmoid(logit(p) + c)) is strictly increasing
        in c, so bisection converges.

        This replaces the closed-form correction because the absences are
        elevation-stratified rather than randomly drawn, which breaks the
        assumption behind the closed form. Uncorrected, the model expected
        337 km2 of flooding at the reference event against 31 km2 observed;
        the closed form over-corrected to 20 km2.
        """
        from hazard import logit, sigmoid

        base = logit(self.calibrator_.predict(self._raw(X_domain).mean(axis=0)))
        target = float(np.clip(target_prevalence, 1e-9, 1 - 1e-9))

        def expected(c: float) -> float:
            return float(sigmoid(base + c).mean())

        lo, hi = -25.0, 25.0
        if expected(lo) > target:
            self.prior_offset_ = lo
            return lo
        if expected(hi) < target:
            self.prior_offset_ = hi
            return hi

        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if expected(mid) < target:
                lo = mid
            else:
                hi = mid

        self.prior_offset_ = 0.5 * (lo + hi)
        return self.prior_offset_

    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "SpatialEnsemble":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=self.n_folds)
        oof = np.full(y.shape, np.nan, dtype=np.float64)
        self.models_ = []

        for k, (tr, te) in enumerate(splitter.split(X, y, groups=groups)):
            model = make_estimator(random_state=self.random_state + k)
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
            self.models_.append(model)

        scored = np.flatnonzero(np.isfinite(oof))

        # Honest reliability: fit the isotonic curve on half the out-of-fold
        # predictions and measure calibration on the untouched half. Reporting
        # reliability of a calibrator on the very points it was fitted to is
        # circular and always looks perfect.
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(scored)
        half = shuffled.size // 2
        fit_idx, eval_idx = shuffled[:half], shuffled[half:]

        probe = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            oof[fit_idx], y[fit_idx]
        )
        self.holdout_calibration_ = (y[eval_idx], probe.predict(oof[eval_idx]))

        # Final calibrator uses all out-of-fold predictions.
        self.calibrator_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
            oof[scored], y[scored]
        )
        self.oof_ = oof
        return self

    def _raw(self, X: np.ndarray) -> np.ndarray:
        """(n_models, n_samples) uncalibrated fold predictions."""
        return np.vstack([m.predict_proba(X)[:, 1] for m in self.models_])

    def _apply_prior(self, p: np.ndarray) -> np.ndarray:
        """Shift calibrated sample probabilities to the population base rate."""
        if self.prior_offset_ == 0.0:
            return p
        from hazard import logit, sigmoid

        return sigmoid(logit(p) + self.prior_offset_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._apply_prior(self.calibrator_.predict(self._raw(X).mean(axis=0)))
        return np.column_stack([1.0 - p, p])

    def predict_balanced_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability *before* the prior correction (for diagnostics)."""
        return self.calibrator_.predict(self._raw(X).mean(axis=0))

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple:
        """
        Return (population-scale probability, across-fold standard deviation).

        The spread is measured on the raw fold predictions, i.e. it reflects
        disagreement between models rather than calibration uncertainty.
        """
        raw = self._raw(X)
        mean = self._apply_prior(self.calibrator_.predict(raw.mean(axis=0)))
        return mean, raw.std(axis=0)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    scheme: str,
    n_splits: int = 5,
    random_state: int = 0,
) -> CVMetrics:
    """Out-of-fold evaluation under either random or spatial-block splitting."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import GroupKFold, StratifiedKFold

    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
    else:
        # GroupKFold keeps every block entirely within one fold.
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(X, y, groups=groups)

    oof = np.full(y.shape, np.nan, dtype=np.float64)
    for k, (tr, te) in enumerate(split_iter):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            LOGGER.warning("  fold %d has a single class; skipped", k)
            continue
        model = make_estimator(random_state=random_state + k)
        model.fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

    scored = np.isfinite(oof)
    if scored.sum() == 0:
        raise RuntimeError(f"No out-of-fold predictions produced for scheme {scheme!r}")

    return CVMetrics(
        scheme=scheme,
        auc_roc=float(roc_auc_score(y[scored], oof[scored])),
        auc_pr=float(average_precision_score(y[scored], oof[scored])),
        brier=float(brier_score_loss(y[scored], oof[scored])),
        n_folds=n_splits,
    )


def calibration_report(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list:
    """
    Reliability table: predicted vs observed frequency per probability bin.

    A calibrated model has predicted ~= observed in every bin. This is the
    check the previous pipeline had no way to pass, because its output was a
    raw score multiplied by a rainfall constant and then clipped to [0, 1].
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append((float(lo), float(hi), float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    return rows


def permutation_importance_report(
    model, X: np.ndarray, y: np.ndarray, features: List[str], n_repeats: int = 5
) -> Dict[str, float]:
    """
    Permutation importance in AUC units.

    Implemented directly rather than via sklearn.inspection because the model
    is a bespoke ensemble, not a fitted sklearn estimator with `classes_`.
    Each feature is shuffled `n_repeats` times and the mean drop in AUC is
    reported; a feature the model does not use scores ~0.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    baseline = roc_auc_score(y, model.predict_proba(X)[:, 1])

    importance = {}
    for j, name in enumerate(features):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(baseline - roc_auc_score(y, model.predict_proba(Xp)[:, 1]))
        importance[name] = float(np.mean(drops))
    return importance


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train(
    n_per_class: int = 60_000,
    seed: int = 42,
    aligned_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> dict:
    """Sample, evaluate, calibrate and persist the susceptibility model."""
    import joblib

    model_dir = model_dir or MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    sample = sample_training_points(n_per_class=n_per_class, seed=seed, aligned_dir=aligned_dir)
    X, y = sample["X"], sample["y"]
    features = list(sample["features"])
    groups = spatial_blocks(sample["row"], sample["col"])

    LOGGER.info(
        "Training set: %d samples, %d features, %d positives (%.1f%%), %d spatial blocks",
        X.shape[0],
        X.shape[1],
        int(y.sum()),
        100 * y.mean(),
        len(np.unique(groups)),
    )

    # ── Honest vs inflated performance ──
    LOGGER.info("Cross-validating (this is the slow part)...")
    random_cv = cross_validate(X, y, groups=None, scheme="random k-fold")
    spatial_cv = cross_validate(X, y, groups=groups, scheme="spatial block")
    LOGGER.info("  %s", random_cv)
    LOGGER.info("  %s   <- the honest number", spatial_cv)
    LOGGER.info(
        "  spatial autocorrelation inflates AUC by %.4f",
        random_cv.auc_roc - spatial_cv.auc_roc,
    )

    # ── Hard case: can the model rank within the low-lying zone? ──
    #
    # District-wide AUC is dominated by the trivial contrast between the
    # coastal plain and the inland hills. The decision a planner actually
    # faces is between two low-lying pixels, so performance is also reported
    # restricted to terrain below the 75th percentile of flooded elevation.
    dem_col = features.index("dem")
    flood_elev = X[y == 1, dem_col]
    low_cut = float(np.percentile(flood_elev, 75))
    low = X[:, dem_col] <= low_cut
    lowlying_cv = None
    if low.sum() > 1000 and len(np.unique(y[low])) == 2:
        lowlying_cv = cross_validate(
            X[low], y[low], groups=groups[low], scheme="spatial (low-lying)"
        )
        LOGGER.info(
            "  %s   <- restricted to DEM <= %.1f m (%d samples, %.0f%% positive)",
            lowlying_cv,
            low_cut,
            int(low.sum()),
            100 * y[low].mean(),
        )

    # ── Final model: spatial-fold ensemble with out-of-fold isotonic calibration ──
    LOGGER.info("Fitting calibrated spatial ensemble...")
    calibrated = SpatialEnsemble(n_folds=5, random_state=seed).fit(X, y, groups)

    # ── Correct for balanced, stratified (case-control) sampling ──
    domain_prevalence = float(sample["domain_prevalence"])
    calibrated.set_prior_offset(domain_prevalence, float(y.mean()))
    closed_form_offset = calibrated.prior_offset_

    LOGGER.info("Calibrating the prior offset against the observed flood extent...")
    X_domain, y_domain, row_domain, col_domain = sample_domain_points(
        aligned_dir=aligned_dir, with_labels=True
    )
    fitted_offset = calibrated.fit_prior_offset(X_domain, domain_prevalence)

    px_km2 = 1e-4
    observed_km2 = sample["presence_pixels"] * px_km2
    expected_km2 = (
        calibrated.predict_proba(X_domain)[:, 1].mean() * sample["domain_pixels"] * px_km2
    )
    LOGGER.info(
        "  domain prevalence %.5f | closed-form offset %.4f -> fitted offset %.4f",
        domain_prevalence,
        closed_form_offset,
        fitted_offset,
    )
    LOGGER.info(
        "  expected flooded area at the reference event: %.1f km2 (observed %.1f km2)",
        expected_km2,
        observed_km2,
    )

    # ── Conformal prediction: distribution-free coverage on the district ──
    #
    # Calibrated on district pixels, not the training set, because the
    # guarantee only transfers to data exchangeable with the calibration
    # sample. The calibration blocks are held out from the evaluation blocks
    # so coverage is not measured on the points that set the threshold.
    conformal_summary = None
    try:
        import conformal

        p_domain = calibrated.predict_proba(X_domain)[:, 1]
        domain_blocks = spatial_blocks(row_domain, col_domain)
        block_ids = np.unique(domain_blocks)
        rng_cf = np.random.default_rng(seed)
        cal_blocks = rng_cf.choice(block_ids, size=max(1, len(block_ids) // 2), replace=False)
        is_cal = np.isin(domain_blocks, cal_blocks)

        enough = is_cal.sum() > 1000 and (~is_cal).sum() > 1000
        both_classes = y_domain[is_cal].sum() > 20 and y_domain[~is_cal].sum() > 20

        if enough and both_classes:
            # Marginal split conformal, then class-conditional. Both are kept:
            # the marginal one is the standard construction, the Mondrian one
            # is what actually protects the flood class, and the gap between
            # them is the finding worth showing.
            LOGGER.info("--- Marginal split conformal ---")
            marginal_t = conformal.fit(p_domain[is_cal], y_domain[is_cal], alpha=0.10)
            marginal_summary = conformal.report(
                p_domain[~is_cal], y_domain[~is_cal], marginal_t, LOGGER
            )

            LOGGER.info("--- Class-conditional (Mondrian) conformal ---")
            mondrian_t = conformal.fit_mondrian(p_domain[is_cal], y_domain[is_cal], alpha=0.10)
            mondrian_summary = conformal.report(
                p_domain[~is_cal], y_domain[~is_cal], mondrian_t, LOGGER
            )

            # The operational layer uses Mondrian, because a guarantee averaged
            # over a 98.6% dry district says nothing useful about flooding.
            conformal_summary = dict(mondrian_summary)
            conformal_summary["marginal_variant"] = marginal_summary
            conformal_summary["calibration_blocks"] = int(len(cal_blocks))
            conformal_summary["evaluation_blocks"] = int(len(block_ids) - len(cal_blocks))
        else:
            LOGGER.warning("Too few domain points/positives for a spatial conformal split")
    except Exception as exc:  # pragma: no cover - diagnostics must not break training
        LOGGER.error("Conformal calibration failed: %s", exc)

    y_hold, p_hold = calibrated.holdout_calibration_
    reliability = calibration_report(y_hold, p_hold)
    LOGGER.info(
        "Reliability on calibration hold-out, balanced scale " "(curve never saw these points):"
    )
    for lo, hi, mean_pred, observed, n in reliability:
        LOGGER.info(
            "  p in [%.2f, %.2f): predicted %.3f, observed %.3f  (n=%d)",
            lo,
            hi,
            mean_pred,
            observed,
            n,
        )
    max_gap = max((abs(m - o) for _, _, m, o, _ in reliability), default=0.0)
    LOGGER.info("  worst calibration gap: %.4f", max_gap)

    # ── Feature importance, measured on held-out spatial blocks ──
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(groups)
    held_blocks = rng.choice(unique_blocks, size=max(1, len(unique_blocks) // 5), replace=False)
    hold = np.isin(groups, held_blocks)
    if hold.sum() < 500 or len(np.unique(y[hold])) < 2:
        hold = np.ones(y.shape, dtype=bool)
    sub = np.flatnonzero(hold)
    if sub.size > 20_000:
        sub = rng.choice(sub, size=20_000, replace=False)
    importance = permutation_importance_report(calibrated, X[sub], y[sub], features)

    ranked = sorted(importance.items(), key=lambda kv: -kv[1])
    LOGGER.info("Permutation importance (AUC drop when shuffled):")
    for name, value in ranked:
        LOGGER.info("  %-14s %.4f", name, value)

    metadata = {
        "features": features,
        "n_samples": int(X.shape[0]),
        "n_positive": int(y.sum()),
        "block_size_px": BLOCK_SIZE_PX,
        "seed": seed,
        "cv": {
            "random": asdict(random_cv),
            "spatial_block": asdict(spatial_cv),
            "spatial_low_lying": asdict(lowlying_cv) if lowlying_cv else None,
        },
        "low_lying_dem_cutoff_m": low_cut,
        "auc_inflation_from_spatial_autocorrelation": float(random_cv.auc_roc - spatial_cv.auc_roc),
        "worst_calibration_gap_balanced_scale": float(max_gap),
        "domain_prevalence": domain_prevalence,
        "domain_pixels": int(sample["domain_pixels"]),
        "presence_pixels": int(sample["presence_pixels"]),
        "prior_logit_offset_closed_form": float(closed_form_offset),
        "prior_logit_offset_fitted": float(fitted_offset),
        "expected_flooded_km2_at_reference": float(expected_km2),
        "observed_flooded_km2": float(observed_km2),
        "conformal": conformal_summary,
        "permutation_importance": importance,
        "reliability": [
            {
                "bin_low": lo,
                "bin_high": hi,
                "predicted": mean_pred,
                "observed": observed,
                "n": n,
            }
            for lo, hi, mean_pred, observed, n in reliability
        ],
    }

    model_path = model_dir / SUSCEPTIBILITY_MODEL
    joblib.dump({"model": calibrated, "metadata": metadata}, model_path)
    LOGGER.info("Saved model -> %s", model_path)

    metrics_path = model_dir / "susceptibility_metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics -> %s", metrics_path)

    return metadata


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def load_model(model_dir: Optional[Path] = None):
    """Load the trained susceptibility model and its metadata."""
    import joblib

    model_dir = model_dir or MODELS_DIR
    path = model_dir / SUSCEPTIBILITY_MODEL
    if not path.exists():
        raise FileNotFoundError(
            f"Susceptibility model not found at {path}. "
            "Run `python src/susceptibility.py --train` first."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def predict_surface(
    output_path: Optional[Path] = None,
    aligned_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
    stripe_rows: int = 256,
) -> Path:
    """
    Score every model-domain pixel and write the susceptibility raster.

    Processed in horizontal stripes so peak memory stays near one stripe
    rather than the 2.2 GB a full-grid feature stack would need.

    `stripe_rows` trades memory for I/O. build_matrix holds all 13 feature
    arrays for the stripe at once, so peak is roughly
    `stripe_rows * width * 13 * 4` bytes: 256 rows over the 7374-wide grid is
    ~100 MB, whereas 512 rows measured at ~2.1 GB private once GDAL's block
    cache and the fold models are included, which pushed this machine into
    paging. Raise it on a larger box.
    """
    import rasterio

    model, metadata = load_model(model_dir)
    features = metadata["features"]
    profile = grid_profile(aligned_dir)
    height, width = profile["height"], profile["width"]

    output_path = output_path or (OUTPUT_DIR / "susceptibility.tif")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    uncertainty_path = output_path.with_name(output_path.stem + "_uncertainty" + output_path.suffix)

    LOGGER.info("Predicting susceptibility over %dx%d grid...", height, width)
    with (
        rasterio.open(output_path, "w", **profile) as dst,
        rasterio.open(uncertainty_path, "w", **profile) as unc_dst,
    ):
        for n, window in enumerate(iter_stripes(height, width, stripe_rows)):
            X, idx, shape = build_matrix(window, aligned_dir, features)
            out = np.full(shape[0] * shape[1], profile["nodata"], dtype=np.float32)
            unc = np.full(shape[0] * shape[1], profile["nodata"], dtype=np.float32)
            if idx.size:
                mean, std = model.predict_with_uncertainty(X)
                out[idx] = mean.astype(np.float32)
                unc[idx] = std.astype(np.float32)
            dst.write(out.reshape(shape), 1, window=window)
            unc_dst.write(unc.reshape(shape), 1, window=window)
            if n % 4 == 0:
                LOGGER.info(
                    "  rows %d-%d (%d domain px)",
                    window.row_off,
                    window.row_off + window.height,
                    idx.size,
                )

    LOGGER.info("Susceptibility written -> %s", output_path)
    LOGGER.info("Ensemble uncertainty written -> %s", uncertainty_path)
    return output_path


def recalibrate_conformal(
    alpha: float = 0.10,
    seed: int = 42,
    aligned_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> dict:
    """
    Redo the conformal calibration against an already-trained model.

    Conformal calibration needs only the fitted model and a labelled sample of
    the district, so changing alpha or the calibration scheme does not require
    refitting the ensemble -- which is the expensive part.
    """
    import joblib

    import conformal

    model_dir = model_dir or MODELS_DIR
    model, metadata = load_model(model_dir)

    LOGGER.info("Sampling the district for conformal calibration...")
    X_domain, y_domain, row_domain, col_domain = sample_domain_points(
        aligned_dir=aligned_dir, with_labels=True
    )
    p_domain = model.predict_proba(X_domain)[:, 1]

    blocks = spatial_blocks(row_domain, col_domain)
    block_ids = np.unique(blocks)
    rng = np.random.default_rng(seed)
    cal_blocks = rng.choice(block_ids, size=max(1, len(block_ids) // 2), replace=False)
    is_cal = np.isin(blocks, cal_blocks)

    LOGGER.info("--- Marginal split conformal ---")
    marginal_t = conformal.fit(p_domain[is_cal], y_domain[is_cal], alpha=alpha)
    marginal_summary = conformal.report(p_domain[~is_cal], y_domain[~is_cal], marginal_t, LOGGER)

    LOGGER.info("--- Class-conditional (Mondrian) conformal ---")
    mondrian_t = conformal.fit_mondrian(p_domain[is_cal], y_domain[is_cal], alpha=alpha)
    mondrian_summary = conformal.report(p_domain[~is_cal], y_domain[~is_cal], mondrian_t, LOGGER)

    summary = dict(mondrian_summary)
    summary["marginal_variant"] = marginal_summary
    summary["calibration_blocks"] = int(len(cal_blocks))
    summary["evaluation_blocks"] = int(len(block_ids) - len(cal_blocks))

    metadata["conformal"] = summary
    joblib.dump({"model": model, "metadata": metadata}, model_dir / SUSCEPTIBILITY_MODEL)
    (model_dir / "susceptibility_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Updated conformal calibration in the saved model and metrics.")
    return summary


def write_conformal_layer(
    susceptibility_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> Optional[Path]:
    """
    Write the conformal decision raster from an existing susceptibility map.

    Cheap: the thresholds are plain probability cut-offs, so this is a
    reclassification of a raster already on disk, not another inference pass.

    Codes are conformal.SET_* -- empty / dry / ambiguous / flood.
    """
    import rasterio

    import conformal

    _, metadata = load_model(model_dir)
    summary = metadata.get("conformal")
    if not summary:
        LOGGER.warning("No conformal calibration in the model metadata; skipping")
        return None

    t = conformal.ConformalThresholds(**summary["thresholds"])

    susceptibility_path = susceptibility_path or (OUTPUT_DIR / "susceptibility.tif")
    output_path = output_path or (OUTPUT_DIR / "conformal_sets.tif")
    if not susceptibility_path.exists():
        raise FileNotFoundError(f"Susceptibility raster not found: {susceptibility_path}")

    with rasterio.open(susceptibility_path) as src:
        p = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nd = src.nodata

    valid = np.isfinite(p) & (p != np.float32(nd))
    codes = conformal.classify(np.clip(p, 0.0, 1.0), t).astype(np.float32)
    out = np.where(valid, codes, nd).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)

    LOGGER.info("Conformal decision raster -> %s", output_path)
    px_km2 = 1e-4
    for code, label in sorted(conformal.SET_LABELS.items()):
        n = int(((codes == code) & valid).sum())
        LOGGER.info("  %-32s %8.1f km2 (%5.2f%%)", label, n * px_km2, 100 * n / max(valid.sum(), 1))
    return output_path


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Flood susceptibility model")
    parser.add_argument("--train", action="store_true", help="Train and persist the model")
    parser.add_argument("--predict", action="store_true", help="Write the susceptibility raster")
    parser.add_argument(
        "--recalibrate-conformal",
        action="store_true",
        help="Redo conformal calibration against the saved model (no refit)",
    )
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="Write the conformal decision raster from an existing susceptibility map",
    )
    parser.add_argument("--alpha", type=float, default=0.10, help="Conformal miscoverage rate")
    parser.add_argument("--samples", type=int, default=60_000, help="Samples per class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging(logging.INFO)

    if not (args.train or args.predict or args.conformal or args.recalibrate_conformal):
        parser.error("Specify --train, --predict, --recalibrate-conformal and/or --conformal")

    if args.train:
        train(n_per_class=args.samples, seed=args.seed)
    if args.recalibrate_conformal:
        recalibrate_conformal(alpha=args.alpha, seed=args.seed)
    if args.predict:
        predict_surface()
    if args.conformal:
        write_conformal_layer()


if __name__ == "__main__":  # pragma: no cover
    main()
