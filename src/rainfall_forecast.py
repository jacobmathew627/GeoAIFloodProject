"""
Short-term rainfall prediction from IMD gauge-based gridded rainfall.

Why this exists
---------------
The project title claims "short-term rainfall prediction". Until now nothing in
the system predicted rainfall: the dashboard had a slider, plus an optional
call to Open-Meteo that *consumes someone else's* forecast. Consuming a
forecast is not making one.

This module predicts the **total rainfall over the next three days** for the
Ernakulam district from the preceding observations and the seasonal cycle. The
three-day horizon is not arbitrary -- it is exactly what the flood model
consumes, because `RAINFALL.reference_event_mm` is a 3-day storm depth. So a
prediction here feeds straight into the hazard map, which is the chain the
title describes.

Data: IMD 0.25 deg daily gridded rainfall, the official Indian gauge-based
analysis, free and scriptable via `imdlib`. No API key, no registration.

What this is and is not
-----------------------
This is a *statistical* short-range forecast trained on rainfall history and
seasonality. It is not numerical weather prediction, and it is not radar
nowcasting -- the latter would need the IMD Kochi Doppler radar feed, which
requires API access that is not currently open to registration.

Honest baselines are the point. Monsoon rainfall is strongly seasonal and
strongly autocorrelated, so climatology and persistence are hard to beat. Any
claim of skill is reported against both, on a **temporal** hold-out: training
stops before the test period begins, because shuffling a time series leaks the
future into the past exactly as a random split leaks space in the flood model.

Run:
    python src/rainfall_forecast.py --train
    python src/rainfall_forecast.py --predict-latest
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import DATA_DIR, MODELS_DIR, setup_logging

LOGGER = logging.getLogger("geoai_flood")

#: Ernakulam footprint, matching src/reference_rainfall.py.
LAT = (9.79, 10.30)
LON = (76.17, 76.84)

#: Forecast horizon, in days. Matches the hazard model's storm window.
HORIZON_DAYS = 3

#: Where imdlib caches the yearly files.
CACHE_DIR = DATA_DIR / "imd_rain"

MODEL_NAME = "rainfall_forecast.joblib"

#: Warning thresholds for the 3-day total, in mm. The lower one is roughly the
#: level at which the hazard model starts showing elevated area; the upper is
#: near the 2019 and 2018 event depths.
WARN_THRESHOLDS = (100.0, 200.0, 400.0)


# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
def load_series(
    start_year: int = 1990,
    end_year: int = 2024,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    District-mean daily rainfall. Returns (dates as datetime64[D], mm).

    Downloads on first use and caches; subsequent calls read from disk.
    """
    import imdlib as imd

    cache_dir = cache_dir or CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    data = imd.get_data(
        "rain", start_year, end_year, fn_format="yearwise", file_dir=str(cache_dir)
    )
    ds = data.get_xarray().sel(lat=slice(*LAT), lon=slice(*LON))

    values = ds["rain"].values.astype(float)
    # IMD marks missing data with a negative value, not NaN.
    values = np.where(values < 0, np.nan, values)
    daily = np.nanmean(values, axis=(1, 2))
    dates = ds["time"].values.astype("datetime64[D]")

    ok = np.isfinite(daily)
    LOGGER.info(
        "IMD series %s..%s: %d days, %d usable, mean %.2f mm/day, max %.1f mm",
        dates[0], dates[-1], daily.size, int(ok.sum()),
        np.nanmean(daily), np.nanmax(daily),
    )
    return dates[ok], daily[ok]


# ──────────────────────────────────────────────
# Features
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    "rain_t", "rain_t1", "rain_t2",
    "sum_3d", "sum_7d", "sum_15d", "sum_30d",
    "wet_days_7d", "max_1d_7d",
    "doy_sin1", "doy_cos1", "doy_sin2", "doy_cos2",
]


def build_features(
    dates: np.ndarray, rain: np.ndarray, horizon: int = HORIZON_DAYS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build (X, y, sample_dates) where y is the total rainfall over the next
    `horizon` days and each row uses only information available at time t.
    """
    n = rain.size
    lookback = 30
    rows, targets, stamps = [], [], []

    doy = (
        dates.astype("datetime64[D]") - dates.astype("datetime64[Y]").astype("datetime64[D]")
    ).astype(int) + 1

    for t in range(lookback, n - horizon):
        window = rain[t - lookback + 1: t + 1]
        angle = 2.0 * np.pi * doy[t] / 365.25
        rows.append([
            rain[t], rain[t - 1], rain[t - 2],
            window[-3:].sum(), window[-7:].sum(), window[-15:].sum(), window.sum(),
            float((window[-7:] > 1.0).sum()), float(window[-7:].max()),
            np.sin(angle), np.cos(angle), np.sin(2 * angle), np.cos(2 * angle),
        ])
        targets.append(rain[t + 1: t + 1 + horizon].sum())
        stamps.append(dates[t])

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(stamps),
    )


# ──────────────────────────────────────────────
# Baselines
# ──────────────────────────────────────────────
def persistence_baseline(X: np.ndarray) -> np.ndarray:
    """The next three days will total what the last three days did."""
    return X[:, FEATURE_NAMES.index("sum_3d")]


def climatology_baseline(
    train_dates: np.ndarray, train_y: np.ndarray, test_dates: np.ndarray
) -> np.ndarray:
    """
    Day-of-year mean of the target, smoothed over a +/-7 day window.

    In a monsoon climate this is a strong baseline and beating it is the real
    test of whether a model has learned anything beyond the calendar.
    """
    def doy_of(d):
        return (
            d.astype("datetime64[D]") - d.astype("datetime64[Y]").astype("datetime64[D]")
        ).astype(int) + 1

    train_doy = doy_of(train_dates)
    means = np.zeros(367)
    for day in range(1, 367):
        # Circular +/-7 day window.
        delta = np.minimum(np.abs(train_doy - day), 366 - np.abs(train_doy - day))
        sel = delta <= 7
        means[day] = train_y[sel].mean() if sel.any() else train_y.mean()
    return means[doy_of(test_dates)]


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train(
    start_year: int = 1990,
    end_year: int = 2024,
    test_from: int = 2015,
    model_dir: Optional[Path] = None,
) -> Dict:
    """Fit and evaluate on a temporal hold-out."""
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

    model_dir = model_dir or MODELS_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    dates, rain = load_series(start_year, end_year)
    X, y, stamps = build_features(dates, rain)

    # Temporal split. Shuffling here would leak the future into the past.
    years = stamps.astype("datetime64[Y]").astype(int) + 1970
    train_mask = years < test_from
    test_mask = ~train_mask
    LOGGER.info(
        "train %d samples (%d..%d), test %d samples (%d..%d)",
        train_mask.sum(), years[train_mask].min(), years[train_mask].max(),
        test_mask.sum(), years[test_mask].min(), years[test_mask].max(),
    )

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=0,
    )
    model.fit(X[train_mask], y[train_mask])

    pred = model.predict(X[test_mask])
    pred = np.clip(pred, 0.0, None)  # rainfall cannot be negative
    truth = y[test_mask]

    persist = np.clip(persistence_baseline(X[test_mask]), 0.0, None)
    clim = climatology_baseline(stamps[train_mask], y[train_mask], stamps[test_mask])

    def scores(p):
        return {
            "mae": float(mean_absolute_error(truth, p)),
            "rmse": float(np.sqrt(mean_squared_error(truth, p))),
            "corr": float(np.corrcoef(truth, p)[0, 1]),
        }

    result = {
        "model": scores(pred),
        "persistence": scores(persist),
        "climatology": scores(clim),
    }

    LOGGER.info("3-day total, held-out %d-%d:", years[test_mask].min(), years[test_mask].max())
    for name in ("model", "persistence", "climatology"):
        s = result[name]
        LOGGER.info(
            "  %-12s MAE %6.2f mm   RMSE %6.2f mm   r %.3f",
            name, s["mae"], s["rmse"], s["corr"],
        )
    skill = 1.0 - result["model"]["mae"] / result["climatology"]["mae"]
    skill_p = 1.0 - result["model"]["mae"] / result["persistence"]["mae"]
    LOGGER.info("  MAE skill vs climatology: %+.1f%%", 100 * skill)
    LOGGER.info("  MAE skill vs persistence: %+.1f%%", 100 * skill_p)
    result["mae_skill_vs_climatology"] = float(skill)
    result["mae_skill_vs_persistence"] = float(skill_p)

    # Warning-threshold discrimination: for flood use, ranking heavy events
    # matters more than the mean absolute error.
    result["exceedance"] = {}
    LOGGER.info("  discrimination for heavy-rain warnings:")
    for thr in WARN_THRESHOLDS:
        label = (truth >= thr).astype(int)
        if label.sum() < 5 or label.sum() == label.size:
            LOGGER.info("    >= %5.0f mm: too few events in the hold-out", thr)
            continue
        entry = {
            "n_events": int(label.sum()),
            "base_rate": float(label.mean()),
            "auc_model": float(roc_auc_score(label, pred)),
            "auc_persistence": float(roc_auc_score(label, persist)),
            "auc_climatology": float(roc_auc_score(label, clim)),
        }
        result["exceedance"][str(int(thr))] = entry
        LOGGER.info(
            "    >= %5.0f mm (%3d events, %.2f%%): AUC model %.3f | persistence %.3f | climatology %.3f",
            thr, entry["n_events"], 100 * entry["base_rate"],
            entry["auc_model"], entry["auc_persistence"], entry["auc_climatology"],
        )

    metadata = {
        "target": f"total rainfall over the next {HORIZON_DAYS} days, mm",
        "source": "IMD 0.25 deg gauge-based gridded rainfall",
        "features": FEATURE_NAMES,
        "train_years": [int(years[train_mask].min()), int(years[train_mask].max())],
        "test_years": [int(years[test_mask].min()), int(years[test_mask].max())],
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "scores": result,
        "caveat": (
            "Statistical short-range forecast from rainfall history and "
            "seasonality. Not NWP, not radar nowcasting. Evaluated on a "
            "temporal hold-out against persistence and climatology."
        ),
    }

    joblib.dump(
        {"model": model, "features": FEATURE_NAMES, "metadata": metadata},
        model_dir / MODEL_NAME,
    )
    (model_dir / "rainfall_forecast_metrics.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved -> %s", model_dir / MODEL_NAME)
    return metadata


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────
def load_model(model_dir: Optional[Path] = None):
    import joblib

    model_dir = model_dir or MODELS_DIR
    path = model_dir / MODEL_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/rainfall_forecast.py --train`."
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"]


def predict_latest(
    start_year: int = 2020, end_year: int = 2024, model_dir: Optional[Path] = None
) -> Dict:
    """Forecast the next 3-day total from the most recent available observations."""
    model, metadata = load_model(model_dir)
    dates, rain = load_series(start_year, end_year)
    X, _, stamps = build_features(dates, rain)

    if X.size == 0:
        raise RuntimeError("Not enough history to build a feature row")

    latest = X[-1:, :]
    value = float(np.clip(model.predict(latest)[0], 0.0, None))
    return {
        "as_of": str(stamps[-1]),
        "horizon_days": HORIZON_DAYS,
        "predicted_total_mm": round(value, 1),
        "last_3_days_mm": round(float(latest[0, FEATURE_NAMES.index("sum_3d")]), 1),
        "last_30_days_mm": round(float(latest[0, FEATURE_NAMES.index("sum_30d")]), 1),
        "note": (
            "Feeds the hazard model directly: this is a 3-day storm depth, the "
            "same quantity RAINFALL.reference_event_mm expresses."
        ),
    }


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Short-term rainfall prediction")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict-latest", action="store_true")
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--test-from", type=int, default=2015)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    if args.train:
        train(args.start_year, args.end_year, args.test_from)
    if args.predict_latest:
        for k, v in predict_latest().items():
            LOGGER.info("  %-22s %s", k, v)
    if not args.train and not args.predict_latest:
        parser.error("Specify --train and/or --predict-latest")


if __name__ == "__main__":  # pragma: no cover
    main()
