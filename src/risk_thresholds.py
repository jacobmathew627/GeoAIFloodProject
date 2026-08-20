"""
Derive the risk-band thresholds from the precision-recall curve.

`RiskThresholds` in config.py is not a set of round numbers chosen for
readability. Each edge is an operating point read off the precision-recall
curve of the reference-event hazard map against the flood inventory, so each
one has an operational meaning: how much of the observed flood it captures,
and how often it cries wolf.

That makes the thresholds **properties of the fitted probabilities**, not
constants. Retraining moves them. Changing the label moves them a lot: the
inventory switch from a single Sentinel-1 scene to the NDEM inventory raised
the domain prevalence from 1.4% to 3.5%, which shifts every operating point.

This module exists so that re-deriving them is a command rather than a
scratch script someone has to reconstruct.

Run:  python src/risk_thresholds.py
      python src/risk_thresholds.py --rainfall 443 --apply
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from config import MODELS_DIR, OUTPUT_DIR, RAINFALL, RISK, setup_logging

LOGGER = logging.getLogger("geoai_flood")

#: Operating points to report, as (band name, criterion, target).
#: The bands are ordered from most inclusive to most selective.
CRITERIA = [
    ("moderate", "recall", 0.95),
    ("high", "recall", 0.80),
    ("severe", "max_f1", None),
    ("critical", "precision", 0.50),
]


def load_surface_and_labels(
    rainfall_mm: Optional[float] = None,
    max_pixels: int = 3_000_000,
    seed: int = 0,
):
    """Hazard at the reference event and the flood inventory, over the domain."""
    import rasterio

    from feature_stack import domain_mask, flood_labels

    rainfall_mm = rainfall_mm or RAINFALL.reference_event_mm
    path = OUTPUT_DIR / f"flood_hazard_{int(round(rainfall_mm))}mm.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/hazard.py` first so the "
            "reference-event raster exists."
        )

    with rasterio.open(path) as src:
        hazard = src.read(1).astype(np.float32)
        nd = src.nodata
    hazard[hazard == np.float32(nd)] = np.nan

    flood, gt_valid = flood_labels()
    domain = domain_mask()

    usable = np.isfinite(hazard) & gt_valid & domain
    idx = np.flatnonzero(usable.ravel())
    if idx.size > max_pixels:
        idx = np.random.default_rng(seed).choice(idx, size=max_pixels, replace=False)

    y = flood.ravel()[idx].astype(np.int8)
    p = hazard.ravel()[idx]
    LOGGER.info(
        "Scoring %d pixels at %.0f mm: %.2f%% flooded",
        idx.size, rainfall_mm, 100 * y.mean(),
    )
    return y, p, float(rainfall_mm)


def derive(rainfall_mm: Optional[float] = None) -> Dict:
    """Read the operating points off the precision-recall curve."""
    from sklearn.metrics import precision_recall_curve

    y, p, rainfall = load_surface_and_labels(rainfall_mm)
    precision, recall, thresholds = precision_recall_curve(y, p)
    # precision/recall carry one extra element relative to thresholds
    precision, recall = precision[:-1], recall[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )

    base_rate = float(y.mean())
    result = {
        "rainfall_mm": rainfall,
        "base_rate": base_rate,
        "n_scored": int(y.size),
        "bands": {},
    }

    for name, criterion, target in CRITERIA:
        if criterion == "max_f1":
            i = int(np.argmax(f1))
        elif criterion == "recall":
            ok = np.flatnonzero(recall >= target)
            if ok.size == 0:
                LOGGER.warning("  %s: recall %.2f unreachable", name, target)
                continue
            i = int(ok[-1])  # recall falls as the threshold rises
        else:  # precision
            ok = np.flatnonzero(precision >= target)
            if ok.size == 0:
                LOGGER.warning("  %s: precision %.2f unreachable", name, target)
                continue
            i = int(ok[0])

        result["bands"][name] = {
            "threshold": round(float(thresholds[i]), 4),
            "precision": round(float(precision[i]), 3),
            "recall": round(float(recall[i]), 3),
            "f1": round(float(f1[i]), 3),
            "lift_over_base_rate": round(float(precision[i]) / max(base_rate, 1e-9), 1),
            "criterion": f"{criterion}={target}" if target else criterion,
        }

    LOGGER.info("Base rate (no-skill precision): %.4f", base_rate)
    LOGGER.info(
        "%-10s %9s %10s %10s %8s %8s",
        "band", "threshold", "precision", "recall", "F1", "lift",
    )
    for name, b in result["bands"].items():
        LOGGER.info(
            "%-10s %9.4f %10.3f %10.3f %8.3f %7.0fx",
            name, b["threshold"], b["precision"], b["recall"], b["f1"],
            b["lift_over_base_rate"],
        )

    current = {
        "safe": RISK.safe, "moderate": RISK.moderate,
        "high": RISK.high, "critical": RISK.critical,
    }
    LOGGER.info("Current config: %s", current)
    LOGGER.info(
        "Suggested config: safe=%.3f moderate=%.3f high=%.3f critical=%.3f",
        result["bands"].get("moderate", {}).get("threshold", RISK.safe),
        result["bands"].get("high", {}).get("threshold", RISK.moderate),
        result["bands"].get("severe", {}).get("threshold", RISK.high),
        result["bands"].get("critical", {}).get("threshold", RISK.critical),
    )

    out = MODELS_DIR / "risk_thresholds.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return result


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Derive risk-band thresholds")
    parser.add_argument("--rainfall", type=float, default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    derive(args.rainfall)
    LOGGER.info(
        "Update RiskThresholds in src/config.py by hand -- the values map to "
        "safe/moderate/high/critical as printed above, and the docstring there "
        "should record what each one buys."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
