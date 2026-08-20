"""
Fit the rainfall sensitivity beta against observed flood extents.

Why
---
`HYDRO.runoff_logit_beta` was the last hand-chosen constant in the hazard model.
Everything else is now measured: the reference event comes from IMD gauge data,
the curve numbers from LULC, the thresholds from the PR curve, the prior offset
from the observed 2018 extent. beta = 1.8 was a plausible guess and nothing more,
and it sets how hard the slider bites -- how much the risk zones move when the
user drags rainfall from 100 mm to 400 mm. Guessing it means the slider's
*shape* is unvalidated even though its endpoints are not.

The NDEM inventory gives three flood events with IMD rainfall attached, so beta
can be fitted instead of assumed.

Method
------
Expected flooded area under the hazard model, with no threshold to choose:

    A(P; beta) = sum_x sigma( logit(S(x)) + beta * ln( Q_x(P) / Q_x(P_ref) ) ) * px_km2

At P = P_ref every shift term is zero, so A(P_ref) = sum_x S(x), which the prior
calibration already pinned to the observed 2018 extent. The reference event
therefore carries *no* information about beta -- it is satisfied for every value.
beta is identified entirely by the off-reference events, and with three events
that means two informative points.

What this can and cannot establish
----------------------------------
2019 fell 412.5 mm and shows 31.3 km2; 2018 fell 443.2 mm and shows 78.7 km2.
A 7% difference in rainfall against a 2.5x difference in mapped extent means
most of the between-event variance is *not* rainfall. NDEM extents depend on
when the satellite happened to pass over relative to the flood peak, and on
which scenes covered the district at all -- the same acquisition-timing problem
that made the 2018 SAR scene (21 Aug, against an IMD peak of 15-17 Aug) unusable
earlier in this project.

So a beta fitted here absorbs acquisition differences along with hydrology. It
is still a better number than a guess, because it is anchored to observed extents
rather than to nothing, but the leave-one-out spread below is the honest measure
of how well it is determined -- not the fit residual.

Run:  python src/fit_beta.py
      python src/fit_beta.py --events 2019 2021
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import ALIGNED_DIR, HYDRO, OUTPUT_DIR, RAINFALL, RASTER, setup_logging

LOGGER = logging.getLogger("geoai_flood")

#: Search bracket for beta. Below 0 the model would make heavier rain safer;
#: above 8 a 2x runoff ratio would move the logit by more than 5, which
#: saturates the sigmoid and makes the slider a step function.
BETA_BOUNDS = (0.0, 8.0)

#: Floor on the runoff ratio, matching hazard.py -- a dry pixel must not send
#: the logit to -inf.
MIN_RUNOFF_RATIO = 1e-6


def _observed_extents(
    events: Optional[Sequence[str]] = None,
    aligned_dir: Optional[Path] = None,
) -> List[Dict]:
    """Observed flooded area per event, from the NDEM label rasters."""
    from feature_stack import domain_mask, read_raster
    from ndem_labels import EVENTS

    aligned_dir = aligned_dir or ALIGNED_DIR
    domain = domain_mask(aligned_dir=aligned_dir)
    px_km2 = (RASTER.cell_size / 1000.0) ** 2

    wanted = list(events) if events else sorted(EVENTS)
    out: List[Dict] = []
    for name in wanted:
        cfg = EVENTS.get(name)
        if cfg is None:
            raise ValueError(f"Unknown event {name!r}. Known: {sorted(EVENTS)}")
        rain = cfg.get("rainfall_mm")
        if rain is None:
            LOGGER.info("  %s: skipped, rainfall not derived", name)
            continue

        path = aligned_dir / f"ndem_flood_{name}_aligned.tif"
        if not path.exists():
            LOGGER.info("  %s: skipped, %s not built", name, path.name)
            continue

        label, valid = read_raster(f"ndem_flood_{name}", aligned_dir=aligned_dir)
        flooded = (label > 0.5) & valid & domain
        out.append({
            "event": name,
            "rainfall_mm": float(rain),
            "observed_km2": float(flooded.sum()) * px_km2,
        })

    if not out:
        raise RuntimeError("No events with both a label raster and IMD rainfall")
    return out


def _load_surface(aligned_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Susceptibility and curve number over the model domain, as flat arrays."""
    from feature_stack import domain_mask, read_raster
    from hydrology import curve_number_from_lulc

    aligned_dir = aligned_dir or ALIGNED_DIR
    domain = domain_mask(aligned_dir=aligned_dir)

    path = OUTPUT_DIR / "susceptibility.tif"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python src/susceptibility.py --predict"
        )

    import rasterio

    with rasterio.open(path) as src:
        s = src.read(1).astype(np.float64)
        nd = src.nodata
    ok = np.isfinite(s) & (s > 0.0) & (s < 1.0)
    if nd is not None:
        ok &= s != nd

    lulc, lulc_valid = read_raster("lulc", aligned_dir=aligned_dir)
    cn = curve_number_from_lulc(lulc, lulc_valid)

    use = domain & ok & lulc_valid & np.isfinite(cn) & (cn > 0)
    if not use.any():
        raise RuntimeError(
            "Susceptibility raster is empty over the domain -- a partially "
            "written file reads as all-nodata; re-run --predict to completion"
        )
    LOGGER.info("  %d domain pixels with susceptibility and curve number", int(use.sum()))
    return s[use], cn[use]


def expected_area_km2(
    susceptibility: np.ndarray,
    curve_number: np.ndarray,
    rainfall_mm: float,
    beta: float,
    reference_mm: Optional[float] = None,
) -> float:
    """
    Expected flooded area under the hazard model, in km2.

    Summing probabilities rather than thresholding them keeps this independent
    of the risk-band cuts, which were derived separately.
    """
    from hydrology import runoff_depth

    reference_mm = reference_mm if reference_mm is not None else RAINFALL.reference_event_mm

    q_now = runoff_depth(rainfall_mm, curve_number)
    q_ref = runoff_depth(reference_mm, curve_number)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(q_ref > 0, q_now / q_ref, np.nan)
    ratio = np.clip(ratio, MIN_RUNOFF_RATIO, None)

    logit_s = np.log(susceptibility / (1.0 - susceptibility))
    hazard = 1.0 / (1.0 + np.exp(-(logit_s + beta * np.log(ratio))))

    px_km2 = (RASTER.cell_size / 1000.0) ** 2
    return float(np.nansum(hazard)) * px_km2


def _loss(
    beta: float,
    surface: Tuple[np.ndarray, np.ndarray],
    events: Sequence[Dict],
) -> float:
    """
    Mean squared error in log area.

    Log space because the extents span 4.1 to 78.7 km2; in linear space the
    biggest event would set beta on its own.
    """
    s, cn = surface
    err = 0.0
    for ev in events:
        pred = expected_area_km2(s, cn, ev["rainfall_mm"], beta)
        err += (np.log(max(pred, 1e-6)) - np.log(max(ev["observed_km2"], 1e-6))) ** 2
    return err / len(events)


def fit(
    surface: Tuple[np.ndarray, np.ndarray],
    events: Sequence[Dict],
    bounds: Tuple[float, float] = BETA_BOUNDS,
    tol: float = 1e-3,
) -> float:
    """
    Golden-section search for the beta minimising log-area error.

    The loss is smooth and unimodal in beta -- expected area is monotone
    increasing in beta above the reference and monotone decreasing below it --
    so a derivative-free line search is enough and avoids a scipy dependency.
    """
    lo, hi = bounds
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - invphi * (b - a)
    d = a + invphi * (b - a)
    fc, fd = _loss(c, surface, events), _loss(d, surface, events)

    while abs(b - a) > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = _loss(c, surface, events)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = _loss(d, surface, events)
    return float((a + b) / 2.0)


def run(
    events: Optional[Sequence[str]] = None,
    aligned_dir: Optional[Path] = None,
) -> Dict:
    """Fit beta, report per-event fit and leave-one-out spread."""
    LOGGER.info("Observed extents from the NDEM inventory:")
    obs = _observed_extents(events, aligned_dir)
    for ev in obs:
        LOGGER.info(
            "  %s  %6.1f mm -> %6.1f km2", ev["event"], ev["rainfall_mm"], ev["observed_km2"]
        )

    LOGGER.info("Loading susceptibility surface...")
    surface = _load_surface(aligned_dir)

    ref = RAINFALL.reference_event_mm
    informative = [e for e in obs if abs(e["rainfall_mm"] - ref) > 1.0]
    LOGGER.info(
        "Reference event %.1f mm carries no information about beta; "
        "%d of %d events are informative",
        ref, len(informative), len(obs),
    )
    if not informative:
        raise RuntimeError(
            f"Every event sits at the reference depth ({ref} mm); beta is "
            "unidentified. Add an event at a different rainfall."
        )

    beta = fit(surface, informative)
    LOGGER.info("Fitted beta = %.3f  (was %.3f, assumed)", beta, HYDRO.runoff_logit_beta)

    LOGGER.info("")
    LOGGER.info("%-8s %10s %12s %12s %12s", "event", "rain mm", "observed", "fitted", "assumed")
    LOGGER.info("-" * 58)
    per_event = []
    s, cn = surface
    for ev in obs:
        pred = expected_area_km2(s, cn, ev["rainfall_mm"], beta)
        old = expected_area_km2(s, cn, ev["rainfall_mm"], HYDRO.runoff_logit_beta)
        marker = "" if ev in informative else "  (reference, fixed by construction)"
        LOGGER.info(
            "%-8s %10.1f %9.1f km2 %8.1f km2 %8.1f km2%s",
            ev["event"], ev["rainfall_mm"], ev["observed_km2"], pred, old, marker,
        )
        per_event.append({
            "event": ev["event"],
            "rainfall_mm": ev["rainfall_mm"],
            "observed_km2": round(ev["observed_km2"], 1),
            "predicted_km2_fitted": round(pred, 1),
            "predicted_km2_assumed": round(old, 1),
            "informative": ev in informative,
        })

    # Leave-one-out: with one parameter and few points, the spread of beta
    # across subsets says more about identifiability than the residual does.
    loo: List[Dict] = []
    if len(informative) > 1:
        LOGGER.info("")
        LOGGER.info("Leave-one-out over the informative events:")
        for held in informative:
            rest = [e for e in informative if e is not held]
            b_loo = fit(surface, rest)
            pred = expected_area_km2(s, cn, held["rainfall_mm"], b_loo)
            LOGGER.info(
                "  hold out %s: beta=%.3f -> %.1f km2 vs %.1f observed (x%.2f)",
                held["event"], b_loo, pred, held["observed_km2"],
                pred / max(held["observed_km2"], 1e-6),
            )
            loo.append({
                "held_out": held["event"],
                "beta": round(b_loo, 3),
                "predicted_km2": round(pred, 1),
                "observed_km2": round(held["observed_km2"], 1),
                "ratio": round(pred / max(held["observed_km2"], 1e-6), 2),
            })
        betas = [d["beta"] for d in loo]
        LOGGER.info("  beta across folds: %.3f to %.3f", min(betas), max(betas))
    else:
        LOGGER.info("")
        LOGGER.info(
            "Only one informative event, so beta is exactly determined and "
            "cannot be cross-validated. Treat it as a point estimate."
        )

    summary = {
        "beta_fitted": round(beta, 3),
        "beta_assumed": HYDRO.runoff_logit_beta,
        "reference_mm": ref,
        "n_events": len(obs),
        "n_informative": len(informative),
        "per_event": per_event,
        "leave_one_out": loo,
        "caveat": (
            "NDEM extents depend on satellite acquisition timing relative to "
            "the flood peak, so beta absorbs acquisition differences alongside "
            "hydrology. 2019 fell 412.5 mm against 2018's 443.2 mm -- a 7% "
            "rainfall difference -- yet shows 2.5x less extent, so most "
            "between-event variance is not rainfall. The leave-one-out spread, "
            "not the fit residual, is the honest measure of certainty."
        ),
    }
    out = OUTPUT_DIR / "beta_fit.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("")
    LOGGER.info("Wrote %s", out.name)
    return summary


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Fit the rainfall sensitivity beta")
    parser.add_argument("--events", nargs="*", default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    run(events=args.events)


if __name__ == "__main__":  # pragma: no cover
    main()
