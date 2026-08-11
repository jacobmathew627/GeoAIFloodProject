"""
Hazard model: combine terrain susceptibility with rainfall-driven runoff.

Susceptibility S(x) answers "if a major storm hits, does this pixel flood?".
It is learned from the 2018 event and does not depend on rainfall. The
rainfall response comes from SCS-CN runoff Q(x, P) (hydrology.py).

They are combined in logit space:

    H(x, P) = sigma( logit(S(x)) + beta * ln( Q(x,P) / Q(x,P_ref) ) )

with P_ref the reference event the susceptibility was calibrated on
(RAINFALL.reference_event_mm). This has four properties the previous formula
lacked:

  * At P = P_ref the log-ratio is zero and the hazard reduces exactly to
    S(x), so the model reproduces the observed 2018 flood extent rather than
    drifting away from it.
  * H is strictly increasing in P at every pixel, because Q is. The shipped
    maps were not: the 100 mm map had a higher mean probability than the
    150 mm map.
  * H stays inside (0, 1) by construction, so nothing has to be clipped and
    no pixel saturates at exactly 1.0 the way the old multiplier table did
    for everything above ~150 mm.
  * The response is a *ratio*, not a difference. This matters: an impervious
    surface loses less of its runoff proportionally as the storm shrinks
    than a forest does, so at 150 mm built-up Kochi (CN 94) stays more
    hazardous than the forested east (CN 84) at equal susceptibility. An
    absolute difference inverts that ordering, because the high-CN surface
    has the larger absolute runoff deficit relative to the reference storm.
    A scalar multiplier could not express either behaviour.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from config import HYDRO, OUTPUT_DIR, RAINFALL, RASTER

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

# Floor on the runoff ratio. Rainfall below the initial abstraction produces
# exactly zero runoff, and ln(0) is -inf; the floor maps that to a hazard of
# effectively zero without propagating infinities through the arithmetic.
_MIN_RUNOFF_RATIO = 1e-4

# Probabilities are squeezed away from the open interval's endpoints before
# taking a logit, so a susceptibility of exactly 0 or 1 does not produce an
# infinite offset.
_EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    # Numerically stable for large |z|.
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def combine(
    susceptibility: np.ndarray,
    curve_number: np.ndarray,
    rainfall_mm: float,
    reference_mm: Optional[float] = None,
    beta: Optional[float] = None,
) -> np.ndarray:
    """
    Rainfall-conditioned flood hazard.

    Args:
        susceptibility: S(x) in [0, 1]; NaN outside the model domain.
        curve_number: CN grid from hydrology.curve_number_from_lulc.
        rainfall_mm: Storm depth to evaluate.
        reference_mm: Calibration event depth (default RAINFALL.reference_event_mm).
        beta: Logit sensitivity to runoff (default HYDRO.runoff_logit_beta).

    Returns:
        Hazard probability array; NaN where susceptibility is NaN.
    """
    from hydrology import runoff_depth

    reference_mm = reference_mm if reference_mm is not None else RAINFALL.reference_event_mm
    beta = beta if beta is not None else HYDRO.runoff_logit_beta

    q_now = runoff_depth(rainfall_mm, curve_number)
    q_ref = runoff_depth(reference_mm, curve_number)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(q_ref > 0, q_now / q_ref, np.nan)
    ratio = np.clip(ratio, _MIN_RUNOFF_RATIO, None)

    shift = beta * np.log(ratio)
    hazard = sigmoid(logit(susceptibility) + shift)

    invalid = ~np.isfinite(susceptibility) | ~np.isfinite(curve_number)
    hazard = hazard.astype(np.float32)
    hazard[invalid] = np.nan
    return hazard


# ──────────────────────────────────────────────
# Raster generation
# ──────────────────────────────────────────────
def generate_hazard_rasters(
    scenarios: Optional[Tuple[float, ...]] = None,
    susceptibility_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    aligned_dir: Optional[Path] = None,
) -> Dict[float, Path]:
    """Write one hazard raster per rainfall scenario."""
    import rasterio

    from feature_stack import compute_curve_number, grid_profile

    scenarios = scenarios or RAINFALL.scenarios
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    susceptibility_path = susceptibility_path or (output_dir / "susceptibility.tif")

    if not susceptibility_path.exists():
        raise FileNotFoundError(
            f"Susceptibility raster not found at {susceptibility_path}. "
            "Run `python src/susceptibility.py --train --predict` first."
        )

    with rasterio.open(susceptibility_path) as src:
        susc = src.read(1).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
    susc[susc == np.float32(nd)] = np.nan

    # A susceptibility raster with no valid pixels means the prediction pass
    # did not finish (rasterio flushes stripe by stripe, so a partially
    # written file opens fine and reads back as all-nodata). Fail here with
    # something actionable rather than deep inside a percentile call.
    if not np.isfinite(susc).any():
        raise ValueError(
            f"{susceptibility_path} contains no valid pixels. The prediction pass "
            "is probably still running or was interrupted; re-run "
            "`python src/susceptibility.py --predict` and wait for it to finish."
        )

    cn, _ = compute_curve_number(aligned_dir=aligned_dir)
    profile = grid_profile(aligned_dir)

    written = {}
    for mm in scenarios:
        hazard = combine(susc, cn, float(mm))
        out = np.where(np.isfinite(hazard), hazard, NODATA).astype(np.float32)

        path = output_dir / f"flood_hazard_{int(mm)}mm.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out, 1)

        valid = hazard[np.isfinite(hazard)]
        px_km2 = _pixel_area_km2(aligned_dir)
        LOGGER.info(
            "%4d mm -> %s | mean=%.4f p95=%.4f max=%.4f | "
            "expected flooded %.0f km2 | area>0.3 = %.1f km2",
            int(mm),
            path.name,
            valid.mean(),
            np.percentile(valid, 95),
            valid.max(),
            float(valid.sum()) * px_km2,
            float((valid > 0.3).sum()) * px_km2,
        )
        written[float(mm)] = path

    return written


def _pixel_area_km2(aligned_dir: Optional[Path] = None) -> float:
    from feature_stack import pixel_area_km2

    return pixel_area_km2(aligned_dir)


# ──────────────────────────────────────────────
# Dashboard support
# ──────────────────────────────────────────────
def blend_scenarios(maps: Dict[float, np.ndarray], rainfall_mm: float) -> Optional[np.ndarray]:
    """
    Interpolate the pre-computed hazard scenarios to an arbitrary rainfall.

    Interpolation is linear in logit space between the two bracketing
    scenarios, which preserves monotonicity and keeps the result a
    probability. Outside the bracket the nearest scenario is returned rather
    than extrapolated.

    Nodata is preserved: a pixel is only produced where both bracketing
    scenarios have data.
    """
    if not maps:
        return None

    depths = sorted(maps)
    if rainfall_mm <= depths[0]:
        return maps[depths[0]].copy()
    if rainfall_mm >= depths[-1]:
        return maps[depths[-1]].copy()

    hi_idx = int(np.searchsorted(depths, rainfall_mm, side="left"))
    lo, hi = depths[hi_idx - 1], depths[hi_idx]
    w = (rainfall_mm - lo) / (hi - lo)

    a, b = maps[lo], maps[hi]
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        wd = min(a.shape[1], b.shape[1])
        a, b = a[:h, :wd], b[:h, :wd]

    valid = (a > -9000) & (b > -9000)
    out = np.full(a.shape, NODATA, dtype=np.float32)
    out[valid] = sigmoid(
        (1.0 - w) * logit(a[valid]) + w * logit(b[valid])
    ).astype(np.float32)
    return out


if __name__ == "__main__":  # pragma: no cover
    import argparse

    from config import setup_logging

    parser = argparse.ArgumentParser(description="Generate rainfall-conditioned hazard rasters")
    parser.add_argument("--scenarios", type=float, nargs="*", default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    generate_hazard_rasters(tuple(args.scenarios) if args.scenarios else None)
