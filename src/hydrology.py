"""
Hydrology module: rainfall to runoff via the SCS Curve Number method.

This replaces the hard-coded rainfall multiplier table that the previous
pipeline used ({50mm: 0.4, 100mm: 0.8, 150mm: 1.6, 200mm: 2.6, ...}). That
table was applied as a straight multiplier on a probability, which meant it
both saturated at 1.0 for anything above ~150 mm and produced maps that were
not even monotonic in rainfall (the shipped 100 mm map had a higher mean
probability than the 150 mm map).

The SCS-CN method gives a runoff depth that is, by construction, continuous
and strictly increasing in rainfall, and it varies in space according to land
cover rather than being a single scalar for the whole district.

References
----------
USDA NRCS, National Engineering Handbook Part 630, Chapter 10 ("Estimation of
Direct Runoff from Storm Rainfall") and Table 2-2 for curve numbers by cover
type and hydrologic soil group.
Woodward et al. (2003), "Runoff Curve Number Method: Examination of the
Initial Abstraction Ratio" -- the basis for the lambda = 0.05 formulation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import HYDRO, RASTER

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value


# ──────────────────────────────────────────────
# Curve number grid
# ──────────────────────────────────────────────
def curve_number_from_lulc(
    lulc: np.ndarray,
    valid: np.ndarray,
    amc: Optional[str] = None,
) -> np.ndarray:
    """
    Build a curve number grid from the LULC classification.

    Args:
        lulc: LULC class raster (float, class codes).
        valid: Boolean mask of pixels inside the study area.
        amc: Antecedent moisture condition, "I" | "II" | "III".
             Defaults to HYDRO.amc.

    Returns:
        Curve number array. Invalid pixels hold NaN.
    """
    amc = amc or HYDRO.amc
    cn = np.full(lulc.shape, np.nan, dtype=np.float32)

    cn[valid] = HYDRO.default_curve_number
    for cls, value in HYDRO.curve_numbers.items():
        cn[valid & (np.round(lulc) == cls)] = value

    return adjust_cn_for_amc(cn, amc)


def adjust_cn_for_amc(cn: np.ndarray, amc: str) -> np.ndarray:
    """
    Convert AMC II curve numbers to AMC I (dry) or AMC III (wet).

    Standard NEH-630 conversions:
        CN_I   = 4.2 * CN_II / (10 - 0.058 * CN_II)
        CN_III = 23  * CN_II / (10 + 0.13  * CN_II)
    """
    if amc == "II":
        return cn

    out = cn.copy()
    m = np.isfinite(cn)
    if amc == "I":
        out[m] = 4.2 * cn[m] / (10.0 - 0.058 * cn[m])
    elif amc == "III":
        out[m] = 23.0 * cn[m] / (10.0 + 0.13 * cn[m])
    else:
        raise ValueError(f"amc must be 'I', 'II' or 'III', got {amc!r}")

    # CN is a percentage-like index bounded by 100 (total runoff).
    out[m] = np.clip(out[m], 30.0, 100.0)
    return out


# ──────────────────────────────────────────────
# Runoff
# ──────────────────────────────────────────────
def potential_retention(cn: np.ndarray) -> np.ndarray:
    """
    Maximum potential retention S in mm from the curve number.

        S = 25400 / CN - 254        (mm, the metric form)

    For lambda = 0.05 the retention must be rescaled, because the curve
    numbers themselves were fitted under the lambda = 0.20 assumption:

        S_005 = 1.33 * S_020 ** 1.15     (Woodward et al., 2003)
    """
    s = np.full(cn.shape, np.nan, dtype=np.float32)
    m = np.isfinite(cn)
    # CN = 100 means zero retention (open water); clamp to avoid 0/0.
    cn_safe = np.clip(cn[m], 30.0, 100.0)
    s020 = 25400.0 / cn_safe - 254.0
    s020 = np.maximum(s020, 0.0)

    if abs(HYDRO.initial_abstraction_ratio - 0.05) < 1e-9:
        s[m] = 1.33 * np.power(s020, 1.15)
    else:
        s[m] = s020
    return s


def runoff_depth(rainfall_mm: float, cn: np.ndarray) -> np.ndarray:
    """
    Direct runoff depth Q (mm) for a storm of `rainfall_mm` over a CN grid.

        Ia = lambda * S
        Q  = (P - Ia)^2 / (P - Ia + S)   for P > Ia
        Q  = 0                            otherwise

    Q is continuous, non-negative, bounded above by P, and strictly
    increasing in P wherever P > Ia. Invalid pixels hold NaN.
    """
    if rainfall_mm < 0:
        raise ValueError(f"rainfall_mm must be >= 0, got {rainfall_mm}")

    s = potential_retention(cn)
    q = np.full(cn.shape, np.nan, dtype=np.float32)
    m = np.isfinite(s)

    ia = HYDRO.initial_abstraction_ratio * s[m]
    excess = rainfall_mm - ia
    q_valid = np.zeros_like(excess, dtype=np.float32)
    pos = excess > 0
    q_valid[pos] = (excess[pos] ** 2) / (excess[pos] + s[m][pos])

    q[m] = q_valid
    return q


def runoff_coefficient(rainfall_mm: float, cn: np.ndarray) -> np.ndarray:
    """Fraction of rainfall converted to runoff, Q / P. NaN where invalid."""
    if rainfall_mm <= 0:
        return np.full(cn.shape, np.nan, dtype=np.float32)
    return runoff_depth(rainfall_mm, cn) / rainfall_mm


# ──────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────
def describe_response(cn_value: float, depths=(50, 100, 150, 200, 300, 400)) -> str:
    """Human-readable runoff response for a single curve number."""
    cn = np.array([[cn_value]], dtype=np.float32)
    rows = [f"CN={cn_value:.0f} (AMC {HYDRO.amc}, lambda={HYDRO.initial_abstraction_ratio})"]
    s = float(potential_retention(cn)[0, 0])
    rows.append(f"  S = {s:.1f} mm, Ia = {HYDRO.initial_abstraction_ratio * s:.1f} mm")
    for p in depths:
        q = float(runoff_depth(p, cn)[0, 0])
        rows.append(f"  P={p:>4.0f} mm -> Q={q:>6.1f} mm  (C={q / p:.2f})")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from config import LULC_CLASS_NAMES

    print("SCS Curve Number runoff response by land cover")
    print("=" * 60)
    for cls, cn_ii in sorted(HYDRO.curve_numbers.items()):
        name = LULC_CLASS_NAMES.get(cls, f"class {cls}")
        cn_adj = float(adjust_cn_for_amc(np.array([[cn_ii]], dtype=np.float32), HYDRO.amc)[0, 0])
        print(f"\n{name} (CN_II={cn_ii:.0f} -> CN_{HYDRO.amc}={cn_adj:.1f})")
        print(describe_response(cn_adj))
