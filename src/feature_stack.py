"""
Feature stack access for the flood susceptibility model.

Every conditioning factor lives in `data_aligned/` on one common grid (see
align_data.py). This module is the single place that knows how to turn those
rasters into a model design matrix, so training and prediction cannot drift
apart in feature order, nodata handling, or derived-feature definitions.

The full grid is 5690 x 7374 (42M pixels). A 13-feature float32 stack for the
whole grid is ~2.2 GB, so nothing here ever materialises one: training reads
one raster at a time and keeps only sampled pixels, and prediction works in
horizontal stripes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from config import (
    ALIGNED_DIR,
    PERMANENT_WATER_CLASS,
    RASTER,
    SUSCEPTIBILITY_FEATURES,
)
from hydrology import curve_number_from_lulc

LOGGER = logging.getLogger("geoai_flood")

NODATA = RASTER.nodata_value

# Features read straight off disk. `curve_number` is derived from lulc.
_RASTER_FEATURES = [f for f in SUSCEPTIBILITY_FEATURES if f != "curve_number"]


class MissingFeatureError(FileNotFoundError):
    """Raised when a required aligned raster is absent."""


# ──────────────────────────────────────────────
# Raster access
# ──────────────────────────────────────────────
def feature_path(name: str, aligned_dir: Optional[Path] = None) -> Path:
    d = aligned_dir or ALIGNED_DIR
    return d / f"{name}_aligned.tif"


def grid_profile(aligned_dir: Optional[Path] = None) -> dict:
    """Profile of the master grid, taken from the LULC raster."""
    path = feature_path("lulc", aligned_dir)
    if not path.exists():
        raise MissingFeatureError(
            f"Aligned data not found at {path}. Run `python align_data.py` first."
        )
    with rasterio.open(path) as src:
        profile = src.profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=NODATA, compress="lzw")
    return profile


def pixel_area_km2(aligned_dir: Optional[Path] = None) -> float:
    """True pixel area from the raster transform, not an assumed cell size."""
    with rasterio.open(feature_path("lulc", aligned_dir)) as src:
        return abs(src.res[0] * src.res[1]) / 1e6


def read_raster(
    name: str,
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read one aligned raster.

    Returns (values, valid_mask). Values at invalid pixels are NaN so that a
    caller who forgets the mask gets NaN rather than -9999 silently entering
    the arithmetic.
    """
    path = feature_path(name, aligned_dir)
    if not path.exists():
        raise MissingFeatureError(
            f"Missing aligned raster: {path}. Run `python align_data.py` first."
        )
    with rasterio.open(path) as src:
        data = src.read(1, window=window).astype(np.float32)
        nd = src.nodata if src.nodata is not None else NODATA
    valid = np.isfinite(data) & (data != np.float32(nd))
    data[~valid] = np.nan
    return data, valid


def domain_mask(
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Pixels the susceptibility model is defined over.

    That is the district footprint minus permanent water. Sentinel-1 cannot
    distinguish permanent water from flood water, so leaving the backwaters in
    makes 80% of the positive labels "this is a lake" -- the model then scores
    beautifully on AUC while being useless for flood warning.
    """
    lulc, valid = read_raster("lulc", window, aligned_dir)
    return valid & (np.round(lulc) != PERMANENT_WATER_CLASS)


def permanent_water_mask(
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> np.ndarray:
    lulc, valid = read_raster("lulc", window, aligned_dir)
    return valid & (np.round(lulc) == PERMANENT_WATER_CLASS)


def flood_labels(
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary flood inventory (1 = observed flooded) and its valid mask."""
    gt, valid = read_raster("ground_truth", window, aligned_dir)
    return (gt > 0.5), valid


# ──────────────────────────────────────────────
# Derived features
# ──────────────────────────────────────────────
def compute_curve_number(
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Curve number grid derived from LULC (see hydrology.py)."""
    lulc, valid = read_raster("lulc", window, aligned_dir)
    cn = curve_number_from_lulc(lulc, valid)
    return cn, np.isfinite(cn)


def read_feature(
    name: str,
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read a model feature by name, dispatching derived features."""
    if name == "curve_number":
        return compute_curve_number(window, aligned_dir)
    return read_raster(name, window, aligned_dir)


# ──────────────────────────────────────────────
# Design matrix
# ──────────────────────────────────────────────
def build_matrix(
    window: Optional[Window] = None,
    aligned_dir: Optional[Path] = None,
    features: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Build the design matrix for every model-domain pixel in `window`.

    Returns (X, flat_index, shape) where:
        X          - (n_pixels, n_features) float32
        flat_index - flat indices into the window's (H, W) grid
        shape      - the window's (H, W)
    """
    features = features or SUSCEPTIBILITY_FEATURES

    mask = domain_mask(window, aligned_dir)
    shape = mask.shape

    # Read each raster once and keep it. The previous form read every feature
    # twice per window -- once to build the mask, once to fill the matrix --
    # which doubled I/O across the whole grid for no benefit.
    #
    # The cost is that all `len(features)` stripe arrays are alive at once, so
    # peak memory is stripe_rows * width * n_features * 4 bytes. Keep the
    # caller's stripe height in mind: 512 rows over the 7374-wide grid pushed
    # this to ~2 GB private and made the machine page, which is why
    # predict_surface defaults to 256.
    columns = []
    for name in features:
        values, valid = read_feature(name, window, aligned_dir)
        mask &= valid
        columns.append(values.ravel())

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return np.empty((0, len(features)), dtype=np.float32), idx, shape

    X = np.empty((idx.size, len(features)), dtype=np.float32)
    for j, column in enumerate(columns):
        X[:, j] = column[idx]
        columns[j] = None  # release the stripe array as soon as it is consumed

    return X, idx, shape


def iter_stripes(
    height: int,
    width: int,
    stripe_rows: int = 512,
) -> Iterator[Window]:
    """Yield full-width horizontal windows covering the grid."""
    for row in range(0, height, stripe_rows):
        rows = min(stripe_rows, height - row)
        yield Window(col_off=0, row_off=row, width=width, height=rows)


# ──────────────────────────────────────────────
# Sampling for training
# ──────────────────────────────────────────────
def sample_domain_points(
    n: int = 400_000,
    seed: int = 7,
    aligned_dir: Optional[Path] = None,
    with_labels: bool = False,
):
    """
    Draw a uniform random sample of the model domain.

    Used for two things that both require an *honestly representative* sample
    rather than the training set:

      * the prior offset, because the training set is deliberately balanced
        and elevation-stratified, so the closed-form case-control correction
        does not apply to it;
      * conformal calibration, whose coverage guarantee only transfers to the
        district if the calibration points are exchangeable with district
        pixels.

    Returns X, or (X, y, row, col) when `with_labels` is set.
    """
    rng = np.random.default_rng(seed)

    domain = domain_mask(aligned_dir=aligned_dir)
    width = domain.shape[1]
    idx = np.flatnonzero(domain.ravel())
    take = min(n, idx.size)
    idx = np.sort(rng.choice(idx, size=take, replace=False))

    X = np.empty((idx.size, len(SUSCEPTIBILITY_FEATURES)), dtype=np.float32)
    for j, name in enumerate(SUSCEPTIBILITY_FEATURES):
        values, _ = read_feature(name, aligned_dir=aligned_dir)
        X[:, j] = values.ravel()[idx]
        del values

    complete = np.isfinite(X).all(axis=1)
    LOGGER.info(
        "  domain sample: %d of %d drawn pixels complete",
        int(complete.sum()), idx.size,
    )

    if not with_labels:
        return X[complete]

    flood, gt_valid = flood_labels(aligned_dir=aligned_dir)
    y = (flood.ravel()[idx] & gt_valid.ravel()[idx]).astype(np.int8)
    kept = idx[complete]
    return (
        X[complete],
        y[complete],
        (kept // width).astype(np.int32),
        (kept % width).astype(np.int32),
    )


def _allocate_stratified(
    members_by_stratum: list,
    total: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Draw `total` samples spread as evenly as possible across strata.

    Strata smaller than their equal share contribute everything they have and
    the remainder is re-offered to the strata that still have spare capacity,
    repeating until the quota is met or the pool is exhausted.
    """
    remaining_capacity = [m.size for m in members_by_stratum]
    quota = [0] * len(members_by_stratum)
    outstanding = total

    while outstanding > 0:
        open_strata = [i for i, c in enumerate(remaining_capacity) if c > 0]
        if not open_strata:
            break
        share = max(1, outstanding // len(open_strata))
        progressed = False
        for i in open_strata:
            if outstanding <= 0:
                break
            take = min(share, remaining_capacity[i], outstanding)
            if take <= 0:
                continue
            quota[i] += take
            remaining_capacity[i] -= take
            outstanding -= take
            progressed = True
        if not progressed:
            break

    chosen = [
        rng.choice(members, size=q, replace=False)
        for members, q in zip(members_by_stratum, quota)
        if q > 0
    ]
    if not chosen:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(chosen)


def sample_training_points(
    n_per_class: int = 60_000,
    absence_buffer_px: int = 5,
    n_strata: int = 10,
    strata_feature: str = "dem",
    seed: int = 42,
    aligned_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Draw a presence/pseudo-absence training set.

    Presence points are the observed 2018 flood pixels inside the model domain
    (permanent water already excluded).

    Pseudo-absences are NOT drawn uniformly at random. Two corrections from
    the recent literature are applied:

      1. A buffer is cut around every presence pixel, so absences are never
         drawn from the uncertain transition zone at the flood margin. This
         addresses the "true negative bias" that dominates presence-only
         hazard modelling.
      2. Absences are stratified across deciles of `strata_feature` (elevation
         by default), so they span the full terrain gradient instead of
         collapsing onto high ground. Without this the classifier only has to
         learn "high ground is dry", which inflates AUC and produces a model
         that cannot rank low-lying pixels against each other -- exactly the
         ranking a planner needs. Elevation is the default because it is the
         dominant predictor here, and stratifying on the dominant predictor
         is what forces the model to learn everything else.

    The draw is balanced 1:1, so `domain_prevalence` is returned alongside it
    for the case-control prior correction the caller must apply.

    Returns a dict with keys: X, y, row, col, features, domain_prevalence,
    domain_pixels, presence_pixels.
    """
    from scipy.ndimage import binary_dilation

    rng = np.random.default_rng(seed)

    LOGGER.info("Sampling training points (seed=%d)", seed)
    domain = domain_mask(aligned_dir=aligned_dir)
    flood, gt_valid = flood_labels(aligned_dir=aligned_dir)

    presence = domain & gt_valid & flood
    LOGGER.info("  presence pixels in domain: %d", presence.sum())
    if presence.sum() == 0:
        raise ValueError("No presence pixels found; check the flood inventory.")

    # 1. Buffer out the flood margin.
    if absence_buffer_px > 0:
        struct = np.ones((3, 3), dtype=bool)
        buffered = binary_dilation(presence, structure=struct, iterations=absence_buffer_px)
    else:
        buffered = presence
    absence_pool = domain & gt_valid & ~buffered
    LOGGER.info(
        "  absence pool after %d-px buffer: %d", absence_buffer_px, absence_pool.sum()
    )

    # 2. Stratify absences across strata of the conditioning variable.
    #
    # Strata are built on percentile edges with duplicates removed, because a
    # zero-inflated variable (HAND is mean 0.17 m here) collapses percentile
    # edges and a naive digitize dumps almost everything into one bin. Any
    # shortfall in a thin stratum is redistributed to the strata that still
    # have capacity -- otherwise the requested absence count is silently
    # under-delivered and the classes end up unbalanced.
    #
    # Strata are matched to the *presence* distribution, not the pool
    # distribution: absences are drawn to span the same range of the
    # stratifying variable that the flood pixels occupy, so the model cannot
    # win by separating "coastal lowland" from "inland hills".
    strat, strat_valid = read_raster(strata_feature, aligned_dir=aligned_dir)
    absence_pool &= strat_valid

    pool_idx = np.flatnonzero(absence_pool.ravel())
    pool_values = strat.ravel()[pool_idx]

    presence_values = strat.ravel()[np.flatnonzero(presence & strat_valid)]
    reference = presence_values if presence_values.size else pool_values

    edges = np.unique(np.percentile(reference, np.linspace(0, 100, n_strata + 1)))
    if edges.size < 2:
        strata = np.zeros(pool_idx.size, dtype=np.int64)
        n_effective = 1
    else:
        strata = np.digitize(pool_values, edges[1:-1])
        n_effective = int(strata.max()) + 1
    if n_effective < n_strata:
        LOGGER.info(
            "  %s supports only %d distinct strata (requested %d)",
            strata_feature,
            n_effective,
            n_strata,
        )

    members_by_stratum = [pool_idx[strata == s] for s in range(n_effective)]
    absence_idx = _allocate_stratified(members_by_stratum, n_per_class, rng)
    LOGGER.info(
        "  absences drawn: %d across %d %s strata (matched to presence range %.1f-%.1f)",
        absence_idx.size,
        n_effective,
        strata_feature,
        float(np.percentile(reference, 1)),
        float(np.percentile(reference, 99)),
    )

    presence_idx = np.flatnonzero(presence.ravel())
    # Keep the classes balanced: an unbalanced draw shifts the base rate and
    # makes the calibrated probabilities mean something other than the
    # inventory frequency.
    take = min(n_per_class, presence_idx.size, absence_idx.size)
    presence_idx = rng.choice(presence_idx, size=take, replace=False)
    if absence_idx.size > take:
        absence_idx = rng.choice(absence_idx, size=take, replace=False)
    LOGGER.info("  presences drawn: %d (balanced 1:1)", presence_idx.size)

    all_idx = np.concatenate([presence_idx, absence_idx])
    y = np.concatenate(
        [np.ones(presence_idx.size, dtype=np.int8), np.zeros(absence_idx.size, dtype=np.int8)]
    )

    # Read features once each, keeping only the sampled pixels.
    width = domain.shape[1]
    X = np.empty((all_idx.size, len(SUSCEPTIBILITY_FEATURES)), dtype=np.float32)
    for j, name in enumerate(SUSCEPTIBILITY_FEATURES):
        values, valid = read_feature(name, aligned_dir=aligned_dir)
        X[:, j] = values.ravel()[all_idx]
        LOGGER.info("  read feature %-12s (%d/%d)", name, j + 1, len(SUSCEPTIBILITY_FEATURES))
        del values, valid

    # Drop any sample with a missing feature.
    complete = np.isfinite(X).all(axis=1)
    LOGGER.info("  complete samples: %d / %d", complete.sum(), complete.size)

    # True prevalence over the model domain. The training set is deliberately
    # balanced 1:1, so probabilities learned from it are calibrated to a 50%
    # base rate rather than the real one. The caller needs this number to
    # apply the case-control prior correction, without which "30% chance of
    # flooding" corresponds to no real-world frequency at all.
    domain_px = int((domain & gt_valid).sum())
    presence_px = int(presence.sum())
    prevalence = presence_px / max(domain_px, 1)
    LOGGER.info(
        "  domain prevalence: %d / %d = %.5f (%.3f%%)",
        presence_px, domain_px, prevalence, 100 * prevalence,
    )

    return {
        "X": X[complete],
        "y": y[complete],
        "row": (all_idx[complete] // width).astype(np.int32),
        "col": (all_idx[complete] % width).astype(np.int32),
        "features": np.array(SUSCEPTIBILITY_FEATURES),
        "domain_prevalence": prevalence,
        "domain_pixels": domain_px,
        "presence_pixels": presence_px,
    }
