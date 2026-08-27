"""
Test the waterlogging layers against documented waterlogging locations.

The problem this addresses
-------------------------
The title claims "urban waterlogging prevention". The calibrated flood layer
demonstrably does not predict urban waterlogging -- its hot zone has median
urban fraction 0.0 against a district median of 1.0 -- and the pluvial index
built to address that has never been tested, because there are no waterlogging
records for this district and the municipal ones need a data request.

This module builds the best test that free sources allow: a small set of
locations *documented in public reporting* as recurrent waterlogging points,
geocoded through Nominatim, and scored against a matched urban background.

What this can and cannot establish
----------------------------------
It can distinguish "no skill" from "clear skill". With ~10 confirmed points the
95% bootstrap interval on AUC is roughly +/- 0.15, so it cannot rank two
similar models, and it is far too small to train on.

Two biases to keep in mind, both of which inflate apparent skill:

  * Reporting bias. Journalists cover waterlogging where it disrupts traffic
    and commerce, so the sample favours arterial junctions in the city centre
    over residential streets that flood just as often.
  * Geocoding error. Nominatim resolves "Jos Junction, Kochi" to a point,
    while the water sits in a specific dip nearby. At 30-100 m resolution an
    error of one or two hundred metres matters, which is why each point is
    scored as the maximum within a small radius rather than at one pixel.

Sources are recorded per point so every label can be audited or discarded.

Run:  python src/waterlogging_validation.py
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MODELS_DIR, setup_logging

LOGGER = logging.getLogger("geoai_flood")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "GeoAI-Flood-Ernakulam/1.0 (academic flood modelling)"

CACHE = MODELS_DIR / "waterlogging_points.json"

#: Locations reported publicly as recurrent waterlogging points in Kochi.
#: `source` records where the claim comes from so it can be checked or dropped.
#: These are documented *reports*, not a survey, and the set is small.
#: Each entry carries several query variants: Nominatim knows Indian
#: neighbourhood names far better than colloquial junction names, so
#: "Jos Junction" misses while the road or landmark beside it resolves.
DOCUMENTED_HOTSPOTS: List[Dict] = [
    # Named in reporting on Operation Breakthrough phase 4, the Irrigation
    # Department's anti-flooding programme for the city.
    {
        "name": "Ernakulam South railway station",
        "source": "Operation Breakthrough phase 4",
        "queries": [
            "Ernakulam Junction railway station, Ernakulam",
            "Ernakulam Junction, Kerala",
            "Ernakulam South, Kochi",
        ],
    },
    {
        "name": "Jos Junction",
        "source": "Operation Breakthrough phase 4",
        "queries": [
            "Jos Junction, Ernakulam",
            "MG Road, Ernakulam, Kerala",
            "Padma Junction, Kochi",
        ],
    },
    {
        "name": "Durbar Hall ground",
        "source": "Operation Breakthrough phase 4",
        "queries": ["Durbar Hall Road, Ernakulam", "Durbar Hall Art Centre, Kochi"],
    },
    {
        "name": "Rajendra Maidan",
        "source": "Operation Breakthrough phase 4",
        "queries": ["Rajendra Maidan, Ernakulam", "Ernakulam Town Hall, Kochi"],
    },
    {
        "name": "High Court Junction",
        "source": "Operation Breakthrough phase 4",
        "queries": ["Kerala High Court, Ernakulam", "High Court of Kerala, Kochi"],
    },
    {
        "name": "Kammattipadam",
        "source": "Operation Breakthrough phase 4",
        "queries": ["Kammattipadam, Ernakulam", "Karanakodam, Kochi"],
    },
    # Named in press reporting of the 2024 monsoon flooding in the city.
    {
        "name": "Kalamassery",
        "source": "press reporting, 2024 monsoon",
        "queries": ["Kalamassery, Ernakulam, Kerala"],
    },
    {
        "name": "Kaloor",
        "source": "press reporting, 2024 monsoon",
        "queries": ["Kaloor, Ernakulam, Kerala"],
    },
    {
        "name": "Edappally",
        "source": "press reporting, 2024 monsoon",
        "queries": ["Edappally, Ernakulam, Kerala"],
    },
    # Kochi's primary drainage arteries; their chronic overflow is the subject
    # of the Operation Breakthrough programme.
    {
        "name": "Mullassery Canal",
        "source": "Operation Breakthrough",
        "queries": ["Mullassery Canal, Ernakulam", "Mullassery Canal Road, Kochi"],
    },
    {
        "name": "Thevara-Perandoor Canal",
        "source": "Operation Breakthrough",
        "queries": [
            "Thevara Perandoor Canal, Kochi",
            "Perandoor Canal, Ernakulam",
            "Thevara, Ernakulam, Kerala",
        ],
    },
    {
        "name": "Kadavanthra",
        "source": "Operation Breakthrough (TP Canal reach)",
        "queries": ["Kadavanthra, Ernakulam, Kerala"],
    },
    {
        "name": "Panampilly Nagar",
        "source": "Operation Breakthrough (TP Canal reach)",
        "queries": ["Panampilly Nagar, Ernakulam, Kerala"],
    },
    {
        "name": "Vyttila",
        "source": "press reporting, recurrent junction flooding",
        "queries": ["Vyttila, Ernakulam, Kerala"],
    },
    {
        "name": "Palarivattom",
        "source": "press reporting, recurrent junction flooding",
        "queries": ["Palarivattom, Ernakulam, Kerala"],
    },
]

#: Score each point as the maximum within this radius, to absorb geocoding
#: error without letting a single mis-placed pixel decide the result.
SEARCH_RADIUS_M = 150.0


# ──────────────────────────────────────────────
# Geocoding
# ──────────────────────────────────────────────
def geocode(query: str, timeout: int = 30) -> Optional[Tuple[float, float]]:
    """Resolve a place name to (lat, lon) via Nominatim. Returns None on miss."""
    url = f"{NOMINATIM}?{urllib.parse.urlencode({'q': query, 'format': 'json', 'limit': 1})}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            hits = json.load(r)
    except Exception as exc:
        LOGGER.warning("  geocode failed for %r: %s", query, exc)
        return None
    if not hits:
        return None
    return float(hits[0]["lat"]), float(hits[0]["lon"])


def resolve_hotspots(use_cache: bool = True) -> List[Dict]:
    """Geocode the documented hotspots, caching the result."""
    if use_cache and CACHE.exists():
        points = json.loads(CACHE.read_text(encoding="utf-8"))
        LOGGER.info("Loaded %d cached hotspot locations", len(points))
        return points

    points = []
    for entry in DOCUMENTED_HOTSPOTS:
        coords, used = None, None
        for query in entry["queries"]:
            coords = geocode(query)
            time.sleep(1.1)  # Nominatim asks for at most one request per second
            if coords is not None:
                used = query
                break
        if coords is None:
            LOGGER.warning("  no match: %s", entry["name"])
            continue
        lat, lon = coords
        points.append(
            {
                "name": entry["name"],
                "source": entry["source"],
                "resolved_query": used,
                "lat": lat,
                "lon": lon,
            }
        )
        LOGGER.info("  %-32s %.5f, %.5f  via %r", entry["name"], lat, lon, used)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(points, indent=2), encoding="utf-8")
    LOGGER.info("Resolved %d/%d hotspots", len(points), len(DOCUMENTED_HOTSPOTS))
    return points


# ──────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────
def sample_at(grid, surface: np.ndarray, lat: float, lon: float) -> float:
    """Maximum value within SEARCH_RADIUS_M of a point. NaN if outside."""
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", grid.crs, always_xy=True).transform(lon, lat)
    col, row = ~grid.transform * (x, y)
    row, col = int(row), int(col)

    radius = max(1, int(round(SEARCH_RADIUS_M / grid.cell_width_m)))
    h, w = surface.shape
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return float("nan")

    window = surface[r0:r1, c0:c1]
    return float(np.nanmax(window)) if np.isfinite(window).any() else float("nan")


def _load_on_grid(grid, name: str) -> np.ndarray:
    import rasterio
    from rasterio.enums import Resampling

    from config import ALIGNED_DIR

    with rasterio.open(ALIGNED_DIR / f"{name}_aligned.tif") as src:
        a = src.read(1, out_shape=grid.shape, resampling=Resampling.average).astype(np.float32)
        nd = src.nodata
    return np.where(np.isfinite(a) & (a != nd), a, np.nan)


def urban_background(
    grid,
    n: int = 4000,
    seed: int = 0,
    elevation_band: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """
    Random built-up pixels, as the comparison set.

    The contrast that matters is hotspot versus ordinary *urban* ground. Using
    the whole district as background would let elevation alone separate the
    classes and manufacture skill.

    `elevation_band` narrows the background to a height range, which is the
    real robustness check: every documented hotspot is a low-lying central
    junction, so a model that only knows "low ground floods" would score well
    against unrestricted urban background. If the skill survives an
    elevation-matched background, it is not just elevation.
    """
    urban = _load_on_grid(grid, "urban_mask")
    mask = np.isfinite(urban) & (urban > 0.5) & np.isfinite(grid.susceptibility)

    if elevation_band is not None:
        dem = _load_on_grid(grid, "dem")
        lo, hi = elevation_band
        mask &= np.isfinite(dem) & (dem >= lo) & (dem <= hi)

    idx = np.flatnonzero(mask.ravel())
    if idx.size == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(idx, size=min(n, idx.size), replace=False)


def hotspot_elevations(grid, points: List[Dict]) -> np.ndarray:
    """Elevation at each hotspot, for building a matched background."""
    dem = _load_on_grid(grid, "dem")
    out = []
    for p in points:
        v = sample_at(grid, dem, p["lat"], p["lon"])
        if np.isfinite(v):
            out.append(v)
    return np.asarray(out)


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """AUC with a bootstrap confidence interval, resampling the positives."""
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    s = np.concatenate([pos, neg])
    point = float(roc_auc_score(y, s))

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        p = rng.choice(pos, size=pos.size, replace=True)
        n = rng.choice(neg, size=neg.size, replace=True)
        yy = np.concatenate([np.ones(p.size), np.zeros(n.size)])
        ss = np.concatenate([p, n])
        if len(np.unique(yy)) < 2:
            continue
        draws.append(roc_auc_score(yy, ss))
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return point, float(lo), float(hi)


def evaluate(rainfall_mm: float = 150.0, use_cache: bool = True) -> Dict:
    """Score both layers at the documented hotspots against urban background."""
    import live_model

    grid = live_model.load()
    points = resolve_hotspots(use_cache)
    if not points:
        raise RuntimeError("No hotspots resolved; cannot evaluate")

    layers = {
        "fluvial_probability": live_model.fluvial_probability(grid, rainfall_mm),
        "pluvial_index": live_model.pluvial_index(grid, rainfall_mm),
    }

    # Two backgrounds: all urban ground, and urban ground at the same
    # elevations as the hotspots. The second is the one that matters.
    elev = hotspot_elevations(grid, points)
    band = (float(np.min(elev)), float(np.percentile(elev, 95))) if elev.size else None
    LOGGER.info(
        "Hotspot elevations: %.1f-%.1f m (median %.1f)",
        elev.min(),
        elev.max(),
        np.median(elev),
    )

    backgrounds = {
        "urban": urban_background(grid),
        "urban_elevation_matched": urban_background(grid, elevation_band=band),
    }
    for label, idx in backgrounds.items():
        LOGGER.info("Background '%s': %d pixels", label, idx.size)

    results = {
        "rainfall_mm": rainfall_mm,
        "n_hotspots_documented": len(DOCUMENTED_HOTSPOTS),
        "hotspot_elevation_band_m": band,
    }

    for layer_name, surface in layers.items():
        pos_vals: list = []
        kept = []
        for p in points:
            v = sample_at(grid, surface, p["lat"], p["lon"])
            if np.isfinite(v):
                pos_vals.append(v)
                kept.append(p["name"])
        pos = np.asarray(pos_vals)
        entry: Dict[str, object] = {
            "n_hotspots_scored": int(pos.size),
            "scored_points": kept,
            "hotspot_median": float(np.median(pos)) if pos.size else None,
        }

        for bg_label, bg_idx in backgrounds.items():
            neg = surface.ravel()[bg_idx]
            neg = neg[np.isfinite(neg)]
            if pos.size < 3 or neg.size < 50:
                LOGGER.warning(
                    "%s vs %s: too few samples (%d pos, %d neg)",
                    layer_name,
                    bg_label,
                    pos.size,
                    neg.size,
                )
                continue

            auc, lo, hi = bootstrap_auc(pos, neg)
            entry[bg_label] = {
                "n_background": int(neg.size),
                "background_median": float(np.median(neg)),
                "auc": auc,
                "auc_ci95": [lo, hi],
                "beats_chance": bool(lo > 0.5),
            }
            LOGGER.info(
                "%-20s vs %-24s AUC %.3f (95%% CI %.3f-%.3f)  "
                "hotspot %.4f vs background %.4f  %s",
                layer_name,
                bg_label,
                auc,
                lo,
                hi,
                np.median(pos),
                np.median(neg),
                "SKILL" if lo > 0.5 else "CHANCE",
            )

        results[layer_name] = entry

    out = MODELS_DIR / "waterlogging_validation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)
    return results


def main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Validate against documented hotspots")
    parser.add_argument("--rainfall", type=float, default=150.0)
    parser.add_argument("--refresh", action="store_true", help="Re-geocode, ignore cache")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    evaluate(args.rainfall, use_cache=not args.refresh)


if __name__ == "__main__":  # pragma: no cover
    main()
