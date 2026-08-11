"""
Derive the reference event depth from ERA5 reanalysis.

`RAINFALL.reference_event_mm` anchors the entire rainfall response: the hazard
model reduces exactly to the fitted susceptibility at that depth, so the
constant decides which storm the observed 2018 flood extent is taken to
represent. It was previously a guess (400 mm). This script replaces the guess
with reanalysis, and exists so the number can be re-derived and audited rather
than trusted.

Run:  python src/reference_rainfall.py
      python src/reference_rainfall.py --event 2019

Caveats worth carrying:

  * ERA5 is a reanalysis at roughly 9-31 km. It systematically under-resolves
    orographic extremes, so these depths are more likely low than high over
    the Western Ghats flank.
  * The 2018 flooding in Ernakulam was not driven by local rainfall alone -
    reservoir releases into the Periyar contributed substantially. No
    rain-gauge product captures that, and the SCS-CN formulation cannot
    represent it. The reference depth is therefore a proxy for total forcing,
    not a measured storm.
  * Pair the window with the antecedent moisture condition deliberately.
    AMC III already encodes a wet antecedent 5 days, so the storm depth
    should be the burst itself (3-day max), not a 5- or 7-day accumulation,
    or the antecedent wetness is counted twice.
"""
from __future__ import annotations

import argparse
import json
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

import numpy as np

from config import MODELS_DIR, setup_logging

LOGGER = logging.getLogger("geoai_flood")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

#: WGS84 footprint of the master grid.
DISTRICT_BOX = {"lat": (9.79, 10.30), "lon": (76.17, 76.84)}

#: Known Kerala flood events with Sentinel-1 era coverage.
EVENTS: Dict[str, Tuple[str, str]] = {
    "2018": ("2018-08-01", "2018-08-31"),
    "2019": ("2019-08-01", "2019-08-31"),
    "2021": ("2021-10-01", "2021-10-31"),
}

#: Storm window, in days. Three pairs correctly with AMC III (see module docs).
STORM_WINDOW_DAYS = 3


def sample_grid(box: Dict, n: int = 3) -> List[Tuple[float, float]]:
    lats = np.linspace(box["lat"][0], box["lat"][1], n)
    lons = np.linspace(box["lon"][0], box["lon"][1], n)
    return [(float(la), float(lo)) for la in lats for lo in lons]


def fetch_daily(points, start: str, end: str, timeout: int = 90):
    """Daily precipitation for each point. Returns (dates, array[points, days])."""
    query = urllib.parse.urlencode({
        "latitude": ",".join(f"{la:.4f}" for la, _ in points),
        "longitude": ",".join(f"{lo:.4f}" for _, lo in points),
        "start_date": start,
        "end_date": end,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    })
    with urllib.request.urlopen(f"{ARCHIVE_URL}?{query}", timeout=timeout) as response:
        payload = json.load(response)

    if isinstance(payload, dict):
        payload = [payload]

    dates = payload[0]["daily"]["time"]
    series = np.array([
        [v if v is not None else 0.0 for v in p["daily"]["precipitation_sum"]]
        for p in payload
    ], dtype=float)
    return dates, series


def max_accumulation(daily: np.ndarray, window: int):
    """Largest `window`-day running total and the index it starts at."""
    if daily.size < window:
        return float("nan"), None
    sums = np.convolve(daily, np.ones(window), mode="valid")
    i = int(np.argmax(sums))
    return float(sums[i]), i


def analyse(event: str = "2018", n_grid: int = 3) -> Dict:
    if event not in EVENTS:
        raise ValueError(f"Unknown event {event!r}. Known: {sorted(EVENTS)}")

    start, end = EVENTS[event]
    points = sample_grid(DISTRICT_BOX, n_grid)
    dates, series = fetch_daily(points, start, end)
    mean_daily = series.mean(axis=0)

    windows = {}
    for w in (1, 2, 3, 5, 7):
        total, i = max_accumulation(mean_daily, w)
        windows[f"max_{w}day_mm"] = round(total, 1)
        windows[f"max_{w}day_window"] = (
            f"{dates[i]}..{dates[i + w - 1]}" if i is not None else None
        )

    peak = int(np.argmax(mean_daily))
    result = {
        "event": event,
        "source": "ERA5 reanalysis via Open-Meteo archive API",
        "n_grid_points": len(points),
        "bbox": DISTRICT_BOX,
        "period": [start, end],
        "wettest_day": dates[peak],
        "wettest_day_district_mean_mm": round(float(mean_daily[peak]), 1),
        "wettest_day_point_max_mm": round(float(series[:, peak].max()), 1),
        "month_total_mm": round(float(mean_daily.sum()), 1),
        "storm_window_days": STORM_WINDOW_DAYS,
        "reference_event_mm": windows[f"max_{STORM_WINDOW_DAYS}day_mm"],
        **windows,
    }
    return result


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Derive the reference event depth")
    parser.add_argument("--event", default="2018", choices=sorted(EVENTS))
    parser.add_argument("--grid", type=int, default=3, help="Sampling grid side")
    parser.add_argument("--all", action="store_true", help="Report every known event")
    args = parser.parse_args()

    setup_logging(logging.INFO)

    events = sorted(EVENTS) if args.all else [args.event]
    results = {}
    for event in events:
        r = analyse(event, args.grid)
        results[event] = r
        LOGGER.info("=" * 62)
        LOGGER.info("%s  (%d ERA5 points over the district)", event, r["n_grid_points"])
        LOGGER.info("=" * 62)
        LOGGER.info(
            "  wettest day   %s  %.1f mm district mean (point max %.1f)",
            r["wettest_day"], r["wettest_day_district_mean_mm"], r["wettest_day_point_max_mm"],
        )
        for w in (1, 2, 3, 5, 7):
            LOGGER.info(
                "  max %d-day     %6.1f mm   (%s)",
                w, r[f"max_{w}day_mm"], r[f"max_{w}day_window"],
            )
        LOGGER.info("  month total   %6.1f mm", r["month_total_mm"])
        LOGGER.info(
            "  -> reference depth (%d-day storm, pairs with AMC III): %.1f mm",
            STORM_WINDOW_DAYS, r["reference_event_mm"],
        )

    out = MODELS_DIR / "reference_rainfall.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", out)


if __name__ == "__main__":  # pragma: no cover
    main()
