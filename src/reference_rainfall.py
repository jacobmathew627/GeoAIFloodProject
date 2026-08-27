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

#: Known Kerala flood events with satellite-derived inundation coverage
#: (Sentinel-1 for 2018 and 2026, NDEM for 2018-2021). Month windows match the
#: month containing each event's acquisition dates (src/ndem_labels.py EVENTS
#: for NDEM, src/acquire_flood_event.py EVENTS for the Sentinel-1 ones).
EVENTS: Dict[str, Tuple[str, str]] = {
    "2018": ("2018-08-01", "2018-08-31"),
    "2019": ("2019-08-01", "2019-08-31"),
    "2020": ("2020-08-01", "2020-08-31"),
    "2021": ("2021-10-01", "2021-10-31"),
    # 2026 Kerala floods: onset late July, peak ~1 Aug. Window starts mid-July
    # rather than the 1st of the month so a late-July onset is not clipped.
    "2026": ("2026-07-15", "2026-08-15"),
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


# ──────────────────────────────────────────────
# IMD gauge-based gridded rainfall (authoritative)
# ──────────────────────────────────────────────
def fetch_imd_daily(year: int, cache_dir: Optional[str] = None):
    """
    District-mean daily rainfall from the IMD 0.25 deg gridded product.

    This is the official Indian gauge-based analysis and is the reference this
    project uses. ERA5 is retained only as a cross-check: measured against IMD
    it under-reads the 3-day maximum by 1.34x in 2018, 2.09x in 2019 and 1.49x
    in 2021 -- a reanalysis smooths orographic extremes, and the Western Ghats
    flank is exactly where that hurts.

    Needs the `imdlib` package and network access.
    """
    import tempfile

    import imdlib as imd

    cache_dir = cache_dir or tempfile.mkdtemp()
    data = imd.get_data("rain", year, year, fn_format="yearwise", file_dir=cache_dir)
    ds = data.get_xarray().sel(
        lat=slice(DISTRICT_BOX["lat"][0], DISTRICT_BOX["lat"][1]),
        lon=slice(DISTRICT_BOX["lon"][0], DISTRICT_BOX["lon"][1]),
    )
    values = ds["rain"].values.astype(float)
    # IMD encodes missing data as a negative value, not NaN.
    values = np.where(values < 0, np.nan, values)
    daily = np.nanmean(values, axis=(1, 2))
    times = [str(t)[:10] for t in ds["time"].values]
    return times, daily


def analyse_imd(event: str = "2018") -> Dict:
    """Reference-event statistics from IMD gridded rainfall."""
    if event not in EVENTS:
        raise ValueError(f"Unknown event {event!r}. Known: {sorted(EVENTS)}")

    start, end = EVENTS[event]
    times, daily = fetch_imd_daily(int(event))

    keep = [i for i, t in enumerate(times) if start <= t <= end]
    sub = daily[keep]
    sub_times = [times[i] for i in keep]

    windows = {}
    for w in (1, 2, 3, 5, 7):
        total, i = max_accumulation(sub, w)
        windows[f"max_{w}day_mm"] = round(total, 1)
        windows[f"max_{w}day_window"] = (
            f"{sub_times[i]}..{sub_times[i + w - 1]}" if i is not None else None
        )

    peak = int(np.nanargmax(sub))
    return {
        "event": event,
        "source": "IMD 0.25 deg gauge-based gridded rainfall (imdlib)",
        "bbox": DISTRICT_BOX,
        "period": [start, end],
        "wettest_day": sub_times[peak],
        "wettest_day_district_mean_mm": round(float(sub[peak]), 1),
        "month_total_mm": round(float(np.nansum(sub)), 1),
        "storm_window_days": STORM_WINDOW_DAYS,
        "reference_event_mm": windows[f"max_{STORM_WINDOW_DAYS}day_mm"],
        **windows,
    }


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


def merge_results(existing: Dict, new: Dict) -> Dict:
    """
    Merge freshly computed event results into whatever is already on disk.

    A single `--event 2020` run used to overwrite the whole
    `reference_rainfall.json`, silently destroying the cached 2018/2019/2021
    results that fit_beta.py and the README both read from -- discovered by
    doing exactly that. `new` always wins for keys it contains (a rerun of an
    event should update it), but keys `new` does not touch are preserved.
    """
    return {**existing, **new}


def _load_existing(path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("could not read existing %s (%s); starting fresh", path, exc)
        return {}


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Derive the reference event depth")
    parser.add_argument("--event", default="2018", choices=sorted(EVENTS))
    parser.add_argument("--grid", type=int, default=3, help="Sampling grid side")
    parser.add_argument("--all", action="store_true", help="Report every known event")
    parser.add_argument(
        "--source", default="imd", choices=["imd", "era5"],
        help="imd = official gauge analysis (default); era5 = reanalysis cross-check",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)

    events = sorted(EVENTS) if args.all else [args.event]
    out = MODELS_DIR / "reference_rainfall.json"
    new_results = {}

    for event in events:
        if args.source == "imd":
            r = analyse_imd(event)
            try:
                era5 = analyse(event, args.grid)
                r["era5_cross_check"] = {
                    "max_3day_mm": era5["max_3day_mm"],
                    "ratio_imd_over_era5": round(
                        r["max_3day_mm"] / max(era5["max_3day_mm"], 1e-6), 2
                    ),
                }
            except Exception as exc:  # pragma: no cover - network path
                LOGGER.warning("ERA5 cross-check unavailable: %s", exc)
        else:
            r = analyse(event, args.grid)
        new_results[event] = r
        LOGGER.info("=" * 62)
        LOGGER.info("%s  -- %s", event, r["source"])
        LOGGER.info("=" * 62)
        LOGGER.info(
            "  wettest day   %s  %.1f mm district mean",
            r["wettest_day"], r["wettest_day_district_mean_mm"],
        )
        for w in (1, 2, 3, 5, 7):
            LOGGER.info(
                "  max %d-day     %6.1f mm   (%s)",
                w, r[f"max_{w}day_mm"], r[f"max_{w}day_window"],
            )
        LOGGER.info("  month total   %6.1f mm", r["month_total_mm"])
        if "era5_cross_check" in r:
            LOGGER.info(
                "  ERA5 cross-check: %.1f mm 3-day -> IMD is %.2fx higher",
                r["era5_cross_check"]["max_3day_mm"],
                r["era5_cross_check"]["ratio_imd_over_era5"],
            )
        LOGGER.info(
            "  -> reference depth (%d-day storm, pairs with AMC III): %.1f mm",
            STORM_WINDOW_DAYS, r["reference_event_mm"],
        )

    merged = merge_results(_load_existing(out), new_results)
    out.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")
    LOGGER.info("Wrote %s (%d events on record)", out, len(merged))


if __name__ == "__main__":  # pragma: no cover
    main()
