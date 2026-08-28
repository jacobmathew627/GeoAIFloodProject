---
title: Ernakulam Flood Risk
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
short_description: Rainfall-conditioned flood hazard for Ernakulam, Kerala
---

# Ernakulam Flood Risk Dashboard

Rainfall-conditioned flood hazard for Ernakulam district, Kerala. Move the
storm-depth slider and the map recomputes from the model rather than
interpolating between pre-rendered scenarios.

**Source and full technical documentation:**
<https://github.com/jacobmathew627/GeoAIFloodProject>

## What the numbers mean

- **Flood Probability (live)** — calibrated probability of riverine/backwater
  inundation. Trained on the NDEM August 2018 inundation inventory,
  spatial-block AUC 0.824, with a conformal coverage guarantee.
- **Waterlogging Index (live)** — a *relative ranking* of rain-driven
  waterlogging pressure, not a probability. Proxy-validated (AUC 0.807 against
  14 documented Kochi hotspots), because no incident records exist for this
  district yet.
- **Storm rainfall** is a **3-day cumulative depth**, not a 24-hour figure. The
  calibration reference is 443 mm (the IMD 3-day maximum for 15–17 Aug 2018).
- **Population exposed** is a real spatial sum over WorldPop 2020, not a
  district-average estimate. **Building value exposed** is replacement-cost
  *exposure* from OpenStreetMap footprints — not a damage prediction.

## Known limitation worth reading before you act on it

The land cover is from 2018, and the district's built-up area has grown
**+23.3% (+161 km²)** since (measured via Google Dynamic World). Built-up
surfaces shed far more runoff, so on land developed after 2018 this model
**understates** present-day risk.

That is not a stale file: 2018 land cover is the *correct* land cover for
training, because the labels are the August 2018 flood inventory. Fixing it
properly means separating the training epoch from the inference epoch, which
is a temporal-transfer design change. The Advanced panel states the gap.

## Disclaimer

Decision support only. Verify against official meteorological and disaster
management authorities before acting on any output.

Data: ESA WorldCover, Copernicus Sentinel-1, NDEM/Bhuvan (NRSC), IMD gridded
rainfall, WorldPop, OpenStreetMap (ODbL), Google Dynamic World.
