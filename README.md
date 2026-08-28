# GeoAI Flood Risk Dashboard — Ernakulam, Kerala

**[Open the live dashboard](https://geoai-waterlogging-project.streamlit.app/)**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-1.42%20GB-blue.svg)](Dockerfile)
[![Tests](https://img.shields.io/badge/tests-452%20passing-green.svg)](tests/)

A flood-risk model for Ernakulam district, Kerala. Move a rainfall slider and the
map recomputes from the model — nothing is interpolated between pre-rendered
images.

**New here?** [**How it works**](docs/HOW_IT_WORKS.md) is a plain-language walkthrough
of the data going in, what the machine learning actually does, what every layer in
the app means, and what comes out. Start there. This file is the technical
reference.

Every number in both files is reproduced from the code or a committed artefact.
Where a figure could not be regenerated, it says so.

---

## Read this first: what it predicts, and what it does not

The dashboard shows **two layers that are never blended**, because they answer
different questions and have very different evidence behind them.

| Layer | What it is | How much to trust it |
|---|---|---|
| **Flood Probability (live)** | Calibrated probability of **riverine / backwater inundation** | Trained on the NDEM August 2018 inundation inventory. Spatial-block AUC **0.824**, isotonic-calibrated, with a conformal coverage guarantee. |
| **Waterlogging Index (live)** | A 0–1 **ranking** of rain-driven waterlogging pressure | Physics only, **not a probability**. Proxy-validated at AUC **0.807** against 14 documented hotspots. No calibration, because there is nothing to calibrate against. |

The distinction is not cosmetic — it decides what the output may be used for.

**In this district the two phenomena happen in different places.** Kochi city did
not experience riverine flooding in August 2018; that flood was inland along the
Periyar (Aluva, Perumbavoor, Kalamassery). All 14 documented street-waterlogging
hotspots fall *outside* the 2018 flood extent. So the flood layer ranks the city
centre as comparatively safe — correct for riverine inundation, and irrelevant to
street ponding.

Averaging the two would have destroyed the only waterlogging signal the project
has, and would have laundered the weaker layer's validation into the stronger
one's credibility. They stay separate, with separate colour ramps, and the
waterlogging layer carries a warning banner in the app.

---

## Try it

**Live:** <https://geoai-waterlogging-project.streamlit.app/>

**Locally** (needs the trained artefacts — see [Rebuilding from scratch](#rebuilding-from-scratch)):

```bash
pip install -r requirements.txt
python serve.py                  # Streamlit on :8501
python serve.py --mode fastapi   # JSON API on :8000
python serve.py --check          # report what is missing, without launching
```

**With Docker** (both services):

```bash
python src/make_display_rasters.py          # one-time, builds display/ (21 MB)
python src/rainfall_forecast.py --snapshot  # one-time, caches the forecast
docker compose up --build                   # dashboard :8501, API :8000
```

---

## How it works

Three things are kept separate that are often conflated:

| Component | Question it answers | Depends on rainfall? |
|---|---|---|
| **Susceptibility** `S(x)` | If a major storm hits, does this pixel flood? | No |
| **Runoff** `Q(x, P)` | How much of a storm of depth `P` becomes runoff here? | Yes |
| **Hazard** `H(x, P)` | Probability this pixel floods in a storm of depth `P` | Yes |

`S(x)` is learned by gradient-boosted trees over 16 terrain, land-cover and
drainage-context factors. `Q(x, P)` is SCS Curve Number runoff. They combine in
logit space:

```
H(x, P) = sigma( logit(S(x)) + beta * ln( Q_routed(x,P) / Q_routed(x,P_ref) ) )
```

At the reference depth `P_ref` this reduces exactly to `S(x)`, so the model
reproduces the observed 2018 extent. Everywhere else it is strictly monotonic in
rainfall and stays inside (0, 1) without clipping — verified across all nine
scenarios, zero pixels decreasing at any step.

**Rainfall is a 3-day cumulative depth, not a 24-hour figure.** The calibration
reference is **443 mm**, the IMD gauge-based 3-day maximum for 15–17 August 2018.
`beta = 3.085`, fitted across four flood events (three of which carry information;
see [`outputs/beta_fit.json`](outputs/beta_fit.json)).

### Why it is fast

Both surfaces are closed-form in rainfall, and both route runoff downslope.
SCS-CN runoff depends on a pixel only through its curve number, and there are
**seven** curve-number classes in this district. So the upstream count of each
class draining through `x` is accumulated once when the cache is built, and a new
rainfall value costs seven scalar evaluations plus a weighted sum over that fixed
basis — no re-routing, no re-fit.

```bash
python src/live_model.py --build       # precompute, writes a 7.6 MB cache
python src/live_model.py --benchmark   # measured: 89-90 ms per rainfall value
```

Both layers draw on the same routed basis, so a rainfall change moves them
consistently.

---

## Performance

Held-out performance of the susceptibility model, from
[`models/susceptibility_metrics.json`](models/susceptibility_metrics.json):

| Cross-validation scheme | AUC-ROC | AUC-PR | Brier |
|---|---|---|---|
| Random k-fold | 0.902 | 0.894 | 0.127 |
| **Spatial block (5 km)** | **0.824** | **0.805** | **0.173** |
| Spatial block, low-lying only (DEM ≤ 21.7 m) | 0.840 | 0.821 | 0.166 |

**Quote the spatial-block number.** Random k-fold overstates AUC by **7.8
points** here: neighbouring pixels of a 10 m raster are near-duplicates, so a
random split leaks the test set into training. The low-lying row is the
operationally relevant case — it measures whether the model can rank two
low-lying pixels against each other, which is the decision a planner actually
faces.

Trained on 117,706 samples (59,108 positive, balanced 1:1 by design).

### A lower AUC than before, and a better model

An earlier version trained on a single Sentinel-1 scene scored 0.919. That number
was almost entirely elevation:

| Label | AUC from elevation **alone** | Full model | Added by everything else |
|---|---|---|---|
| Sentinel-1, 21 Aug 2018 | **0.912** | 0.919 | **+0.007** |
| NDEM, 17–18 Aug 2018 | **0.763** | 0.824 | **+0.061** |

The Sentinel-1 inventory was 94% non-urban backwater fringe at a median elevation
of 4.0 m, so "low ground is wet" solved it. The NDEM inventory is 43% urban at a
median 13.9 m — much closer to the district as a whole — and the conditioning
factors now contribute roughly eight times more over that baseline.

### Calibration

Probabilities are isotonic-calibrated out-of-fold, with the isotonic curve fitted
on one half of the out-of-fold predictions and measured on the other, so the
reliability figure is not circular. Worst predicted-vs-observed deviation across
ten probability bins: **0.031**.

Training is balanced 1:1, but only **3.53%** of the sampling domain (786,504 of
22,299,912 pixels) is NDEM-flooded. Probabilities are shifted back to the
population base rate. The closed-form case-control offset assumes randomly drawn
absences, which these are not — they are elevation-stratified — so the offset is
solved by bisection instead:

| | Prior logit offset | Expected flooded area at 443 mm |
|---|---|---|
| Closed-form (assumes random absences) | −3.3175 | — |
| **Fitted by bisection** | **−3.3779** | **78.65 km²** (target 78.65 km²) |

That match is exact **by construction** — bisection solves for it — so it
confirms the fit converged, not that the model is right. The independent check is
against the written raster, pixel by pixel: the 443 mm hazard raster sums to
74.6 km² of expected flooding against 77.5 km² actually flooded (ratio 0.96).

### Risk bands

Band edges are read off the precision-recall curve of the reference-event hazard
map against the 2018 inventory — not chosen as round numbers
([`models/risk_thresholds.json`](models/risk_thresholds.json)):

| Band | Lower edge | Captures of observed flooding | Precision | Lift over base rate |
|---|---|---|---|---|
| Moderate | 0.024 | 95.4% | 0.078 | 2.1× |
| High | 0.056 | 81.4% | 0.134 | 3.7× |
| Severe | 0.125 | 38.6% (max F1) | 0.281 | 7.7× |
| Critical | 0.269 | 4.1% | 0.539 | 14.8× |

District base rate is 3.64% at 443 mm. **Re-derive these whenever the model is
retrained** (`python src/risk_thresholds.py`) — they are properties of the fitted
probabilities, not constants.

### Uncertainty: conformal prediction

Split + Mondrian conformal prediction gives a distribution-free coverage
guarantee at α = 0.10, calibrated on 185,303 held-out pixels:

| | Coverage |
|---|---|
| Target | 0.90 |
| Marginal (achieved) | 0.87 |
| Class-conditional, flood | **0.93** |
| Class-conditional, dry | 0.87 |

Mean prediction-set size 1.21. The class-conditional split is the point: a
*marginal* guarantee is a trap for a rare hazard, because a model can hit 90%
overall while badly under-covering the flood class. The marginal-only variant in
the same file achieves 0.85 marginal coverage but just **0.003** on the flood
class — it satisfies the headline number by abandoning the class that matters.

### What the app reports

At a given rainfall the alert banner reports real spatial sums, not
district averages:

- **Population exposed** — WorldPop 2020 100 m counts, summed over cells above
  the critical threshold ([`src/population.py`](src/population.py)).
- **Building value exposed** — OpenStreetMap footprints (176,318 buildings,
  25.9 km² of mapped floor area) priced at the Kerala PWD 2025 urban
  construction rate ([`src/building_exposure.py`](src/building_exposure.py)).
  This is **replacement-cost exposure, not a damage prediction** — no
  India-specific depth-damage function was available to discount it to expected
  loss, so it is not called damage.

Note on population totals: WorldPop over the full administrative district sums to
~3.66M, but the **model domain is 2,114 km² of the district's 3,068 km²**, and
population within that domain sums to ~2.84M. The app sums over the model domain.

---

## Validation against urban waterlogging

Both layers were scored at **14 locations** documented in public reporting as
recurrent Kochi waterlogging points (Operation Breakthrough phase-4 works, the TP
and Mullassery canal reaches, 2024 monsoon press coverage), geocoded via
Nominatim and sampled as the maximum within 150 m, at 150 mm of rain.

| Layer | vs urban background | vs **elevation-matched** urban background |
|---|---|---|
| Flood probability | AUC 0.324 | **0.388** — worse than chance |
| **Waterlogging index** | AUC 0.839 | **0.807** (95% CI 0.698–0.908) |

Only the waterlogging index has skill, and it survives the elevation-matched
control — the test that matters, since every documented hotspot is a low-lying
junction, and a model knowing only "low ground floods" would ace an unrestricted
comparison and collapse against a matched one.

Proximity to a mapped drain or canal is itself a real signal: scored by the same
procedure, **AUC 0.698** (95% CI 0.555–0.840). Kochi's canals are tidal and back
up, so being **closer** is worse, not better — which is the opposite of the
intuition that drains reduce flooding.

Two caveats on that 0.698, both worth stating rather than burying. The
`src/live_model.py` and `tests/test_osm_drainage.py` docstrings record an
earlier run at 0.713; re-deriving it here with the committed helpers gives
0.698, and the original figure is not regenerable from any committed artefact,
so treat ~0.70 as the number and the *direction* as the finding. And the
sampling takes the maximum within 150 m, which is right for a hazard surface
but means the score is not a symmetric ranking — you cannot read the
"far from a channel" case off as one minus this.

This is why `src/pluvial.py` must never acquire a hand-signed drainage term: the
sign would come from the same 14 points used to validate it. A test enforces
that.

### What this does not license

- **n = 14.** The intervals are wide by construction — enough to separate skill
  from chance, and nothing finer.
- **Reporting bias.** Journalists cover junctions that stall traffic, so the
  sample favours arterial city-centre roads over residential streets that flood
  just as often. Some of the measured skill may be "near a canal or major road"
  rather than "waterlogs".
- **Not a probability.** A physics-derived ranking with no calibration.
- **It is a test set, not a training set.** Nothing was fitted to it.

Regenerate with `python src/waterlogging_validation.py`; every point carries its
source in [`models/waterlogging_validation.json`](models/waterlogging_validation.json),
so any label can be audited or dropped.

---

## Known limitations

These are the things most likely to make an output wrong. They are listed because
they are real, not as boilerplate.

### 1. The land cover is from 2018, and the district has grown 23% since

Measured, not asserted — Google Dynamic World, modal class per year
([`models/landcover_drift.json`](models/landcover_drift.json)):

| Year | Built-up area | vs 2018 |
|---|---|---|
| 2018 | 691.0 km² | — |
| 2021 | 795.7 km² | +15.2% |
| 2025 | 852.1 km² | **+23.3% (+161 km²)** |

Built-up surfaces carry a much higher curve number, so on the 161 km² developed
since 2018 the model routes less runoff than the ground now produces and
**understates present-day risk there**.

This is **not a stale file**. 2018 land cover is the *correct* land cover for
training: the labels are the August 2018 flood inventory, and the features must
describe the surface that produced those floods. Swapping in 2025 land cover and
retraining would misalign features from labels — a worse error than the one it
fixes. Doing it properly means separating the training epoch from the inference
epoch, which is a temporal-transfer design change and is not attempted here. The
app states the gap in its Advanced panel.

### 2. Water entering from upstream is not modelled

The DEM is clipped to the district; water is not. The largest catchment the
shipped flow network can resolve is a fraction of what actually drains here:

| Basin draining into Ernakulam | Area |
|---|---|
| Periyar | ~5,398 km² |
| Chalakudy | ~1,704 km² |
| Muvattupuzha | ~1,554 km² |

Every river enters across a nodata edge carrying nothing. The August 2018 flood
was driven largely by Periyar discharge and reservoir releases from a catchment
more than twenty times larger than anything the model sees.

**Attempted fix, and where it actually stands.**
[`src/upstream_dem.py`](src/upstream_dem.py) builds a DEM over the full
contributing area — **25,085 km²** against the district DEM's 2,427 km² —
verified against known elevations (Kochi 4 m, Munnar 1,455 m, Anamudi 2,465 m).
That part works.

[`src/upstream_routing.py`](src/upstream_routing.py) then routes it with
WhiteboxTools `breach_depressions_least_cost` (breaching, not filling — filling a
25,000 km² mountainous grid creates flats whose tie-break disperses flow). This
is a large improvement on the earlier plain priority-flood, which returned 0 km²
at Aluva. It is still **not good enough to ship, and is not wired into the
model** — `aligned_raster: null` in
[`GeoAI_New/routing_work/routing_validation.json`](GeoAI_New/routing_work/):

| Probe | Expected | Found | Ratio |
|---|---|---|---|
| Periyar at Aluva | ~5,000 km² | 1,188 km² | 0.24 |
| Periyar at Neriamangalam | ~3,300 km² | 17 km² | 0.005 |
| Chalakudy at Chalakudy town | ~1,400 km² | 1,016 km² | **0.73 (passes)** |
| Muvattupuzha at Muvattupuzha | ~1,100 km² | 2,884 km² | 2.62 |

**1 of 4 probes passes.** The build refuses to write an aligned raster unless the
probes agree, which is why nothing downstream consumes it.

So the *pattern* knows about flow — HAND, TWI, flow accumulation, river distance
and `upstream_cn` all encode convergence — and the *rainfall response* is routed
within the district. But nothing responds to rain falling in the Western Ghats.

### 3. Street-scale ponding is below the DEM's resolution

A fill-spill depression model was implemented first and abandoned. The DEM is
30 m horizontally with **1 m vertical quantisation**; street ponding is 0.1–0.5 m
deep. Filling produced depressions with a median depth of 3.0 m and a maximum of
28 m — regional basins, not urban hollows — and put 2 m of standing water in
central Ernakulam at 50 mm of rain. `pluvial.fill_depressions` is kept and tested
because it is correct; the DEM is what cannot support it. Street-scale ponding
needs a LiDAR DEM, ideally 1 m with sub-decimetre vertical accuracy.

### 4. Trained on one event

Susceptibility is trained on the 2018 inventory alone. The prior offset is fitted
to reproduce that same event, so "expected area equals observed area" is a
consistency check, not independent validation. `beta` is fitted across four
events, but only three carry information.

### 5. Other honest caveats

- **The rainfall forecast trails the calendar.** IMD's yearwise archive ships a
  fixed-size binary per year, and the in-progress year fails its parser, so the
  most recent usable series ends at the last closed year. A property of the free
  archive format, not a caching bug.
- **The D8 network is approximate.** Its contributing area agrees with the
  shipped flow-accumulation raster at only Spearman ρ ≈ 0.73. `upstream_cn` is a
  catchment *average*, robust to modest routing error, and is kept because it
  measurably improves held-out AUC — not because the routing is known exact.
- **No incident records exist for this district.** That is the single missing
  dataset that would let the waterlogging layer be calibrated rather than merely
  ranked. A request is documented in
  [`docs/data-requests/`](docs/data-requests/).

---

## The graph model was built, tested, and rejected

Adding context *features* is a workaround; the architectural version is to make
the flow network the computation. [`src/catchment_graph.py`](src/catchment_graph.py)
delineates real sub-catchments and [`src/graph_model.py`](src/graph_model.py) runs
directed GraphSAGE over them, with separate weight matrices for upstream and
downstream neighbours.

The comparison is like-for-like — identical node features, identical
spatial-block folds, and an ablation toggling edges within the *same*
architecture, so the graph's contribution is separated from the model family:

| Graph scale | Model | AUC-ROC | AUC-PR |
|---|---|---|---|
| 1,155 nodes / 840 edges | boosted trees, no graph | 0.9249 | 0.3732 |
| | same net, edges **off** | 0.9177 | 0.5382 |
| | same net, edges **on** | 0.9148 | 0.4720 |
| 11,669 nodes / 11,118 edges | boosted trees, no graph | 0.9487 | 0.6415 |
| | same net, edges **off** | 0.9541 | 0.5931 |
| | same net, edges **on** | 0.9010 | 0.4370 |

**Turning the edges on makes it worse, and the damage grows with graph density.**
The edges-off control rules out the obvious confound: the same network without
the graph is competitive with boosted trees, so this is the graph failing, not
neural nets losing to trees. The most likely reason is that the graph is nearly a
tree (~0.95 edges per node), so mean-aggregating over ~1 neighbour adds almost no
context while still smoothing the node's own signal.

The harness is kept and tested — a negative result is only trustworthy if the
machinery behind it is correct. Reproduce with
`python src/run_graph_experiment.py` (needs `pip install -r requirements-legacy.txt`).

---

## Rebuilding from scratch

```bash
# 1. Align every conditioning factor onto one grid (~3 min, writes ~2.6 GB)
python align_data.py

# 2. Derive drainage-network and multi-scale context features (~30 s)
python src/derive_features.py

# 3. Train the susceptibility model, incl. conformal calibration (~6 min)
python src/susceptibility.py --train

# 4. Score the full grid, with per-pixel ensemble uncertainty (~25 min)
python src/susceptibility.py --predict

# 5. Conformal decision raster (seconds - it is a reclassification)
python src/susceptibility.py --conformal

# 6. Rainfall-conditioned hazard rasters (~40 s)
python src/hazard.py

# 7. Live-evaluation cache the dashboard reads (~30 s)
python src/live_model.py --build

# 8. Display rasters for deployment (21 MB)
python src/make_display_rasters.py
```

Conformal calibration can be redone without refitting the ensemble:

```bash
python src/susceptibility.py --recalibrate-conformal --alpha 0.05
```

Optional data-acquisition steps (need an Earth Engine project):

```bash
python src/population.py --project <ee-project>        # WorldPop exposure
python src/building_exposure.py --build                # OSM footprints
python src/landcover_drift.py --project <ee-project>   # measure the 2018 gap
python src/osm_drainage.py --build                     # drain/canal network
```

---

## Repository layout

```
├── app.py                        Streamlit dashboard
├── serve.py                      Launcher for either service
├── align_data.py                 Raster alignment + inventory preparation
├── Dockerfile                    builder -> runtime -> api -> app
├── docker-compose.yml            Both services
├── src/
│   ├── config.py                 All configuration and thresholds
│   ├── hydrology.py              SCS Curve Number runoff
│   ├── routing.py                D8 flow network over the filled DEM
│   ├── pluvial.py                Routed runoff, waterlogging index
│   ├── hazard.py                 Susceptibility x runoff -> hazard
│   ├── live_model.py             Precomputed cache for live evaluation
│   ├── susceptibility.py         Training, spatial CV, calibration
│   ├── feature_stack.py          Design matrix + training-point sampling
│   ├── conformal.py              Split + Mondrian conformal prediction
│   ├── derive_features.py        Drainage-network + multi-scale context
│   ├── fit_beta.py               Fits the rainfall sensitivity beta
│   ├── risk_thresholds.py        Derives the risk band edges
│   ├── population.py             WorldPop exposure raster
│   ├── building_exposure.py      OSM building footprints + replacement cost
│   ├── landcover_drift.py        Measures the 2018-to-present land-cover gap
│   ├── osm_drainage.py           Drain/ditch/canal network from OSM
│   ├── soil_hsg.py               Hydrologic soil group from SoilGrids
│   ├── ndem_labels.py            NDEM flood inventory labels
│   ├── reference_rainfall.py     IMD gauge-based reference depths
│   ├── rainfall_forecast.py      3-day rainfall prediction from IMD data
│   ├── upstream_dem.py           DEM over the full contributing area
│   ├── upstream_routing.py       WhiteboxTools breaching (not yet shipped)
│   ├── waterlogging_validation.py  Scoring against documented hotspots
│   ├── make_display_rasters.py   Display-resolution layers for deployment
│   ├── catchment_graph.py        Sub-catchment delineation (experiment)
│   ├── graph_model.py            Directed GraphSAGE (tested, not adopted)
│   ├── data_loading.py           Raster I/O for display
│   ├── visualization.py          Colormaps, legends, statistics, alerts
│   ├── ui_components.py          Streamlit widgets
│   └── backend.py                FastAPI service
├── deploy/
│   ├── push_to_space.py          Publish to a Hugging Face Space
│   └── SPACE_README.md           Front-matter for that Space
├── display/                      Display rasters (built, gitignored)
├── data_aligned/                 Aligned features (built, gitignored)
├── models/                       Metrics, thresholds and the live cache
├── outputs/                      Susceptibility and hazard rasters
├── GeoAI_New/                    Source rasters (Git LFS)
└── tests/                        452 tests
```

---

## API

```
GET /api/health            service and data readiness
GET /api/scenarios         available rainfall depths
GET /api/model             model card, CV metrics, assumptions
GET /api/conformal         coverage guarantee and per-stratum coverage
GET /api/map/{mm}          hazard overlay (base64 PNG) + WGS84 bounds
GET /api/risk_stats/{mm}   risk-class breakdown with real areas in km²
GET /api/runoff            SCS-CN runoff for a depth and curve number
GET /api/places            known place coordinates
```

`{mm}` accepts any depth from 0 to 2000. If a pre-generated full-resolution
raster exists it is served; otherwise the hazard is evaluated live from the same
cache the dashboard uses. The response's `resolution` field says which path
produced it, since statistics from the display grid (~0.39M cells) and the full
grid (~42M cells) agree closely without being identical.

---

## Testing and CI

```bash
pytest tests/                    # 452 tests
pytest tests/ -m "not requires_model"   # what CI runs, no artefacts needed
```

The suite includes regression tests for previously-shipped defects: nodata
sentinels surviving post-processing, a 30 m vs 10 m cell-size error, layer nodata
rules that never matched, non-monotonic rainfall response, a colour ramp that
hid every real value in two layers, and a rainfall unit mismatch between the
24-hour label and the 3-day model input.

CI runs lint (ruff), format (black), types (mypy, non-blocking), the test suite
on Python 3.10, a Docker `runtime` build with a GDAL probe, and a live-deployment
check. The Docker job builds the `runtime` stage rather than the full app,
because the app stage needs artefacts that are gitignored and cannot exist in a
clean CI checkout.

Tests requiring `torch` are skipped automatically when the optional
`requirements-legacy.txt` extra is absent.

---

## Deployment

The dashboard image is **1.42 GB** and the API image **1.38 GB**. Both run as a
non-root user with a healthcheck.

The images deliberately contain the model's *outputs*, not its *inputs*:

| Excluded | Size | Why the running app does not need it |
|---|---|---|
| `GeoAI_New/` | 3.7 GB | Full-resolution conditioning factors. Every static layer is read at 1000 px, so the image ships `display/` (21 MB) instead — verified pixel-for-pixel identical across all 14 layers. |
| `data_aligned/` | 1.3 GB | Training inputs. The app reads the precomputed `live_model.npz`. |
| `data/imd_rain/` | 874 MB | The forecast ships as a cached JSON snapshot. |
| `models/*.pth`, `*.h5` | 248 MB | Archived U-Net weights; nothing active loads them. |

To publish to a Hugging Face Space (needs a PRO subscription for Docker Spaces):

```bash
python deploy/push_to_space.py --dry-run
python deploy/push_to_space.py --space <user>/<name> --token hf_...
```

---

## Data sources

ESA WorldCover (land cover) · Copernicus Sentinel-1 (SAR flood inventory) ·
NDEM via Bhuvan/NRSC (inundation inventory) · IMD 0.25° gridded rainfall ·
WorldPop (population) · OpenStreetMap contributors, ODbL (drainage, buildings) ·
Google Dynamic World (land-cover drift) · SoilGrids (hydrologic soil group) ·
USDA NRCS NEH-630 (curve number method) · Open-Meteo (live weather)

---

**Disclaimer.** Decision support only. Verify against official meteorological and
disaster management authorities before acting on any output.
