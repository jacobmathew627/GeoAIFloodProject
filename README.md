# GeoAI Flood Risk Dashboard — Ernakulam, Kerala

> A GeoAI framework for flood susceptibility mapping and rainfall-conditioned
> waterlogging risk in Ernakulam District, Kerala, India.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

## Live rainfall evaluation

The two model layers are computed **on demand**. Moving the rainfall slider
re-evaluates the model at that depth in roughly 60–90 ms; nothing is
interpolated between pre-rendered scenario rasters.

That is possible because both surfaces are closed-form in rainfall:

```
fluvial(x, P) = sigma( logit(S(x)) + beta * ln( Q(x,P) / Q(x,P_ref) ) )
pluvial(x, P) = f( routed SCS-CN runoff(P), local gradient )
```

`S` is the learned susceptibility and does not depend on rainfall, so it is
loaded once. `Q` is SCS-CN runoff, which depends on a pixel only through its
curve number — and there are just seven curve numbers in this district. So a
new rainfall value costs seven scalar evaluations plus array arithmetic on the
display grid, with **no re-routing and no model re-fit**.

```bash
python src/live_model.py --build       # precompute, ~30 s, writes a 5.9 MB cache
python src/live_model.py --benchmark   # 60-90 ms per rainfall value
```

Verified by driving the app: at 50 / 120 / 200 / 300 mm the rendered overlay is
four distinct images, and the reported expected flooded area moves 1 → 4 → 12 →
25 km².

### Two layers, deliberately not blended

| Layer | What it is | Trust |
|---|---|---|
| **Flood Probability (live)** | Probability of riverine/backwater inundation | Calibrated. Spatial-block AUC 0.919, conformal coverage guarantee. |
| **Waterlogging Index (live)** | Rain-driven waterlogging pressure, 0–1 | **Unvalidated.** Physics only. |

They answer different questions and only one is calibrated. Averaging them
would launder the unvalidated one into the other's credibility, so the app
keeps them as separate layers with separate colour ramps, and the waterlogging
layer carries a warning banner.

The point query returns both, alongside the physical quantities behind them —
land cover, curve number, runoff depth and runoff coefficient — so a number can
always be traced to the reason for it.

### Why the waterlogging index is not a probability

It is routed SCS-CN runoff over local gradient: a topographic wetness index in
which real runoff replaces catchment area, so it responds to storm depth and to
land cover rather than to terrain alone. It is honest physics and it ranks
locations, but it has never been tested against waterlogging, because **there
are no urban waterlogging records for this district**. Measured against the one
label that does exist — the 2018 inventory — it scores AUC 0.53 inside built-up
areas, which is chance. That is expected, since that inventory contains almost
no urban flooding, but it means nothing is demonstrated.

A fill-spill depression model was implemented first and abandoned. The DEM is
30 m horizontally with **1 m vertical quantisation**, and street ponding is
0.1–0.5 m deep. Filling it produced depressions with a median depth of 3.0 m
and a maximum of 28 m — regional basins, not urban hollows — and put 2 m of
standing water in central Ernakulam at 50 mm of rain. `pluvial.fill_depressions`
is kept and tested because it is correct; the DEM is what cannot support it.
Street-scale ponding needs a LiDAR DEM, ideally 1 m with sub-decimetre vertical
accuracy.

## The biggest open gap: water entering from upstream

The DEM is clipped to the district, and water is not. Measured on the shipped
flow network, **the largest catchment the model can resolve is 248 km²**.
For scale:

| Basin draining into Ernakulam | Area |
|---|---|
| Periyar | ~5,398 km² |
| Chalakudy | ~1,704 km² |
| Muvattupuzha | ~1,554 km² |

Every river enters the district across a nodata edge and arrives carrying
nothing. The August 2018 flood was driven largely by Periyar discharge and
reservoir releases from a catchment more than twenty times bigger than
anything the model can see.

What *is* routed, and what is not:

| Component | Routed? |
|---|---|
| `upstream_cn` feature | Yes — catchment-average curve number |
| Waterlogging index | Yes — `Σ_k Q_k(P)·N_k(x)` |
| **Flood probability's rainfall response** | **No — pointwise** |

So the *pattern* knows about flow, because HAND, TWI, flow accumulation and
river distance encode convergence. The *rainfall response* does not: a pixel's
forcing is the rain that fell on it, not what its catchment delivers. The
learned susceptibility absorbs the 2018 pattern, which is why the maps look
plausible, but nothing in the system responds to rain falling in the Western
Ghats.

### Attempted fix, and where it stands

[src/upstream_dem.py](src/upstream_dem.py) builds a DEM over the full
contributing area from open terrain tiles — **25,085 km² against the district
DEM's 2,427 km²**, verified against known elevations (Kochi 4 m, Munnar
1,455 m, Anamudi 2,465 m). That part works.

Routing it does not, and is **not wired into the model**. Probed at points
with published catchment areas the extended network returns 0 km² for the
Periyar at Aluva, 2 km² at Neriamangalam and 1 km² for the Chalakudy — its
largest accumulation sits on the grid edge rather than on any channel. The
cause is depression conditioning: a plain priority-flood is fine over the
2,400 km² district but across 25,000 km² of Western Ghats it creates flats up
to 124 m deep, and flow disperses instead of concentrating.

Finishing it needs breaching rather than pure filling, plus flat resolution
that imposes a gradient toward the outlet — WhiteboxTools
(`BreachDepressionsLeastCost`, `D8FlowAccumulation`) or RichDEM do this
correctly. Hand-rolling it at this scale is the wrong use of effort.

## Urban waterlogging: what has now been tested

Both layers were scored at **14 locations documented in public reporting** as
recurrent Kochi waterlogging points (Operation Breakthrough phase-4 works, the
TP and Mullassery canal reaches, and press coverage of the 2024 monsoon),
geocoded via Nominatim and sampled as the maximum within 150 m.

| Layer | vs urban background | vs **elevation-matched** urban background |
|---|---|---|
| Flood probability | AUC 0.865 | **0.801** (95% CI 0.690–0.889) |
| Waterlogging index | AUC 0.839 | **0.807** (95% CI 0.698–0.908) |

Every documented hotspot is a low-lying central junction, so the second column
is the one that counts: a model that only knew "low ground floods" would score
well against unrestricted urban background and collapse against an
elevation-matched one. **The skill survives**, and both intervals exclude 0.5.

So the layers do **rank** documented waterlogging locations above comparable
urban ground. That is a real, testable claim, and it is the strongest statement
the free data supports.

What it does **not** license:

- **The probabilities remain wrong for waterlogging.** The flood layer ranks
  hotspots well while assigning them ~0.16% absolute probability. AUC is
  rank-based and blind to calibration. Use the ordering, never the number.
- **n = 14.** The intervals are wide by construction, enough to separate
  "skill" from "chance" and nothing finer.
- **Reporting bias.** Journalists cover junctions that stall traffic, so the
  sample favours arterial city-centre roads over residential streets that
  flood as often. Some of the measured skill may be "near a canal or major
  road" rather than "waterlogs".
- **It is a test set, not a training set.** Nothing was fitted to it.

Regenerate with `python src/waterlogging_validation.py`; every point carries
its source in `models/waterlogging_validation.json` so any label can be
audited or dropped.

## Read this first: what the model does and does not predict

**It does not predict urban waterlogging.** It predicts the extent of
riverine and backwater inundation. Those are different phenomena, and the
distinction is not cosmetic — it decides what the output can be used for.

The evidence, from the shipped rasters:

| | Top 0.1% of predicted hazard | Whole district |
|---|---|---|
| Median elevation | **0.0 m** | 29.4 m |
| Median urban fraction | **0.0** | **1.0** |
| Median distance to drainage | 3,146 m | 1,083 m |

The district is majority built-up, yet the model's highest-hazard zone is
almost entirely *not* built-up: sea-level paddy and wetland fringing the
Vembanad backwaters. Sampled at named places, at the reference event:

| Place | Hazard @332 mm | Band |
|---|---|---|
| Aluva (on the Periyar) | 0.029 | moderate |
| Perumbavoor | 0.007 | safe |
| Vyttila | 0.003 | safe |
| Kaloor | 0.002 | safe |
| Ernakulam / MG Road | 0.001 | safe |
| Edappally | 0.0005 | safe |

Kochi's urban core reads as essentially zero risk. That is not a bug — it is
what the training label contains. The inventory is Sentinel-1 open water at a
single overpass during an event driven by Periyar flooding and reservoir
releases. Street-level waterlogging is largely invisible to C-band SAR (tree
canopy, buildings, and it often drains before the satellite passes), and it is
governed by storm-drain capacity, canal blockage and tidal locking of
outfalls — none of which are inputs to this model.

**What the output is legitimately good for:** ranking the low-lying rural and
peri-urban floodplain by relative inundation susceptibility, with calibrated
probabilities and a coverage guarantee.

**What it is not good for:** answering "will this street flood if 120 mm falls
tomorrow". See [Known limitations](#known-limitations) for the full list and
[What would close the gap](#what-would-close-the-gap) for what it would take.

## Overview

The system separates two things that are physically distinct and were
previously conflated:

| Component | Question it answers | Depends on rainfall? |
|---|---|---|
| **Susceptibility** `S(x)` | If a major storm hits, does this pixel flood? | No |
| **Runoff** `Q(x, P)` | How much of a storm of depth `P` becomes runoff here? | Yes |
| **Hazard** `H(x, P)` | Probability this pixel floods in a storm of depth `P` | Yes |

`S(x)` is learned from the August 2018 Sentinel-1 flood inventory using
gradient-boosted trees over 15 terrain, land-cover and **drainage-network
context** factors. `Q(x, P)` comes from the SCS Curve Number method. They
combine in logit space:

```
H(x, P) = sigma( logit(S(x)) + beta * ln( Q(x, P) / Q(x, P_ref) ) )
```

At the reference event `P_ref` this reduces exactly to `S(x)`, so the model
reproduces the observed 2018 extent. Elsewhere it is strictly monotonic in
rainfall and stays inside (0, 1) without clipping.

## Model performance

Held-out performance of the susceptibility model, from
`models/susceptibility_metrics.json`:

| Cross-validation scheme | AUC-ROC | AUC-PR | Brier |
|---|---|---|---|
| Random k-fold | 0.977 | 0.976 | 0.057 |
| **Spatial block (5 km)** | **0.919** | **0.911** | **0.119** |
| Spatial block, low-lying only (DEM ≤ 7 m) | 0.906 | 0.898 | 0.131 |

Random k-fold **overstates AUC by 5.8 points** here. Neighbouring pixels of a
10 m raster are near-duplicates, so a random split leaks the test set into
training. The spatial-block number is the one to quote. The low-lying row is
the operationally relevant case: it measures whether the model can rank two
low-lying pixels against each other, which is the decision a planner actually
faces.

### Calibration

Probabilities are isotonic-calibrated out-of-fold. The isotonic curve is fitted
on one half of the out-of-fold predictions and measured on the other, so the
reliability figure is not circular; the worst predicted-vs-observed deviation
across ten probability bins is **0.033**.

Training is deliberately balanced 1:1, but only **1.40%** of the model domain
(312,781 of 22.3M pixels, i.e. **31.3 km²** of **2,230 km²**) actually flooded
in 2018. Probabilities are therefore shifted back to the population base rate.
The closed-form case-control offset (−4.231) assumes randomly drawn absences,
which ours are not — they are elevation-stratified. The offset is instead
**solved by bisection** against a uniform 400k-pixel sample of the district so
that the expected flooded area equals the observed extent:

| | Expected flooded area at 400 mm | vs observed |
|---|---|---|
| No correction | 337 km² | 10.8× too high |
| Closed-form offset (−4.231) | 20 km² | 0.65× too low |
| **Fitted offset (−3.730)** | **30 km²** | **0.99×** |

Measured on the written rasters: over the 21.14M-pixel model domain
(2,114 km²) the 2018 inventory holds 297,069 flooded pixels (30 km²) and the
400 mm hazard raster sums to 295,052 expected flooded pixels — a ratio of
**0.993**. Against that same inventory the hazard surface scores AUC-ROC 0.977
and AUC-PR 0.429, versus a 0.014 no-skill baseline at this prevalence.

### Monotonicity

Every pixel is verified non-decreasing across all seven scenarios:

| Transition | Pixels where hazard decreases |
|---|---|
| 50 → 100 mm | 0 |
| 100 → 150 mm | 0 |
| 150 → 200 mm | 0 |
| 200 → 250 mm | 0 |
| 250 → 300 mm | 0 |
| 300 → 400 mm | 0 |

The maps this replaces were not monotonic: the shipped 100 mm map had mean
probability 0.124 against the 150 mm map's 0.025.

### Risk bands

The band edges are read off the precision-recall curve of the reference-event
hazard map against the 2018 inventory, not chosen as round numbers:

| Band | Lower edge | Captures of observed flooding | Precision | Area at 400 mm |
|---|---|---|---|---|
| Moderate | 0.022 | 95.1% | 0.13 | 221 km² |
| High | 0.070 | 80.6% | 0.22 | 110 km² |
| Severe | 0.133 | 53.7% (max F1) | 0.33 | 49 km² |
| Critical | 0.271 | 24.5% | 0.53 | 14 km² |

District base rate is 1.4%, so the critical band runs ~38× the no-skill rate.
The previous thresholds (0.10 / 0.20 / 0.30 / 0.50) came from the uncalibrated
score; carried onto the corrected scale they classified the actual 2018
catastrophe as "monitoring active". **Re-derive these whenever the model is
retrained** — they are properties of the fitted probabilities, not constants.

The resulting alert ladder: MONITORING to 250 mm, WARNING at 300–350 mm,
CRITICAL at 400 mm (the reference event).

Permutation importance (AUC drop when shuffled): elevation 0.097, distance to
built-up 0.073, slope 0.067, **`dem_rel_1km` 0.060**, curve number 0.040,
distance to drainage 0.038, NDWI 0.030, HAND 0.025, **`upstream_cn` 0.021**.
`urban_mask` scores 0.000 — fully redundant with `urban_dist` and
`curve_number`, and should be dropped at the next retrain.

## Pipeline

```bash
# 1. Align every conditioning factor onto one grid (~3 min, writes ~2.6 GB)
python align_data.py

# 2. Derive drainage-network and multi-scale context features (~30 s)
python src/derive_features.py

# 3. Train the susceptibility model, incl. conformal calibration (~6 min)
python src/susceptibility.py --train

# 4. Score the full grid, with per-pixel ensemble uncertainty (~25 min)
python src/susceptibility.py --predict

# 5. Conformal decision raster (seconds — it is a reclassification)
python src/susceptibility.py --conformal

# 6. Generate rainfall-conditioned hazard rasters (~40 s)
python src/hazard.py

# 7. Launch
python serve.py                 # Streamlit on :8501
python serve.py --mode fastapi  # FastAPI on :8000
python serve.py --check         # report what is missing without launching
```

Conformal calibration can be redone without refitting the ensemble, which is
the expensive part:

```bash
python src/susceptibility.py --recalibrate-conformal --alpha 0.05
```

## Repository layout

```
├── align_data.py               # Raster alignment + flood inventory preparation
├── app.py                      # Streamlit dashboard
├── serve.py                    # Launcher
├── src/
│   ├── config.py               # All configuration
│   ├── hydrology.py            # SCS Curve Number runoff
│   ├── routing.py              # D8 flow network over the filled DEM
│   ├── derive_features.py      # Drainage-network + multi-scale context
│   ├── catchment_graph.py      # Sub-catchment delineation (experiment)
│   ├── graph_model.py          # Directed GraphSAGE (tested, not adopted)
│   ├── run_graph_experiment.py # Graph vs tabular, with edge ablation
│   ├── conformal.py            # Split + Mondrian conformal prediction
│   ├── feature_stack.py        # Design matrix + training-point sampling
│   ├── susceptibility.py       # Model training, spatial CV, calibration
│   ├── hazard.py               # Susceptibility x runoff -> hazard
│   ├── data_loading.py         # Raster I/O for display
│   ├── visualization.py        # Colormaps, legends, statistics, alerts
│   ├── ui_components.py        # Streamlit widgets
│   ├── backend.py              # FastAPI
│   └── inference_final.py      # Archived PyTorch U-Net (superseded)
├── data_aligned/               # Build artefact from align_data.py (gitignored)
├── models/                     # Model weights and metrics
├── outputs/                    # susceptibility.tif, flood_hazard_*.tif
├── GeoAI_New/                  # Source rasters
└── tests/
```

## Beyond pixel-independent scoring

Grid models score each cell in isolation, and flooding is not an
isolated-cell phenomenon. Recent work quantifies the cost: a GraphSAGE model
over a watershed connectivity graph reached **AUC 0.978 against 0.881** for
the best pixel-independent ensemble on a comparable Sentinel-1 inventory, the
gap attributed to upstream-downstream propagation that raster models
structurally cannot see ([arXiv:2603.15681](https://arxiv.org/abs/2603.15681)).

Rather than replace the model with a GNN, the same signal is recovered in the
feature layer by [src/routing.py](src/routing.py) and
[src/derive_features.py](src/derive_features.py):

- **`upstream_cn`** — catchment-average curve number, accumulated down a D8
  flow network built on the depression-filled 30 m DEM. This answers "what
  kind of land sheds water onto me", which is a property of the flow network
  rather than of Euclidean distance: a cell 200 m from Kochi's rooftops but
  across a divide receives none of their runoff, and no neighbourhood filter
  can express that.
- **`dem_rel_1km`** — elevation relative to the ~1 km neighbourhood mean. TPI
  is already in the stack but operates at much shorter range; this captures
  "am I in a regional basin", the scale at which the coastal plain drowns.

Measured effect of adding the two (same seed, same sampling):

| | 13 features | 15 features | Δ |
|---|---|---|---|
| Spatial-block AUC-ROC | 0.9096 | **0.9191** | **+0.0095** |
| Spatial-block AUC-PR | 0.9032 | 0.9112 | +0.0080 |
| Spatial-block Brier | 0.1236 | 0.1186 | −0.0050 |
| **Low-lying AUC-ROC** | 0.8918 | **0.9058** | **+0.0140** |

The gain is largest in the low-lying zone — the hard case, where elevation
alone cannot discriminate and drainage context is the only thing left to use.
`dem_rel_1km` ranks 4th by permutation importance and `upstream_cn` 9th, and
both draw importance away from the raw pixel features rather than duplicating
them.

**Honest caveat on the routing.** The D8 network is acyclic and validated for
that, but its contributing area agrees with the shipped flow-accumulation
raster at only Spearman ρ ≈ 0.73. I could not validate channel positions
independently (`Ernakulam_Streams_Grid.tif` is a direction/order code, not a
channel mask). `upstream_cn` is a catchment *average*, which is robust to
modest routing error, and it is kept because it measurably improves held-out
AUC — not because the routing is known to be exact.

## The graph model was built, tested, and rejected

Adding context *features* is a workaround. The architectural version of the
idea is to make the flow network the computation itself, so that is what
[src/catchment_graph.py](src/catchment_graph.py) and
[src/graph_model.py](src/graph_model.py) do: delineate real sub-catchments
(junction-to-junction reaches plus their hillslopes), then run directed
GraphSAGE message passing over them, with **separate weight matrices for
upstream and downstream neighbours** because a sub-catchment is affected by
what lies above it in a completely different way from what lies below it.

Run it with `python src/run_graph_experiment.py`. The comparison is
like-for-like: identical node features, identical spatial-block folds, and an
ablation that toggles the edges on and off within the *same* architecture, so
the graph's contribution is separated from the model family.

| Graph scale | Model | AUC-ROC | AUC-PR |
|---|---|---|---|
| 1,155 nodes / 840 edges | boosted trees, no graph | 0.9249 | 0.3732 |
| | same net, edges **off** | 0.9177 | 0.5382 |
| | same net, edges **on** | 0.9148 | 0.4720 |
| 11,669 nodes / 11,118 edges | boosted trees, no graph | 0.9487 | 0.6415 |
| | same net, edges **off** | 0.9541 | 0.5931 |
| | same net, edges **on** | 0.9010 | 0.4370 |

**Turning the edges on makes it worse, and the damage grows with graph
density** (−0.053 AUC-ROC, −0.156 AUC-PR at the fine scale). The edges-off
control rules out the obvious confound: the same network without the graph is
competitive with boosted trees, so this is the graph failing, not neural nets
losing to trees.

The published +0.097 gain does not reproduce here. Most likely reasons, in
rough order of confidence:

1. **The graph is nearly a tree** — ~0.95 edges per node, versus ~3.7 in the
   published study (460 nodes, 1,700 edges). Mean-aggregating over ~1
   neighbour adds almost no context while still smoothing the node's own
   signal.
2. **Aggregation destroys the discriminative detail.** Flooding here is
   spatially sharp; collapsing a 1,200-cell sub-catchment to its mean throws
   away exactly the within-node contrast a 10 m pixel model exploits.
3. **One event, not six years.** 604 positive nodes at best. Propagation
   patterns are visible across many events; a single snapshot mostly shows
   where the low ground is.
4. **The routing is imperfect** (ρ ≈ 0.73), and a noisy graph injects noise.

**Decision: the GNN is not adopted.** The pixel model with `upstream_cn`
stays, because it injects drainage context *without* the aggregation loss —
and it demonstrably helps where the GNN demonstrably hurts. The experiment
harness and its tests are kept so the conclusion is reproducible and can be
revisited if a multi-event inventory becomes available.

## Uncertainty: conformal prediction

[src/conformal.py](src/conformal.py) adds split conformal prediction, which
gives a **distribution-free, finite-sample coverage guarantee**: at α = 0.10
the prediction set contains the true label for ≥ 90% of exchangeable district
pixels, regardless of how wrong the model is. The ensemble spread already
reported is only model disagreement and guarantees nothing.

### Marginal coverage is a trap for a rare hazard

Standard split conformal met its target and was **useless**:

| Calibration | Marginal coverage | Coverage on *actually flooded* pixels |
|---|---|---|
| Marginal split conformal | 0.875 | **0.010** |
| **Class-conditional (Mondrian)** | 0.879 | **0.933** |

The marginal guarantee is an average over a district that is 98.6% dry, so it
is satisfied almost entirely by correctly declaring dry land dry — while
flagging **1% of the pixels that genuinely flooded**. The headline number
looks fine and the map is worthless.

Mondrian conformal calibrates a separate quantile *within each true class*, so
the guarantee becomes "of the pixels that really flood, at least 90% are
flagged" — the statement a planner actually needs. That is the calibration the
operational layer uses; both are computed and reported so the gap stays
visible.

Each pixel gets a prediction set, written to `outputs/conformal_sets.tif`:

| Set | Meaning | Area |
|---|---|---|
| `{1}` | confidently flood-prone | 171 km² (8.1%) |
| `{0}` | confidently not | 1,843 km² (87.2%) |
| `{}` | indeterminate — neither label clears its bar | 100 km² (4.7%) |
| `{0,1}` | ambiguous — both labels admitted | 0 km² |

The indeterminate class is the point: a single thresholded map forces every
pixel into a decision, this one marks 100 km² where it cannot support one.

Calibration uses a **uniform sample of the district**, not the balanced
training set — the guarantee only transfers to data exchangeable with the
calibration sample. Calibration and evaluation use disjoint spatial blocks,
which is also why marginal coverage lands at 0.879 rather than exactly 0.90:
spatial dependence breaks the exchangeability the guarantee assumes, and a
random split would have hidden that.

Coverage is also reported **per probability stratum**, because the Himachal
Pradesh study found 82.9% marginal coverage collapsing to 45–59% in high-risk
zones under SAR label noise. The same pattern appears here (0.40 in the top
stratum) and is expected: that band is where the model asserts flooding, so
the ~92% of it that did not flood in 2018 is not covered. It is the
precision/recall trade-off made explicit rather than hidden.

## Conditioning factors

Elevation, slope, HAND, TWI, TPI, SPI, flow accumulation, distance to
drainage, distance to built-up, NDVI, NDWI, urban mask, curve number (derived
from land cover), plus the two context features above.

### Land-cover classes

The LULC raster does **not** use standard ESA WorldCover codes. Class
identities were established empirically by cross-tabulating each class against
NDVI, NDWI, elevation, slope, the urban mask and the flood inventory:

| Code | Class | Share | NDVI | Mean DEM | Flooded in 2018 | CN (AMC II, HSG C) |
|---|---|---|---|---|---|---|
| 1 | Permanent water | 7.5% | 0.05 | 5.5 m | 70.8% | 100 |
| 2 | Tree cover | 35.3% | 0.48 | 93.4 m | 0.2% | 70 |
| 4 | Wetland / paddy | 0.9% | 0.21 | 4.4 m | 28.0% | 90 |
| 5 | Cropland / grassland | 7.5% | 0.41 | 20.9 m | 10.0% | 80 |
| 7 | Built-up | 46.6% | 0.36 | 24.7 m | 0.2% | 88 |
| 8 | Bare / sparse | <0.1% | 0.06 | 62.0 m | 27.6% | 89 |
| 11 | Shrubland | 2.2% | 0.34 | 113.8 m | 7.4% | 74 |

Class 7 coincides pixel-for-pixel with the independent urban mask, which is
what identifies it as built-up.

## Assumptions worth auditing

These are modelling choices, not measurements. Each is a single constant in
`src/config.py`.

- **Reference event = 332 mm** (`RAINFALL.reference_event_mm`) — now *derived*,
  not assumed. ERA5 reanalysis on a 3×3 grid over the mapped district
  ([src/reference_rainfall.py](src/reference_rainfall.py)) gives, for August
  2018: max 1-day 157.9 mm (15 Aug), 2-day 255.9, **3-day 331.6 (14–16 Aug)**,
  5-day 420.1, month 786.8. The 3-day window is chosen because `HYDRO.amc` is
  III, which already encodes a wet antecedent 5 days — using a 5-day storm
  total would count antecedent wetness twice.

  The previous value, 400 mm, was a guess that happened to land near the 5-day
  total. Correcting it does not disturb the calibration (hazard equals
  susceptibility at the reference depth, whatever that depth is) but it makes
  every other scenario more severe, because the 2018 extent is now attributed
  to a smaller storm: a 400 mm scenario went from 30 km² to 40 km² of expected
  flooding, and 150 mm from 6 km² to 8 km². The system was previously
  understating hazard at every rainfall depth.

  Caveat: ERA5 under-resolves orographic extremes, and no rainfall product
  captures the Periyar reservoir releases that contributed to the 2018
  inundation. Treat it as a proxy for total forcing, not a measured storm.
- **Hydrologic soil group C** for the whole district. Kerala's uplands are
  laterite (HSG C) and the coastal strip is alluvium (HSG B); C is the
  conservative single-group choice. A real HSG raster would improve this.
- **AMC III (wet)** antecedent moisture, appropriate for a monsoon-season
  flood-forecasting product.
- **Initial abstraction ratio 0.05** rather than the classic 0.20, following
  Woodward et al. (2003), with the retention rescaled accordingly.
- **beta = 1.8** logit units per log-unit of runoff ratio — the sensitivity of
  flood odds to rainfall. Not fitted against multi-event data, because only
  one flood inventory is available.
- **Population and damage figures** in the alert panel are district-average
  density and a flat per-km² damage rate. They are labelled "planning
  estimate" in the UI and are not model outputs.

## Known limitations

- **One flood event.** Susceptibility is calibrated on August 2018 alone. A
  second inventory (2019 or 2021) would allow `beta` to be fitted rather than
  assumed, and would give a genuine out-of-event validation. As it stands the
  prior offset is fitted to reproduce the same event the model was trained on,
  so "expected area equals observed area" is a consistency check, not
  independent validation.
- **`urban_mask` contributes nothing** (permutation importance 0.000). It is
  redundant with `urban_dist` and `curve_number`. Deliberately *not* removed
  yet: dropping a feature forces a retrain, which forces the risk thresholds to
  be re-derived (they are properties of the fitted probabilities, not
  constants), which forces the full-grid prediction, hazard and conformal
  rasters to be regenerated. That is ~45 minutes of compute for no measurable
  change in skill, and it would leave figures built from two different feature
  sets. Fold it into the next retrain, which a second flood inventory will
  require anyway.
- **Presence-only inventory.** SAR observes water, not "dry". Absences are
  pseudo-absences: buffered 5 px away from observed flooding and stratified
  across elevation deciles matched to the presence distribution. This is the
  main source of irreducible label noise.
- **Permanent water is excluded from the model domain.** It accounted for
  80.3% of the raw flood inventory; including it trains a lake detector.
- **The archived `.pth` checkpoints are stale.** They were trained against
  feature rasters whose nodata sentinels had been clipped into the valid
  range. `src/inference_final.py` still loads them for reproducibility, but
  their output should not be used.
- **No hydrodynamic routing.** Runoff is generated per-pixel and not routed
  downslope, so the model gives flood *susceptibility*, not inundation depth.
- **The inventory understates the event.** Sentinel-1 caught 30 km² of
  non-permanent-water flooding across Ernakulam, which is small for an event
  that displaced people district-wide — SAR sees open water at one overpass,
  not peak inundation, and it cannot see under vegetation or inside built-up
  areas. Every absolute area and exposure figure the system reports inherits
  that floor, which is why the population and damage numbers in the alert
  panel are labelled planning estimates rather than predictions.

## What would close the gap

Ordered by how much each one moves the system toward actually answering
"will this place waterlog at X mm", not by effort.

1. **A waterlogging-specific label.** The single blocking issue. SAR open-water
   extent is the wrong target. Municipal complaint logs, KSDMA incident
   reports, traffic-police road-closure records or geolocated news reports
   would give a few hundred points of the *right* phenomenon. Even a small
   validation set would show whether the current surface has any skill at
   street level — right now that is untested, and the place table above
   suggests it does not.
2. **More events, to fit `beta`.** The rainfall sensitivity
   (`HYDRO.runoff_logit_beta = 1.8`) is currently *assumed*. The shape of the
   response is physically constrained by SCS-CN, but its magnitude is a hand-set
   constant, so every non-reference scenario is an extrapolation along a guessed
   curve. Two more inventories turn it into a fitted parameter and give genuine
   out-of-event validation. ERA5 says 2019 and 2021 were 197 mm and 117 mm
   3-day events against 2018's 332 mm — usefully different magnitudes.
   `src/acquire_flood_event.py` is ready once Earth Engine is authenticated.
3. **Route the runoff in the hazard step.** `combine()` currently applies
   SCS-CN *pointwise*: a pixel's forcing is the rain that fell on it, and none
   of what its catchment delivers. The D8 network in `src/routing.py` already
   exists and is only used for a static feature. Accumulating `Q(x, P)`
   downslope would make the rainfall response genuinely spatial. This is the
   cheapest real improvement on the list.
4. **Spatially variable rainfall.** Scenarios apply one depth everywhere.
   Accepting a rainfall raster (IMD gridded, or a forecast field) instead of a
   scalar is a small interface change with a large realism gain.
5. **Drainage infrastructure and tide.** Storm-drain capacity, canal network,
   and outfall tide-locking are what actually determine urban waterlogging in a
   coastal backwater city. Two pixels with identical terrain and one blocked
   drain between them currently receive identical predictions.
6. **Depth, not just probability.** Requires either hydrodynamic modelling or a
   depth-labelled inventory. Furthest away, and only worth it after 1-3.

## Outstanding engineering

- Three `outputs/flood_prob_*.tif` remain modified but uncommitted; they predate
  this work and are superseded.
- The old `fig1`–`fig6` PNGs and `paper_metrics.json` are left in place. Nothing
  regenerates them; deleting them is a manuscript decision.
- The archived `.pth` checkpoints were trained on the pre-fix corrupted rasters
  and should not be used.
- `urban_mask` (importance 0.000) is still in the feature list; fold its removal
  into the next retrain.
- The Streamlit app's data path, rendering and every static layer are
  smoke-tested, and the FastAPI routes are exercised end to end, but neither
  server has been run under a browser in this environment. The Docker build and
  CI workflow have not been executed either.

## API

```
GET /api/health            service and data readiness
GET /api/scenarios         available rainfall scenarios
GET /api/model             model card, CV metrics, assumptions
GET /api/conformal         coverage guarantee, thresholds, per-stratum coverage
GET /api/map/{mm}          hazard overlay (base64 PNG) + WGS84 bounds
GET /api/risk_stats/{mm}   risk-class breakdown with real areas in km2
GET /api/runoff            SCS-CN runoff for a rainfall depth and curve number
GET /api/places            known place coordinates
```

## Results figures

```bash
python src/benchmark_models.py          # real baselines, identical folds
python evaluation/generate_figures.py   # figures from measured numbers
```

| Figure | Shows |
|---|---|
| `fig1_roc_spatial_cv.png` | ROC curves computed from out-of-fold predictions |
| `fig2_cv_inflation.png` | Spatial-block vs random k-fold AUC, per model |
| `fig3_reliability.png` | Predicted vs observed frequency on a held-out split |
| `fig4_threshold_derivation.png` | The PR curve the risk band edges were read off |
| `fig5_conformal_coverage.png` | Marginal vs class-conditional coverage |
| `fig6_graph_ablation.png` | The graph negative result, with the edges-off control |
| `fig7_feature_importance.png` | Permutation importance |
| `fig8_reference_rainfall.png` | ERA5 accumulations behind the reference depth |

### Why the previous figures were replaced

`evaluation/generate_all_charts.py` is deprecated and now refuses to run. It did
not plot measured results:

- `synthetic_roc(auc, seed)` **generated** a ROC curve shaped to hit a target
  AUC. The curves in `fig2_roc_curves.png` came from that function, not from any
  model's predictions.
- The confusion matrix was built from the comment *"Simulate realistic confusion
  matrix values from precision=0.712, recall=0.632"* — back-derived from the
  numbers it appeared to demonstrate.
- The baseline rows (Logistic Regression, SVM, Random Forest, 3-Layer CNN,
  Standard U-Net) were hardcoded literals with no corresponding run.

`paper_metrics.json` is also internally inconsistent: it reports the Attention
U-Net at F1 = 0.670 while its own `training_history` shows validation F1 peaking
at **0.008**, and lists Logistic Regression and SVM with AUC 0.798/0.884 but
precision, recall, F1 and IoU all exactly 0.0. The channel count is 12 in the
literature table, 7 in the ablation and 13 in `config.py`.

The old `fig1`–`fig6` PNGs are left in place rather than deleted — that is a
call for whoever owns the manuscript — but nothing regenerates them.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
pytest tests/ --cov=src --cov-report=term-missing
```

The suite includes regression tests for previously-shipped defects: nodata
sentinels surviving post-processing, the 30 m vs 10 m cell-size error, layer
nodata rules that never matched, and non-monotonic rainfall response.

## Docker

```bash
docker build --target app -t geoai-flood .
docker run -p 8501:8501 geoai-flood

docker build --target fastapi -t geoai-flood-api .
docker run -p 8000:8000 geoai-flood-api
```

The image expects `data_aligned/` to exist in the build context — run
`python align_data.py` first.

## Acknowledgments

ESA WorldCover (land cover), Copernicus Sentinel-1 (SAR flood inventory),
Open-Meteo (forecasts), USDA NRCS NEH-630 (curve number method).

---

**Disclaimer**: decision support only. Verify against official meteorological
and disaster management authorities before acting on any output.
