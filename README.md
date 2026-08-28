# GeoAI Flood Risk Dashboard — Ernakulam, Kerala

> A GeoAI framework for flood susceptibility mapping and rainfall-conditioned
> waterlogging risk in Ernakulam District, Kerala, India.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

## Live rainfall evaluation

The two model layers are computed **on demand**. Moving the rainfall slider
re-evaluates the model at that depth in roughly 90–96 ms; nothing is
interpolated between pre-rendered scenario rasters.

That is possible because both surfaces are closed-form in rainfall, and both
now route runoff downslope rather than evaluating it pointwise:

```
fluvial(x, P) = sigma( logit(S(x)) + beta * ln( Q_routed(x,P) / Q_routed(x,P_ref) ) )
pluvial(x, P) = f( routed SCS-CN runoff(P), local gradient )
```

`S` is the learned susceptibility and does not depend on rainfall, so it is
loaded once. `Q_routed(x, P)` is `pluvial.routed_runoff_ratio()`: SCS-CN
runoff depends on a pixel only through its curve number — seven curve numbers
in this district — so the upstream *count* of each class draining through x
(`grid.basis`) is accumulated once at cache-build time, and a new rainfall
value costs seven scalar evaluations plus a weighted sum over that fixed
basis, with **no re-routing and no model re-fit**. Both the fluvial layer and
the pluvial index draw on the same basis, so a rainfall change moves them
consistently rather than the fluvial layer reacting locally while the pluvial
one reacts to the catchment.

```bash
python src/live_model.py --build       # precompute, ~30 s, writes a 5.9 MB cache
python src/live_model.py --benchmark   # 90-96 ms per rainfall value
```

Verified by driving the app (2026-08-27, headless Chromium against the running
server, not just an import-and-check): at 50 / 120 / 200 / 300 mm the rendered
overlay is four distinct images, the alert banner moves from MONITORING to
CRITICAL, and the reported expected flooded area on the live display grid
moves 0.0 → 0.7 → 4.8 → 20.5 km².

### Two layers, deliberately not blended

| Layer | What it is | Trust |
|---|---|---|
| **Flood Probability (live)** | Probability of riverine/backwater inundation | Calibrated. Spatial-block AUC 0.824, conformal coverage guarantee. |
| **Waterlogging Index (live)** | Rain-driven waterlogging pressure, 0–1 | **Proxy-validated.** AUC 0.807 (95% CI 0.698–0.908) against 14 documented hotspots vs. an elevation-matched urban background — not yet against real incident records. |

They answer different questions and neither is validated against the thing it
would ultimately need to be: real waterlogging incident records. Averaging
them would launder the weaker validation into the stronger one's credibility,
so the app keeps them as separate layers with separate colour ramps, and the
waterlogging layer carries a warning banner explaining the proxy control.

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
| Flood probability | AUC 0.294 | **0.353** — worse than chance |
| **Waterlogging index** | AUC 0.839 | **0.807** (95% CI 0.698–0.908) |

Only the **waterlogging index** has skill here, and it survives the
elevation-matched control — the test that matters, since every documented
hotspot is a low-lying junction and a model knowing only "low ground floods"
would ace an unrestricted comparison and collapse against a matched one.

### The flood layer is anti-correlated with waterlogging, and that is correct

The flood probability layer scores **0.35** — documented hotspots rank *below*
ordinary urban ground. That is not a defect. It is geography:

| | Mean column (of 7374, west→east) | Median elevation |
|---|---|---|
| Documented Kochi hotspots | **1176–1666** (far west) | ~8 m |
| NDEM 2018 flood extent | **2849** (inland) | 13.9 m |

**Kochi city did not experience riverine flooding in August 2018.** The flood
was inland along the Periyar — Aluva, Perumbavoor, Kalamassery. **All 14
hotspots fall outside the NDEM flood extent.** A model trained to predict that
event therefore ranks the city centre as comparatively safe, which is right for
riverine inundation and irrelevant to street ponding.

The two phenomena are not merely different mechanisms. In this district they
happen in **different places**. That is why the layers are kept separate and
never blended: averaging them would have destroyed the only waterlogging
signal the project has.

Note this reverses an earlier result. Trained on the Sentinel-1 inventory the
flood layer scored 0.801 here, which looked like waterlogging skill. That
inventory was coastal-fringe-heavy (mean column 2373, median elevation 4.0 m)
and so sat nearer the city by accident. The better-timed label removed the
coincidence.

What the surviving result does **not** license:

- **n = 14.** The intervals are wide by construction, enough to separate
  "skill" from "chance" and nothing finer.
- **Reporting bias.** Journalists cover junctions that stall traffic, so the
  sample favours arterial city-centre roads over residential streets that
  flood as often. Some of the measured skill may be "near a canal or major
  road" rather than "waterlogs".
- **The index is not a probability.** It is a physics-derived 0–1 ranking with
  no calibration, because there is nothing to calibrate it against.
- **It is a test set, not a training set.** Nothing was fitted to it.

Regenerate with `python src/waterlogging_validation.py`; every point carries
its source in `models/waterlogging_validation.json` so any label can be
audited or dropped.

## Read this first: what the model does and does not predict

**The *fluvial* layer does not predict urban waterlogging.** It predicts the
extent of riverine and backwater inundation. Those are different phenomena,
and the distinction is not cosmetic — it decides what the output can be used
for. (A separate *pluvial* layer, covered below, is built specifically for
street-level ponding and does show real skill at it.)

The evidence, from the current rasters (NDEM-trained, `osm_drain_dist` and
`osm_drain_density` included as inputs, hazard at the 443 mm reference event):

| | Top 0.1% of predicted hazard | Whole district |
|---|---|---|
| Median elevation | **7.9 m** | 29.4 m |
| Median urban fraction | **0.0** | **1.0** |
| Median distance to drainage | 912 m | 1,083 m |

The district is majority built-up, yet the model's highest-hazard zone is
still almost entirely *not* built-up: low-lying paddy and wetland fringing the
Vembanad backwaters, now somewhat closer to a mapped drainage channel than the
district median rather than farther from it. Sampled at named places, at the
reference event:

| Place | Hazard @443 mm | Band |
|---|---|---|
| Perumbavoor | 0.065 | high |
| Aluva (on the Periyar) | 0.008 | safe |
| Kaloor | 0.005 | safe |
| Vyttila | 0.003 | safe |
| Edappally | 0.003 | safe |
| Ernakulam / MG Road | 0.002 | safe |

Kochi's urban core still reads as low fluvial risk. That is not a bug — it is
what the training label contains. NDEM's 2018 inundation footprint is
inland-and-lowland, not the city centre (see "Why this is lower than the
earlier 0.919" below), so the fluvial model correctly learns that this
particular kind of flooding — river and backwater inundation — favours
low-lying non-urban land. Drainage proximity *is* now an input to this
model (`osm_drain_dist` ranks 5th of 16 features, `osm_drain_density` 6th),
but it does not flip the urban core to high risk here: at the 14 documented
waterlogging hotspots the fluvial layer still scores near chance, AUC 0.388
(95% CI 0.269–0.527) against an elevation-matched background. Street-level
ponding is a different mechanism — governed by storm-drain capacity, canal
backup on the tide, and blockage — and that mechanism is what the pluvial
layer targets instead.

**What the fluvial output is legitimately good for:** ranking the low-lying
rural and peri-urban floodplain by relative inundation susceptibility, with
calibrated probabilities and a coverage guarantee.

**What it is not good for on its own:** answering "will this street flood if
120 mm falls tomorrow" — use the pluvial index for that question instead. See
[Known limitations](#known-limitations) for the full list and
[What would close the gap](#what-would-close-the-gap) for what remains.

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
| Random k-fold | 0.882 | 0.871 | 0.139 |
| **Spatial block (5 km)** | **0.822** | **0.804** | **0.173** |
| Spatial block, low-lying only (DEM ≤ 21.7 m) | 0.840 | 0.822 | 0.163 |

### Why this is lower than the earlier 0.919, and why it is a better model

The model previously trained on a single Sentinel-1 scene and scored 0.919.
That number was almost entirely elevation:

| Label | AUC from elevation **alone** | Full 15-feature model | Added by the other 14 |
|---|---|---|---|
| Sentinel-1, 21 Aug 2018 | **0.912** | 0.919 | **+0.007** |
| NDEM, 17–18 Aug 2018 | **0.763** | 0.822 | **+0.059** |

Fourteen conditioning factors were buying 0.007 of AUC over a single one. The
Sentinel-1 inventory was 94% non-urban backwater fringe with a median elevation
of 4.0 m, so "low ground is wet" solved it. The NDEM inventory is 43% urban at
a median 13.9 m — much closer to the district as a whole (50% urban, 30.6 m) —
and the model now contributes **eight times more** over that baseline.

A lower headline AUC against a harder, better-timed, more representative label
is the better model. Calibration improved too: the worst predicted-versus-
observed gap fell from 0.033 to **0.020**, and expected flooded area at the
reference event matches the observed extent exactly (78.7 km²).

The table above is a snapshot of that one ablation. The model has since grown
to 16 features — `osm_drain_dist` and `osm_drain_density` added, `urban_mask`
dropped as dead weight — and the NDEM spatial-block AUC held at **0.8240** to
four decimals through both changes, so none of this section's conclusions
change. Current permutation importance for the full feature set is in
`models/susceptibility_metrics.json` and `evaluation/fig7_feature_importance.png`.

Random k-fold **overstates AUC by 6.0 points** here. Neighbouring pixels of a
10 m raster are near-duplicates, so a random split leaks the test set into
training. The spatial-block number is the one to quote. The low-lying row is
the operationally relevant case: it measures whether the model can rank two
low-lying pixels against each other, which is the decision a planner actually
faces.

### Calibration

Probabilities are isotonic-calibrated out-of-fold. The isotonic curve is fitted
on one half of the out-of-fold predictions and measured on the other, so the
reliability figure is not circular; the worst predicted-vs-observed deviation
across ten probability bins is **0.0309** (`models/susceptibility_metrics.json`
→ `worst_calibration_gap_balanced_scale`).

Training is deliberately balanced 1:1, but only **3.53%** of the sampling
domain (786,504 of 22.3M pixels) is NDEM-flooded in 2018. Probabilities are
therefore shifted back to the population base rate. The closed-form
case-control offset assumes randomly drawn absences, which ours are not — they
are elevation-stratified. The offset is instead **solved by bisection** against
a uniform 400k-pixel sample of the district so that the expected flooded area
equals the observed extent:

| | Prior logit offset | Expected flooded area at 443 mm |
|---|---|---|
| Closed-form (assumes random absences) | −3.3175 | — |
| **Fitted by bisection** | **−3.3779** | **78.65 km²** (target: 78.65 km²) |

That match is exact by construction — bisection is solving for the offset that
produces it — so it confirms the fit converged, not that the model is right.
The independent check is against the *written raster*, pixel by pixel, over the
2,114.5 km² intersection of the model domain and valid 2018 label: the
hazard-at-443mm raster sums to 74.56 km² of expected flooding against 77.5 km²
actually flooded (ratio **0.962**), and scores **AUC-ROC 0.887, AUC-PR 0.254**
against a 0.037 no-skill baseline at this prevalence.

### Monotonicity

Every pixel is verified non-decreasing across all nine current scenarios:

| Transition | Pixels where hazard decreases |
|---|---|
| 50 → 100 mm | 0 |
| 100 → 150 mm | 0 |
| 150 → 200 mm | 0 |
| 200 → 250 mm | 0 |
| 250 → 300 mm | 0 |
| 300 → 400 mm | 0 |
| 400 → 443 mm | 0 |
| 443 → 500 mm | 0 |

The maps this replaces were not monotonic: the shipped 100 mm map had mean
probability 0.124 against the 150 mm map's 0.025.

### Risk bands

The band edges are read off the precision-recall curve of the reference-event
hazard map against the 2018 inventory, not chosen as round numbers:

| Band | Lower edge | Captures of observed flooding | Precision | Cumulative area at 443 mm |
|---|---|---|---|---|
| Moderate | 0.024 | 95.4% | 0.078 | 852.5 km² |
| High | 0.056 | 81.4% | 0.134 | 412.5 km² |
| Severe | 0.125 | 38.6% (max F1) | 0.281 | 106.3 km² |
| Critical | 0.269 | 4.1% | 0.539 | 5.9 km² |

Area is cumulative — "≥ critical" rather than the critical slice alone. District
base rate is 3.64% (NDEM, 443 mm), so the critical band runs ~15× the no-skill
rate. The previous thresholds, against the Sentinel-1 inventory (base rate
1.4%), were 0.022 / 0.070 / 0.133 / 0.271; before that, an uncalibrated score's
0.10 / 0.20 / 0.30 / 0.50 classified the actual 2018 catastrophe as "monitoring
active". **Re-derive these whenever the model is retrained**
(`python src/risk_thresholds.py`) — they are properties of the fitted
probabilities, not constants. Re-derived 2026-08-27 after routing the
fluvial rainfall response and refitting beta; the edges moved only slightly
(0.023/0.050/0.134/0.297 → 0.024/0.056/0.125/0.269), consistent with the
routing change itself moving the 443 mm reference-event hazard almost not at
all — the drift here is more likely PR-curve sampling noise than a real shift,
and was re-derived anyway rather than judged close enough to skip.

Permutation importance (AUC drop when shuffled), current 16-feature model:
elevation (`dem`) 0.1357, **`dem_rel_1km` 0.0983**, **`upstream_cn` 0.0700**,
distance to river 0.0504, **`osm_drain_dist` 0.0466**,
**`osm_drain_density` 0.0409**, slope 0.0164, distance to built-up 0.0161,
flow accumulation 0.0158, curve number 0.0145, HAND 0.0136, TWI 0.0105. Full
list in `models/susceptibility_metrics.json`.
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

- **Reference event = 443.2 mm** (`RAINFALL.reference_event_mm`) — *derived*,
  not assumed, and re-derived twice. First from ERA5 reanalysis (3-day max
  331.6 mm, 14–16 Aug 2018), then superseded by IMD 0.25° gauge-based gridded
  rainfall over the district bbox
  ([src/reference_rainfall.py](src/reference_rainfall.py) `--source imd`):
  max 1-day 191.1 mm (16 Aug), 2-day 334.9, **3-day 443.2 (15–17 Aug)**, 5-day
  526.3, month 919.9. IMD runs 1.34× ERA5 here, consistent with ERA5
  under-resolving monsoon extremes. The 3-day window is kept because
  `HYDRO.amc` is III, which already encodes a wet antecedent 5 days — a 5-day
  storm total would count antecedent wetness twice.

  Caveat: IMD's 0.25° grid still under-resolves any sub-grid orographic
  extreme, and no rainfall product captures the Periyar reservoir releases
  that contributed to the 2018 inundation. Treat it as a proxy for total
  forcing, not a measured storm.
- **Soil hydrologic group.** `src/soil_hsg.py` exists, is tested, and reads
  real SoilGrids texture — but it is **not wired into the model**. Measured
  entropy across the district is 0.12 bits out of a possible 2.0 (98.5% of the
  domain classifies to group D from a median clay-loam texture at 250 m
  resolution), so adopting it acts almost like a uniform curve-number bump
  (+2.16 median, roughly +12.7% runoff at 100 mm) rather than real spatial
  heterogeneity, and it would require re-running the prior offset, the risk
  thresholds and `beta` to stay calibrated. `HydrologyConfig.curve_numbers`
  still uses the single group-C table it always did. See the module docstring
  for the full finding before wiring it in.
- **AMC III (wet)** antecedent moisture, appropriate for a monsoon-season
  flood-forecasting product.
- **Initial abstraction ratio 0.05** rather than the classic 0.20, following
  Woodward et al. (2003), with the retention rescaled accordingly.
- **beta = 3.078** logit units per log-unit of runoff ratio — the sensitivity
  of flood odds to rainfall. *Fitted* (`python src/fit_beta.py`) jointly on
  three NDEM events (2019, 2020, 2021 — 2018 is fixed by construction and
  carries no information). Leave-one-out now runs **2.815 to 4.963**, down
  from the two-event fit's 2.77–8.00, and no longer hits the search bound —
  identifiability improved with the third event, not just the point estimate.
  Still not fully resolved: the joint fit overpredicts both 2019 and 2020 by
  a similar factor (~1.9–2.0x) while underpredicting 2021 (0.73x). For 2019
  that's traced to acquisition-timing coverage loss (see
  [Known limitations](#known-limitations)); for 2020 the same check was run
  and *ruled that out* — its NDEM footprint bbox (2,120 km²) is comparable to
  2018's, not truncated like 2019's — so something about the response curve
  itself, not coverage, is unaccounted for at that depth. A fourth event
  would help distinguish "one global beta is the wrong model" from
  "acquisition noise across all three."
- **Population and damage figures** in the alert panel are district-average
  density and a flat per-km² damage rate. They are labelled "planning
  estimate" in the UI and are not model outputs.

## Known limitations

- **The land cover is from 2018, and the district has since grown 23%.**
  Measured, not asserted: Google Dynamic World puts Ernakulam's built-up area
  at 691 km² in 2018 and 852 km² in 2025 — **+23.3%, +161 km²**
  (`python src/landcover_drift.py`, cached in `models/landcover_drift.json`).
  Built-up surfaces carry a far higher curve number, so on ground developed
  since 2018 the model routes less runoff than the surface now produces and
  **understates present-day risk there**.

  The fix is not a data refresh. 2018 land cover is the *correct* land cover
  for training: the labels are the August 2018 flood inventory, and the
  features have to describe the surface that produced those floods. Swapping in
  2025 land cover and retraining would misalign features from labels — a worse
  error than the one it corrects. Doing this properly means separating the
  training epoch from the inference epoch (train the susceptibility surface on
  2018, evaluate runoff over current land cover), which is a temporal-transfer
  design change and is not attempted here. The app states the gap in its
  Advanced panel rather than letting a low number for a newly built-up ward
  pass without comment.

- **Susceptibility is trained on 2018 alone; `beta` leans on 2021 alone.**
  Training uses `ndem_flood_2018` exclusively — the prior offset is fitted to
  reproduce the same event the model was trained on, so "expected area equals
  observed area" is a consistency check, not independent validation. NDEM does
  supply three more events (2019: 31.3 km² at 412.5 mm; 2020: 11.1 km², IMD
  rainfall not yet derived; 2021: 4.1 km² at 173.7 mm), and those went into
  fitting `beta` — but 2019 is discounted (7% less rainfall than 2018, yet 60%
  less extent, and its label footprint stops 1,477 columns short of 2018's,
  which is acquisition coverage, not hydrology), which leaves 2021 doing
  essentially all the work. A genuinely independent, well-covered multi-event
  validation is still open.
- **Presence-only inventory.** NDEM/SAR-derived inundation observes water, not
  "dry". Absences are pseudo-absences: buffered 5 px away from observed
  flooding and stratified across elevation deciles matched to the presence
  distribution. This is the main source of irreducible label noise.
- **Permanent water is excluded from the model domain.** It accounted for
  80.3% of the raw Sentinel-1 flood inventory that preceded the NDEM switch;
  including it trains a lake detector.
- **The archived `.pth` checkpoints are stale.** They were trained against
  feature rasters whose nodata sentinels had been clipped into the valid
  range. `src/inference_final.py` still loads them for reproducibility, but
  their output should not be used.
- **No local hydrodynamic routing.** `combine()` still applies SCS-CN
  pointwise — a pixel's forcing is the rain that fell on it, not what its
  catchment delivers — so the model gives flood *susceptibility*, not
  inundation depth. (Basin-scale upstream contributing area is now built
  separately by `src/upstream_routing.py`; see below for whether it validated
  and whether it is wired in.)
- **The inventory understates the event.** NDEM's 2018 footprint is 78.7 km²
  of non-permanent-water flooding across Ernakulam, small for an event that
  displaced people district-wide — satellite-derived inundation catches open
  water at one overpass, not peak extent, and cannot see under vegetation or
  inside built-up areas. Every absolute area and exposure figure the system
  reports inherits that floor, which is why the population and damage numbers
  in the alert panel are labelled planning estimates rather than predictions.
- **Soil is one group everywhere**, by design for now — see
  [Assumptions worth auditing](#assumptions-worth-auditing). A real per-pixel
  raster exists (`src/soil_hsg.py`) but carries too little spatial information
  at 250 m to be worth the recalibration cost of adopting it as measured.

## What would close the gap

Ordered by how much each one moves the system toward actually answering
"will this place waterlog at X mm", not by effort.

1. **A waterlogging-specific label.** Still the single blocking issue, and now
   tested twice against 14 documented hotspots: Sentinel-1 open-water extent
   scores 0/14, NDEM inundation scores 0/14 (fluvial AUC 0.388, chance).
   Neither satellite product sees 20 cm of water between buildings. Municipal
   complaint logs, KSDMA incident reports, traffic-police road-closure records
   or geolocated news reports would give a few hundred points of the *right*
   phenomenon and let the pluvial index (currently AUC 0.807 against an
   elevation-matched proxy control, not against real waterlogging incidents)
   be validated against the thing it actually claims to predict. A formal
   request is drafted at
   [docs/data-requests/ksdma-waterlogging-records.md](docs/data-requests/ksdma-waterlogging-records.md).
2. ~~More events, to fit `beta`.~~ **Done, twice now.** `python src/fit_beta.py`
   fits the rainfall sensitivity against NDEM extents instead of assuming it
   (1.8 → 2.8 on two events → **3.078 on three**, once 2020's IMD rainfall was
   derived). Leave-one-out narrowed from 2.77–8.00 to **2.815–4.963** — real
   progress, no longer hitting the search bound — but the joint fit still
   overpredicts 2019 and 2020 by a similar factor and underpredicts 2021, and
   the 2020 miss isn't explained by the acquisition-coverage issue that
   explains 2019's (checked and ruled out). A fourth event is what would
   actually resolve whether that's residual noise or a real curve-shape gap.
3. ~~Route the runoff in the hazard step.~~ **Done, with a modest result.**
   `combine()` no longer applies SCS-CN pointwise by default; it accepts a
   `runoff_ratio` computed by `pluvial.routed_runoff_ratio()`, which
   accumulates rainfall-dependent runoff downslope through the same D8
   network `PluvialModel` already used for the pluvial index, reused rather
   than reimplemented at 10 m. `hazard.generate_hazard_rasters()`,
   `live_model.fluvial_probability()` (so the live slider and the batch
   rasters answer the same question) and `fit_beta.py` all now route by
   default.

   The honest result: it barely changed anything. Beta refit against the
   routed ratio moved 3.078 → 3.085, and the routed hazard surface at 200 mm
   correlates 0.9987 with the pointwise one (mean pixel difference +0.00002,
   max 0.042). Susceptibility already carries most of the catchment signal
   through features like `upstream_cn` and `river_dist`, so the *rainfall-
   response ratio* — the marginal change in that signal away from the
   reference event — had comparatively little left to add. Worth having
   anyway: it replaces an assumption with a measurement, which is a different
   thing from confirming nothing changed. `src/upstream_routing.py`'s
   *basin-scale* contributing-area layer (Periyar, Chalakudy, Muvattupuzha)
   remains a separate, still-open gap — bringing water in from beyond the
   district boundary, which this change does not touch.
4. **Spatially variable rainfall.** Scenarios apply one scalar depth
   everywhere; `runoff_depth()` takes a single `rainfall_mm` float, not a
   raster. Accepting a rainfall raster (IMD gridded, or a forecast field)
   instead of a scalar is a small interface change with a large realism gain.
5. **Drainage infrastructure and tide — partially done.** OSM drainage
   (`src/osm_drainage.py`) is now in both models: it ranks 5th/6th of 16
   features in the fluvial susceptibility model, and proximity-to-canal is the
   strongest single signal in the pluvial index (AUC 0.713 alone). What is
   still missing is capacity and state — pipe diameter, invert level,
   blockage, and outfall tide-locking, none of which OSM records. Two pixels
   with identical terrain and drainage geometry but one blocked outfall still
   receive identical predictions.
6. **Depth, not just probability.** Requires either hydrodynamic modelling or a
   depth-labelled inventory. Furthest away, and only worth it after 1–3.

## Outstanding engineering

- Which artefacts are current versus superseded — including the three
  `outputs/flood_prob_*.tif` that predate this work, the archived `.pth`
  checkpoints, and the old `fig1`–`fig6` PNGs / `paper_metrics.json` — is
  tracked in [docs/artefacts.md](docs/artefacts.md) rather than duplicated
  here. Nothing regenerates the old figures automatically; deleting them is a
  manuscript decision.
- `urban_mask` (permutation importance 0.000, then −0.0000, then −0.0001 across
  three retrains) has been dropped from `SUSCEPTIBILITY_FEATURES`. Confirmed
  free: spatial-block AUC held at 0.8240 to four decimals with it removed.
- Upstream basin-scale routing (`src/upstream_routing.py`, WhiteboxTools
  breach-then-D8 over the 25,085 km² contributing area) is running as of this
  writing; it refuses to write an aligned raster unless it validates against
  published catchment areas at four gauging points, so either it passed and
  `upstream_area_aligned.tif` exists, or it did not and there is no new
  raster to accidentally trust. Check
  `GeoAI_New/routing_work/routing_validation.json` for the outcome. Either
  way it is not yet an input to `combine()` — see item 3 above.
- The Streamlit app's data path, rendering and every static layer are
  smoke-tested, and the FastAPI routes are exercised end to end. Both servers
  have since been run for real: the dashboard under a browser (all 17 layers,
  the rainfall slider, the 2018 event and the advanced panel), and the API by
  requesting all nine routes against the real artefacts. The Docker build and
  the CI lint/format/type steps have all been executed too.
- The rainfall forecast's "as of" date trails live conditions by however much
  of the current calendar year has passed. IMD's yearwise archive ships a
  fixed-size binary per year and the in-progress year fails its parser, so the
  most recent usable series ends at the last closed year. This is a property of
  the free archive format, not a caching bug.

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

## Docker and deployment

```bash
# One-time: build the display-resolution rasters the image ships (21 MB).
python src/make_display_rasters.py

# Optional but recommended: cache the rainfall forecast so the container
# never reaches for the 874 MB IMD archive.
python src/rainfall_forecast.py --snapshot

# Both services.
docker compose up --build          # dashboard :8501, API :8000
docker compose up dashboard        # dashboard only
```

Or per-image:

```bash
docker build --target app -t geoai-flood-dashboard .   # Streamlit, :8501
docker build --target api -t geoai-flood-api .         # FastAPI,   :8000
```

### What ships, and what does not

The dashboard image deliberately does not contain the model's *inputs*, only
its *outputs*. That distinction is what makes it deployable:

| Excluded | Size | Why the running app does not need it |
|---|---|---|
| `GeoAI_New/` | 3.7 GB | Full-resolution conditioning factors. Every static layer is read through `read_downsampled()` at 1000 px, so the image ships `display/` (21 MB) instead — identical pixels, 48× smaller. Needed only by `align_data.py`. |
| `data_aligned/` | 1.3 GB | Training/feature inputs. The app reads the precomputed `models/live_model.npz`. |
| `data/imd_rain/` | 874 MB | Rainfall archive. The forecast ships as `models/rainfall_forecast_latest.json`; the prediction is anchored to the last closed calendar year, so a snapshot loses nothing. |
| `models/*.pth`, `*.h5` | 245 MB | Archived U-Net weights (`docs/artefacts.md`). Nothing in the active pipeline loads them. |
| `.venv/`, `.git/` | ~10 GB | Were previously being sent to the daemon on every build — there was no `.dockerignore`. |

The API image is separate because it serves the pre-generated per-scenario
rasters (`outputs/flood_hazard_*.tif`, ~530 MB) that the dashboard does not
open — the dashboard evaluates hazard live instead. Running both from one
container would mean shipping both data sets and supervising two servers under
PID 1; `docker-compose.yml` runs them as two services.

### Publishing to Hugging Face Spaces

```bash
pip install huggingface_hub
python deploy/push_to_space.py --dry-run                      # list the payload first
python deploy/push_to_space.py --space <user>/<name> --token hf_...
```

The token needs *write* scope (<https://huggingface.co/settings/tokens>); it is
never written to disk. `--dry-run` assembles and lists the payload without
creating or pushing anything.

Two things this has to work around, both verified rather than assumed:

- **Spaces builds a Dockerfile with no `--target`**, taking whichever stage ends
  the file. The `app` stage is therefore last on purpose. Simulated locally with
  a bare `docker build .`: it produces the dashboard, serving on 8501 as uid
  1000. Reordering does not affect `--target api` or `docker-compose.yml`.
- **The Space needs artefacts this repo does not track.** `display/`,
  `models/live_model.npz` and `outputs/conformal_sets.tif` are derived and
  gitignored. Rather than reversing that, `deploy/push_to_space.py` assembles a
  self-contained Space repo in a temp directory — tracked source plus 33.5 MB of
  runtime artefacts, with `GeoAI_New/`, `data_aligned/`, `data/`, `tests/` and
  the per-scenario hazard rasters excluded. GitHub stays source-only.

The Space's `README.md` comes from `deploy/SPACE_README.md`, whose YAML
front-matter (`sdk: docker`, `app_port: 8501`) is what tells Spaces how to run
it. That front-matter is required and its absence is why the March 2026
Dockerfile-only attempt could not have worked.

### Deployment notes

- Both images run as a non-root user and declare a `HEALTHCHECK`.
- The API's `CORSConfig` is `allow_origins=["*"]` with `allow_methods=["GET"]`
  and `allow_credentials=False`. That is intentional for a read-only public
  data API; tighten `APIConfig.cors_origins` if it is ever put behind auth.
- Neither service holds state, so both scale horizontally. The API's
  `load_hazard` LRU-caches up to 8 rasters of ~42 M float32 pixels, which is
  what the 4 GB memory limit in `docker-compose.yml` accounts for.

## Acknowledgments

ESA WorldCover (land cover), Copernicus Sentinel-1 (SAR flood inventory),
Open-Meteo (forecasts), USDA NRCS NEH-630 (curve number method).

---

**Disclaimer**: decision support only. Verify against official meteorological
and disaster management authorities before acting on any output.
