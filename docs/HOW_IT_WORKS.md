# How it works: data in, AI in the middle, maps out

A plain-language walkthrough of the Ernakulam flood model — what goes in, what
the machine learning actually does, and what comes out.

For the short version: **the model looks at 16 things about every 10-metre square
of the district, learns which combinations were under water during the August
2018 flood, and then predicts how likely each square is to flood at a rainfall
depth you choose.**

Every figure here is taken from the code or a committed results file. Sources are
named so anything can be checked.

**Contents**

1. [The one-paragraph version](#1-the-one-paragraph-version)
2. [Inputs: the data](#2-inputs-the-data)
3. [The 17 layers in the app, explained](#3-the-17-layers-in-the-app-explained)
4. [The AI: what is actually learned](#4-the-ai-what-is-actually-learned)
5. [From susceptibility to a rainfall-specific map](#5-from-susceptibility-to-a-rainfall-specific-map)
6. [A second, separate model: rainfall forecasting](#6-a-second-separate-model-rainfall-forecasting)
7. [Outputs: what you get](#7-outputs-what-you-get)
8. [How good is it, honestly](#8-how-good-is-it-honestly)
9. [Glossary](#9-glossary)

---

## 1. The one-paragraph version

The district is cut into a grid of 10-metre squares — about 42 million of them.
For each square the system records 16 measurements: how high it is, how steep,
what is built on it, how far to the nearest river, how much land drains into it,
and so on. It is then shown a map of what actually flooded in August 2018 and
learns, from 117,706 example squares, which combinations of those 16 measurements
went under water. That learned pattern is called **susceptibility** — "if a big
storm hits, does this square flood?" — and it does not depend on rainfall.
Rainfall is added separately, using standard hydrology (the SCS Curve Number
method) to work out how much of a given storm becomes runoff. Combine the two and
you get a probability for any rainfall depth you ask about.

---

## 2. Inputs: the data

### The grid everything sits on

Every input is resampled onto one common grid so the model can compare like with
like:

| Property | Value |
|---|---|
| Cell size | 10 m |
| Grid | 7,374 × 5,690 cells |
| Coordinate system | EPSG:32643 (UTM zone 43N) |
| Model domain | 2,114 km² of the district's 3,068 km² |

The domain is smaller than the district because permanent water bodies and
invalid cells are excluded — they are not "at risk of flooding" in any useful
sense.

### The 16 things the model measures about each cell

| # | Feature | Plain meaning | Where it comes from |
|---|---|---|---|
| 1 | `dem` | Height above sea level | SRTM-derived elevation |
| 2 | `slope` | Steepness | Computed from the DEM |
| 3 | `hand` | Height above the nearest drainage channel | Computed from the DEM |
| 4 | `twi` | Topographic wetness — how much water gathers here | Computed from the DEM |
| 5 | `tpi` | Is this a local dip or a local rise? | Computed from the DEM |
| 6 | `spi` | Stream power — erosive force of flow | Computed from the DEM |
| 7 | `flow` | How many cells drain through this one | Computed from the DEM |
| 8 | `river_dist` | Distance to the nearest river | Derived from the river network |
| 9 | `urban_dist` | Distance to the nearest built-up area | Derived from land cover |
| 10 | `ndvi` | How much vegetation | Sentinel-2 satellite imagery |
| 11 | `ndwi` | How much surface water/moisture | Sentinel-2 satellite imagery |
| 12 | `curve_number` | How much rain runs off rather than soaking in | Land cover + soil group |
| 13 | `upstream_cn` | Average curve number of everything draining into this cell | D8 flow routing |
| 14 | `dem_rel_1km` | Height relative to the surrounding 1 km | Computed from the DEM |
| 15 | `osm_drain_dist` | Distance to the nearest mapped drain or canal | OpenStreetMap |
| 16 | `osm_drain_density` | Length of drain per km² nearby | OpenStreetMap |

Features 12–16 do not appear as layers in the app because they are *derived* —
built by the pipeline rather than downloaded.

**Why 13 and 14 exist.** A flood is not a property of one cell in isolation.
`upstream_cn` answers "what kind of land sheds water onto me", which no
neighbourhood average can express: a cell 200 m from rooftops but across a ridge
receives none of their runoff. `dem_rel_1km` answers "am I sitting in a regional
basin", the scale at which a coastal plain drowns. Together they rank 2nd and 3rd
in importance — see [section 4](#feature-importance).

### The answer sheet: what actually flooded

The model learns from the **NDEM inundation inventory for 17–18 August 2018**
(National Database for Emergency Management, via Bhuvan/NRSC). Every cell is
labelled flooded or not.

An earlier version used a single Sentinel-1 radar scene from 21 August instead.
That was replaced, and it matters:

| Label | Flooded area | Of which urban |
|---|---|---|
| Sentinel-1, 21 Aug | 31.3 km² | 2.0 km² (6.3%) |
| **NDEM, 17–18 Aug (used)** | **78.7 km²** | **33.8 km² (42.9%)** |

The Sentinel-1 scene was taken days after the peak and caught mostly rural
backwater fringe. The NDEM inventory is on the right dates and has **17× more
urban signal** — which is the part a city cares about.

---

## 3. The 17 layers in the app, explained

The sidebar offers 17 layers. They are not all the same kind of thing: three are
model *outputs*, eleven are model *inputs*, and three are context.

### Model outputs (what the system produces)

| Layer | What it shows |
|---|---|
| **Flood Probability (live)** | The main result. Probability that a cell floods at the rainfall you selected. Calibrated — the number means what it says. |
| **Waterlogging Index (live)** | A 0–1 *ranking* of street-flooding pressure. **Not a probability.** Use it to compare places, not to read off a risk percentage. |
| **Conformal Confidence** | Where the model is confident enough to support a decision at 90% confidence. Independent of rainfall. |

### Model inputs (these feed the prediction)

| Layer | What it shows | Importance rank |
|---|---|---|
| **DEM** | Ground elevation. Low ground floods — the single strongest signal. | 1st |
| **Distance to Water** | How far to the nearest river or backwater. | 4th |
| **Slope** | Steepness. Flat ground holds water; steep ground sheds it. | 7th |
| **Distance to Built-up** | Proximity to development. | 8th |
| **Flow Accumulation** | How much upslope land drains through each cell — the river network appears where this is high. | 9th |
| **HAND** | Height above the nearest drainage channel. A cell 2 m above a river behaves very differently from one 20 m above it. | 11th |
| **TWI** | Topographic wetness index — combines catchment size and slope into "how wet does this place tend to be". | 12th |
| **SPI** | Stream power index — the erosive energy of water flowing through. | 13th |
| **TPI** | Topographic position — whether a cell sits in a hollow or on a rise relative to its surroundings. | 14th |
| **NDWI (Water)** | Satellite water/moisture index. Highlights standing water and wet soil. | 15th |
| **NDVI (Vegetation)** | Satellite greenness index. Vegetation slows runoff; bare and built surfaces speed it up. | 16th |

### Context layers (displayed, but not fed to the model)

| Layer | What it shows | Why it is not a feature |
|---|---|---|
| **LULC** | Land cover — water, trees, crops, built-up, and so on. | Used **indirectly**: `curve_number` is derived from it, and that *is* a feature. Feeding both would be duplication. |
| **Urban Mask** | Where the district is built up. | Tested and **dropped**: permutation importance measured 0.000 across three retrains — `urban_dist` and `curve_number` already carry its signal. It is still used to define the urban background when validating, and to measure the urban share of a flood. |
| **Sentinel-1 Ground Truth** | The *older* flood inventory from 21 Aug 2018. | **Superseded.** Kept visible so the label the model was moved away from can still be inspected and compared. |

---

## 4. The AI: what is actually learned

### The model

**Gradient-boosted decision trees** — scikit-learn's
`HistGradientBoostingClassifier`. Not a neural network. It builds hundreds of
small decision trees in sequence, each correcting the previous one's mistakes.
This suits tabular data with mixed units far better than deep learning, and it
trains in minutes rather than hours.

Trained on **117,706 sample cells** (59,108 flooded, balanced 1:1 so the model
sees enough flood examples to learn from).

### The part most flood models get wrong

Training and testing are split by **5 km spatial blocks**, not randomly.

This matters enormously. Neighbouring cells of a 10 m raster are almost
identical. Split them randomly and a cell's own neighbour ends up in the test
set — the model has effectively seen the answer, and the score is inflated:

| Split method | AUC | Verdict |
|---|---|---|
| Random | 0.902 | **Inflated — do not quote this** |
| **Spatial block (5 km)** | **0.824** | The honest number |

The gap is **7.8 points of AUC** bought purely by leakage. The published score is
the lower one.

### Making the probabilities mean something

Two corrections turn a raw score into a usable probability:

1. **Isotonic calibration.** The raw output is stretched so that "0.3" really
   does mean roughly a 30% chance. The calibration curve is fitted on one half of
   held-out predictions and *measured* on the other, so the check is not
   circular. Worst error across ten probability bins: **0.031**.

2. **Base-rate correction.** Training is balanced 1:1, but only **3.53%** of the
   district actually flooded. Without correction every probability would be far
   too high. The standard textbook correction assumes randomly chosen non-flood
   examples; ours are elevation-stratified, so instead the offset is solved
   numerically until the model's expected flooded area matches the observed
   78.65 km².

### Saying "I don't know"

**Conformal prediction** gives a distribution-free guarantee: with 90% target
coverage, calibrated on 185,303 held-out cells.

| | Coverage |
|---|---|
| Target | 0.90 |
| Achieved (overall) | 0.87 |
| **On flooded cells** | **0.93** |
| On dry cells | 0.87 |

The per-class split is the point. A single overall number is a trap for a rare
hazard: a model can hit 90% overall while almost never covering the flood class.
The comparison variant in the same results file does exactly that — 0.85 overall,
but **0.003** on flooded cells. It passes the headline test by abandoning the
class that matters.

### Feature importance

Measured by shuffling each feature and seeing how much the AUC drops:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `dem` (elevation) | 0.1357 |
| 2 | `dem_rel_1km` | 0.0983 |
| 3 | `upstream_cn` | 0.0704 |
| 4 | `river_dist` | 0.0504 |
| 5 | `osm_drain_dist` | 0.0468 |
| 6 | `osm_drain_density` | 0.0411 |
| 7–16 | slope, urban_dist, flow, curve_number, hand, twi, spi, tpi, ndwi, ndvi | 0.0164 → 0.0033 |

Elevation dominates, which is expected. The notable result is that the two
*derived* context features rank 2nd and 3rd — ahead of every classic terrain
index — and the two OpenStreetMap drainage features rank 5th and 6th.

---

## 5. From susceptibility to a rainfall-specific map

Susceptibility answers "does this place flood in a big storm". It says nothing
about *how big*. Rainfall is added separately, and deliberately so — a learned
model cannot be trusted to extrapolate to storm depths it never saw.

**Step 1 — how much rain becomes runoff.** The SCS Curve Number method, a
standard hydrology technique. Each cell has a curve number from its land cover
and soil: tarmac sheds almost everything, forest soaks up a lot.

**Step 2 — where that runoff goes.** Runoff is routed downhill along the flow
network, so a cell's water includes what its catchment delivers, not only the
rain that fell on it.

**Step 3 — combine.**

```
H(x, P) = sigma( logit(S(x)) + beta * ln( Q(x,P) / Q(x,P_ref) ) )
```

In words: take the learned susceptibility, and shift it up or down depending on
whether this storm produces more or less runoff than the reference storm.
`beta = 3.085` controls how strongly rainfall moves the answer, fitted across
four real flood events.

Three properties this guarantees:

- At the reference storm the formula reduces **exactly** to the learned map, so
  the model reproduces the observed 2018 extent.
- More rain never decreases risk anywhere — verified, zero cells decreasing
  across every scenario.
- The result always stays between 0 and 1 without clipping.

**The reference storm is 443 mm over 3 days** (IMD gauge data, 15–17 Aug 2018).
The slider is a **3-day total**, not a 24-hour figure.

### Why it responds instantly

Runoff depends on a cell only through its curve number, and the district has
just **seven** curve-number classes. So how much of each class drains into every
cell is computed **once** when the cache is built. A new rainfall value then
costs seven small calculations plus a weighted sum — **89–90 ms** for the whole
district. Nothing is pre-rendered and nothing is interpolated.

---

## 6. A second, separate model: rainfall forecasting

Independent of the flood model, a second ML model predicts **rainfall itself** —
the total for the next 3 days, which is exactly what the flood model consumes.

- **Model:** `HistGradientBoostingRegressor` (gradient-boosted trees again)
- **Data:** IMD 0.25° gauge-based gridded rainfall
- **Trained on** 9,101 days, **tested on** 3,650 later days — split by time, never
  shuffled, because shuffling a time series leaks the future into the past

| | This model | Persistence | Climatology |
|---|---|---|---|
| Mean absolute error | **18.1 mm** | 19.8 mm | 19.4 mm |
| Correlation | **0.60** | 0.56 | 0.51 |
| AUC for >100 mm events | **0.873** | 0.863 | 0.826 |

Honest reading: it beats "tomorrow is like today" and "tomorrow is like the
seasonal average" — by **7–9%** on error. That is a real but modest gain.
Monsoon rainfall is strongly seasonal and strongly autocorrelated, so those
baselines are hard to beat, and the model is reported against both rather than
in isolation. It is a statistical forecast, not numerical weather prediction.

---

## 7. Outputs: what you get

### On screen

Pick a layer, move the rainfall slider, and the map recomputes. At any rainfall
the alert banner reports:

- **Critical and elevated risk area** in km², from the raster's own geometry
- **Population exposed** — a real spatial sum of WorldPop 2020 counts over
  at-risk cells, not a district average multiplied out
- **Building value exposed** — OpenStreetMap footprints (176,318 buildings,
  25.9 km²) priced at the Kerala PWD 2025 construction rate

That last figure is **replacement-cost exposure, not predicted damage**. No
India-specific depth-damage curve was available to convert exposure into expected
loss, so the report does not pretend otherwise.

Clicking the map gives a point readout with the physical quantities behind the
number — land cover, curve number, runoff depth, runoff coefficient — so any
value can be traced to its reason.

### Files the pipeline writes

| File | What it is |
|---|---|
| `outputs/susceptibility.tif` | The learned rainfall-independent map |
| `outputs/flood_hazard_{mm}mm.tif` | Hazard at each pre-generated storm depth |
| `outputs/conformal_sets.tif` | Where the model is confident at 90% |
| `models/live_model.npz` | 7.6 MB cache that makes live evaluation instant |
| `models/susceptibility_metrics.json` | Every performance number quoted here |

### An API

```
GET /api/map/{mm}          hazard overlay as a PNG + map bounds
GET /api/risk_stats/{mm}   risk-class breakdown in km²
GET /api/model             model card and CV metrics
GET /api/conformal         the coverage guarantee
```

`{mm}` accepts any depth from 0 to 2000.

---

## 8. How good is it, honestly

**What is solid.** Spatial-block AUC **0.824** against a well-timed inventory,
calibrated probabilities, a working uncertainty guarantee, and a rainfall
response that is monotonic by construction rather than by luck.

**What is not.** Four things a reader should weigh before using an output:

1. **The land cover is from 2018 and the district has grown 23% since**
   (+161 km² built-up, measured via Google Dynamic World). New buildings shed
   more water than the model believes, so it **understates** risk on land
   developed after 2018. Note that 2018 land cover is *correct for training* —
   the labels are the 2018 flood — so this is a temporal-transfer problem, not a
   stale file.

2. **Water arriving from upstream is not modelled.** The Periyar drains
   ~5,398 km²; the model can only see inside the district. Rivers enter across
   the boundary carrying nothing. An attempted fix reaches only 1 of 4 validation
   probes and is deliberately not wired in.

3. **The waterlogging layer is a ranking, not a probability.** It scores AUC
   0.807 against 14 documented hotspots — real skill — but 14 points is a small
   sample, and there are no official incident records for this district to
   calibrate against.

4. **Trained on one event.** Susceptibility comes from August 2018 alone.

**Use it to compare places. Do not read a single number as a guarantee.**

---

## 9. Glossary

| Term | Meaning |
|---|---|
| **AUC** | Probability the model ranks a random flooded cell above a random dry one. 0.5 = coin flip, 1.0 = perfect. |
| **Calibration** | Whether "30%" really happens about 30% of the time. |
| **Conformal prediction** | A method giving coverage guarantees without assuming anything about the data's distribution. |
| **Curve Number (SCS-CN)** | Standard hydrology score for how much rain runs off rather than soaking in. |
| **DEM** | Digital Elevation Model — a raster of ground heights. |
| **Fluvial** | Flooding from rivers overflowing. |
| **HAND** | Height Above Nearest Drainage. |
| **NDVI / NDWI** | Satellite indices for vegetation and water. |
| **Pluvial** | Flooding from rain falling faster than it can drain — street waterlogging. |
| **Spatial-block CV** | Testing on geographically separate areas so neighbouring cells cannot leak answers. |
| **Susceptibility** | How flood-prone a place is, independent of any particular storm. |
| **TWI / SPI / TPI** | Terrain indices: wetness, stream power, and position in the landscape. |

---

*Full technical detail, including the negative results and everything that was
tried and rejected, is in the [main README](../README.md).*
