# Which artefacts are live, and which are not

`outputs/` and `models/` used to hold about 2.3 GB of files from earlier
versions of this project alongside the current ones. As of 2026-08-27, the
~1.4 GB of superseded rasters listed under "Removed" below is gone — deleted
with explicit approval once nothing in the live pipeline referenced them (that
check caught a real bug: `evaluation/generate_figures.py` still read one of
them, `flood_hazard_332mm.tif`, for fig4; see the commit that fixed it before
this deletion). The `.pth`/`.h5` legacy model checkpoints are a separate,
deliberate case — still loadable, still referenced by code — and are not
touched here.

Rule of thumb: if it is not in the "live" table below, do not cite it, do not
put it in a paper, and do not show it to anyone as a prediction. If it is not
in *any* table below, it no longer exists.

---

## Live: produced by the current model

| File | Produced by | Notes |
|---|---|---|
| `outputs/susceptibility.tif` | `susceptibility.py --predict` | Calibrated rainfall-independent susceptibility |
| `outputs/susceptibility_uncertainty.tif` | `susceptibility.py --predict` | Ensemble spread across the 5 spatial folds |
| `outputs/conformal_sets.tif` | `susceptibility.py --conformal` | Mondrian class-conditional prediction sets |
| `outputs/flood_hazard_<P>mm.tif` | `hazard.py` | Hazard at rainfall depth P |
| `models/susceptibility_model.joblib` | `susceptibility.py --train` | The spatial ensemble plus isotonic calibrator |
| `models/susceptibility_metrics.json` | `susceptibility.py --train` | CV, calibration, permutation importance |
| `models/live_model.npz` | `live_model.py --build` | Slider cache: class-basis runoff decomposition |
| `models/rainfall_forecast.joblib` | `rainfall_forecast.py --train` | Next-3-day rainfall total |
| `models/risk_thresholds.json` | `risk_thresholds.py` | The four band cuts, read off the PR curve |
| `models/reference_rainfall.json` | `reference_rainfall.py` | IMD-derived 443.2 mm reference event |
| `models/waterlogging_validation.json` | `waterlogging_validation.py` | Hotspot AUC against elevation-matched control |
| `models/benchmark.json` | `benchmark.py` | Timings behind the real-time claim |
| `outputs/beta_fit.json` | `fit_beta.py` | Fitted rainfall sensitivity, replacing the assumed 1.8 |

`data_aligned/` is entirely regenerable from `align_data.py` and is gitignored.

---

## Removed (2026-08-27): what they were and why they were misleading

Kept here as institutional memory — the *reasoning* stays useful even though
the files are gone. None of this is recoverable from the working tree; the
git-tracked ones (all but the reference-depth raster) are still in LFS history
if ever genuinely needed.

### The `_supercharged` rasters were not model output

| File | Size | What it actually was |
|---|---|---|
| `outputs/flood_prob_100mm_supercharged.tif` | 160 MB | Mock MCDA |
| `outputs/flood_prob_150mm_supercharged.tif` | 34 MB | Mock MCDA |
| `outputs/flood_prob_200mm_supercharged.tif` | 160 MB | Mock MCDA |

These came from `src/generate_intelligent_predictions.py`, which described
itself in its own header as a "Mock-Simulation" — a hand-weighted multi-criteria
overlay with no training and no validation. They were previously presented as
CNN output. That script now raises `SystemExit` on import for exactly this
reason. **These three files were the single most misleading thing in the
repository**, which is why they were the first candidates when asked to clean up.

### Legacy CNN inference output

| File | Size | Why superseded |
|---|---|---|
| `outputs/flood_prob_100mm.tif` | 83 MB | Pre-calibration, 30 m grid assumption |
| `outputs/flood_prob_150mm.tif` | 83 MB | Same |
| `outputs/flood_prob_200mm.tif` | 5 MB | Same |
| `outputs/flood_prob_*_robust.tif` | 254 MB each | Same, 6-channel variant |

All of these predated three corrections that change the numbers materially: the
cell size was 30 m when the master grid is actually 10 m (so every area figure
was 9x wrong), the flood label was 80% permanent water, and there was no prior
correction, so expected extent ran about 11x observed.

### Stale reference depth

`outputs/flood_hazard_332mm.tif` — 332 mm was the ERA5-derived reference event,
since replaced by 443.2 mm from IMD gauge data. This one was gitignored, not
LFS-tracked, and it was the one genuinely live dependency in the batch:
`evaluation/generate_figures.py` still read it for fig4's precision-recall
curve, which meant that figure was silently drawn from a different model
generation than the threshold dots plotted on top of it. Fixed to read
`RAINFALL.reference_event_mm` from config instead of a hardcoded filename
before this file was deleted.

### Wrong reduction for flood mapping

`outputs/SAR_VV_filtered.tif`, `outputs/SAR_VH_filtered.tif` — median
composites over a date range spanning the flood. A median over 10-25 Aug 2018
averages flooded and unflooded scenes together, which washes the flood out.
Superseded by `src/acquire_flood_event.py`, which does pre/post change detection
on individual acquisitions.

---

## Legacy models: still loadable, deliberately kept

These are referenced by `src/inference.py`, `src/inference_final.py` and
`config.MODEL_FILES`, and the channel orders in `config.py` exist so they stay
loadable. They are not part of the current pipeline.

| File | Size | Architecture |
|---|---|---|
| `models/flood_model_real2018.pth` | 1.8 MB | 4-channel UNet |
| `models/flood_model_robust_sar.pth` | 1.8 MB | 6-channel |
| `models/flood_model_supercharged.pth` | 1.8 MB | 9-channel |
| `models/flood_model.pth` | 1.8 MB | Earliest checkpoint |
| `models/flood_model_quick.pth` | 30 MB | Short training run |
| `models/flood_model_enhanced.pth` | 89 MB | — |
| `models/geoai_flood_final.pth` | 29 MB | 64 base channels, **not loadable** by `inference_final.py` |
| `models/Ernakulam_Flood_UNet_Ultra.h5` | 91 MB | Keras UNet, `train_keras_unet.py` |

`models/final_metrics.json` and `models/graph_experiment.json` are empty
(0 bytes). The graph experiment's actual result — that adding edges *costs*
0.053 AUC, which is why there is no GNN in this project — is recorded in the
commit history and in `tests/test_graph.py`, not in that file.

---

## Deprecated scripts

Both raise `SystemExit` on import rather than silently producing output:

- `src/generate_intelligent_predictions.py` — the mock MCDA described above
- `evaluation/generate_all_charts.py` — drew figures from hardcoded literature
  values mixed with a `paper_metrics.json` that was internally inconsistent.
  Replaced by `evaluation/generate_figures.py`, which reads only measured JSON.

## Space reclaimed 2026-08-27

The ~1.4 GB above is gone from the working tree:

```
outputs/flood_prob_*_supercharged.tif    354 MB   never valid
outputs/flood_prob_*_robust.tif          762 MB   superseded
outputs/flood_prob_100mm.tif             83 MB    superseded
outputs/flood_prob_150mm.tif             83 MB    superseded
outputs/flood_prob_200mm.tif             5 MB     superseded
outputs/SAR_V*_filtered.tif              20 MB    wrong reduction
outputs/flood_hazard_332mm.tif           25 MB    stale reference depth
```

Deleted with explicit approval, not automatically — nothing in this repo
deletes files as a side effect of running a script. The Git LFS-tracked ones
(everything except `flood_hazard_332mm.tif`, which was gitignored) are staged
as removed and still recoverable from LFS history if ever genuinely needed;
this does not shrink the repository's git history on its own.

Left in place, and out of scope for this pass: `GeoAI_Data/` (2.2 GB),
`processed/` (4.8 GB) and `archive/*.tif` (2.3 GB, not `archive/reports/` —
those are the project's paper drafts and stay regardless). All three are
confirmed off every current code path — `align_data.py` reads from
`GeoAI_New/` and writes `data_aligned/`, and the only things that reference
`GeoAI_Data/`/`processed/` are `GEOAI_DATA_DIR` (defined, never imported) and
already-deprecated legacy scripts — but they are being moved to external
storage rather than deleted outright.
