# Which artefacts are live, and which are not

`outputs/` and `models/` hold about 2.3 GB of files from earlier versions of
this project alongside the current ones. Nothing here is deleted — some of it is
still loadable and a couple of files are the only record of how an earlier
approach behaved — but several of the superseded rasters are **actively
misleading if mistaken for model output**, so this page says plainly which is
which.

Rule of thumb: if it is not in the "live" tables below, do not cite it, do not
put it in a paper, and do not show it to anyone as a prediction.

---

## Live: produced by the current model

| File | Produced by | Notes |
|---|---|---|
| `outputs/susceptibility.tif` | `susceptibility.py --predict` | Calibrated rainfall-independent susceptibility |
| `outputs/susceptibility_uncertainty.tif` | `susceptibility.py --predict` | Ensemble spread across the 5 spatial folds |
| `outputs/conformal_sets.tif` | `susceptibility.py --conformal` | Mondrian class-conditional prediction sets |
| `outputs/flood_hazard_<P>mm.tif` | `hazard.py` | Hazard at rainfall depth P, except `332mm` (see below) |
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

## Superseded: do not cite

### The `_supercharged` rasters are not model output

| File | Size | What it actually is |
|---|---|---|
| `outputs/flood_prob_100mm_supercharged.tif` | 160 MB | Mock MCDA |
| `outputs/flood_prob_150mm_supercharged.tif` | 34 MB | Mock MCDA |
| `outputs/flood_prob_200mm_supercharged.tif` | 160 MB | Mock MCDA |

These came from `src/generate_intelligent_predictions.py`, which described
itself in its own header as a "Mock-Simulation" — a hand-weighted multi-criteria
overlay with no training and no validation. They were previously presented as
CNN output. That script now raises `SystemExit` on import for exactly this
reason. **These three files are the single most misleading thing in the
repository.**

### Legacy CNN inference output

| File | Size | Why superseded |
|---|---|---|
| `outputs/flood_prob_100mm.tif` | 83 MB | Pre-calibration, 30 m grid assumption |
| `outputs/flood_prob_150mm.tif` | 83 MB | Same |
| `outputs/flood_prob_200mm.tif` | 5 MB | Same |
| `outputs/flood_prob_*_robust.tif` | 254 MB each | Same, 6-channel variant |

All of these predate three corrections that change the numbers materially: the
cell size was 30 m when the master grid is actually 10 m (so every area figure
was 9x wrong), the flood label was 80% permanent water, and there was no prior
correction, so expected extent ran about 11x observed.

### Stale reference depth

`outputs/flood_hazard_332mm.tif` — 332 mm was the ERA5-derived reference event,
since replaced by 443.2 mm from IMD gauge data. Kept only so the two can be
compared; it is not a scenario anyone asked for.

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

## Reclaiming the space

Nothing above is deleted by any script; removing files is a deliberate
decision, not a side effect. If you want the space back, the superseded
rasters are the ~1.4 GB to consider, and every one of them is either
regenerable or something you should not be using:

```
outputs/flood_prob_*_supercharged.tif    354 MB   never valid
outputs/flood_prob_*_robust.tif          762 MB   superseded
outputs/flood_prob_100mm.tif             83 MB    superseded
outputs/flood_prob_150mm.tif             83 MB    superseded
outputs/flood_prob_200mm.tif             5 MB     superseded
outputs/flood_hazard_332mm.tif           25 MB    stale reference depth
```

Note these are tracked via Git LFS, so deleting the working copies does not
shrink the repository history.
