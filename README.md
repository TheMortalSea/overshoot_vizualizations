[Launch on the jupyterhub](https://wiemip-hub.org/hub/user-redirect/git-pull?repo=https%3A%2F%2Fgithub.com%2FTheMortalSea%2Fovershoot_vizualizations&branch=main&urlpath=lab%2Ftree%2Fovershoot_vizualizations%2Fnotebooks%2Frun_pipeline.ipynb)

# WIEMIP permafrost charts

Scripts for comparing land-model output from the WIEMIP climate-overshoot experiments inside the
RECCAP2 permafrost region. Three scripts make the charts:

| Script | What it draws |
|---|---|
| `wiemip_extent_charts.py` | **Permafrost extent** over time (million km²): one chart per scenario with a line per model, plus one chart per model with a line per scenario |
| `wiemip_soilc_charts.py` | **Soil carbon** over time (Pg C), above 1 m, below 1 m, or total: one chart per model with all its scenarios |
| `wiemip_veg_charts.py` | **Vegetation carbon** over time (Pg C): one chart per model with all its scenarios |

---

## 1. What you need

**Files** (all in the same `code/` folder):

| File | Role |
|---|---|
| `wiemip_io.py` | Reads data through the WIEMIP registry (`retrieve`, `standardise`, `Run`, ...). Required by all three scripts. |
| `wiemip_charts.py` | Shared helpers: the RECCAP mask, area weighting, caching. Required by all three scripts. |
| `wiemip_extent_charts.py`, `wiemip_soilc_charts.py`, `wiemip_veg_charts.py` | The three chart scripts. |

**Expected folder layout** :

```
overshoot_vizualizations/
├── code/                  <- all the .py files above
├── data/
│   ├── RECCAP2_permafrost_regions_isimip3.nc     <- the region mask
│   └── processed/         <- cached results, created automatically
└── figures/               <- charts, created automatically
```

**Python environment:** `wiemip_registry` (the data registry), `numpy`, `pandas`, `xarray`,
`matplotlib`, and `dask` (needed by the extent script only).
---

## 2. Quick start (in a notebook)

If for some reason the run_pipline notebook does not look right or not working correctly. Make sure that the setup structure looks somewhat similar to the described below.
(I am planning on setting up a different pipeline script for the single CLM tests we did at the beginning and the newly structured model comparisons, just not the priority yet)

Use the same pattern for every script. This example uses the vegetation script.

**Cell 1: import and reload.** Reloading matters: Python keeps old versions of modules in memory, so after
you edit or replace a file you must reload it (or restart the kernel).

```python
import importlib
import wiemip_charts
import wiemip_veg_charts as wvc
importlib.reload(wiemip_charts)
importlib.reload(wvc)
```

**Cell 2: look before you load anything.** `--check-only` shows which models and scenarios exist, without
reading any data. It is fast.

```python
wvc.main(["--io-module", "wiemip_io", "--forcing", "ukesm", "--scenarios", "all", "--check-only"])
```

**Cell 3: make the charts.**

```python
res = wvc.main(["--io-module", "wiemip_io", "--forcing", "ukesm", "--scenarios", "all", "--history"])
```

Charts are saved in `figures/` and also displayed in the notebook. Everything in `main([...])` is a
command-line flag, so the same flags work from a terminal: `python wiemip_veg_charts.py --forcing ukesm ...`

---

## 3. The scripts

### `wiemip_veg_charts.py`: vegetation carbon

Regional total of `cVeg` (Pg C) for each model that has it, one chart per model, all scenarios on it.

```python
import wiemip_veg_charts as wvc
res = wvc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
                "--history", "--anomaly"])
```

Saved to `figures/vegc_<forcing>/`. Returns `res["series"][model][scenario]` (annual series) and
`res["hist"][model]`.

### `wiemip_soilc_charts.py`: soil carbon

Regional soil carbon (Pg C), one chart per model **and** depth selection. Choose the selection with `--parts`:

| `--parts` value | Meaning |
|---|---|
| `above1m` | Carbon in the top 1 m (default) |
| `below1m` | Carbon below 1 m (default) |
| `total` | Whole-column soil carbon (`cSoil`), available for most models |

```python
import wiemip_soilc_charts as wsc
res = wsc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
                "--parts", "above1m", "below1m", "--history", "--anomaly"])
```

How above/below 1 m is obtained (`--source`, default `auto`):

- **direct**: the model supplies `cSoilAbove1m` and `cSoilBelow1m` (e.g. CLM, CoLM, CLM_FATES).
- **layers**: derived from soil carbon by layer (e.g. JULES). A layer that straddles 1 m is split in
  proportion, assuming carbon is spread evenly within the layer.
- Models with neither only get the `total` chart.

The console prints a **consistency check** per model: above + below should equal the total soil carbon. A
ratio far from 1 (the line ends in `<-- CHECK`) means one of those variables is not what its name says.

Saved to `figures/soilc_<forcing>/`. Returns `res["series"][(model, part)][scenario]`.

### `wiemip_extent_charts.py`: permafrost extent

The generation of these charts does take significantly longer than the other two. That is just the nature of the extra calculation.

**Definition:** a grid cell is permafrost in a year if its annual-mean soil temperature at `--depth`
(default 10 m) is at or below `--threshold` (default 0 °C), interpolated between soil layers. Extent is the
area of those cells inside the region.

```python
import wiemip_extent_charts as wec
wec.main(["--io-module", "wiemip_io", "--check-only"])          # who qualifies, which scenarios
res = wec.main(["--io-module", "wiemip_io", "--forcing", "gfdl",
                "--scenarios", "l", "m", "ml", "hl", "--history"])
```

Two sets of charts are made, saved to `figures/ensemble_<forcing>/`:

1. **One per scenario:** a line per model (needs at least `--min-models` models, default 2).
2. **One per model:** a line per scenario, in the registry's scenario colours.

Things specific to this script:

- **Not every model can be used.** A model needs soil temperature down to about the chosen depth. The
  `--check-only` output lists each model as `INCLUDED`, `INCLUDED (proxy)` or `excluded`, with the reason. A
  *proxy* model (dashed line, `*` in the label) uses its deepest layer because the target depth falls
  inside that layer. 
- **It reads the files itself, not through the registry**, to keep memory low. This needs `dask`, only works
  for **monthly** files, and takes the start year from file names like `...1951-2023.nc`. It does not fall
  back to loading everything: if a file can't be opened lazily you get a clear error.
- **List scenarios explicitly** (`--scenarios l m hl ...`). Unlike the other two scripts, it does not accept
  `all`.
- **Single-model test:** one model can't make a per-scenario chart, so add `--min-models 1`.
- Depth rules and extras: `--depth`, `--tolerance`, `--threshold`, `--layer-depths MODEL=z1,z2,...` (for
  models whose files lack layer depths), `--mean` (adds an ensemble-mean line), `--block-years` (lower it if
  memory is tight).

---

## 4. Options common to the scripts

| Flag | Default | What it does |
|---|---|---|
| `--io-module` | `wiemip_io` | Name of the module that reads the data |
| `--models` | all registry models (extent: CLM, JULES, CoLM, CLM_FATES) | Models to include. Use the registry spelling, e.g. `LPX_Bern` |
| `--forcing` | `ukesm` | Which climate forcing: `ukesm`, `ipsl`, `gfdl` (the registry also lists `stable`) |
| `--scenarios` | `l m hl hl_cf ml ml_cf vl vl_cf` | Scenarios to look for, or `all` (veg and soil carbon only). Models only get the ones they have |
| `--region` | `reccap` | `reccap` (permafrost region mask) or `global` |
| `--mask-regions` | all | Only use these RECCAP region IDs, e.g. `--mask-regions 101 103 106 108` |
| `--history` | off | Also compute and draw the historical run, in front of the scenarios |
| `--hist-start` | `1950` | First historical year shown (`0` = everything). Applied when plotting only |
| `--anomaly` | off | Plot change since `--base-year` instead of raw values (veg and soil carbon only) |
| `--base-year` | `2024` | Reference year for `--anomaly` |
| `--share-y` | off | Veg/soil carbon: put all charts on one y-axis. Extent shares by default; use `--free-y` to turn it off |
| `--check-only` | off | Print what would be made; load no data |
| `--refresh` | off | Ignore cached results and recompute |

**Anomaly mode:** each scenario has its own base-year value subtracted, so every scenario line starts at
zero. The historical run has the mean of the scenarios' base-year values subtracted, so it joins them at
zero. Cached results stay as raw values, so switching it on or off never recomputes anything.

---

## 5. Ideas that explain the numbers

- **Region:** the RECCAP2 permafrost mask (`data/RECCAP2_permafrost_regions_isimip3.nc`) is a 0.5° map of
  region IDs (101, 103, 106, 108, 113). By default all IDs are combined into one permafrost region. The mask
  is mapped onto each model's own grid by exact area overlap, so no model data is regridded.
- **Totals:** Pg C values are cell area × region fraction × the model value, summed over the region.
- **Scenarios:** names come from the registry (`l`, `m`, `ml`, `vl`, `hl`, ...). The suffix `_cf` marks a
  counterfactual run of that scenario. Line colours always come from the registry's scenario colours.
- **Forcing:** the climate pattern driving the experiment. Charts for different forcings are not
  interchangeable, and not every model has every forcing.
- **Historical run:** `hist`, which normally ends where the scenarios start (2024). A dotted line or shaded band marks
  where the scenarios begin.

---

## 6. Caching: why a second run is fast

Each model/scenario result is saved as a small CSV, so re-plotting is instant. Files live in
`data/processed/<model>_<forcing>_<scenario>/` (in lower case, e.g. `clm_gfdl_hl`) and the file name includes the variable and a short code for the
mask, for example `cVeg_reccap-1a2b3c4d.csv` or `pfextent_z10_t0_reccap-1a2b3c4d.csv`.

- Changing the mask or the region IDs creates new cache files automatically.
- Changing the **permafrost depth or threshold** also creates new files (they are in the name).
- Use `--refresh` if the **underlying data** changed, or if you suspect a stale result.

---

## 7. Troubleshooting

| What you see | Likely cause and fix |
|---|---|
| Behaviour doesn't match the code you just saved, or `main` is missing | Python is using an old copy. Restart the kernel, or reload `wiemip_charts` first and then the script. Check `module.__file__` points to the right folder |
| `unexpected keyword argument 'mask_regions'` | An old `wiemip_charts.py` is loaded. Replace the file and restart the kernel |
| `Nothing to chart ... add --min-models 1` (extent) | Fewer than two eligible models share a scenario. Use `--min-models 1` or add models |
| `no variables found` / a model is skipped | Check the spelling against the registry (`wiemip_registry.models`), for example `LPX_Bern`, not `LPX-BERN` |
| The mask line says nearly all cells are inside, or an area near 510 million km² | The mask is being read as the whole globe. Use the latest `wiemip_charts.py` |
| Region area printed per model differs a lot from the native mask area | Something is wrong with the grid or mask; send the console output to the maintainer |
| `FAILED (n): ...` lines at the end | A run failed. The **first** failure prints a full traceback: read that one |
| `could not open soilT lazily` (extent) | `dask` missing, or the files aren't monthly / don't carry years in their names. The script stops on purpose instead of loading everything |
| Out of memory | Lower `--block-years` (extent and soil carbon), run fewer models at once, or restart the kernel to free memory |
| A model has no chart | It doesn't have that variable or scenario for the chosen forcing. Run `--check-only` to see why |

**Reading the console output:** each script first prints a *plan* (which models and scenarios it found), then
a `Mask ...` line (which region it is using), then *Computing...* with one line per model, and finally the
saved filenames. A line ending in `<-- CHECK` or `FAILED` needs attention.

---