"""Ensemble permafrost-extent charts: one chart per shared scenario, one line per model,
plus one chart per model with one line per scenario (registry scenario colours).

Built on wiemip_charts.py (keep both files in the same folder): the RECCAP mask,
grid weighting, config and path handling come from there.

Definition (all adjustable)
---------------------------
A cell is permafrost in a year if its annual-mean soil temperature at --depth
(default 10 m) is <= --threshold (default 0 C). Temperature at that depth is linearly
interpolated between layer centres. Extent = sum of cell area x region fraction over
permafrost cells, in million km2, inside the RECCAP mask (or the globe with --region global).

Model eligibility (--depth, --tolerance)
----------------------------------------
* Layers bracket the depth                 -> interpolate ("interp").
* Deepest layer centre is within --tolerance
  (default 1.5 m) above the depth          -> use the deepest layer as a proxy, flagged in the
                                              legend ("deepest").
* Otherwise                                -> model is excluded, with the reason printed.
Models whose files lack layer depths can be given them in LAYER_DEPTHS below or with
--layer-depths MODEL=z1,z2,... (centres in metres, top to bottom).

Charts
------
* One figure per scenario that at least --min-models eligible models share
  (one line per model; a bar along the top edge shows the scenario's registry colour).
* One figure per model (one line per scenario, coloured with the registry's SCENARIO_COLORS).
* With --history, the historical run (--hist-sim, default "hist") is prepended to every line.

Speed and memory
----------------
file.read() in the registry takes no arguments and always loads the whole soilT array
(~20 GB for CLM). This script therefore opens the files itself with xr.open_mfdataset
(lazy, dask-backed, time-chunked), selects the 1-2 soil layers needed for the depth, and
computes a few years (--block-years) at a time. Requires dask.

Because this bypasses the registry's read()/convert step, check each new model against a
registry-based result before trusting it. It only works for monthly files. The start year is
taken from file names like "...1951-2023.nc" (verified against the number of time steps); if
names carry no years it falls back to the time units and labels that "(unverified)".

Notebook use:
    import wiemip_extent_charts as wec
    wec.main(["--io-module", "wiemip_io", "--check-only"])
    wec.main(["--io-module", "wiemip_io", "--models", "CLM", "JULES", "CoLM", "CLM_FATES",
              "--forcing", "gfdl", "--scenarios", "l", "m", "ml", "hl", "--history",
              "--mask-regions", "101", "103", "106", "108", "--min-models", "1", "--mean"])
"""

from __future__ import annotations

import argparse
import gc
import importlib
import re
import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
except NameError:               # pasted into a notebook cell
    sys.path.insert(0, str(Path.cwd()))
import wiemip_charts as wc

try:
    from wiemip_registry.const import SCENARIO_COLORS
except Exception:                      # registry too old / not importable
    SCENARIO_COLORS = {}


def scen_colour(s: str) -> str:
    return SCENARIO_COLORS.get(s, "#555555")


NOLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="f8")
SPATIAL = {"time", "lat", "lon", "latitude", "longitude"}
METRES = {"m", "meter", "meters", "metre", "metres"}
DEFAULT_MODELS = ("CLM", "JULES", "CoLM", "CLM_FATES")
SOILT_NAMES = ("soilT", "tsl", "tsoil")     # names a soil temperature may have inside a file
LAYER_BOTTOM: dict[str, float] = {}         # model -> bottom of the deepest layer (m), filled from file bounds

# Layer-centre depths (m) for models whose files don't carry usable depths, e.g.
#   "JSBACH": [0.065, 0.254, 0.913, 2.902, 5.700]     (illustrative: take these from the model docs)
LAYER_DEPTHS: dict[str, list[float]] = {}

HIST: dict[str, pd.Series] = {}             # model -> historical extent series
_W_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Layer depths and eligibility
# ---------------------------------------------------------------------------

def depths_from_coord(coord):
    """Depth values (m) from a layer coordinate, or None if they look like indices / unknown units."""
    try:
        v = np.asarray(coord.values, dtype="f8")
    except Exception:
        return None
    units = str(coord.attrs.get("units", "")).strip().lower()
    if units and units not in METRES:
        return None
    if not units and np.array_equal(v, np.arange(v.size)):      # 0..n-1: a layer index, not a depth
        return None
    return v


def peek_depths(io, model, forcing, sim, factorial):
    """(layer depths, bottom of deepest layer) from the file (lazy, metadata only), or (None, None)."""
    file = io.retrieve("soilT", model, forcing, sim, factorial)
    path = getattr(file, "path", None)
    if not path or not Path(path).exists():
        return None, None
    with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
        name = next((n for n in SOILT_NAMES if n in ds.data_vars), None)
        if name is None:
            return None, None
        extra = [d for d in ds[name].dims if d not in SPATIAL]
        if len(extra) != 1 or extra[0] not in ds.coords:
            return None, None
        coord = ds[extra[0]]
        z = depths_from_coord(coord)
        bottom = None
        bnds = coord.attrs.get("bounds")
        if z is not None and bnds in ds.variables:
            try:
                bottom = float(np.nanmax(np.asarray(ds[bnds].values, dtype="f8")))
            except Exception:
                bottom = None
        return z, bottom


def depth_rule(z, depth, tol, bottom=None):
    """(mode, message): mode is 'interp', 'deepest' or None (not eligible)."""
    z = np.asarray(z, dtype="f8")
    if z.size < 2 or not np.all(np.isfinite(z)):
        return None, "no usable layer depths"
    lo, hi = float(z.min()), float(z.max())
    if lo <= depth <= hi:
        return "interp", f"layers span {lo:g}-{hi:g} m (interpolated to {depth:g} m)"
    if bottom is not None and hi < depth <= bottom:
        return "deepest", (f"deepest layer centre {hi:g} m, but that layer extends to {bottom:g} m "
                           f"and contains {depth:g} m (bottom layer used)")
    if bottom is None and hi < depth <= hi + tol:
        return "deepest", f"deepest layer centre {hi:g} m used as proxy for {depth:g} m (bounds unknown)"
    extent = f", layer bottom {bottom:g} m" if bottom is not None else ""
    return None, f"deepest layer centre {hi:g} m{extent}, short of {depth:g} m"


def precheck(io, cfg, model, depth, tol):
    """(eligible, mode, message, depths-known). Never loads data arrays."""
    z, bottom = LAYER_DEPTHS.get(model), None
    if z is None:
        for sim in cfg.scenarios:
            try:
                z, bottom = peek_depths(io, model, cfg.forcing, sim, cfg.factorial)
            except Exception:
                continue
            break
    if z is None:
        return True, None, "layer depths not readable from file; checked when loaded", False
    if bottom is not None:
        LAYER_BOTTOM[model] = bottom
    mode, msg = depth_rule(z, depth, tol, bottom)
    return mode is not None, mode, msg, True


# ---------------------------------------------------------------------------
# Lazy opening
# ---------------------------------------------------------------------------

def pick_soilt(obj):
    """The soil temperature DataArray, whatever it is called inside the object."""
    if isinstance(obj, xr.Dataset):
        name = next((n for n in SOILT_NAMES if n in obj.data_vars), None)
        if name is None:
            cand = [n for n, v in obj.data_vars.items() if v.ndim == 4]
            if not cand:
                raise ValueError(f"no soil temperature variable in {list(obj.data_vars)}")
            name = cand[0]
        return obj[name]
    return obj


def _name_years(p: Path):
    m = re.search(r"(\d{4})[-_](\d{4})", p.name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def open_soilt_lazy(file, block_years):
    """Lazy, time-chunked soilT. Sets attrs['_start_year']. Raises if years can't be verified."""
    path = Path(file.path)
    stem = re.split(r"[._]\d{4}", path.name)[0]
    parts = sorted(path.parent.glob(f"{stem}*.nc")) or [path]
    ds = xr.open_mfdataset(
        parts, decode_times=False, decode_timedelta=False,
        combine="nested", concat_dim="time",
        data_vars="minimal", coords="minimal", compat="override",
        chunks={"time": 12 * block_years})
    da = pick_soilt(ds)
    n = da.sizes["time"]
    if n % 12:
        raise ValueError(f"{n} time steps is not a whole number of monthly years")

    spans = [_name_years(p) for p in parts]
    if all(spans):
        for (a1, b1), (a2, b2) in zip(spans, spans[1:]):
            if a2 != b1 + 1:
                raise ValueError(f"chunk files are not contiguous: {spans}")
        expected = sum(b - a + 1 for a, b in spans) * 12
        if expected != n:
            raise ValueError(f"file names imply {expected} months but data has {n}")
        start, src = spans[0][0], "file names"
    else:
        units = str(ds["time"].attrs.get("units", ""))
        m = re.search(r"since\s+(\d{4})", units)
        if not m:
            raise ValueError(f"no years in file names and cannot parse time units {units!r}")
        start, src = int(m.group(1)), "time units (unverified)"
    da.attrs["_start_year"] = start
    da.attrs["_start_source"] = src
    da.attrs["_files"] = [p.name for p in parts]
    return da


def cached_weights(mask, lat, lon):
    """wc.grid_weights, computed once per (mask, grid)."""
    key = (id(mask), lat.tobytes(), lon.tobytes())
    if key not in _W_CACHE:
        _W_CACHE[key] = wc.grid_weights(mask, lat, lon)
    return _W_CACHE[key]


def with_hist(model, ser):
    """Historical years followed by the scenario series (scenario wins on overlap)."""
    h = HIST.get(model)
    if h is None:
        return ser
    return pd.concat([h[h.index < ser.index.min()], ser])


# ---------------------------------------------------------------------------
# Extent per year
# ---------------------------------------------------------------------------

def extent_series(io, da, model, mask, depth, tol, threshold_c, block_years, notes, announced):
    """Annual permafrost extent (million km2) from a lazy soilT array."""
    extra = [d for d in da.dims if d not in SPATIAL]
    if len(extra) != 1:
        raise ValueError(f"expected one soil-depth dimension, found {extra}")
    zd = extra[0]
    if model in LAYER_DEPTHS:
        z = np.asarray(LAYER_DEPTHS[model], dtype="f8")
        if z.size != da.sizes[zd]:
            raise ValueError(f"LAYER_DEPTHS['{model}'] has {z.size} values but '{zd}' has {da.sizes[zd]}")
    else:
        z = depths_from_coord(da[zd]) if zd in da.coords else None
        if z is None:
            raise ValueError(f"no usable depth values for dimension '{zd}'; "
                             f"give them with LAYER_DEPTHS['{model}'] or --layer-depths")
    mode, msg = depth_rule(z, depth, tol, LAYER_BOTTOM.get(model))
    if mode is None:
        raise ValueError(f"not eligible: {msg}")
    notes.setdefault(model, ("proxy", msg) if mode == "deepest" else ("ok", msg))

    order = np.argsort(z)
    zs = z[order]
    if mode == "interp":
        k = int(np.clip(np.searchsorted(zs, depth, side="right"), 1, zs.size - 1))
        t = (depth - zs[k - 1]) / (zs[k] - zs[k - 1])
        layer_idx, layer_w = [int(order[k - 1]), int(order[k])], np.array([1 - t, t])
    else:
        layer_idx, layer_w = [int(order[-1])], np.array([1.0])

    if "_start_year" in da.attrs:                       # lazy path: years from position
        n = da.sizes["time"]
        years = da.attrs["_start_year"] + np.arange(n) // 12
        months = np.arange(n) % 12 + 1
    else:
        years, months = io._time_years(da), io._time_months(da)
    uy = np.unique(years)

    is_c = {"degc": True, "c": True, "celsius": True, "k": False, "kelvin": False}.get(
        str(da.attrs.get("units", "")).strip().lower())
    w = None
    out_years, out_vals = [], []
    for i in range(0, uy.size, block_years):
        ys = uy[i:i + block_years]
        idx = np.flatnonzero(np.isin(years, ys))
        blk = io.standardise(da.isel({"time": idx, zd: layer_idx})).transpose("time", zd, "lat", "lon")
        if w is None:
            lat = np.asarray(blk["lat"].values, dtype="f8")
            lon = np.asarray(blk["lon"].values, dtype="f8")
            frac, area = cached_weights(mask, lat, lon)
            w = frac * area
            if model not in announced:
                announced.add(model)
                lazy = da.chunks is not None
                line = (f"  {model}: {msg}; soilT {da.nbytes / 1e9:.1f} GB on disk, "
                        f"{'lazy' if lazy else 'ALREADY LOADED IN MEMORY'}; "
                        f"years {int(uy[0])}-{int(uy[-1])} "
                        f"(from {da.attrs.get('_start_source', 'time axis')})")
                if mask is not None:
                    line += (f"; region {(w.sum() / 1e12):.2f} million km2 "
                             f"(native mask {mask.native_area_mkm2:.2f})")
                print(line)
        arr = np.asarray(blk.values, dtype="f4")                       # (time, 1-2 layers, lat, lon)
        if is_c is None:
            is_c = bool(np.nanmedian(arr) < 100)
        thr = threshold_c if is_c else threshold_c + 273.15
        yy, mm = years[idx], months[idx]
        for y in ys:
            sel = yy == y
            wt = NOLEAP[mm[sel] - 1]
            layers = np.tensordot(wt / wt.sum(), arr[sel], axes=(0, 0))          # (layers, lat, lon)
            T = np.tensordot(layer_w, layers, axes=(0, 0))                       # (lat, lon)
            with np.errstate(invalid="ignore"):
                perma = np.isfinite(T) & (T <= thr)
            out_years.append(int(y))
            out_vals.append(float((perma * w).sum() / 1e12))
        del blk, arr
    gaps = np.setdiff1d(np.arange(uy.min(), uy.max() + 1), uy)
    if gaps.size:
        raise ValueError(f"missing years {gaps[:10].tolist()}")
    return pd.Series(out_vals, index=out_years, name="extent")


def get_extent(io, cfg, mask, model, sim, depth, tol, threshold_c, block_years, notes, announced):
    use_mask = mask if cfg.region == "reccap" else None
    tag = f"reccap-{mask.signature}" if use_mask is not None else "global"
    run = io.Run(model, cfg.forcing, sim)
    cache = io.PROCESSED_DIR / run.tag / f"pfextent_z{depth:g}_t{threshold_c:g}_{tag}.csv"
    if cache.exists() and not cfg.refresh:
        df = pd.read_csv(cache, index_col=0)
        return pd.Series(df.iloc[:, 0].values, index=df.index.values, name="extent")

    file = io.retrieve("soilT", model, cfg.forcing, sim, cfg.factorial)
    try:
        da = open_soilt_lazy(file, block_years)
    except Exception as exc:
        raise RuntimeError(
            f"{model}/{sim}: could not open soilT lazily ({type(exc).__name__}: {exc}). "
            "Not falling back to file.read(), which loads the whole array.") from exc
    try:
        s = extent_series(io, da, model, use_mask, depth, tol, threshold_c, block_years, notes, announced)
    finally:
        try:
            da.close()
        except Exception:
            pass
        del da
        gc.collect()
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.rename_axis("year").to_csv(cache)
    return s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _footer(fig, args):
    fig.text(0.99, 0.01, f"annual-mean soil T at {args.depth:g} m <= {args.threshold:g} °C",
             ha="right", va="bottom", fontsize=7, color="0.4")


def plot_scenario(io, cfg, scenario, data, models, notes, args, ylim):
    """One scenario: one line per model (history prepended when available)."""
    colours = {m: f"C{i % 10}" for i, m in enumerate(cfg.models)}
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    frame = {}
    for m in models:
        s = with_hist(m, data[(m, scenario)])
        proxy = notes.get(m, ("",))[0] == "proxy"
        label = m + (" (bottom layer used)" if proxy else "")
        ax.plot(s.index, s.values, lw=1.8, color=colours[m], label=label,
                ls="--" if proxy else "-")
        frame[m] = s
    if args.mean and len(frame) > 1:
        df = pd.DataFrame(frame)
        mean = df.mean(axis=1)[df.count(axis=1) >= 2]
        ax.plot(mean.index, mean.values, color="black", lw=2.6, label="Ensemble mean")
    if HIST:
        ax.axvline(int(min(data[(m, scenario)].index.min() for m in models)),
                   color="0.6", ls=":", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    region = "RECCAP2 permafrost region" if cfg.region == "reccap" else "global"
    ax.set_title(f"Permafrost extent - {scenario.upper()} - {region} ({cfg.forcing.upper()})", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Permafrost extent (million km$^2$)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    _footer(fig, args)
    # bar along the top edge in the scenario's registry colour
    ax.add_patch(plt.Rectangle((0, 1.0), 1, 0.012, transform=ax.transAxes,
                               color=scen_colour(scenario), clip_on=False))
    fig.tight_layout()

    out_dir = io.FIGURE_DIR / f"ensemble_{cfg.forcing}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ("_" + "-".join(str(int(r)) for r in cfg.mask_regions)
           if cfg.mask_regions and cfg.region == "reccap" else "")
    hist = "_hist" if HIST else ""
    path = out_dir / f"pfextent_{scenario}_z{args.depth:g}_{cfg.region}{ids}{hist}_{date.today():%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    print(f"  saved {path.name}")
    return fig


def plot_model(io, cfg, model, data, scenarios, args, ylim):
    """One model: history in the 'hist' colour, then one line per scenario in registry colours."""
    present = [s for s in scenarios if (model, s) in data]
    if not present:
        return None
    order = {s: i for i, s in enumerate(SCENARIO_COLORS)}          # registry order (vl ... hl)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    h = HIST.get(model)
    if h is not None:
        ax.plot(h.index, h.values, lw=1.8, color=scen_colour("hist"), label="HIST")
    for s in sorted(present, key=lambda s: order.get(s, 99)):
        ser = data[(model, s)]
        ax.plot(ser.index, ser.values, lw=1.8, color=scen_colour(s), label=s.upper())
    if ylim:
        ax.set_ylim(*ylim)
    region = "RECCAP2 permafrost region" if cfg.region == "reccap" else "global"
    ax.set_title(f"Permafrost extent - {model} - {region} ({cfg.forcing.upper()})", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel("Permafrost extent (million km$^2$)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _footer(fig, args)
    fig.tight_layout()

    out_dir = io.FIGURE_DIR / f"ensemble_{cfg.forcing}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ("_" + "-".join(str(int(r)) for r in cfg.mask_regions)
           if cfg.mask_regions and cfg.region == "reccap" else "")
    path = out_dir / f"pfextent_{model}_z{args.depth:g}_{cfg.region}{ids}_{date.today():%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path.name}")
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-module", default="wiemip_io")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--forcing", default="ukesm")
    ap.add_argument("--scenarios", nargs="+", default=list(wc.DEFAULT_SCENARIOS))
    ap.add_argument("--region", default="reccap", choices=["reccap", "global"])
    ap.add_argument("--mask", default=wc.DEFAULT_MASK)
    ap.add_argument("--mask-var", default=None)
    ap.add_argument("--mask-regions", nargs="+", type=float, default=None)
    ap.add_argument("--depth", type=float, default=10.0, help="soil depth (m) used for the definition")
    ap.add_argument("--tolerance", type=float, default=1.5,
                    help="max shortfall (m) of the deepest layer centre before a model is excluded")
    ap.add_argument("--threshold", type=float, default=0.0, help="permafrost if annual-mean T <= this (°C)")
    ap.add_argument("--block-years", type=int, default=5, help="years read at a time (lower = less memory)")
    ap.add_argument("--layer-depths", nargs="+", default=[], metavar="MODEL=z1,z2,...",
                    help="layer-centre depths (m) for models whose files lack them")
    ap.add_argument("--min-models", type=int, default=2, help="a scenario needs this many models to get a chart")
    ap.add_argument("--mean", action="store_true", help="also draw the ensemble mean")
    ap.add_argument("--free-y", action="store_true", help="each chart picks its own y range")
    ap.add_argument("--history", action="store_true", help="also compute and prepend the historical run")
    ap.add_argument("--hist-sim", default="hist", help="registry simulation name for the historical run")
    ap.add_argument("--factorial", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached series")
    ap.add_argument("--check-only", action="store_true", help="report eligibility and scenarios; load no data")
    args = ap.parse_args(argv)

    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    for p in (here, here.parent, here.parent / "src", Path.cwd()):
        sys.path.insert(0, str(p))
    io = importlib.import_module(args.io_module)

    for item in args.layer_depths:
        name, _, vals = item.partition("=")
        LAYER_DEPTHS[name] = [float(v) for v in vals.split(",")]

    cfg = wc.ChartConfig(models=tuple(args.models), forcing=args.forcing, scenarios=tuple(args.scenarios),
                         region=args.region, mask_var=args.mask_var,
                         mask_regions=tuple(args.mask_regions) if args.mask_regions else None,
                         factorial=args.factorial, refresh=args.refresh)

    # ---- eligibility + scenario availability (no data loaded) ----
    print(f"Soil temperature at {args.depth:g} m (tolerance {args.tolerance:g} m), {cfg.forcing.upper()} forcing:")
    eligible, notes, available = [], {}, {}
    for m in cfg.models:
        ok, mode, msg, known = precheck(io, cfg, m, args.depth, args.tolerance)
        sims = []
        for s in cfg.scenarios:
            try:
                io.retrieve("soilT", m, cfg.forcing, s, cfg.factorial)
                sims.append(s)
            except Exception:
                pass
        available[m] = sims
        flag = ("INCLUDED" if known else "CHECK ON LOAD") if ok else "excluded"
        flag += " (proxy)" if mode == "deepest" else ""
        print(f"  {m:<12} {flag:<20} {msg}")
        print(f"  {'':<12} scenarios with soilT: {', '.join(sims) or 'none'}")
        if ok and sims:
            eligible.append(m)
            if mode == "deepest":
                notes[m] = ("proxy", msg)

    shared = [s for s in cfg.scenarios if sum(s in available[m] for m in eligible) >= args.min_models]
    print(f"\nModels to compute: {', '.join(eligible) or 'none'}  "
          f"(any marked CHECK ON LOAD are verified once their data is read)")
    print(f"Scenarios shared by >= {args.min_models} eligible models: {', '.join(shared) or 'none'}")
    results = {"eligible": eligible, "shared": shared, "series": {}, "figures": {}}
    if not shared and not args.check_only:
        print(f"Nothing to chart: no scenario has {args.min_models}+ eligible models. "
              f"For a single-model test, add --min-models 1.")
    if args.check_only or not shared:
        return results

    mask = None
    if cfg.region == "reccap":
        mpath = wc.resolve_path(io, args.mask)
        mask = wc.load_region_mask(mpath, cfg.mask_var, cfg.mask_regions)
        print(f"\nMask {mpath.name}: {mask.description}")

    # ---- compute (cached) ----
    print("\nComputing permafrost extent...")
    data, missing, errors, announced = {}, [], {}, set()
    for m in eligible:
        for s in shared:
            if s not in available[m]:
                continue
            try:
                data[(m, s)] = get_extent(io, cfg, mask, m, s, args.depth, args.tolerance,
                                          args.threshold, args.block_years, notes, announced)
            except FileNotFoundError:
                missing.append(f"{m}/{s}")
            except Exception as e:
                errors.setdefault(f"{type(e).__name__}: {str(e)[:200]}", []).append(f"{m}/{s}")
    if missing:
        print(f"  not found ({len(missing)}): {', '.join(missing)}")
    for msg, runs in errors.items():
        print(f"  FAILED ({len(runs)}): {', '.join(runs)} -> {msg}")
    results["series"] = data

    # ---- historical runs ----
    HIST.clear()
    if args.history:
        print(f"\nHistorical runs ({args.hist_sim})...")
        for m in eligible:
            try:
                HIST[m] = get_extent(io, cfg, mask, m, args.hist_sim, args.depth, args.tolerance,
                                     args.threshold, args.block_years, notes, announced)
            except FileNotFoundError:
                print(f"  {m}: no {args.hist_sim} soilT found")
            except Exception as e:
                print(f"  {m}: FAILED {type(e).__name__}: {str(e)[:200]}")
        results["hist"] = dict(HIST)

    # ---- charts ----
    vals = [v for s in [*data.values(), *HIST.values()] for v in (s.min(), s.max())]
    ylim = None
    if vals and not args.free_y:
        pad = 0.05 * ((max(vals) - min(vals)) or 1.0)
        ylim = (max(0.0, min(vals) - pad), max(vals) + pad)

    print("\nPer-scenario charts:")
    for s in shared:
        models = [m for m in eligible if (m, s) in data]
        if len(models) < args.min_models:
            print(f"  {s}: only {len(models)} model(s) computed, skipped")
            continue
        results["figures"][s] = plot_scenario(io, cfg, s, data, models, notes, args, ylim)

    print("\nPer-model scenario charts:")
    for m in eligible:
        plot_model(io, cfg, m, data, shared, args, ylim)
    return results


if __name__ == "__main__":
    main()