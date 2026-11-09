"""Soil carbon charts: one chart per model and depth selection, with all of the model's scenarios.

Built on wiemip_charts.py (keep both in the same folder): RECCAP mask, grid weighting,
config, caching and path handling come from there.

Depth selections (--parts)
--------------------------
  above1m   soil carbon in the top 1 m
  below1m   soil carbon below 1 m
  total     whole-column soil carbon (cSoil), available for every model that has it
Values are regional totals in Pg C (RECCAP permafrost region by default, --region global
for the globe).

Where above/below 1 m comes from (--source)
-------------------------------------------
  direct    the registry variables cSoilAbove1m / cSoilBelow1m (CLM, CoLM, CLM_FATES)
  layers    derived from layered soil carbon (cSoilLayers or cPoolVr; JULES, and CLM/CoLM when
            forced). A layer straddling 1 m is split in proportion to the thickness on each
            side, assuming carbon is spread evenly within the layer. Layer bounds come from
            LAYER_INTERFACES below, then the file's bounds variable, and only then midpoints
            between layer centres (approximate; flagged in the output).
  auto      direct if the model has it, otherwise layers (default)
Models with neither only get the 'total' chart.

Scenarios and colours
---------------------
--scenarios all uses every future scenario in the registry (wiemip_registry.const.SCENARIO_COLORS,
minus hist/ctrl). A model only gets the scenarios it actually has. Lines use the registry's
scenario colours and are ordered as in the registry (vl ... hl).

History (--history)
-------------------
Also computes the historical run (--hist-sim, default "hist") for every model/part that has it
and draws it in the registry's "hist" colour ahead of the scenarios. Do not put "hist" in
--scenarios.

A consistency check is printed for each model: above + below should equal the total
(cSoil). A ratio far from 1 means one of the variables is not what its name says.

Notebook use:
    import wiemip_soilc_charts as wsc
    wsc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
              "--history", "--check-only"])
    wsc.main(["--io-module", "wiemip_io", "--forcing", "gfdl", "--scenarios", "all",
              "--history", "--mask-regions", "101", "103", "106", "108", "--share-y"])
"""

from __future__ import annotations

import argparse
import gc
import importlib
import inspect
import sys
import traceback
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
except Exception:               # registry too old / not importable
    SCENARIO_COLORS = {}


def scen_colour(s: str) -> str:
    """Registry colour; a scenario missing from the registry (e.g. hl_cf) borrows its base scenario's colour."""
    if s in SCENARIO_COLORS:
        return SCENARIO_COLORS[s]
    base = s[:-3] if s.endswith("_cf") else s
    return SCENARIO_COLORS.get(base, "#555555")


CUT = 1.0                                   # depth (m) separating "above" from "below"
NOLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31], dtype="f8")
SPATIAL = {"time", "lat", "lon", "latitude", "longitude"}
METRES = {"m", "meter", "meters", "metre", "metres"}
LAYER_VARS = ("cSoilLayers", "cPoolVr")

PARTS = {
    "above1m": dict(direct="cSoilAbove1m", title="Soil carbon above 1 m"),
    "below1m": dict(direct="cSoilBelow1m", title="Soil carbon below 1 m"),
    "total": dict(direct="cSoil", title="Soil carbon (total)"),
}


def register_charts():
    """Register the soil-carbon variables in the shared chart table (regional totals, Pg C).

    Called at import AND at the start of main(), so reloading wiemip_charts (which replaces
    wc.CHARTS with a fresh dict) cannot leave these keys missing.
    """
    for p in PARTS.values():
        wc.CHARTS[p["direct"]] = dict(variable=p["direct"], kind="total", title=p["title"],
                                      ylabel=f"{p['title']} (Pg C)")


register_charts()

# Layer interfaces (m, top to bottom, surface = first value) for models whose files lack bounds.
# CLM5 soil grid, taken from the CLM_FATES file's depth_bnds (same layer structure).
LAYER_INTERFACES: dict[str, list[float]] = {
    "CLM": [0, 0.02, 0.06, 0.12, 0.2, 0.32, 0.48, 0.68, 0.92, 1.2, 1.52, 1.88, 2.28, 2.72,
            3.26, 3.9, 4.64, 5.48, 6.42, 7.46, 8.6, 10.99, 15.67, 23.3, 34.44, 49.56],
}


# ---------------------------------------------------------------------------
# Layer handling
# ---------------------------------------------------------------------------

def depths_from_coord(coord):
    """Layer-centre depths (m), or None if they look like indices / have non-metre units."""
    try:
        v = np.asarray(coord.values, dtype="f8")
    except Exception:
        return None
    units = str(coord.attrs.get("units", "")).strip().lower()
    if units and units not in METRES:
        return None
    if not units and np.array_equal(v, np.arange(v.size)):
        return None
    return v


def peek_bounds(file, zd, n):
    """Layer bounds (n, 2) from the underlying file's bounds variable, or None."""
    path = getattr(file, "path", None)
    if not path or not Path(path).exists():
        return None
    try:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            names = []
            if zd in ds.coords and ds[zd].attrs.get("bounds"):
                names.append(ds[zd].attrs["bounds"])
            names += [f"{zd}_bnds", "depth_bnds", "layer_bnds", "lev_bnds", "levgrnd_bnds", "levsoi_bnds"]
            for nm in names:
                if nm in ds.variables:
                    v = np.asarray(ds[nm].values, dtype="f8")
                    if v.shape == (n, 2):
                        return v
    except Exception:
        return None
    return None


def midpoint_bounds(z):
    """Approximate bounds: midpoints between layer centres, surface at 0."""
    order = np.argsort(z)
    zs = z[order]
    edges = np.concatenate([[0.0], (zs[1:] + zs[:-1]) / 2.0, [zs[-1] + (zs[-1] - zs[-2]) / 2.0]])
    b_sorted = np.column_stack([edges[:-1], edges[1:]])
    b = np.empty_like(b_sorted)
    b[order] = b_sorted
    return b


def get_bounds(file, model, z, zd):
    """((n,2) bounds aligned with z, source label)."""
    n = z.size

    def ok(b):
        return (b is not None and b.shape == (n, 2) and np.all(b[:, 1] > b[:, 0])
                and np.all((z >= b[:, 0] - 1e-6) & (z <= b[:, 1] + 1e-6)))
    if model in LAYER_INTERFACES:
        iface = np.asarray(LAYER_INTERFACES[model], dtype="f8")
        if iface.size - 1 >= n:
            b = np.column_stack([iface[:n], iface[1:n + 1]])
            if ok(b):
                return b, "known layer interfaces"
    b = peek_bounds(file, zd, n)
    if ok(b):
        return b, "bounds from file"
    return midpoint_bounds(z), "midpoints between layer centres (approximate)"


def read_lean(file, block_years):
    """file.read(), asking for time-chunked (lazy) data if the registry accepts `chunks`."""
    try:
        params = inspect.signature(file.read).parameters
    except (TypeError, ValueError):
        params = {}
    kw = {"chunks": {"time": 12 * block_years}} if "chunks" in params else {}
    return file.read(**kw)


def pick_var(obj, name):
    if isinstance(obj, xr.Dataset):
        if name in obj.data_vars:
            return obj[name]
        cand = [n for n, v in obj.data_vars.items() if v.ndim >= 4] or list(obj.data_vars)
        return obj[cand[0]]
    return obj


def layered_series(io, cfg, mask, model, sim, var, block_years, announced) -> pd.DataFrame:
    """Annual regional soil C (Pg C) above/below 1 m, plus the layer total, from layered carbon."""
    use_mask = mask if cfg.region == "reccap" else None
    tag = f"reccap-{mask.signature}" if use_mask is not None else "global"
    run = io.Run(model, cfg.forcing, sim)
    cache = io.PROCESSED_DIR / run.tag / f"soilc-{var}_{tag}.csv"
    if cache.exists() and not cfg.refresh:
        return pd.read_csv(cache, index_col=0)

    file = io.retrieve(var, model, cfg.forcing, sim, cfg.factorial)
    da = pick_var(read_lean(file, block_years), var)
    try:
        extra = [d for d in da.dims if d not in SPATIAL]
        if len(extra) != 1:
            raise ValueError(f"expected one soil-depth dimension, found {extra}")
        zd = extra[0]
        z = depths_from_coord(da[zd]) if zd in da.coords else None
        if z is None:
            raise ValueError(f"no usable depth values for dimension '{zd}'")
        bounds, src = get_bounds(file, model, z, zd)
        top, bot = bounds[:, 0], bounds[:, 1]
        if bot.max() < CUT:
            raise ValueError(f"layers end at {bot.max():g} m, short of {CUT:g} m")
        f_above = np.clip((CUT - top) / (bot - top), 0.0, 1.0)         # share of each layer above the cut
        f_below = 1.0 - f_above

        years, months = io._time_years(da), io._time_months(da)
        uy = np.unique(years)
        w = None
        rows = []
        for i in range(0, uy.size, block_years):
            ys = uy[i:i + block_years]
            idx = np.flatnonzero(np.isin(years, ys))
            blk = io.standardise(da.isel(time=idx)).transpose("time", zd, "lat", "lon")
            if w is None:
                lat = np.asarray(blk["lat"].values, dtype="f8")
                lon = np.asarray(blk["lon"].values, dtype="f8")
                frac, area = wc.grid_weights(use_mask, lat, lon)
                w = frac * area
                if model not in announced:
                    announced.add(model)
                    split = [j for j in range(z.size) if 0 < f_above[j] < 1]
                    lazy = da.chunks is not None or getattr(da.variable, "_in_memory", True) is False
                    print(f"  {model}: {var}, {z.size} layers to {bot.max():g} m, bounds: {src}; "
                          + (f"layer(s) split at {CUT:g} m: " +
                             ", ".join(f"{top[j]:g}-{bot[j]:g} m ({f_above[j]:.0%} above)" for j in split)
                             if split else f"{CUT:g} m falls on a layer boundary")
                          + f"; {da.nbytes / 1e9:.1f} GB, {'lazy' if lazy else 'ALREADY LOADED by read()'}")
            arr = np.asarray(blk.values, dtype="f4")
            yy, mm = years[idx], months[idx]
            for y in ys:
                sel = yy == y
                wt = NOLEAP[mm[sel] - 1]
                layers = np.nan_to_num(np.tensordot(wt / wt.sum(), arr[sel], axes=(0, 0)))   # (layer, lat, lon)
                a = np.tensordot(f_above, layers, axes=(0, 0))
                b = np.tensordot(f_below, layers, axes=(0, 0))
                rows.append((int(y), float((a * w).sum() / 1e12), float((b * w).sum() / 1e12)))
            del blk, arr
        out = pd.DataFrame(rows, columns=["year", "above1m", "below1m"]).set_index("year")
        out["layers_total"] = out["above1m"] + out["below1m"]
        gaps = np.setdiff1d(np.arange(uy.min(), uy.max() + 1), uy)
        if gaps.size:
            raise ValueError(f"missing years {gaps[:10].tolist()}")
    finally:
        del da
        gc.collect()
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


# ---------------------------------------------------------------------------
# Planning which model can do what
# ---------------------------------------------------------------------------

def plan_model(io, cfg, model, source, hist_sim=None):
    """What each model can do. Scenario lists never contain the historical run;
    p['hist'][part] says whether the historical run is available for that part."""
    sims_to_check = list(cfg.scenarios) + ([hist_sim] if hist_sim else [])
    avail_all = {}
    for v in ("cSoil", "cSoilAbove1m", "cSoilBelow1m") + LAYER_VARS:
        sims = []
        for s in sims_to_check:
            try:
                io.retrieve(v, model, cfg.forcing, s, cfg.factorial)
                sims.append(s)
            except Exception:
                pass
        if sims:
            avail_all[v] = sims
    avail = {v: [s for s in sims if s != hist_sim] for v, sims in avail_all.items()}
    avail = {v: sims for v, sims in avail.items() if sims}

    direct = "cSoilAbove1m" in avail and "cSoilBelow1m" in avail
    layered = next((v for v in LAYER_VARS if v in avail), None)
    if source == "direct":
        layered = None
    if source == "layers":
        direct = False
    use = "direct" if direct else ("layers" if layered else None)

    scen = {"total": avail.get("cSoil", [])}
    hist = {"total": bool(hist_sim) and hist_sim in avail_all.get("cSoil", [])}
    if use == "direct":
        both = [s for s in avail["cSoilAbove1m"] if s in avail["cSoilBelow1m"]]
        scen["above1m"] = scen["below1m"] = both
        h = bool(hist_sim) and hist_sim in avail_all["cSoilAbove1m"] and hist_sim in avail_all["cSoilBelow1m"]
        hist["above1m"] = hist["below1m"] = h
    elif use == "layers":
        scen["above1m"] = scen["below1m"] = avail[layered]
        h = bool(hist_sim) and hist_sim in avail_all[layered]
        hist["above1m"] = hist["below1m"] = h
    else:
        scen["above1m"] = scen["below1m"] = []
        hist["above1m"] = hist["below1m"] = False
    return dict(avail=avail, use=use, layered=layered, scen=scen, hist=hist)


def describe_plan(model, p, hist_sim=None):
    if p["use"] == "direct":
        how = "above/below 1 m: cSoilAbove1m + cSoilBelow1m" + (
            f" (layers also available: {p['layered']})" if p["layered"] else "")
    elif p["use"] == "layers":
        how = f"above/below 1 m: derived from {p['layered']}"
    else:
        how = "above/below 1 m: not available"
    tot = "total: cSoil" if "cSoil" in p["avail"] else "total: not available"
    hs = ""
    if hist_sim:
        have = [k for k, v in p["hist"].items() if v]
        hs = f"; {hist_sim}: {'/'.join(have) if have else 'none'}"
    return f"  {model:<22} {how}; {tot}{hs}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def scenario_styles(scenarios) -> dict:
    """Registry colour per scenario. Counterfactuals the registry doesn't list are dashed."""
    return {s: dict(color=scen_colour(s),
                    ls="--" if (s.endswith("_cf") and s not in SCENARIO_COLORS) else "-")
            for s in scenarios}


def plot_model(io, cfg, model, part, series, hist, styles, ylim):
    title = PARTS[part]["title"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    order = {s: i for i, s in enumerate(SCENARIO_COLORS)}
    if hist is not None:
        ax.plot(hist.index, hist.values, lw=1.8, color=scen_colour("hist"), label="HIST")
    for s in sorted((s for s in cfg.scenarios if s in series), key=lambda s: order.get(s, 99)):
        ser = series[s]
        ax.plot(ser.index, ser.values, lw=2.4 if s.startswith("ml") else 1.8,
                label=s.upper(), **styles[s])
    if hist is not None and series:
        start = min(int(s.index.min()) for s in series.values())
        ax.axvline(start, color="0.6", ls=":", lw=1)
    if ylim:
        ax.set_ylim(*ylim)
    region = "RECCAP2 permafrost region" if cfg.region == "reccap" else "global"
    ax.set_title(f"{model} - {title} - {region} ({cfg.forcing.upper()})", fontsize=11)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{title} (Pg C)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2 if len(series) > 4 else 1, fontsize=9, title="Scenario")
    fig.tight_layout()
    out_dir = io.FIGURE_DIR / f"soilc_{cfg.forcing}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = ("_" + "-".join(str(int(r)) for r in cfg.mask_regions)
           if cfg.mask_regions and cfg.region == "reccap" else "")
    h = "_hist" if hist is not None else ""
    path = out_dir / f"{model}_{part}_{cfg.region}{ids}{h}_{date.today():%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    print(f"  saved {path.name}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-module", default="wiemip_io")
    ap.add_argument("--models", nargs="+", default=None, help="default: every registered model")
    ap.add_argument("--parts", nargs="+", default=["above1m", "below1m"], choices=list(PARTS))
    ap.add_argument("--source", default="auto", choices=["auto", "direct", "layers"])
    ap.add_argument("--forcing", default="ukesm")
    ap.add_argument("--scenarios", nargs="+", default=list(wc.DEFAULT_SCENARIOS),
                    help="scenarios to look for, or 'all' for every future scenario in the registry")
    ap.add_argument("--region", default="reccap", choices=["reccap", "global"])
    ap.add_argument("--mask", default=wc.DEFAULT_MASK)
    ap.add_argument("--mask-var", default=None)
    ap.add_argument("--mask-regions", nargs="+", type=float, default=None)
    ap.add_argument("--block-years", type=int, default=2, help="years read at a time for layered data")
    ap.add_argument("--share-y", action="store_true", help="share the y range across models (per part)")
    ap.add_argument("--history", action="store_true", help="also compute and draw the historical run")
    ap.add_argument("--hist-sim", default="hist", help="registry simulation name for the historical run")
    ap.add_argument("--factorial", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached series")
    ap.add_argument("--check-only", action="store_true", help="show the plan for each model; load no data")
    args = ap.parse_args(argv)

    register_charts()          # wiemip_charts may have been reloaded since import

    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    for p in (here, here.parent, here.parent / "src", Path.cwd()):
        sys.path.insert(0, str(p))
    io = importlib.import_module(args.io_module)

    hist_sim = args.hist_sim if args.history else None
    if args.scenarios == ["all"]:
        scenarios = [s for s in SCENARIO_COLORS if not s.startswith(("hist", "ctrl"))]
        if not scenarios:
            scenarios = list(getattr(io.wr, "overshoot_simulations", wc.DEFAULT_SCENARIOS))
    else:
        scenarios = list(args.scenarios)
    scenarios = [s for s in scenarios if s != hist_sim]          # history is handled by --history

    models = args.models or list(getattr(io.wr, "models", ()))
    if not models:
        sys.exit("No models: pass --models.")
    cfg = wc.ChartConfig(models=tuple(models), forcing=args.forcing, scenarios=tuple(scenarios),
                         region=args.region, mask_var=args.mask_var,
                         mask_regions=tuple(args.mask_regions) if args.mask_regions else None,
                         factorial=args.factorial, refresh=args.refresh)

    print(f"Soil carbon plan ({cfg.forcing.upper()}, source: {args.source}):")
    plans = {}
    for m in cfg.models:
        plans[m] = plan_model(io, cfg, m, args.source, hist_sim)
        print(describe_plan(m, plans[m], hist_sim))
    todo = {m: [part for part in args.parts if plans[m]["scen"][part]] for m in cfg.models}
    todo = {m: parts for m, parts in todo.items() if parts}
    listing = ", ".join(m + ": " + "/".join(v) for m, v in todo.items()) or "none"
    print(f"\nCharts: {sum(len(v) for v in todo.values())} ({listing})")
    results = {"plans": plans, "series": {}, "hist": {}, "figures": {}}
    if args.check_only or not todo:
        return results

    mask = None
    if cfg.region == "reccap":
        mpath = wc.resolve_path(io, args.mask)
        mask = wc.load_region_mask(mpath, cfg.mask_var, cfg.mask_regions)
        print(f"\nMask {mpath.name}: {mask.description}")

    announced, wc_announced = set(), set()

    def fetch(m, s, part, p):
        """Annual series for one model, simulation and part."""
        if p["use"] != "layers" or part == "total":
            return wc.get_series(io, cfg, mask, m, s, PARTS[part]["direct"], wc_announced)
        df = layered_series(io, cfg, mask, m, s, p["layered"], args.block_years, announced)
        return df[part]

    print("\nComputing...")
    data: dict = {}            # (model, part) -> {scenario: Series}
    hist_data: dict = {}       # (model, part) -> Series
    errors: dict = {}
    shown_tb = False

    def record(exc, label):
        """Collect an error; print the full traceback for the first one only."""
        nonlocal shown_tb
        if not shown_tb:
            shown_tb = True
            print(f"  first failure ({label}), full traceback:")
            traceback.print_exc(file=sys.stdout)
        errors.setdefault(f"{type(exc).__name__}: {str(exc)[:130]}", []).append(label)

    for m, parts in todo.items():
        p = plans[m]
        for part in parts:
            for s in p["scen"][part]:
                try:
                    data.setdefault((m, part), {})[s] = fetch(m, s, part, p)
                except Exception as e:
                    record(e, f"{m}/{s}/{part}")
            if hist_sim and p["hist"].get(part):
                try:
                    hist_data[(m, part)] = fetch(m, hist_sim, part, p)
                except Exception as e:
                    record(e, f"{m}/{hist_sim}/{part}")
        # consistency check on one scenario: above + below vs cSoil total
        if p["use"] and "cSoil" in p["avail"]:
            s0 = next((s for s in p["scen"]["above1m"] if s in p["avail"]["cSoil"]), None)
            try:
                if s0:
                    if p["use"] == "direct":
                        ab = (wc.get_series(io, cfg, mask, m, s0, "cSoilAbove1m", wc_announced)
                              + wc.get_series(io, cfg, mask, m, s0, "cSoilBelow1m", wc_announced))
                        what = "above + below"
                    else:
                        ab = layered_series(io, cfg, mask, m, s0, p["layered"], args.block_years, announced)["layers_total"]
                        what = "sum of layers"
                    tot = wc.get_series(io, cfg, mask, m, s0, "cSoil", wc_announced)
                    yr = ab.index.intersection(tot.index)[0]
                    ratio = ab.loc[yr] / tot.loc[yr]
                    flag = "" if 0.98 <= ratio <= 1.02 else "   <-- CHECK: does not match the total"
                    print(f"  {m} check ({s0}, {yr}): {what} = {ab.loc[yr]:.1f} Pg C, cSoil = {tot.loc[yr]:.1f} Pg C, "
                          f"ratio {ratio:.3f}{flag}")
            except Exception as e:
                print(f"  {m} check skipped: {type(e).__name__}: {str(e)[:100]}")
                traceback.print_exc(file=sys.stdout)
    for msg, runs in errors.items():
        print(f"  FAILED ({len(runs)}): {', '.join(runs[:8])}{' ...' if len(runs) > 8 else ''} -> {msg}")
    results["series"], results["hist"] = data, hist_data

    print("\nCharts:")
    styles = scenario_styles(scenarios)
    ylims = {}
    if args.share_y:
        for part in args.parts:
            vals = [v for (m, pt), d in data.items() if pt == part for s in d.values() for v in (s.min(), s.max())]
            vals += [v for (m, pt), s in hist_data.items() if pt == part for v in (s.min(), s.max())]
            if vals:
                pad = 0.05 * ((max(vals) - min(vals)) or 1.0)
                ylims[part] = (max(0.0, min(vals) - pad), max(vals) + pad)
    for (m, part), series in data.items():
        results["figures"][(m, part)] = plot_model(
            io, cfg, m, part, series, hist_data.get((m, part)), styles, ylims.get(part))
    return results


if __name__ == "__main__":
    main()