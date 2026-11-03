#!/usr/bin/env python
"""WIEMIP intercomparability diagnostics.

Probes every (model, forcing, simulation, variable) combination through your IO
module / the wiemip_registry, records metadata (units, grid, time axis, fill
values, value magnitudes, auxiliary fields, ...), then analyses which models
can be compared directly and which need regridding / unit conversion / time
alignment.

Outputs (in --out-dir):
    records.csv              one row per probed combination (the raw metadata)
    coverage.csv             model x (forcing/simulation): variables found / tried
    variable_comparison.csv  one representative row per model per variable
    flags.csv                every detected incompatibility or oddity
    report.md                human-readable summary of all of the above

Examples
--------
    python wiemip_diagnostics.py                         # everything the registry knows
    python wiemip_diagnostics.py --models CLASSIC CLM --variables cVeg tas
    python wiemip_diagnostics.py --totals                # also pull latitudinal_sum() totals
    python wiemip_diagnostics.py --reuse diagnostics/.../records.csv   # re-analyse, no I/O

Set --io-module to the name of the module that holds your `retrieve`,
`standardise`, ... helpers (default: wiemip_io).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import sys
import time as _time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Defaults -- edit to taste
# ---------------------------------------------------------------------------

DEFAULT_IO_MODULE = "wiemip_io"
DEFAULT_FORCINGS = ["ukesm", "ipsl", "gfdl"]
DEFAULT_SIMULATIONS = ["l", "m", "hl", "hl_cf", "hist", "ctrl"]
DEFAULT_VARIABLES = [
    # carbon stocks
    "cVeg", "cSoil", "cLitter", "cTotal",
    # carbon fluxes
    "gpp", "npp", "ra", "rh", "nbp",
    # structure / climate
    "lai", "tas", "pr",
]

STR_COLS = [
    "status", "error", "path", "note", "raw_dims", "raw_shape", "dtype", "units",
    "units_norm", "long_name", "standard_name", "cell_methods", "fill_value",
    "raw_lat_order", "raw_lon_range", "raw_time_dtype", "raw_time_first",
    "raw_time_last", "file_time_units", "file_calendar", "dims", "extra_dims",
    "grid_sig", "grid_hash", "freq", "steps_per_year", "file_vars",
    "landfrac_var", "area_var", "area_units", "institution", "source", "title",
    "total_units", "total_shape",
]

# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"([a-zA-Z%µ]+)\s*(?:\^)?\s*([+-]?\d+)?")


def normalise_units(u: str) -> str:
    """Canonical form so 'kg m-2 s-1', 'kg/m2/s' and 'kg m^-2 s^-1' compare equal."""
    u = (u or "").strip().lower()
    if not u:
        return ""
    if u in ("1", "none", "unitless", "-", "fraction", "frac"):
        return u
    try:
        s = u.replace("**", "^").replace("·", " ").replace("*", " ")
        s = s.replace("(", " ").replace(")", " ")
        segments = s.split("/")
        exps: dict[str, int] = {}
        for i, seg in enumerate(segments):
            sign = 1 if i == 0 else -1
            for name, exp in _TOKEN.findall(seg):
                e = int(exp) if exp else 1
                exps[name] = exps.get(name, 0) + sign * e
        parts = [f"{k}{v}" if v != 1 else k for k, v in sorted(exps.items()) if v != 0]
        return " ".join(parts) if parts else u
    except Exception:
        return u


# ---------------------------------------------------------------------------
# Probing a single run/variable
# ---------------------------------------------------------------------------

def _first_attr(da, keys):
    for k in keys:
        if k in da.attrs:
            return str(da.attrs[k])
    return ""


def _pick_dataarray(obj, variable, rec):
    if isinstance(obj, xr.Dataset):
        if variable in obj.data_vars:
            return obj[variable]
        names = list(obj.data_vars)
        rec["note"] = f"variable not in file; used '{names[0]}' of {names}"
        return obj[names[0]]
    return obj


def describe_raw(da: xr.DataArray) -> dict:
    """Metadata as stored, BEFORE standardise() changes anything."""
    out = {
        "raw_dims": ",".join(map(str, da.dims)),
        "raw_shape": "x".join(map(str, da.shape)),
        "dtype": str(da.dtype),
        "units": _first_attr(da, ["units"]),
        "long_name": _first_attr(da, ["long_name"]),
        "standard_name": _first_attr(da, ["standard_name"]),
        "cell_methods": _first_attr(da, ["cell_methods"]),
    }
    fill = da.encoding.get("_FillValue", da.attrs.get("_FillValue", da.encoding.get("missing_value", "")))
    out["fill_value"] = str(fill)

    for name in ("lat", "latitude"):
        if name in da.coords and da[name].size > 1:
            v = da[name].values
            out["raw_lat_order"] = "descending" if v[0] > v[-1] else "ascending"
            break
    for name in ("lon", "longitude"):
        if name in da.coords:
            v = np.asarray(da[name].values, dtype="f8")
            out["raw_lon_range"] = f"{v.min():.2f}..{v.max():.2f}"
            break

    if "time" in da.coords and da["time"].size:
        t = da["time"]
        out["raw_time_dtype"] = str(t.dtype)
        out["raw_time_first"] = str(t.values[0])
        out["raw_time_last"] = str(t.values[-1])
    return out


def describe_std(std: xr.DataArray, io) -> dict:
    """Grid + time structure after standardise()."""
    out = {"dims": ",".join(map(str, std.dims))}
    extra = [d for d in std.dims if d not in ("time", "lat", "lon")]
    out["extra_dims"] = ";".join(f"{d}={std.sizes[d]}" for d in extra)

    if "lat" in std.coords and "lon" in std.coords:
        lat = np.asarray(std["lat"].values, dtype="f8")
        lon = np.asarray(std["lon"].values, dtype="f8")
        dlat = float(np.abs(np.diff(lat)).mean()) if lat.size > 1 else np.nan
        dlon = float(np.abs(np.diff(lon)).mean()) if lon.size > 1 else np.nan
        out.update(
            nlat=lat.size, nlon=lon.size, dlat=dlat, dlon=dlon,
            lat_min=lat.min(), lat_max=lat.max(), lon_min=lon.min(), lon_max=lon.max(),
        )
        try:
            out["regular_grid"] = bool(
                np.allclose(np.diff(lat), np.diff(lat)[0], atol=1e-4)
                and np.allclose(np.diff(lon), np.diff(lon)[0], atol=1e-4)
            )
        except Exception:
            out["regular_grid"] = False
        out["grid_sig"] = f"{lat.size}x{lon.size}@{dlat:.3f}x{dlon:.3f}"
        out["grid_hash"] = hashlib.md5(
            np.round(lat, 4).tobytes() + np.round(lon, 4).tobytes()
        ).hexdigest()[:8]

    if "time" in std.dims:
        years = io._time_years(std)
        uy, counts = np.unique(years, return_counts=True)
        spy = counts[0] if len(set(counts)) == 1 else None
        out.update(
            n_steps=int(std.sizes["time"]),
            year_min=int(uy.min()), year_max=int(uy.max()), n_years=int(uy.size),
            year_gaps=int(np.setdiff1d(np.arange(uy.min(), uy.max() + 1), uy).size),
        )
        out["steps_per_year"] = str(spy) if spy else f"irregular({counts.min()}-{counts.max()})"
        out["freq"] = {1: "annual", 12: "monthly", 365: "daily", 360: "daily(360)"}.get(
            spy, f"{spy}/yr" if spy else "irregular"
        )
    else:
        out["freq"] = "static"
    return out


def describe_values(std: xr.DataArray, io) -> dict:
    """Value statistics on the first time slice (cheap, but enough to spot unit/mask issues)."""
    first = io.first_step(std).load()
    vals = np.asarray(first.values, dtype="f8")
    finite = vals[np.isfinite(vals)]
    out = {"nan_frac": float(1.0 - finite.size / vals.size) if vals.size else np.nan}
    if finite.size:
        nz = finite[finite != 0]
        out.update(
            v_min=float(finite.min()), v_max=float(finite.max()), v_mean=float(finite.mean()),
            p50_abs=float(np.median(np.abs(nz))) if nz.size else 0.0,
            n_huge=int((np.abs(finite) > 1e15).sum()),
        )
    return out


def describe_file_aux(path) -> dict:
    """Peek at the underlying NetCDF: variables present, landfrac/area, global attrs, time encoding."""
    out: dict = {}
    if not path or not Path(path).exists():
        return out
    try:
        with xr.open_dataset(path, decode_times=False, decode_timedelta=False) as ds:
            out["file_vars"] = ";".join(list(ds.data_vars)[:25])
            for name in ("landfrac", "sftlf"):
                if name in ds:
                    out["landfrac_var"] = name
                    out["landfrac_max"] = float(ds[name].max())
                    break
            if "area" in ds:
                out["area_var"] = "area"
                out["area_units"] = str(ds["area"].attrs.get("units", ""))
            for k in ("institution", "source", "title"):
                out[k] = str(ds.attrs.get(k, ""))[:80]
            if "time" in ds.variables:
                tv = ds["time"]
                out["file_time_units"] = str(tv.attrs.get("units", ""))
                out["file_calendar"] = str(tv.attrs.get("calendar", tv.attrs.get("calender", "")))
    except Exception as e:  # metadata only; never fail the probe because of this
        out["note"] = f"aux read failed: {type(e).__name__}"
    return out


def describe_total(file) -> dict:
    out: dict = {}
    res = file.latitudinal_sum()
    if hasattr(res, "compute"):
        res = res.compute()
    out["total_units"] = str(getattr(res, "attrs", {}).get("units", ""))
    out["total_shape"] = "x".join(map(str, getattr(res, "shape", ())))
    if hasattr(res, "dims"):
        other = [d for d in res.dims if d != "time"]
        if other:
            res = res.sum(other)
    arr = np.asarray(getattr(res, "values", res), dtype="f8").ravel()
    if arr.size:
        out["total_first"], out["total_last"] = float(arr[0]), float(arr[-1])
    return out


def probe(io, model, forcing, simulation, variable, factorial, do_stats, do_totals) -> dict:
    rec = {"model": model, "forcing": forcing, "simulation": simulation,
           "variable": variable, "status": "ok", "error": ""}
    try:
        file = io.retrieve(variable, model, forcing, simulation, factorial)
    except FileNotFoundError as e:
        rec.update(status="missing", error=str(e))
        return rec
    except Exception as e:
        rec.update(status="registry_error", error=f"{type(e).__name__}: {e}")
        return rec

    rec["path"] = str(file.path)
    try:
        raw = _pick_dataarray(file.read(), variable, rec)
        rec.update(describe_raw(raw))
        rec["units_norm"] = normalise_units(rec.get("units", ""))
        std = io.standardise(raw)
        rec.update(describe_std(std, io))
        if do_stats:
            rec.update(describe_values(std, io))
        rec.update(describe_file_aux(file.path))
        if do_totals:
            try:
                rec.update(describe_total(file))
            except Exception as e:
                rec["note"] = (rec.get("note", "") + f" total failed: {type(e).__name__}: {e}").strip()
    except Exception as e:
        rec.update(status="read_error", error=f"{type(e).__name__}: {e}")
    return rec


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def md_table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out)


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["combo"] = df["forcing"] + "/" + df["simulation"]
    ok = df[df.status == "ok"].groupby(["model", "combo"]).size()
    tot = df.groupby(["model", "combo"]).size()
    cov = pd.DataFrame({"ok": ok, "tot": tot}).fillna(0).astype(int)
    cov["cell"] = cov["ok"].astype(str) + "/" + cov["tot"].astype(str)
    return cov["cell"].unstack("combo").fillna("0/0")


def analyse(df: pd.DataFrame, magnitude_ratio: float):
    """Returns (flags DataFrame, representative-rows DataFrame, groups dict, overlap rows)."""
    ok = df[df.status == "ok"].copy()
    flags: list[dict] = []
    groups: dict = {}
    overlaps: list[list] = []
    reps = []

    def flag(level, variable, topic, detail):
        flags.append({"level": level, "variable": variable, "topic": topic, "detail": detail})

    # Failures worth knowing about
    for (model, status), g in df[df.status != "ok"].groupby(["model", "status"]):
        if status == "missing":
            continue  # shown in the coverage matrix instead
        flag("WARN", "*", status, f"{model}: {len(g)} combos, e.g. {g.iloc[0]['error'][:150]}")

    for v, sub in ok.groupby("variable"):
        sub = sub.sort_values(["model", "forcing", "simulation"])
        rep = sub.groupby("model").first().reset_index()
        rep["variable"] = v
        reps.append(rep)
        models = list(rep["model"])

        # Within-model consistency across forcing / simulation
        for m in models:
            msub = sub[sub.model == m]
            for col in ("units_norm", "grid_hash", "freq"):
                vals = msub[col].dropna().unique()
                if len(vals) > 1:
                    flag("WARN", v, f"{col} varies within {m}", f"{list(vals)}")

        if len(models) < 2:
            flag("INFO", v, "single model", f"only {models[0] if models else 'none'} provides this variable")
            continue

        # Units
        by_norm = rep.groupby("units_norm")["model"].apply(list).to_dict()
        if len(by_norm) > 1:
            flag("WARN", v, "UNITS differ",
                 "; ".join(f"'{u}': {', '.join(ms)}" for u, ms in by_norm.items()))
        elif rep["units"].nunique() > 1:
            flag("INFO", v, "unit notation differs",
                 "; ".join(f"{r.model}='{r.units}'" for r in rep.itertuples()))
        if (rep["units"] == "").any():
            flag("WARN", v, "units missing",
                 ", ".join(rep.loc[rep["units"] == "", "model"]))

        # Grid
        by_grid = rep.groupby("grid_sig")["model"].apply(list).to_dict()
        if len(by_grid) > 1:
            flag("WARN", v, "GRID differs (regrid needed)",
                 "; ".join(f"{g}: {', '.join(ms)}" for g, ms in by_grid.items()))
        elif rep["grid_hash"].nunique() > 1:
            flag("WARN", v, "grid coordinates differ (same shape)",
                 "same sig, different lat/lon values (offset / cell-centre vs edge?)")

        # Frequency / extra dims
        if rep["freq"].nunique() > 1:
            flag("WARN", v, "TIME FREQUENCY differs",
                 "; ".join(f"{r.model}={r.freq}" for r in rep.itertuples()))
        if rep["extra_dims"].nunique() > 1:
            flag("WARN", v, "extra dimensions differ",
                 "; ".join(f"{r.model}=[{r.extra_dims}]" for r in rep.itertuples()))

        # Raw conventions that standardise() hides
        for col, label in (("raw_lat_order", "raw lat order"), ("raw_lon_range", "raw lon range"),
                           ("raw_time_dtype", "raw time encoding"), ("file_calendar", "calendar")):
            if col in rep and rep[col].replace("", np.nan).nunique() > 1:
                flag("INFO", v, f"{label} differs",
                     "; ".join(f"{r.model}={getattr(r, col)}" for r in rep.itertuples()))

        # Magnitude sanity (catches g-vs-kg, per-day vs per-second, fraction vs percent, ...)
        if "p50_abs" in rep:
            p = rep[["model", "p50_abs", "units"]].dropna()
            p = p[p["p50_abs"] > 0]
            if len(p) > 1 and p["p50_abs"].max() / p["p50_abs"].min() > magnitude_ratio:
                ratio = p["p50_abs"].max() / p["p50_abs"].min()
                flag("WARN", v, f"magnitude spread x{ratio:.3g}",
                     "; ".join(f"{r.model}={r.p50_abs:.3g} [{r.units}]" for r in p.itertuples())
                     + " -- possible unit/scale mismatch")
        if "n_huge" in rep and (rep["n_huge"] > 0).any():
            flag("WARN", v, "huge values (unmasked fill?)",
                 ", ".join(rep.loc[rep["n_huge"] > 0, "model"]))

        # Directly comparable groups
        groups[v] = [
            (key, list(g["model"]))
            for key, g in rep.groupby(["units_norm", "grid_hash", "freq"])
        ]

        # Year overlap per forcing/simulation
        for (f, s), g in sub.groupby(["forcing", "simulation"]):
            per_model = g.groupby("model").agg(lo=("year_min", "min"), hi=("year_max", "max"))
            if len(per_model) < 2:
                continue
            lo, hi = per_model["lo"].max(), per_model["hi"].min()
            span = f"{int(lo)}-{int(hi)}" if lo <= hi else "NONE"
            overlaps.append([v, f"{f}/{s}", len(per_model), span,
                             ", ".join(f"{m}:{int(r.lo)}-{int(r.hi)}" for m, r in per_model.iterrows())])
            if lo > hi:
                flag("WARN", v, f"no common years ({f}/{s})", overlaps[-1][-1])

    flags_df = pd.DataFrame(flags, columns=["level", "variable", "topic", "detail"])
    reps_df = pd.concat(reps, ignore_index=True) if reps else pd.DataFrame()
    return flags_df, reps_df, groups, overlaps


def build_report(df, cov, flags, reps, groups, overlaps, meta) -> str:
    L: list[str] = []
    L.append(f"# WIEMIP intercomparability diagnostics ({date.today():%Y-%m-%d})\n")
    L.append(f"Models: {', '.join(meta['models'])}  ")
    L.append(f"Forcings: {', '.join(meta['forcings'])}  ")
    L.append(f"Simulations: {', '.join(meta['simulations'])}  ")
    L.append(f"Variables: {', '.join(meta['variables'])}\n")
    st = df["status"].value_counts().to_dict()
    L.append("Probe results: " + ", ".join(f"{k}={v}" for k, v in st.items()) + "\n")

    L.append("## 1. Coverage (variables found / tried)\n")
    L.append(md_table(["model"] + list(cov.columns),
                      [[m] + list(r) for m, r in cov.iterrows()]) + "\n")

    ok = df[df.status == "ok"]
    L.append("## 2. Grids per model\n")
    rows = []
    for m, g in ok.groupby("model"):
        for sig, gg in g.groupby("grid_sig"):
            r = gg.iloc[0]
            rows.append([m, sig, f"{r.get('lat_min', np.nan):.2f}..{r.get('lat_max', np.nan):.2f}",
                         f"{r.get('lon_min', np.nan):.2f}..{r.get('lon_max', np.nan):.2f}",
                         r.get("regular_grid", ""), r.get("raw_lat_order", ""),
                         r.get("raw_lon_range", ""), r.get("landfrac_var", "") or "-",
                         r.get("area_var", "") or "-", ", ".join(sorted(gg["variable"].unique()))[:60]])
    L.append(md_table(["model", "grid (nlat x nlon @ dlat x dlon)", "lat (std)", "lon (std)", "regular",
                       "raw lat", "raw lon", "landfrac", "area", "variables"], rows) + "\n")

    L.append("## 3. Flags\n")
    if flags.empty:
        L.append("No issues detected.\n")
    else:
        for level in ("WARN", "INFO"):
            sub = flags[flags.level == level]
            if sub.empty:
                continue
            L.append(f"### {level}\n")
            L.append(md_table(["variable", "topic", "detail"],
                              sub[["variable", "topic", "detail"]].values.tolist()) + "\n")

    L.append("## 4. Per-variable comparison (first available run per model)\n")
    for v, g in reps.groupby("variable") if not reps.empty else []:
        L.append(f"### {v}\n")
        L.append(md_table(
            ["model", "units", "grid", "freq", "years", "extra dims", "median |x|", "nan frac"],
            [[r.model, r.units or "?", r.grid_sig, r.freq,
              f"{int(r.year_min)}-{int(r.year_max)}" if pd.notna(r.year_min) else "-",
              r.extra_dims or "-",
              f"{r.p50_abs:.3g}" if "p50_abs" in g and pd.notna(r.p50_abs) else "-",
              f"{r.nan_frac:.2f}" if "nan_frac" in g and pd.notna(r.nan_frac) else "-"]
             for r in g.itertuples()]) + "\n")

    L.append("## 5. Directly comparable groups (same units, grid, frequency)\n")
    for v, gl in groups.items():
        L.append(f"- **{v}**: " + " | ".join(
            f"[{', '.join(ms)}] ({k[0] or '?'}, grid {k[1]}, {k[2]})" for k, ms in gl))
    L.append("")

    L.append("## 6. Common year ranges\n")
    L.append(md_table(["variable", "forcing/sim", "n models", "common years", "per model"], overlaps) + "\n")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def discover_models(wr) -> list[str]:
    m = getattr(wr, "models", None)
    try:
        if callable(m):
            m = m()
        if isinstance(m, dict):
            return sorted(map(str, m))
        if m is not None:
            return sorted(x if isinstance(x, str) else str(getattr(x, "name", x)) for x in m)
    except Exception:
        pass
    return []


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-module", default=DEFAULT_IO_MODULE)
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--forcings", nargs="+", default=DEFAULT_FORCINGS)
    ap.add_argument("--simulations", nargs="+", default=DEFAULT_SIMULATIONS)
    ap.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    ap.add_argument("--factorial", default=None)
    ap.add_argument("--no-stats", action="store_true", help="skip reading values (metadata only)")
    ap.add_argument("--totals", action="store_true", help="also call latitudinal_sum() (slower)")
    ap.add_argument("--magnitude-ratio", type=float, default=20.0)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--reuse", type=Path, help="existing records.csv: skip probing, just re-analyse")
    ap.add_argument("--quiet", action="store_true", help="do not print the full report")
    args = ap.parse_args(argv)

    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    for p in (here, here.parent, here.parent / "src", Path.cwd()):
        sys.path.insert(0, str(p))
    io = importlib.import_module(args.io_module)

    out_dir = args.out_dir or (io.PROJECT_ROOT / "diagnostics" / f"wiemip_{date.today():%Y-%m-%d}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse:
        df = pd.read_csv(args.reuse)
        for c in STR_COLS:
            if c in df:
                df[c] = df[c].fillna("")
        models = sorted(df["model"].unique())
    else:
        models = args.models or discover_models(io.wr)
        if not models:
            sys.exit("Could not discover models from wiemip_registry; pass --models.")
        combos = [(m, f, s, v) for m in models for f in args.forcings
                  for s in args.simulations for v in args.variables]
        print(f"Probing {len(combos)} combinations across {len(models)} models...")
        records, t0 = [], _time.time()
        for i, (m, f, s, v) in enumerate(combos, 1):
            rec = probe(io, m, f, s, v, args.factorial, not args.no_stats, args.totals)
            records.append(rec)
            print(f"[{i:>5}/{len(combos)}] {m}/{f}/{s}/{v}: {rec['status']}", flush=True)
        print(f"Done in {_time.time() - t0:.0f}s")
        df = pd.DataFrame(records)
        for c in STR_COLS:
            if c in df:
                df[c] = df[c].fillna("")
        df.to_csv(out_dir / "records.csv", index=False)

    cov = coverage_table(df)
    cov.to_csv(out_dir / "coverage.csv")
    flags, reps, groups, overlaps = analyse(df, args.magnitude_ratio)
    flags.to_csv(out_dir / "flags.csv", index=False)
    reps.to_csv(out_dir / "variable_comparison.csv", index=False)

    meta = dict(models=models, forcings=sorted(df["forcing"].unique()),
                simulations=sorted(df["simulation"].unique()),
                variables=sorted(df["variable"].unique()))
    report = build_report(df, cov, flags, reps, groups, overlaps, meta)
    (out_dir / "report.md").write_text(report)
    if not args.quiet:
        print("\n" + report)
    print(f"\nWrote outputs to {out_dir}")
    return dict(df=df, coverage=cov, flags=flags, reps=reps, groups=groups,
                overlaps=overlaps, report=report, out_dir=out_dir)


if __name__ == "__main__":
    main()