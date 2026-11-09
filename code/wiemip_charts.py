"""Model-comparison charts for the WIEMIP permafrost analysis (first version).

Charts so far: regional air temperature ("tas") and vegetation carbon ("cVeg"),
for a region (RECCAP2 permafrost mask, or global), over time, one line per
model, one panel per scenario (or the reverse with --panel-by model).

Design
------
* Everything is driven by a ChartConfig (models, forcing, scenarios, region, ...),
  so switching forcing is a one-field change.
* Each (model, scenario, variable, region) is reduced once to an annual series and
  cached as a small CSV in the run's processed dir. Re-plotting is then instant.
  Use --refresh to recompute (e.g. after changing the mask file).
* The RECCAP mask (any grid) is mapped onto each model's native grid as an exact
  area-overlap fraction per cell, so no field is ever regridded.
* Reads go through your IO module (load_series / reduce_in_blocks), in time blocks.

Notebook use
------------
    import wiemip_charts as wc
    res = wc.main(["--io-module", "wiemip_io", "--charts", "tas"])
    res = wc.main(["--io-module", "wiemip_io", "--charts", "tas", "cVeg",
                   "--models", "CLM", "JULES", "LPX_Bern", "CoLM"])
    res = wc.main([... , "--check-totals"])     # compare our cVeg totals with global_total()
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import sys
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D

EARTH_R = 6.371e6

DEFAULT_MASK = "overshoot_vizualizations/data/RECCAP2_permafrost_regions_isimip3.nc"
DEFAULT_MODELS = ("CLM", "JULES", "LPX_Bern", "CoLM")
DEFAULT_SCENARIOS = ("l", "m", "hl", "hl_cf", "ml", "ml_cf", "vl", "vl_cf")

# kind: "mean" = area-weighted regional mean; "total" = regional sum, kg -> Pg
# To add a chart (e.g. cSoil), add a line here.
CHARTS = {
    "tas": dict(variable="tas", kind="mean", title="Air temperature", ylabel="Air temperature (°C)"),
    "cVeg": dict(variable="cVeg", kind="total", title="Vegetation carbon", ylabel="Vegetation carbon (Pg C)"),
}


@dataclass(frozen=True)
class ChartConfig:
    models: tuple = DEFAULT_MODELS
    forcing: str = "ukesm"
    scenarios: tuple = DEFAULT_SCENARIOS
    region: str = "reccap"          # "reccap" or "global"
    panel_by: str = "scenario"      # "scenario" or "model"
    mask_var: str | None = None
    mask_regions: tuple | None = None
    factorial: str | None = None
    refresh: bool = False


# ---------------------------------------------------------------------------
# Region mask -> per-model-grid area fraction
# ---------------------------------------------------------------------------

LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")


@dataclass
class MaskInfo:
    M: np.ndarray            # (nlat, nlon) 0/1, lat ascending, lon 0-360 ascending
    lat_edges: np.ndarray
    lon_edges: np.ndarray
    native_area_mkm2: float
    description: str

    @property
    def signature(self) -> str:
        """Short hash of the mask contents, used in cache file names."""
        return hashlib.md5(np.ascontiguousarray(self.M).tobytes()).hexdigest()[:8]


def _find(ds, names):
    for n in names:
        if n in ds.coords or n in ds.dims:
            return n
    raise ValueError(f"none of {names} found in mask file (dims: {list(ds.dims)})")


def _edges(c):
    c = np.asarray(c, dtype="f8")
    mid = (c[1:] + c[:-1]) / 2.0
    return np.concatenate([[c[0] - (c[1] - c[0]) / 2.0], mid, [c[-1] + (c[-1] - c[-2]) / 2.0]])


def _lat_overlap(src_edges, dst_edges):
    """Overlap of lat bands in sin(lat) units, shape (n_dst, n_src)."""
    s = np.sin(np.deg2rad(np.clip(src_edges, -90, 90)))
    d = np.sin(np.deg2rad(np.clip(dst_edges, -90, 90)))
    lo = np.maximum(d[:-1, None], s[None, :-1])
    hi = np.minimum(d[1:, None], s[None, 1:])
    return np.clip(hi - lo, 0, None)


def _lon_overlap(src_edges, dst_edges):
    """Overlap of lon bands in radians, shape (n_dst, n_src), with wrap-around."""
    total = 0.0
    for shift in (-360.0, 0.0, 360.0):
        d = dst_edges + shift
        lo = np.maximum(d[:-1, None], src_edges[None, :-1])
        hi = np.minimum(d[1:, None], src_edges[None, 1:])
        total = total + np.clip(hi - lo, 0, None)
    return np.deg2rad(total)


FILL_THRESHOLD = 1e30     # netCDF default fill is 9.97e36; treat anything this large as "no data"


def load_region_mask(path, var=None, regions=None) -> MaskInfo:
    """Read the RECCAP mask.

    A cell is 'inside' if its value is a valid region ID: finite, > 0 and below the
    fill-value threshold. If `regions` (a list of IDs) is given, only those IDs count.
    If the file has extra dimensions (e.g. one mask per region), the union is used.
    """
    with xr.open_dataset(path, decode_times=False) as ds:
        latn, lonn = _find(ds, LAT_NAMES), _find(ds, LON_NAMES)
        cands = [v for v in ds.data_vars if latn in ds[v].dims and lonn in ds[v].dims]
        if not cands:
            raise ValueError(f"no gridded variable in {path}; variables: {list(ds.data_vars)}")
        name = var or cands[0]
        da = ds[name].load()
        lat = np.asarray(ds[latn].values, dtype="f8")
        lon = np.asarray(ds[lonn].values, dtype="f8") % 360.0

    da = da.where(da < FILL_THRESHOLD)                  # fill value -> NaN
    valid_ids = np.unique(da.values[np.isfinite(da.values) & (da.values > 0)])
    if regions:
        missing = sorted(set(regions) - set(valid_ids.tolist()))
        if missing:
            raise ValueError(f"region IDs {missing} not in mask; available: {valid_ids.tolist()}")
        inside = da.isin(list(regions))
    else:
        inside = da.fillna(0) > 0
    extra = [d for d in da.dims if d not in (latn, lonn)]
    if extra:
        inside = inside.any(extra)
    M = inside.transpose(latn, lonn).values.astype("f8")

    ilat, ilon = np.argsort(lat), np.argsort(lon)
    M, lat, lon = M[np.ix_(ilat, ilon)], lat[ilat], lon[ilon]
    latE, lonE = np.clip(_edges(lat), -90, 90), _edges(lon)

    s = np.sin(np.deg2rad(latE))
    area = EARTH_R**2 * (s[1:] - s[:-1])[:, None] * np.deg2rad(lonE[1:] - lonE[:-1])[None, :]
    native = float((M * area).sum() / 1e12)

    ids = [int(v) if float(v).is_integer() else float(v) for v in valid_ids[:20]]
    used = f"using IDs {list(regions)}" if regions else "using all IDs"
    desc = (f"mask variable '{name}' (dims {da.dims}); region IDs {ids}, {used}; "
            f"{int(M.sum())} of {M.size} cells inside, native area {native:.2f} million km2"
            + (f"; union over {extra}" if extra else "")
            + (f"  [other candidates: {[c for c in cands if c != name]}]" if len(cands) > 1 else ""))
    return MaskInfo(M, latE, lonE, native, desc)


def grid_weights(mask: MaskInfo | None, lat, lon):
    """(region fraction, cell area m2) on a model grid. mask=None -> whole globe."""
    latE = np.clip(_edges(lat), -90, 90)
    lonE = _edges(lon)
    s = np.sin(np.deg2rad(latE))
    dlat_s = s[1:] - s[:-1]
    dlon_r = np.deg2rad(lonE[1:] - lonE[:-1])
    denom = dlat_s[:, None] * dlon_r[None, :]
    area = EARTH_R**2 * denom
    if mask is None:
        return np.ones_like(area), area
    num = _lat_overlap(mask.lat_edges, latE) @ mask.M @ _lon_overlap(mask.lon_edges, lonE).T
    return np.clip(num / denom, 0.0, 1.0), area


# ---------------------------------------------------------------------------
# Per-run annual series (cached)
# ---------------------------------------------------------------------------

def make_reducer(io, kind, mask, label, announced):
    """Reducer for io.load_series: collapse lat/lon to a region mean or total, block by block."""

    def reducer(da):
        lat = np.asarray(da["lat"].values, dtype="f8")
        lon = np.asarray(da["lon"].values, dtype="f8")
        frac, area = grid_weights(mask, lat, lon)
        if mask is not None and label not in announced:
            announced.add(label)
            print(f"  {label}: {lat.size}x{lon.size} grid, region area "
                  f"{(frac * area).sum() / 1e12:.2f} million km2 (native mask {mask.native_area_mkm2:.2f})")
        w = xr.DataArray(frac * area, dims=("lat", "lon"), coords={"lat": da["lat"], "lon": da["lon"]})

        def block(b):
            b = b.astype("float64")
            num = (b.fillna(0.0) * w).sum(("lat", "lon"))
            if kind == "total":
                return num * 1e-12                      # kg -> Pg
            den = (b.notnull() * w).sum(("lat", "lon"))
            return num / den.where(den > 0)

        out = io.reduce_in_blocks(da, block)
        if kind == "mean":
            units = str(da.attrs.get("units", "")).lower()
            if units in ("k", "kelvin") or float(out.mean()) > 150:
                out = out - 273.15
        return out

    return reducer


def get_series(io, cfg, mask, model, sim, chart, announced):
    spec = CHARTS[chart]
    run = io.Run(model, cfg.forcing, sim)
    use_mask = mask if cfg.region == "reccap" else None
    tag = f"reccap-{mask.signature}" if use_mask is not None else cfg.region
    cache = io.PROCESSED_DIR / run.tag / f"{spec['variable']}_{tag}.csv"
    if cache.exists() and not cfg.refresh:
        df = pd.read_csv(cache, index_col=0)
        return pd.Series(df.iloc[:, 0].values, index=df.index.values, name=chart)

    reducer = make_reducer(io, spec["kind"], use_mask, model, announced)
    da = io.load_series(spec["variable"], model, cfg.forcing, sim, cfg.factorial, reducer=reducer)
    s = pd.Series(np.asarray(da.values, dtype="f8"), index=np.asarray(da["time"].values).astype(int), name=chart)
    cache.parent.mkdir(parents=True, exist_ok=True)
    s.rename_axis("year").to_csv(cache)
    return s


def collect(io, cfg, mask, chart):
    data, missing, errors = {}, [], {}
    announced: set = set()
    for model in cfg.models:
        for sim in cfg.scenarios:
            try:
                data[(model, sim)] = get_series(io, cfg, mask, model, sim, chart, announced)
            except FileNotFoundError:
                missing.append(f"{model}/{sim}")
            except Exception as e:
                errors.setdefault(f"{type(e).__name__}: {str(e)[:120]}", []).append(f"{model}/{sim}")
    if missing:
        print(f"  not found ({len(missing)}): {', '.join(missing)}")
    for msg, runs in errors.items():
        print(f"  FAILED ({len(runs)}): {', '.join(runs)} -> {msg}")
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chart(io, data, cfg, chart, mask):
    spec = CHARTS[chart]
    models = [m for m in cfg.models if any(k[0] == m for k in data)]
    scens = [s for s in cfg.scenarios if any(k[1] == s for k in data)]
    if not data:
        print(f"  nothing to plot for {chart}")
        return None
    if cfg.panel_by == "scenario":
        panels, lines, key = scens, models, (lambda p, l: (l, p))
    else:
        panels, lines, key = models, scens, (lambda p, l: (p, l))

    ncol = min(4, len(panels))
    nrow = math.ceil(len(panels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.3 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    colors = {l: f"C{i}" for i, l in enumerate(lines)}
    for ax, p in zip(axes.flat, panels):
        for l in lines:
            s = data.get(key(p, l))
            if s is not None:
                ax.plot(s.index, s.values, color=colors[l], lw=1.5)
        ax.set_title(p.upper() if cfg.panel_by == "scenario" else p, fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel(spec["ylabel"])
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")

    labels = (lambda l: l) if cfg.panel_by == "scenario" else (lambda l: l.upper())
    fig.legend(handles=[Line2D([0], [0], color=colors[l], lw=2, label=labels(l)) for l in lines],
               loc="lower center", ncol=min(len(lines), 6), frameon=False)
    region = "RECCAP2 permafrost region" if cfg.region == "reccap" else "global"
    fig.suptitle(f"{spec['title']} - {region} ({cfg.forcing.upper()})", fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    out_dir = io.FIGURE_DIR / f"compare_{cfg.forcing}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = "_" + "-".join(str(int(r)) for r in cfg.mask_regions) if cfg.mask_regions and cfg.region == "reccap" else ""
    path = out_dir / f"{chart}_{cfg.region}{ids}_{date.today():%Y-%m-%d}.png"
    fig.savefig(path, dpi=200)
    print(f"  saved {path}")
    return fig


# ---------------------------------------------------------------------------
# Sanity check against the registry's own totals
# ---------------------------------------------------------------------------

def check_totals(io, cfg, variable="cVeg", sim="hl"):
    """Compare our global area-weighted totals to the registry's global_total().

    A ratio far from 1 usually means the weighting differs (e.g. a stock stored per
    land area rather than per grid-cell area) and the registry weighting should win.
    """
    print(f"\nGlobal {variable} total check ({cfg.forcing}/{sim}), first value, ours vs registry:")
    gcfg = replace(cfg, region="global", refresh=True)
    for m in cfg.models:
        try:
            ours = get_series(io, gcfg, None, m, sim, variable, set())
            theirs = io.global_total(variable, m, cfg.forcing, sim, cfg.factorial)
            if hasattr(theirs, "compute"):
                theirs = theirs.compute()
            if hasattr(theirs, "dims"):
                other = [d for d in theirs.dims if d != "time"]
                if other:
                    theirs = theirs.sum(other)
            t0 = float(np.asarray(getattr(theirs, "values", theirs), dtype="f8").ravel()[0])
            print(f"  {m:<12} ours {ours.iloc[0]:9.2f}   registry {t0:9.2f}   ratio {ours.iloc[0] / t0:6.3f}")
        except Exception as e:
            print(f"  {m:<12} failed: {type(e).__name__}: {str(e)[:100]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_path(io, p) -> Path:
    p = Path(p)
    root = Path(io.PROJECT_ROOT)
    tried = [p] if p.is_absolute() else [Path.cwd() / p, root / p, root.parent / p, root / "data" / p.name]
    for c in tried:
        if c.exists():
            return c
    raise FileNotFoundError("mask file not found; tried:\n  " + "\n  ".join(map(str, tried)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--io-module", default="wiemip_io")
    ap.add_argument("--charts", nargs="+", default=["tas", "cVeg"], choices=list(CHARTS))
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--forcing", default="ukesm")
    ap.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    ap.add_argument("--region", default="reccap", choices=["reccap", "global"])
    ap.add_argument("--panel-by", default="scenario", choices=["scenario", "model"])
    ap.add_argument("--mask", default=DEFAULT_MASK)
    ap.add_argument("--mask-var", default=None, help="variable name in the mask file (default: first gridded one)")
    ap.add_argument("--mask-regions", nargs="+", type=float, default=None,
                    help="only use these region IDs from the mask (default: all valid IDs)")
    ap.add_argument("--factorial", default=None)
    ap.add_argument("--refresh", action="store_true", help="ignore cached series")
    ap.add_argument("--check-totals", action="store_true", help="compare cVeg totals with global_total()")
    args = ap.parse_args(argv)

    here = Path(globals().get("__file__", Path.cwd())).resolve().parent
    for p in (here, here.parent, here.parent / "src", Path.cwd()):
        sys.path.insert(0, str(p))
    io = importlib.import_module(args.io_module)

    cfg = ChartConfig(models=tuple(args.models), forcing=args.forcing, scenarios=tuple(args.scenarios),
                      region=args.region, panel_by=args.panel_by, mask_var=args.mask_var,
                      mask_regions=tuple(args.mask_regions) if args.mask_regions else None,
                      factorial=args.factorial, refresh=args.refresh)

    mask = None
    if cfg.region == "reccap":
        mpath = resolve_path(io, args.mask)
        mask = load_region_mask(mpath, cfg.mask_var, cfg.mask_regions)
        print(f"Mask {mpath.name}: {mask.description}")

    results = {"series": {}, "figures": {}}
    for chart in args.charts:
        print(f"\n== {chart} ({cfg.region}, {cfg.forcing}) ==")
        data = collect(io, cfg, mask, chart)
        results["series"][chart] = data
        results["figures"][chart] = plot_chart(io, data, cfg, chart, mask)

    if args.check_totals:
        check_totals(io, cfg)
    return results


if __name__ == "__main__":
    main()