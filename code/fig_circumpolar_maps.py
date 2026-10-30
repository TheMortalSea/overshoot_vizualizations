"""Circumpolar maps of one carbon pool and of near-surface air temperature.

Notebook:
    fig_circumpolar_maps.main(run)                  # cSoilAbove1m (default)
    fig_circumpolar_maps.main(run, "cSoilBelow1m")
    fig_circumpolar_maps.main(run, "tas")           # temperature map

Both maps: first year | GMST peak year | last year | last minus first. Carbon as
stock, temperature as warming above 1850-1900. Reads the raw annual carbon files
and monthly tas directly (~30 s); carbon_series.csv from the run's processed
folder is used only for the year range.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import xarray as xr

import calc_carbon_series
import maps
import masks
import plotting
import tas_baseline
import wiemip_io
from wiemip_io import Run

CARBON_VAR_DEFAULT = "cSoilAbove1m"
TEMP_VAR = "tas"

SHORT_LABELS = {
    "cSoilAbove1m": "Soil carbon above 1 m",
    "cSoilBelow1m": "Soil carbon below 1 m",
    "cSoilTotal": "Total soil carbon",
    "cSoil": "Total soil carbon",
    "cVeg": "Vegetation carbon",
    "cSoilVeg": "Soil + vegetation carbon",
}

DERIVED = {
    "cSoilTotal": ("cSoilAbove1m", "cSoilBelow1m"),
    "cSoilVeg": ("cSoilAbove1m", "cSoilBelow1m", "cVeg"),
}


def say(*parts):
    print(*parts, flush=True)


def configure_maps(run: Run):
    """maps.py formats its headings from module-level run constants (I haven't
    seen it), so set whichever of them exist. Remove once maps.row takes a run."""
    for attr, val in [
        ("MODEL", run.model),
        ("FORCING", run.forcing),
        ("SIMULATION", run.simulation),
        ("SCENARIO", run.scenario_label),
        ("SCENARIO_LABEL", run.scenario_label),
        ("DEFAULT_SCENARIO", run.scenario_label),
    ]:
        if hasattr(maps, attr):
            setattr(maps, attr, val)


def load_component(component: str, run: Run) -> xr.DataArray:
    """Load a spatial variable for the run, reduced to annual means."""
    da = wiemip_io.load(component, run.model, run.forcing, run.simulation)
    if "time" in da.dims and len(da.time) > 1:
        da = wiemip_io.to_annual(da)
    return da


def carbon_cube(variable: str, run: Run) -> np.ndarray:
    """Every year of a carbon stock, (year, lat, lon), kg C m-2 of land.

    Derived variables are summed across their components.
    """
    total = None
    for component in DERIVED.get(variable, (variable,)):
        field = load_component(component, run).values
        total = field if total is None else total + field
    return total


def pi_climatology(run: Run) -> np.ndarray | None:
    """Per-cell 1850-1900 mean tas (K), from this model's historical run."""
    first, last = getattr(tas_baseline, "BASELINE_PERIOD", (1850, 1900))

    # 1. Historical run via the registry. Time is integer years after to_annual,
    #    so slice with ints (string slices don't work on an integer axis).
    for exp in ("hist", "historical"):
        try:
            hist_da = wiemip_io.load(TEMP_VAR, run.model, run.forcing, exp)
            ann = wiemip_io.to_annual(hist_da)
            return ann.sel(time=slice(first, last)).mean("time").values
        except Exception:
            pass

    # 2. Disk scan fallback for historical tas NetCDF
    for raw_dir in (wiemip_io.PROJECT_ROOT / "data" / "raw",):
        if raw_dir.exists():
            for match in raw_dir.rglob("*hist*tas*.nc"):
                with xr.open_dataset(match, decode_timedelta=False) as ds:
                    ann = wiemip_io.to_annual(wiemip_io.standardise(ds[TEMP_VAR]))
                    return ann.sel(time=slice(first, last)).mean("time").values

    return None


def land_and_permafrost(grid: xr.DataArray):
    """(lat, lon, landmask, pf_frac) for the map panels."""
    lat, lon = grid.lat.values, grid.lon.values

    if "landmask" in grid:
        landmask = grid["landmask"].values
    else:
        lf = wiemip_io.get_landfrac(grid)
        landmask = lf.values if lf is not None else np.ones((len(lat), len(lon)))

    try:
        pf_frac = masks.permafrost_mask(lat, lon).values
    except Exception:
        pf_frac = masks.build_permafrost_mask(grid).values
    return lat, lon, landmask, pf_frac


def year_axis(run: Run, n_fallback: int) -> np.ndarray:
    """Year range from the run's carbon_series.csv, else 2024 onward."""
    try:
        cached = calc_carbon_series.require_cached(run)
        return next(iter(cached.values()))[0]
    except FileNotFoundError:
        say("  NOTE: carbon_series.csv missing; assuming years start at 2024")
        return np.arange(2024, 2024 + n_fallback)


def peak_year_of(years: np.ndarray, run: Run) -> int:
    gmst = tas_baseline.gmst_series(years, run.scenario_label)
    if gmst is None:
        say(f"  NOTE: no GMST pathway for {run.scenario_label!r}; "
            "using a quarter of the way through the series as 'peak'")
        return int(years[len(years) // 4])
    return int(years[int(np.nanargmax(gmst))])


def snapshot_titles(years, peak):
    snaps = [int(years[0]), peak, int(years[-1])]
    titles = [str(snaps[0]), f"peak warming, {peak}", str(snaps[2]),
              f"{snaps[2]} − {snaps[0]}"]
    return snaps, titles


def temperature_map(run: Run):
    say(f"\nCircumpolar temperature map ({run.model} · {run.forcing.upper()} · "
        f"{run.scenario_label}) ...")
    tas_annual = load_component(TEMP_VAR, run)
    lat, lon, landmask, pf_frac = land_and_permafrost(tas_annual)
    ocean = landmask < 0.5

    def mask_ocean(field):
        return np.where(ocean, np.nan, field)

    years = year_axis(run, len(tas_annual.time))
    peak = peak_year_of(years, run)
    snapshots, titles = snapshot_titles(years, peak)

    pi_t = pi_climatology(run)
    if pi_t is None:
        raise FileNotFoundError(
            f"no historical {TEMP_VAR} for {run.model}/{run.forcing}: cannot "
            "compute the 1850-1900 baseline (the old code silently used the "
            "first model year instead, which is a different zero)")

    fields = [mask_ocean(tas_annual.isel(time=y - years[0]).values - pi_t)
              for y in snapshots]
    values = np.concatenate([f.ravel() for f in fields])
    vmax = float(np.nanpercentile(values, 99))
    vmin = min(float(np.nanpercentile(values, 0.5)), -0.5)
    warming = plotting.warming_cmap(vmin, vmax)
    unit = "°C above 1850–1900"
    residual = fields[2] - fields[0]
    lim = maps.symmetric_limit(residual)

    out = run.figure_path("map_circumpolar_tas")
    maps.row(
        [(f, t, warming, vmin, vmax, unit) for f, t in zip(fields, titles)]
        + [(residual, titles[3], plotting.DIVERGING, -lim, lim, "Δ °C")],
        lat, lon, landmask, pf_frac, out,
        name="Near-surface air temperature, warming above 1850–1900",
        note=f"Single years; baseline is the 1850–1900 mean. "
             f"Peak = GMST maximum of the FaIR {run.scenario_label} pathway.",
    )
    say(f"      wrote {out}")
    return out


def carbon_map(run: Run, carbon_var: str):
    cached = calc_carbon_series.require_cached(run)
    if carbon_var not in cached:
        raise KeyError(f"{carbon_var} not in carbon_series.csv for {run.tag}; "
                       f"available: {sorted(cached)}")
    years, carbon = cached[carbon_var]

    say(f"{carbon_var} ({run.model} · {run.forcing.upper()} · "
        f"{run.scenario_label}): {years[0]}-{years[-1]}  n={len(years)}")
    say(f"  {carbon[0]:.1f} -> {carbon[-1]:.1f} PgC "
        f"({100.0 * carbon[-1] / carbon[0]:.2f}% of start)")

    probe = DERIVED.get(carbon_var, (carbon_var,))[0]
    grid = load_component(probe, run)
    lat, lon, landmask, pf_frac = land_and_permafrost(grid)
    ocean = landmask < 0.5

    def mask_ocean(field):
        return np.where(ocean, np.nan, field)

    peak = peak_year_of(years, run)
    snapshots, titles = snapshot_titles(years, peak)

    say(f"\n[1/1] {carbon_var} map ...")
    cube = carbon_cube(carbon_var, run)
    stocks = [mask_ocean(cube[y - years[0]]) for y in snapshots]
    change = stocks[2] - stocks[0]
    vmax = float(np.nanpercentile(np.concatenate([s.ravel() for s in stocks]), 99))
    lim = maps.symmetric_limit(change)
    unit = "kg C m-2"
    pool = plotting.pool_cmap(carbon_var)
    long_name = SHORT_LABELS.get(carbon_var, grid.attrs.get("long_name", carbon_var))

    out = run.figure_path(f"map_circumpolar_{carbon_var}")
    maps.row(
        [(s, t, pool, 0.0, vmax, unit) for s, t in zip(stocks, titles)]
        + [(change, titles[3], plotting.carbon_change_cmap(carbon_var),
            -lim, lim, f"Δ {unit}")],
        lat, lon, landmask, pf_frac, out,
        name=long_name,
        note=f"Single years, per m² of land. Peak = GMST maximum of the FaIR "
             f"{run.scenario_label} pathway.",
    )
    say(f"      wrote {out}")
    return out


def main(run: Run, carbon_var: str = CARBON_VAR_DEFAULT):
    configure_maps(run)
    if carbon_var == TEMP_VAR:
        return temperature_map(run)
    return carbon_map(run, carbon_var)