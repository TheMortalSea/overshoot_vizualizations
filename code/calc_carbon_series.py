"""Calculate annual carbon stock time series over the permafrost domain.

Usage (notebook):
    calc_carbon_series.main(run, recompute=True)

Usage (CLI):
    python -u code/calc_carbon_series.py --model CLM --forcing gfdl --simulation ml [--recompute]

Output:
    data/processed/<model>_<forcing>_<simulation>/carbon_series.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import xarray as xr

import masks
import wiemip_io
from wiemip_io import PROCESSED_DIR, Run

DEFAULT_VARS = [
    "cSoilAbove1m",
    "cSoilBelow1m",
    "cSoilTotal",
    "cVeg",
    "cSoilVeg",
]

LEGACY_MASK_NC = PROCESSED_DIR / "permafrost_mask.nc"


def say(*parts):
    print(*parts, flush=True)


def csv_path(run: Run):
    return run.processed_dir / "carbon_series.csv"


def mask_path(run: Run):
    return run.processed_dir / "permafrost_mask.nc"


def require_cached(run: Run) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {variable: (years, values_PgC)} or raise FileNotFoundError."""
    path = csv_path(run)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run calc_carbon_series.main(run) first"
        )
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    years = np.array([int(r["year"]) for r in rows])
    return {
        col: (years, np.array([float(r[col]) for r in rows]))
        for col in reader.fieldnames
        if col != "year"
    }


def load_mask(grid: xr.DataArray, run: Run) -> xr.DataArray:
    """2024 permafrost mask aligned to grid coordinates."""
    path = mask_path(run)
    if not path.exists() and LEGACY_MASK_NC.exists():
        say(f"  WARNING: {path.name} not found for {run.tag}; using shared "
            f"{LEGACY_MASK_NC.name} (only valid if it was built for this run)")
        path = LEGACY_MASK_NC

    if path.exists():
        with xr.open_dataset(path) as ds:
            m = ds["permafrost"]
            if "year" in m.dims:
                m = m.sel(year=2024).squeeze()
            return xr.DataArray(
                np.nan_to_num(m.values.astype("f8"), nan=0.0),
                coords={"lat": grid.lat, "lon": grid.lon},
                dims=("lat", "lon"),
            )

    say("  no permafrost mask found; falling back to RECCAP2 boundary...")
    frac = masks.build_permafrost_mask(grid, include_icecap=False, cache=False)
    return (frac > 0.5).astype("f8")


def load_var(var: str, run: Run) -> xr.DataArray:
    """Load a variable, synthesizing composite pools when missing."""
    m, f, s = run.model, run.forcing, run.simulation

    if var == "cSoilTotal":
        try:
            return wiemip_io.load("cSoilTotal", m, f, s)
        except Exception:
            say("  Synthesizing cSoilTotal (cSoilAbove1m + cSoilBelow1m)...")
            above = wiemip_io.load("cSoilAbove1m", m, f, s)
            below = wiemip_io.load("cSoilBelow1m", m, f, s)
            tot = above + below
            tot.attrs = dict(above.attrs)
            tot.attrs["wiemip_variable"] = "cSoilTotal"
            return tot

    if var == "cSoilVeg":
        try:
            return wiemip_io.load("cSoilVeg", m, f, s)
        except Exception:
            say("  Synthesizing cSoilVeg (cSoilTotal + cVeg)...")
            soil = load_var("cSoilTotal", run)
            veg = wiemip_io.load("cVeg", m, f, s)
            tot = soil + veg
            tot.attrs = dict(soil.attrs)
            tot.attrs["wiemip_variable"] = "cSoilVeg"
            return tot

    return wiemip_io.load(var, m, f, s)


def integrate_variable(
    da: xr.DataArray,
    mask_da: xr.DataArray,
    area_m2: xr.DataArray,
    landfrac: xr.DataArray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate kg C / m2 over the permafrost domain into an annual PgC series."""
    land = landfrac if landfrac is not None else xr.ones_like(mask_da)
    weights = (area_m2 * land * mask_da).fillna(0.0)

    if "time" in da.dims and len(da.time) > 1:
        annual_da = wiemip_io.to_annual(da)
    else:
        annual_da = da

    weighted = (annual_da * weights).sum(dim=("lat", "lon")) / 1.0e12

    if "year" in weighted.coords:
        years = weighted["year"].values
    elif "time" in weighted.coords:
        t = weighted["time"]
        if hasattr(t, "dt") and hasattr(t.dt, "year"):
            years = t.dt.year.values
        else:
            years = np.array([int(str(v)[:4]) for v in t.values])
    else:
        n_steps = len(weighted) if weighted.ndim > 0 else 1
        years = np.arange(2024, 2024 + n_steps)

    years = np.atleast_1d(np.asarray(years, dtype=int))
    values = np.atleast_1d(np.asarray(weighted.values, dtype=float))
    return years, values


def main(run: Run, variables: list[str] | None = None, recompute: bool = False):
    out = csv_path(run)

    if not recompute and out.exists():
        say(f"Cached {out.relative_to(PROCESSED_DIR.parent)} exists. "
            "Pass recompute=True to rebuild.")
        return

    vars_to_run = variables or DEFAULT_VARS
    say(f"Calculating carbon series for {run.tag}...")

    base_grid = load_var("cSoilAbove1m", run)
    area_m2 = wiemip_io.get_cell_area(base_grid)
    landfrac = wiemip_io.get_landfrac(base_grid)
    mask_da = load_mask(base_grid, run)

    results: dict[str, dict[int, float]] = {}
    all_years: set[int] = set()

    for v in vars_to_run:
        say(f"\n--- Variable: {v} ---")
        try:
            da = load_var(v, run)
            yrs, vals = integrate_variable(da, mask_da, area_m2, landfrac)
            results[v] = dict(zip(yrs.tolist(), vals.tolist()))
            all_years.update(yrs.tolist())
            say(f"  {v}: {vals[0]:.1f} PgC ({yrs[0]}) -> {vals[-1]:.1f} PgC ({yrs[-1]})")
        except Exception as exc:
            say(f"  Failed to calculate {v}: {exc}")

    if not results:
        raise RuntimeError(f"Could not compute carbon series for any of {vars_to_run}")

    cols = [v for v in vars_to_run if v in results]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year"] + cols)
        for y in sorted(all_years):
            writer.writerow([y] + [f"{results[c].get(y, np.nan):.3f}" for c in cols])

    say(f"\nWrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calculate carbon series.")
    p.add_argument("--model", required=True)
    p.add_argument("--forcing", required=True)
    p.add_argument("--simulation", required=True)
    p.add_argument("--recompute", action="store_true")
    a = p.parse_args()
    main(Run(a.model, a.forcing, a.simulation), recompute=a.recompute)