"""Calculate permafrost extent and masks across all time chunks (2024-2300).

Notebook:
    calc_permafrost_extent.main(run, method="soilt", recompute=True)

CLI:
    python -u code/calc_permafrost_extent.py --method soilt \
        --model CLM --forcing gfdl --simulation ml [--recompute]

Outputs (per run, in data/processed/<model>_<forcing>_<simulation>/):
  - permafrost_mask.nc
  - permafrost_extent.csv
  - permafrost_extent_method.txt   (which method built the cache)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import xarray as xr

import masks
import wiemip_io
from wiemip_io import Run

DEEPEST_M = 9.795     # CLM level 21 (~9.8 m zero-annual-amplitude depth)
FREEZING_K = 273.15


def say(*parts):
    print(*parts, flush=True)


def mask_path(run: Run) -> Path:
    return run.processed_dir / "permafrost_mask.nc"


def extent_csv(run: Run) -> Path:
    return run.processed_dir / "permafrost_extent.csv"


def method_path(run: Run) -> Path:
    return run.processed_dir / "permafrost_extent_method.txt"


def cached_method(run: Run) -> str | None:
    p = method_path(run)
    return p.read_text().strip() if p.exists() else None


def require_cached(run: Run) -> tuple[np.ndarray, np.ndarray]:
    """Load the run's permafrost_extent.csv or raise FileNotFoundError."""
    path = extent_csv(run)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path} -- run calc_permafrost_extent.main(run) first"
        )
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    years = np.array([int(r["year"]) for r in rows])
    extent = np.array([float(r["extent_Mkm2"]) for r in rows])
    return years, extent


# --------------------------------------------------------------------------
# Option 1: Static RECCAP2 Outline
# --------------------------------------------------------------------------

def compute_reccap2(grid, area_m2, landfrac) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Static permafrost domain using the RECCAP2 benchmark outline (> 0.5)."""
    say("  Building static permafrost boundary from RECCAP2...")
    reccap_frac = masks.build_permafrost_mask(grid, include_icecap=False, cache=False)
    binary_mask = (reccap_frac > 0.5).astype("f8")

    land_w = landfrac if landfrac is not None else xr.ones_like(area_m2)
    weight_mkm2 = (area_m2 * land_w * binary_mask).fillna(0.0).values / 1.0e12
    total_mkm2 = float(weight_mkm2.sum())

    years = np.arange(2024, 2301)
    extents = np.full_like(years, total_mkm2, dtype=float)

    say(f"  RECCAP2 permafrost cells: {int(binary_mask.sum())}")
    say(f"  Fixed permafrost area:    {total_mkm2:.2f} M km²")
    return binary_mask, years, extents


# --------------------------------------------------------------------------
# Option 2: Dynamic Soil Temperature at 9.8 m (Multi-chunk 2024-2300)
# --------------------------------------------------------------------------

def _extract_chunk_years(da_chunk: xr.DataArray, path: Path, n_years: int) -> np.ndarray | None:
    """Resolve annual coordinates from xarray time metadata or file stem."""
    if "time" in da_chunk.coords:
        t = da_chunk["time"]
        try:
            if hasattr(t, "dt") and hasattr(t.dt, "year"):
                y = np.unique(t.dt.year.values)
                if len(y) == n_years:
                    return y.astype(int)
        except Exception:
            pass
        try:
            first_val = t.values[0]
            if hasattr(first_val, "year"):
                y = np.array([val.year for val in t.values[::12]])
                if len(y) == n_years:
                    return y.astype(int)
        except Exception:
            pass

    match = re.search(r"(\d{4})[-_](\d{4})", path.name)
    if match:
        start_year = int(match.group(1))
        return np.arange(start_year, start_year + n_years, dtype=int)

    return None


def compute_soilt(
    grid: xr.DataArray,
    area_m2: xr.DataArray,
    landfrac: xr.DataArray | None,
    run: Run,
) -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """Dynamic extent across all time chunks: soil at 9.795 m < 0 °C all 12 months."""
    say(f"  Resolving soilT file via registry for {run.tag}...")
    file = wiemip_io.retrieve(
        variable="soilT",
        model=run.model,
        forcing=run.forcing,
        simulation=run.simulation,
    )

    file_path = Path(file.path)
    parent_dir = file_path.parent
    match = re.search(r"(\.|\_)\d{4}.*", file_path.name)
    prefix = file_path.name[: match.start()] if match else file_path.stem

    # Locate and sort all chunk files (e.g. 2024-2100, 2101-2200, 2201-2300)
    chunk_files = sorted(parent_dir.glob(f"{prefix}*.nc"))
    if not chunk_files:
        chunk_files = [file_path]

    say(f"  Found {len(chunk_files)} time-chunk file(s):")
    for f in chunk_files:
        say(f"    - {f.name}")

    with xr.open_dataset(chunk_files[0], decode_times=False) as ds_first:
        levels = ds_first["levgrnd"].values
        level_idx = int(np.argmin(np.abs(levels - DEEPEST_M)))
        selected_depth = float(levels[level_idx])
        say(f"  Selected level {level_idx + 1} at depth {selected_depth:.3f} m (target {DEEPEST_M} m)")

    csoil_da = grid.isel(time=0) if "time" in grid.dims else grid
    has_soil = (csoil_da.values > 0.0)
    land_w = landfrac.values if landfrac is not None else np.ones(grid.shape[-2:])
    nh_mask = (grid.lat.values[:, None] > 0.0) if grid.lat.ndim == 1 else (grid.lat.values > 0.0)
    is_nh_land = nh_mask & (land_w > 0.0) & has_soil
    cell_area_mkm2 = (wiemip_io.get_cell_area(grid).values * land_w) / 1.0e12

    all_is_pf = []
    all_years = []

    for chunk_p in chunk_files:
        say(f"  Processing chunk {chunk_p.name}...")
        with xr.open_dataset(chunk_p, decode_times=False) as ds_chunk:
            soil_t_9m = ds_chunk["soilT"].isel(levgrnd=level_idx)
            soil_t_9m = wiemip_io.standardise(soil_t_9m)

            n_months = soil_t_9m.sizes["time"]
            n_years = n_months // 12
            if n_years == 0:
                continue

            chunk_years = _extract_chunk_years(soil_t_9m, chunk_p, n_years)
            if chunk_years is None or len(chunk_years) != n_years:
                start_yr = all_years[-1][-1] + 1 if all_years else 2024
                chunk_years = np.arange(start_yr, start_yr + n_years, dtype=int)

            t_data = soil_t_9m.isel(time=slice(0, n_years * 12)).values.astype("f4")
            warmest_month = t_data.reshape(n_years, 12, *soil_t_9m.shape[-2:]).max(axis=1)
            is_pf_chunk = (warmest_month < FREEZING_K) & is_nh_land

            all_is_pf.append(is_pf_chunk)
            all_years.append(chunk_years)

    is_pf = np.concatenate(all_is_pf, axis=0)
    years = np.concatenate(all_years, axis=0)
    extents = (is_pf * cell_area_mkm2).sum(axis=(1, 2))

    idx_2024 = np.where(years == 2024)[0]
    mask_2024_arr = is_pf[idx_2024[0]] if len(idx_2024) > 0 else is_pf[0]
    mask_2024 = xr.DataArray(
        mask_2024_arr.astype("f8"),
        coords={"lat": grid.lat, "lon": grid.lon},
        dims=("lat", "lon"),
    )

    say(f"  Total series: {years[0]} to {years[-1]} ({len(years)} years)")
    say(f"  2024 permafrost cells: {int(mask_2024.sum())}")
    say(
        f"  Extent: {extents[0]:.2f} M km² ({years[0]}) -> "
        f"{extents[np.argmin(extents)]:.2f} M km² (min at {years[np.argmin(extents)]}) -> "
        f"{extents[-1]:.2f} M km² ({years[-1]})"
    )

    return mask_2024, years, extents


# --------------------------------------------------------------------------
# Main Controller
# --------------------------------------------------------------------------

def main(run: Run, method: str = "reccap2", recompute: bool = False):
    if method not in ("reccap2", "soilt"):
        raise ValueError(f"Method must be 'reccap2' or 'soilt', got: {method}")

    mask_nc, csv_out = mask_path(run), extent_csv(run)

    if not recompute and mask_nc.exists() and csv_out.exists():
        built_with = cached_method(run)
        if built_with in (None, method):
            say(f"Cached permafrost mask/extent exist for {run.tag} "
                f"(method: {built_with or 'unknown'}). Use recompute=True to rebuild.")
            return
        say(f"Cache for {run.tag} was built with '{built_with}', "
            f"but '{method}' was requested -- recomputing.")

    say(f"Calculating permafrost extent for {run.tag} (method: {method})...")
    grid = wiemip_io.load(
        variable="cSoilAbove1m",
        model=run.model,
        forcing=run.forcing,
        simulation=run.simulation,
    )
    area_m2 = wiemip_io.get_cell_area(grid)
    landfrac = wiemip_io.get_landfrac(grid)

    if method == "reccap2":
        mask_2024, years, extents = compute_reccap2(grid, area_m2, landfrac)
    else:
        mask_2024, years, extents = compute_soilt(grid, area_m2, landfrac, run)

    if not isinstance(mask_2024, xr.DataArray):
        mask_2024 = xr.DataArray(
            mask_2024,
            coords={"lat": grid.lat, "lon": grid.lon},
            dims=("lat", "lon"),
        )

    ds_mask = xr.Dataset({"permafrost": mask_2024.expand_dims(year=[2024])})
    ds_mask.to_netcdf(mask_nc)
    say(f"  Wrote {mask_nc}")

    with open(csv_out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "extent_Mkm2"])
        writer.writerows(zip(years, extents))
    say(f"  Wrote {csv_out}")

    method_path(run).write_text(method + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Calculate permafrost extent and mask.")
    p.add_argument("--method", choices=["reccap2", "soilt"], default="reccap2")
    p.add_argument("--model", required=True)
    p.add_argument("--forcing", required=True)
    p.add_argument("--simulation", required=True)
    p.add_argument("--recompute", action="store_true", help="Force recomputation")
    a = p.parse_args()
    main(Run(a.model, a.forcing, a.simulation), method=a.method, recompute=a.recompute)