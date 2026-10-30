"""I/O helpers for WIEMIP climate-overshoot model output.

Data is retrieved through the `wiemip_registry` package rather than by globbing
and parsing raw filenames. The registry knows, for each model, how to build the
path to a variable on the WIEMIP bucket, so nothing here depends on filename
conventions any more.

Identifying a run
-----------------
A run is addressed by four things:

    model       registered model name, e.g. "CLASSIC", "CLM" (see `wr.models`)
    forcing     PRIME GCM pattern: "ukesm", "ipsl" or "gfdl"
    simulation  overshoot simulation: "l", "m", "hl", "hl_cf", "hist", "ctrl", ...
    variable    WIEMIP variable name, e.g. "cVeg", "tas"

Historical and control runs ("hist", "ctrl") use observed CRUJRA forcing rather
than a PRIME pattern, but the registry still takes a `forcing` argument for them,
so pass whichever pattern you are pairing them with.

Counterfactual runs are no longer a separate flag: HL-CF is simulation "hl_cf".
`simulation_name()` converts the old (scenario, counterfactual) pair if needed.
"""

# THIS SCRIPT HAS BEEN MIGRATED TO THE WIEMIP REGISTRY METHOD

from __future__ import annotations

from pathlib import Path

import numpy as np
import wiemip_registry as wr
import xarray as xr
from wiemip_registry.core import WIEFile

from dataclasses import dataclass
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "figures"

EARTH_RADIUS_M = 6.371e6

# The protocol specifies a noleap calendar, so February is always 28 days.
NOLEAP_MONTH_DAYS = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

@dataclass(frozen=True)
class Run:
    """One model / forcing / simulation combination. Single source of truth."""
    model: str
    forcing: str
    simulation: str

    @property
    def tag(self) -> str:
        return f"{self.model}_{self.forcing}_{self.simulation}".lower()

    @property
    def scenario_label(self) -> str:
        return self.simulation.upper()

    @property
    def processed_dir(self) -> Path:
        d = PROCESSED_DIR / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def figure_dir(self) -> Path:
        d = FIGURE_DIR / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figure_path(self, stem: str) -> Path:
        return self.figure_dir / f"{stem}_{date.today():%Y-%m-%d}.png"

@dataclass(frozen=True)
class RunSet:
    """Several models under one forcing and simulation, for comparison figures."""
    models: tuple[str, ...]
    forcing: str
    simulation: str

    @property
    def runs(self) -> list[Run]:
        return [Run(m, self.forcing, self.simulation) for m in self.models]

    @property
    def tag(self) -> str:
        return f"compare_{self.forcing}_{self.simulation}".lower()

    @property
    def scenario_label(self) -> str:
        return self.simulation.upper()

    @property
    def figure_dir(self) -> Path:
        d = FIGURE_DIR / self.tag
        d.mkdir(parents=True, exist_ok=True)
        return d

    def figure_path(self, stem: str) -> Path:
        return self.figure_dir / f"{stem}_{date.today():%Y-%m-%d}.png"
# Retrieval

def simulation_name(scenario: str, counterfactual: bool = False) -> str:
    """Map an old-style (scenario, counterfactual) pair to a registry simulation.

    ("hl", True) -> "hl_cf";  ("m", False) -> "m".
    """
    scenario = scenario.lower()
    return f"{scenario}_cf" if counterfactual else scenario


def retrieve(
    variable: str,
    model: str,
    forcing: str,
    simulation: str,
    factorial: str | None = None,
) -> WIEFile:
    """Resolve one overshoot variable to a `WIEFile` and check that it exists.

    Raises FileNotFoundError (with the resolved path) if the registry builds a
    path that is not actually on the bucket.
    """
    file = wr.retrieve_overshoot_variable(
        model=model,
        forcing=forcing,
        simulation=simulation,
        variable=variable,
        factorial=factorial,
    )
    if not file.exists():
        raise FileNotFoundError(
            f"{model}/{forcing}/{simulation}/{variable}: not found at {file.path}"
        )
    return file



# Standardisation

def standardise(obj):
    """Put a WIEMIP Dataset or DataArray on the same conventions.

    lat ascending, lon in 0-360, and `time` as integer calendar years when the
    time axis is numeric. Some models store integer years; others store float
    mid-years (2024.5), which are floored.
    """
    renames = {}
    if "latitude" in obj.dims or "latitude" in obj.coords:
        renames["latitude"] = "lat"
    if "longitude" in obj.dims or "longitude" in obj.coords:
        renames["longitude"] = "lon"
    if renames:
        obj = obj.rename(renames)

    if "lat" in obj.coords and obj.lat.size > 1 and obj.lat[0] > obj.lat[-1]:
        obj = obj.sortby("lat")

    if "lon" in obj.coords:
        obj = obj.assign_coords(lon=(obj.lon % 360)).sortby("lon")

    if "time" in obj.coords and np.issubdtype(obj.time.dtype, np.floating):
        obj = obj.assign_coords(time=np.floor(obj.time.values).astype(int))

    return obj

def load(
    variable: str,
    model: str,
    forcing: str,
    simulation: str,
    factorial: str | None = None,
) -> xr.DataArray:
    """Read a variable (native units) from the registry with standardised coords."""
    file = retrieve(variable, model, forcing, simulation, factorial)
    da = standardise(file.read())
    da.attrs.update(
        {
            "wiemip_model": model,
            "wiemip_forcing": forcing,
            "wiemip_simulation": simulation,
            "wiemip_variable": variable,
            "wiemip_path": str(file.path),
        }
    )

    # Attach auxiliary 2D variables (landfrac, area) as coordinates if present in the file
    if hasattr(file, "path") and Path(file.path).exists():
        try:
            with xr.open_dataset(file.path, decode_timedelta=False) as ds:
                for name in ("landfrac", "sftlf"):
                    if name in ds and "landfrac" not in da.coords:
                        lf = standardise(ds[name].load())
                        if float(lf.max()) > 1.0:
                            lf = lf / 100.0
                        da = da.assign_coords(landfrac=(("lat", "lon"), lf.values))
                        break
                if "area" in ds and "area" not in da.coords:
                    ar = standardise(ds["area"].load())
                    da = da.assign_coords(area=(("lat", "lon"), ar.values))
        except Exception:
            pass

    return da


def global_total(
    variable: str,
    model: str,
    forcing: str,
    simulation: str,
    factorial: str | None = None,
):
    """Global total via the registry's own weighting (Pg C for carbon stocks).

    This uses `WIEFile.latitudinal_sum()`, which applies the weighting the
    modelling group intended, so prefer it over hand-rolled area weighting.
    """
    return retrieve(variable, model, forcing, simulation, factorial).latitudinal_sum()


# --------------------------------------------------------------------------
# Grid helpers
# --------------------------------------------------------------------------

def get_cell_area(obj) -> xr.DataArray:
    """Grid-cell area in m^2.

    If the object carries its own `area` field (km^2, as CLM ships) it is used.
    Otherwise falls back to the exact spherical formula from the lat/lon grid.
    For weighted aggregation of registry data, prefer `WIEFile.latitudinal_sum()`.
    """
    has_area = (
        "area" in obj.coords
        or "area" in getattr(obj, "data_vars", {})
        or "area" in getattr(obj, "variables", {})
    )
    if has_area:
        return (obj["area"] * 1.0e6).rename("cell_area_m2")  # km^2 -> m^2

    lat = obj["lat"]
    lon = obj["lon"]
    dlat = float(np.abs(np.diff(lat.values)).mean())
    dlon = float(np.abs(np.diff(lon.values)).mean())

    # Exact spherical band area: R^2 * dlon * (sin(lat_north) - sin(lat_south)).
    north = np.deg2rad(lat.values + dlat / 2.0)
    south = np.deg2rad(lat.values - dlat / 2.0)
    band = EARTH_RADIUS_M**2 * np.deg2rad(dlon) * (np.sin(north) - np.sin(south))

    return xr.DataArray(
        np.repeat(band[:, None], lon.size, axis=1),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="cell_area_m2",
    )


def get_landfrac(obj):
    """Land fraction per cell, or None when the object does not provide one."""
    for container in (getattr(obj, "variables", {}), getattr(obj, "coords", {})):
        if "landfrac" in container:
            return obj["landfrac"]
        if "sftlf" in container:
            return obj["sftlf"]

    # Fallback: check underlying NetCDF dataset if obj carries its source path
    path = getattr(obj, "attrs", {}).get("wiemip_path")
    if path and Path(path).exists():
        try:
            with xr.open_dataset(path, decode_timedelta=False) as ds:
                for name in ("landfrac", "sftlf"):
                    if name in ds:
                        lf = standardise(ds[name].load())
                        if float(lf.max()) > 1.0:
                            lf = lf / 100.0
                        return lf
        except Exception:
            pass

    return None


# --------------------------------------------------------------------------
# Time handling
# --------------------------------------------------------------------------

def _time_years(da) -> np.ndarray:
    """Integer calendar year for each timestep, whatever the time encoding."""
    time = da["time"]
    if np.issubdtype(time.dtype, np.datetime64) or time.dtype == object:
        return time.dt.year.values.astype(int)
    return np.floor(np.asarray(time.values, dtype="f8")).astype(int)


def _time_months(da) -> np.ndarray:
    """Month number (1-12) for each timestep."""
    time = da["time"]
    if np.issubdtype(time.dtype, np.datetime64) or time.dtype == object:
        return time.dt.month.values.astype(int)
    # Numeric time with several steps per year: assume they run in month order.
    return (np.arange(time.size) % 12 + 1).astype(int)


def to_annual(da: xr.DataArray) -> xr.DataArray:
    """Reduce to one value per calendar year, weighting by noleap month length.

    Annual input is returned unchanged apart from an integer-year time axis.
    """
    years = _time_years(da)
    if len(np.unique(years)) == len(years):
        return da.assign_coords(time=years)

    months = _time_months(da)
    weights = xr.DataArray(
        NOLEAP_MONTH_DAYS[months - 1].astype("f8"),
        dims="time",
        coords={"time": da["time"]},
    )
    year_coord = xr.DataArray(
        years, dims="time", coords={"time": da["time"]}, name="year"
    )
    numerator = (da * weights).groupby(year_coord).sum("time")
    denominator = weights.groupby(year_coord).sum("time")
    annual = numerator / denominator
    return annual.rename({"year": "time"})

def reduce_in_blocks(da: xr.DataArray, func, block_size: int = 120) -> xr.Dataset:
    """Apply a reduction function block-by-block along the time dimension to conserve memory."""
    parts = []
    n_time = da.sizes.get("time", 0)
    for i in range(0, n_time, block_size):
        block = da.isel(time=slice(i, i + block_size))
        res = func(block)
        if hasattr(res, "compute"):
            res = res.compute()
        parts.append(res)
    return xr.concat(parts, dim="time")


# --------------------------------------------------------------------------
# Series loading
# --------------------------------------------------------------------------

def load_series(
    variable: str,
    model: str,
    forcing: str,
    simulation: str,
    factorial: str | None = None,
    reducer=None,
) -> xr.DataArray:
    """Return an annual series for one variable of one run.

    The registry resolves the run to its data, so there is no chunk stitching
    here. `reducer(da) -> DataArray` is applied before annual averaging; pass
    one for gridded monthly fields such as `tas` to collapse space first.
    (Previously the reducer received `(da, ds)`; the dataset is no longer
    available, and `get_cell_area(da)` works on the DataArray directly.)

    The result is still checked for duplicated and missing years, since a
    silent gap or overlap would corrupt any downstream trend.
    """
    da = load(variable, model, forcing, simulation, factorial)
    if reducer is not None:
        da = reducer(da)
    series = to_annual(da).sortby("time")
    if hasattr(series.data, "compute"):
        series = series.compute()

    years = series["time"].values
    duplicates = len(years) - len(np.unique(years))
    if duplicates:
        raise ValueError(f"{variable}: {duplicates} duplicated years")
    gaps = np.setdiff1d(np.arange(years.min(), years.max() + 1), years)
    if gaps.size:
        raise ValueError(f"{variable}: missing years {gaps[:10]}")

    return series

# HELPER FUNCTIONS TO POINT ERRORS

def first_step(da: xr.DataArray) -> xr.DataArray:
    """Return the first time slice (or the array itself if no time axis)."""
    return da.isel(time=0) if "time" in da.dims else da

def land_fraction(obj, *args, **kwargs):
    """Alias for get_landfrac."""
    return get_landfrac(obj)


