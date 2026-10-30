"""Permafrost masks.

permafrost_mask(lat, lon, run=run) is the region every total in this project is
summed over: cells where soil at 9.795 m stays below 0 °C all year in 2024,
built per run by calc_permafrost_extent.py into
data/processed/<model>_<forcing>_<simulation>/permafrost_mask.nc. The cell set
is fixed at 2024, so totals over time are not affected by the region shrinking.

build_permafrost_mask() is the RECCAP2 / ISIMIP3 region file. It works on any
regular lat/lon grid: 1 degree grids use an exact 2x2 mean of the 0.5 degree
mask, other grids are binned by cell edges. It is used for the 2024 definition
comparison and for the RECCAP-based regional temperature series.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

import wiemip_io
from wiemip_io import PROCESSED_DIR

RAW_DIR = wiemip_io.PROJECT_ROOT / "data" / "raw"
RECCAP2_FILE = RAW_DIR / "RECCAP2_permafrost_regions_isimip3.nc"
LEGACY_MASK_NC = PROCESSED_DIR / "permafrost_mask.nc"   # shared, pre-Run file
MASK_YEAR = 2024


def _extract_grid(target_lat, target_lon):
    """Allow passing a DataArray/Dataset directly or separate lat/lon arrays."""
    if target_lon is None and hasattr(target_lat, "coords"):
        return target_lat["lat"].values, target_lat["lon"].values
    lat = target_lat.values if hasattr(target_lat, "values") else np.asarray(target_lat)
    lon = target_lon.values if hasattr(target_lon, "values") else np.asarray(target_lon)
    return lat, lon


# --------------------------------------------------------------------------
# Dynamic 2024 permafrost mask (per run)
# --------------------------------------------------------------------------

def permafrost_mask(target_lat, target_lon=None, year=MASK_YEAR, run=None) -> xr.DataArray:
    """1 where the cell is permafrost in `year`, else 0, on the model grid.

    Accepts separate lat/lon coordinates or a single DataArray/Dataset.
    Pass `run` (wiemip_io.Run) to read that run's own mask. Without it the old
    shared data/processed/permafrost_mask.nc is used, which is only valid if it
    was built for the same run and grid.
    """
    target_lat, target_lon = _extract_grid(target_lat, target_lon)

    if run is not None:
        path = run.processed_dir / "permafrost_mask.nc"
        label = run.tag
        hint = f"calc_permafrost_extent.main(run, method='soilt')"
    else:
        path = LEGACY_MASK_NC
        label = "shared (no run given)"
        hint = "calc_permafrost_extent.main(run, method='soilt')"

    if not path.exists():
        raise FileNotFoundError(f"missing {path} -- build it with:\n    {hint}")

    with xr.open_dataset(path) as ds:
        mask = ds["permafrost"]
        if "year" in mask.dims:
            mask = mask.sel(year=year)
        mask = mask.astype("f8").load()

    same_shape = (mask.lat.size == len(target_lat)) and (mask.lon.size == len(target_lon))
    if not (same_shape
            and np.allclose(mask.lat.values, target_lat)
            and np.allclose(mask.lon.values, target_lon)):
        raise ValueError(
            f"permafrost mask for {label} is {mask.lat.size}x{mask.lon.size} and does "
            f"not match the target grid {len(target_lat)}x{len(target_lon)}")

    # Re-assign exact target coordinates to prevent floating point alignment mismatches
    mask = mask.assign_coords(lat=target_lat, lon=target_lon)
    if "year" in mask.coords:
        mask = mask.drop_vars("year")
    return mask.rename("permafrost")


# --------------------------------------------------------------------------
# RECCAP2 region mask (any grid)
# --------------------------------------------------------------------------

REGION_NAMES = {
    101: "North_America",
    103: "Europe",
    106: "North_Asia",
    108: "East_Asia",
    113: "Ice_Cap",
}
PERMAFROST_REGIONS = [101, 103, 106, 108]  # Ice_Cap excluded by default
FILL_THRESHOLD = 1.0e30  # the file stores 9.969e36 as missing


def _edges(c):
    """Cell edges for a 1D array of cell centres."""
    c = np.asarray(c, dtype="f8")
    m = (c[1:] + c[:-1]) / 2
    return np.concatenate([[c[0] - (m[0] - c[0])], m, [c[-1] + (c[-1] - m[-1])]])


def _bin_to_grid(binary: xr.DataArray, target_lat, target_lon) -> np.ndarray:
    """Mean of fine 0/1 cells falling inside each target cell (any regular grid).

    `binary` must have ascending lat and lon in 0-360. Longitude is treated as
    cyclic; fine cells outside the target latitude range are ignored.
    """
    target_lat = np.asarray(target_lat, dtype="f8")
    target_lon = np.asarray(target_lon, dtype="f8")
    binary = binary.transpose("lat", "lon")

    la_e, lo_e = _edges(target_lat), _edges(target_lon)
    ilat = np.searchsorted(la_e, binary.lat.values) - 1
    ilon = (np.searchsorted(lo_e, binary.lon.values) - 1) % len(target_lon)
    ok_lat = (ilat >= 0) & (ilat < len(target_lat))

    II, JJ = np.meshgrid(np.where(ok_lat, ilat, 0), ilon, indexing="ij")
    valid = np.broadcast_to(ok_lat[:, None], II.shape)

    vals = np.nan_to_num(binary.values, nan=0.0)
    sums = np.zeros((len(target_lat), len(target_lon)))
    cnt = np.zeros_like(sums)
    np.add.at(sums, (II[valid], JJ[valid]), vals[valid])
    np.add.at(cnt, (II[valid], JJ[valid]), 1.0)
    return np.divide(sums, cnt, out=np.zeros_like(sums), where=cnt > 0)


def build_permafrost_mask(
    target_lat,
    target_lon=None,
    include_icecap: bool = False,
    cache: bool = True,
    regions=None,
) -> xr.DataArray:
    """Fractional permafrost-region weight (0-1) on the target grid.

    Accepts separate lat/lon coordinates or a single DataArray/Dataset.
    Ice_Cap (region 113) is excluded by default: it holds negligible soil carbon
    and including it only dilutes the permafrost signal.
    """
    target_lat, target_lon = _extract_grid(target_lat, target_lon)
    nlat, nlon = len(target_lat), len(target_lon)

    suffix = "_icecap" if include_icecap else ""
    cache_path = PROCESSED_DIR / f"pf_mask_{nlat}x{nlon}{suffix}.nc"
    if cache and regions is None and cache_path.exists():
        with xr.open_dataarray(cache_path) as cached:
            if (cached.sizes.get("lat") == nlat and cached.sizes.get("lon") == nlon
                    and np.allclose(cached.lat.values, target_lat, atol=1e-6)
                    and np.allclose(cached.lon.values, target_lon, atol=1e-6)):
                return cached.load()

    if not RECCAP2_FILE.exists():
        raise FileNotFoundError(f"RECCAP2 reference file not found at: {RECCAP2_FILE}")

    with xr.open_dataset(RECCAP2_FILE, decode_timedelta=False) as ds:
        region = ds["permafrost_region_mask"].rename(
            {"latitude": "lat", "longitude": "lon"}).load()
    region = region.where(region < FILL_THRESHOLD)

    # `regions` overrides the default selection, so a single RECCAP2 region can
    # be pulled out on its own -- e.g. regions=[113] for the ice caps.
    wanted = (
        list(regions)
        if regions is not None
        else PERMAFROST_REGIONS + ([113] if include_icecap else [])
    )
    binary = xr.where(region.isin(wanted), 1.0, 0.0)

    # Match the target conventions before reducing: lat ascending, lon 0-360.
    binary = binary.sortby("lat")
    binary = binary.assign_coords(lon=(binary.lon % 360)).sortby("lon")

    native_dlat = float(np.abs(np.diff(binary.lat.values)).mean())
    if not np.isclose(native_dlat, 0.5):
        raise ValueError(f"expected a 0.5 deg RECCAP2 mask, found {native_dlat:g} deg")

    dlat = float(np.abs(np.diff(target_lat)).mean())
    dlon = float(np.abs(np.diff(target_lon)).mean())

    if np.isclose(dlat, 1.0) and np.isclose(dlon, 1.0):
        frac = binary.coarsen(lat=2, lon=2, boundary="exact").mean()
        if frac.sizes["lat"] != nlat or frac.sizes["lon"] != nlon:
            raise ValueError(
                f"coarsened mask is {frac.sizes['lat']}x{frac.sizes['lon']}, "
                f"target is {nlat}x{nlon}")
        if not np.allclose(frac.lat.values, target_lat, atol=1e-6):
            raise ValueError("coarsened mask latitudes do not line up with the target grid")
        if not np.allclose(frac.lon.values, target_lon, atol=1e-6):
            raise ValueError("coarsened mask longitudes do not line up with the target grid")
        note = "0.5 deg RECCAP2 mask reduced to 1 deg by exact 2x2 mean"
    else:
        frac = xr.DataArray(
            _bin_to_grid(binary, target_lat, target_lon),
            coords={"lat": target_lat, "lon": target_lon},
            dims=("lat", "lon"),
        )
        note = (f"0.5 deg RECCAP2 mask binned onto a {dlat:g}x{dlon:g} deg grid "
                "(fraction of fine cells inside each target cell)")

    frac = frac.assign_coords(lat=target_lat, lon=target_lon)
    frac.name = "permafrost_fraction"
    frac.attrs = {
        "long_name": "fractional permafrost-region coverage",
        "source": RECCAP2_FILE.name,
        "regions": ", ".join(f"{r}:{REGION_NAMES[r]}" for r in wanted),
        "note": note,
    }

    if cache and regions is None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        frac.to_netcdf(cache_path)

    return frac


if __name__ == "__main__":
    # Sanity check on CLM's 1 degree grid, then on a non-1-degree grid.
    da = wiemip_io.load(variable="cSoilAbove1m", model="CLM", forcing="gfdl", simulation="hl")

    for icecap, expected in ((False, 4709), (True, 5488)):
        frac = build_permafrost_mask(da, include_icecap=icecap, cache=not icecap)
        n = int((frac > 0.5).sum())
        flag = "OK" if n == expected else f"MISMATCH (expected {expected})"
        print(f"include_icecap={icecap!s:5s} cells>0.5: {n:5d}  {flag}")

    area_km2 = wiemip_io.get_cell_area(da) / 1e6
    pf_area = float((area_km2 * build_permafrost_mask(da)).sum())
    print(f"permafrost-region area: {pf_area:.3g} km2  (expect 1.6-1.8e7)")

    lat = np.arange(-89.375, 90, 1.25)      # JULES-like 1.25 x 1.875 grid
    lon = np.arange(0.0, 360, 1.875)
    coarse = build_permafrost_mask(lat, lon, cache=False)
    area = wiemip_io.get_cell_area(xr.Dataset(coords={"lat": lat, "lon": lon}))
    print(f"1.25x1.875 grid: cells>0.5: {int((coarse > 0.5).sum())}, "
          f"area {float((area * coarse).sum()) / 1e6:.3g} km2 (should be close to the 1 deg value)")