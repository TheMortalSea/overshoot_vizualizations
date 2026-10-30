"""Area-weighted aggregation over a region mask.

The weight for each grid cell is  area x permafrost_fraction  and, when the
model reports per-land-area values, x landfrac as well. Whether that last term
belongs is decided empirically by `check_landfrac.py` -- never assumed.
"""

from __future__ import annotations

import xarray as xr

KG_PER_PG = 1.0e12


def cell_weights(area_m2, frac=None, landfrac=None) -> xr.DataArray:
    """Combined per-cell weight in m^2."""
    weights = area_m2
    if frac is not None:
        weights = weights * frac
    if landfrac is not None:
        weights = weights * landfrac
    return weights


def regional_total(da, area_m2, frac=None, landfrac=None) -> xr.DataArray:
    """Sum a kg C m-2 field over a region, returning PgC."""
    weights = cell_weights(area_m2, frac, landfrac)
    return (da * weights).sum(dim=("lat", "lon")) / KG_PER_PG


def regional_mean(da, area_m2, frac=None, landfrac=None) -> xr.DataArray:
    """Area-weighted mean of an intensive field (e.g. tas) over a region."""
    weights = cell_weights(area_m2, frac, landfrac)
    return da.weighted(weights.fillna(0.0)).mean(dim=("lat", "lon"))


def regional_area(area_m2, frac=None, landfrac=None) -> float:
    """Total weighted area in km^2 -- the sanity check on the weighting."""
    return float(cell_weights(area_m2, frac, landfrac).sum()) / 1.0e6
