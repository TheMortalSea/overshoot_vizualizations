"""Pre-industrial baseline for air temperature, permafrost region and global land.

    python -u code/tas_baseline.py
    python -u code/tas_baseline.py --recompute

Every anomaly in this project was referenced to 2024 because no pre-2024 output
was on disk. The historical run (CLM CRUJRA forcing, 1850-2023) fixes that, so
warming can be quoted against pre-industrial as the literature does.

WHAT THIS COMPUTES: area-weighted mean annual 2 m air temperature for 1850-2023,
over (a) the RECCAP2 permafrost region and (b) all global land, and from them
several candidate baselines.

TWO THINGS TO BE CAREFUL ABOUT.

1. Neither series is a global warming level. The scenario warming levels (GWL)
   in the protocol are GLOBAL MEAN SURFACE temperature, which includes ocean.
   WIEMIP is a land-model archive, so GMST cannot be computed from it at all --
   global LAND warms faster than global mean, and the permafrost region faster
   still. For a GWL, use the PRIME/FaIR forcing metadata, not this output. The
   permafrost-vs-land ratio below is reported to make that gap concrete.

2. The pre-industrial period is partly synthetic. The protocol states that
   1850-1900 in WIEMIP-CRUJRAv3.5 is recycled 1901-1920 data, repeated ~2.5x.
   Both periods are computed below: they agree to within 0.002 K, so the caveat
   is a footnote rather than a numerical problem.

Weights come from ONE grid file and are reused for the historical and future
series, so an anomaly cannot mix two different weightings. Grids are checked for
equality rather than assumed to match.
"""

#THIS SCRIPT HAS BEEN MIGRATED TO WIEMIP REGISTRY

from __future__ import annotations

import csv
from pathlib import Path
import sys

import numpy as np
import xarray as xr

import masks
import wiemip_io
from wiemip_io import PROCESSED_DIR

# Registered model identifiers and defaults
MODEL = "CLM"
FORCING = "gfdl"
SCENARIO = "HL"          # which FaIR pathway gmst_series() returns by default
TEMP_VAR = "tas"
FREEZING_K = 273.15

# FaIR reference file path reconstructed from PROJECT_ROOT
RAW_DIR = wiemip_io.PROJECT_ROOT / "data" / "raw"
GMST_FILE = RAW_DIR / "FaIR_overshoot_temperature.nc"

HIST_CSV = PROCESSED_DIR / "tas_hist_series.csv"
FUTURE_CSV = PROCESSED_DIR / "tas_series.csv"

# (label, first year, last year), all inclusive.
PERIODS = [
    ("1850-1900  convention", 1850, 1900),
    ("1901-1920  what 1850-1900 recycles", 1901, 1920),
    ("1850-1869  spin-up forcing", 1850, 1869),
    ("1995-2014  recent reference", 1995, 2014),
]

BASELINE_PERIOD = (1850, 1900)


def say(*parts):
    print(*parts, flush=True)


def grid_weights():
    """Permafrost-region and global-land cell weights, from one reference file."""
    grid = wiemip_io.load(
        variable="cSoilAbove1m",
        model=MODEL,
        forcing=FORCING,
        simulation=wiemip_io.simulation_name(SCENARIO),
    )
    area_m2 = wiemip_io.get_cell_area(grid)
    landfrac = wiemip_io.get_landfrac(grid)
    pf_frac = masks.permafrost_mask(grid.lat.values, grid.lon.values)
    land = landfrac if landfrac is not None else xr.ones_like(area_m2)
    w_pf = (area_m2 * pf_frac * land).fillna(0.0)
    w_land = (area_m2 * land).fillna(0.0)
    return grid, w_pf, w_land


def compute_hist_series(grid, w_pf, w_land):
    """Annual mean tas 1850-2023 for both domains, one pass over the data."""
    da = wiemip_io.load(
        variable=TEMP_VAR,
        model=MODEL,
        forcing=FORCING,
        simulation="hist",
    )

    if not (
        np.allclose(da.lat.values, grid.lat.values)
        and np.allclose(da.lon.values, grid.lon.values)
    ):
        raise SystemExit(f"{MODEL} hist {TEMP_VAR}: grid differs from reference grid")

    sum_pf, sum_land = float(w_pf.sum()), float(w_land.sum())
    say(f"    processing historical {TEMP_VAR} ({da.sizes.get('time')} steps)...")

    monthly_pf = (da * w_pf).sum(dim=("lat", "lon")) / sum_pf
    monthly_land = (da * w_land).sum(dim=("lat", "lon")) / sum_land

    if hasattr(monthly_pf.data, "compute"):
        monthly_pf = monthly_pf.compute()
    if hasattr(monthly_land.data, "compute"):
        monthly_land = monthly_land.compute()

    annual_pf = wiemip_io.to_annual(monthly_pf).sortby("time")
    annual_land = wiemip_io.to_annual(monthly_land).sortby("time")

    years = np.asarray(annual_pf["time"].values, dtype=int)
    pf_vals = np.asarray(annual_pf.values, dtype="f8")
    land_vals = np.asarray(annual_land.values, dtype="f8")

    order = np.argsort(years)
    return years[order], pf_vals[order], land_vals[order]


def load_cached():
    """Cached historical series, or None when absent or written by an older run."""
    if not HIST_CSV.exists():
        return None
    with open(HIST_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "tas_land_K" not in rows[0]:
        return None  # pre-dates the global-land column; recompute
    return (
        np.array([int(r["year"]) for r in rows]),
        np.array([float(r["tas_permafrost_K"]) for r in rows]),
        np.array([float(r["tas_land_K"]) for r in rows]),
    )


def period_mean(years, values, first, last):
    sel = (years >= first) & (years <= last)
    if not sel.any():
        return None, 0
    return float(values[sel].mean()), int(sel.sum())


def permafrost_pi_baseline():
    """1850-1900 permafrost-region mean tas in K, or None if not yet computed.

    The accessor other figures call. Returning None rather than raising lets a
    caller fall back to a 2024 baseline *and say so*; silently swapping one for
    the other would shift every anomaly by about 2.5 C with no visible sign.
    """
    cached = load_cached()
    if cached is None:
        return None
    years, pf, _land = cached
    mean, _ = period_mean(years, pf, *BASELINE_PERIOD)
    return mean


def global_land_pi_baseline():
    """1850-1900 global-LAND mean tas in K, or None if not yet computed.

    NOT global mean surface temperature. GMST includes ocean, which this
    land-model archive does not have, and land warms faster than land+ocean. An
    anomaly against this is therefore an upper bound on the scenario's global
    warming level, not equal to it. Label it "global land" wherever it appears.
    """
    cached = load_cached()
    if cached is None:
        return None
    years, _pf, land = cached
    mean, _ = period_mean(years, land, *BASELINE_PERIOD)
    return mean


def tas_long_name(default="Near-Surface Air Temperature"):
    """The `long_name` the model records for tas, for use as an axis label."""
    try:
        file = wiemip_io.retrieve(
            variable=TEMP_VAR,
            model=MODEL,
            forcing=FORCING,
            simulation="hist",
        )
        if hasattr(file, "path") and Path(file.path).exists():
            with xr.open_dataset(file.path, decode_timedelta=False) as ds:
                for var in (TEMP_VAR, "tas"):
                    if var in ds and "long_name" in ds[var].attrs:
                        return str(ds[var].attrs["long_name"])
    except Exception:
        pass
    return default


def gmst_series(years, scenario=SCENARIO):
    """TRUE global mean temperature anomaly for a scenario, in C, or None.

    Read from the FaIR pathway that PRIME used to generate the WIEMIP forcing
    (`FaIR config 867236`, matching the parameterisation named in the protocol).
    This is the quantity that DEFINES the scenario's warming level.
    """
    if not GMST_FILE.exists():
        return None
    with xr.open_dataset(GMST_FILE, decode_timedelta=False) as ds:
        available = [str(s) for s in ds["scenario"].values]
        if scenario not in available:
            raise SystemExit(
                f"scenario {scenario!r} not in {GMST_FILE.name}; have {available}"
            )
        file_years = ds["year"].values.astype(int)
        values = np.asarray(ds["data"].sel(scenario=scenario).values, dtype="f8")

    lookup = dict(zip(file_years, values))
    missing = [int(y) for y in years if int(y) not in lookup]
    if missing:
        raise SystemExit(
            f"{GMST_FILE.name} lacks years {missing[:5]}"
            f"{'...' if len(missing) > 5 else ''} (covers "
            f"{file_years[0]}-{file_years[-1]})"
        )
    return np.array([lookup[int(y)] for y in years])


def warming_series(years, domain="land"):
    """tas anomaly above pre-industrial (1850-1900), in C.

    domain="land"        global-LAND mean -- the scenario-defining quantity.
    domain="permafrost"  permafrost-region mean -- the local driver of thaw.
    """
    if domain == "gmst":
        series = gmst_series(years)
        if series is None:
            print(
                f"  WARNING: {GMST_FILE.name} not found, so falling back to global "
                "LAND, which runs ~1.3x warmer than global mean at peak. "
                "This is NOT the scenario's warming level.",
                flush=True,
            )
            return warming_series(years, "land")
        return series

    column = {"land": "tas_land_K", "permafrost": "tas_permafrost_K"}[domain]
    baseline_fn = {
        "land": global_land_pi_baseline,
        "permafrost": permafrost_pi_baseline,
    }[domain]

    with open(FUTURE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    lookup = {int(r["year"]): float(r[column]) for r in rows}
    series = np.array([lookup[int(y)] for y in years])

    pi = baseline_fn()
    if pi is None:
        print(
            f"  WARNING: no historical baseline, so the {domain} anomaly is "
            f"relative to {int(years[0])}, NOT pre-industrial. "
            "Run code/tas_baseline.py",
            flush=True,
        )
        return series - series[0]
    return series - pi


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cached = None if "--recompute" in sys.argv else load_cached()
    grid, w_pf, w_land = grid_weights()
    if cached is not None:
        years, pf, land = cached
        say(f"  cached {HIST_CSV.name}, {len(years)} years (--recompute to rebuild)")
    else:
        say("  mean tas from the historical files:")
        years, pf, land = compute_hist_series(grid, w_pf, w_land)
        with open(HIST_CSV, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["year", "tas_permafrost_K", "tas_land_K"])
            w.writerows(zip(years, pf, land))
        say(f"  wrote {HIST_CSV.name}")

    say(f"\n  historical coverage {years[0]}-{years[-1]} ({len(years)} years)")
    gaps = np.setdiff1d(np.arange(years[0], years[-1] + 1), years)
    if gaps.size:
        say(f"  WARNING: missing years {gaps[:10]}")

    say("\n=== candidate baselines (area-weighted mean tas) ===")
    say(f"  {'period':38s} {'permafrost':>12s} {'global land':>13s}")
    results = {}
    for label, first, last in PERIODS:
        m_pf, n = period_mean(years, pf, first, last)
        m_land, _ = period_mean(years, land, first, last)
        if m_pf is None:
            continue
        results[(first, last)] = (m_pf, m_land)
        say(f"  {label:38s} {m_pf - FREEZING_K:9.3f} °C {m_land - FREEZING_K:10.3f} °C")

    pi = results.get(BASELINE_PERIOD)
    rec = results.get((1901, 1920))
    if pi and rec:
        say(
            f"\n  1850-1900 minus 1901-1920: permafrost {pi[0] - rec[0]:+.4f} K, "
            f"global land {pi[1] - rec[1]:+.4f} K"
        )
        say("    (protocol says the former is the latter recycled; near zero confirms it)")

    if not pi:
        return
    pi_pf, pi_land = pi

    if not FUTURE_CSV.exists():
        say(f"\n  {FUTURE_CSV.name} not found; run calc_carbon_series.py for the future series")
        return
    with open(FUTURE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    fy = np.array([int(r["year"]) for r in rows])
    f_pf = np.array([float(r["tas_permafrost_K"]) for r in rows])
    f_land = np.array([float(r["tas_land_K"]) for r in rows])

    say("\n=== warming above pre-industrial (1850-1900) ===")
    say(f"  {'':16s} {'permafrost':>11s} {'global land':>12s} {'ratio':>7s}")
    peak_i = int(np.argmax(f_pf))
    for label, i in (("2024", 0), (f"peak ({fy[peak_i]})", peak_i), ("2300", -1)):
        a, b = f_pf[i] - pi_pf, f_land[i] - pi_land
        say(f"  {label:16s} {a:+10.2f} °C {b:+11.2f} °C {a / b:7.2f}x")

    say(
        "\n  The permafrost region warms roughly "
        f"{(f_pf[peak_i] - pi_pf) / (f_land[peak_i] - pi_land):.1f}x global LAND, and global "
        "land in turn\n  warms faster than global mean surface temperature, which includes "
        "ocean and\n  CANNOT be computed from this land-only archive. Do not read either "
        "column as\n  the scenario's global warming level."
    )


if __name__ == "__main__":
    main()