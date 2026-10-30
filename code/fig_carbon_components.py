"""Soil carbon above 1 m, below 1 m and total, with temperature below.

Notebook:
    fig_carbon_components.main(run)

Change since the first year rather than absolute stock: the three stocks sit at
very different levels, so on an absolute axis the changes would be invisible.
Absolute first-year stocks are in the legend.

Reads carbon_series.csv from the run's processed folder. The two model air
temperature diagnostics are optional: if tas_series.csv (columns year,
tas_land_K, tas_permafrost_K) exists in the run folder they are drawn,
otherwise only GMST is shown.
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

import calc_carbon_series
import plotting
import tas_baseline
from plotting import INK_MUTED, INK_PRIMARY, INK_SECONDARY
from wiemip_io import Run

# Colours come from plotting.pool_color so they match every other figure:
# a light -> dark brown ramp, all three being soil.
COMPONENTS = [
    ("cSoilAbove1m", "Above 1 m", plotting.pool_color("cSoilAbove1m")),
    ("cSoilBelow1m", "Below 1 m", plotting.pool_color("cSoilBelow1m")),
    ("cSoilTotal", "Total", plotting.pool_color("cSoilTotal")),
]


def say(*parts):
    print(*parts, flush=True)


def read_model_tas(run: Run, years):
    """(tas_land_K, tas_permafrost_K) aligned to `years`, or None if not built."""
    path = run.processed_dir / "tas_series.csv"
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    by_year = {int(r["year"]): r for r in rows}
    if not all(int(y) in by_year for y in years):
        say(f"  NOTE: {path.name} does not cover all years; model tas not plotted")
        return None
    land = np.array([float(by_year[int(y)]["tas_land_K"]) for y in years])
    pf = np.array([float(by_year[int(y)]["tas_permafrost_K"]) for y in years])
    return land, pf


def main(run: Run):
    plotting.apply_style()

    cached = calc_carbon_series.require_cached(run)
    missing = [v for v, _l, _c in COMPONENTS if v not in cached]
    if missing:
        raise KeyError(
            f"carbon_series.csv for {run.tag} lacks {missing}; "
            f"available: {sorted(cached)}")

    years = cached["cSoilTotal"][0]
    for var, _l, _c in COMPONENTS:
        if not np.array_equal(cached[var][0], years):
            raise ValueError(f"{var} covers different years")

    # GMST on the 1850-1900 baseline, the same reference the FaIR pathway uses.
    gmst = tas_baseline.gmst_series(years, run.scenario_label)
    if gmst is None:
        raise FileNotFoundError(
            f"no GMST pathway for scenario {run.scenario_label!r} -- check "
            "data/raw/FaIR_overshoot_temperature.nc")

    # Optional model diagnostics, referenced to the same 1850-1900 baseline.
    land_anom = pf_anom = None
    model_tas = read_model_tas(run, years)
    if model_tas is not None:
        pi_land = tas_baseline.global_land_pi_baseline()
        pi_pf = tas_baseline.permafrost_pi_baseline()
        if pi_land is None or pi_pf is None:
            say("  NOTE: PI tas baselines unavailable; model tas not plotted")
        else:
            land_anom = model_tas[0] - pi_land
            pf_anom = model_tas[1] - pi_pf
    else:
        say(f"  NOTE: no tas_series.csv for {run.tag}; temperature panel shows GMST only")

    # Peak year follows GMST, the quantity that defines the scenario.
    peak_year = int(years[int(np.argmax(gmst))])

    fig, (ax_c, ax_t) = plt.subplots(
        2, 1, figsize=(9.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.12})

    ax_c.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    ax_c.axvline(peak_year, color=INK_MUTED, linewidth=0.8)

    for var, label, colour in COMPONENTS:
        stock = cached[var][1]
        change = stock - stock[0]
        ax_c.plot(years, change, color=colour,
                  label=f"{label}  ({stock[0]:.0f} PgC in {years[0]})")
        ax_c.annotate(f"{change[-1]:+.1f}", xy=(years[-1], change[-1]),
                      xytext=(6, 0), textcoords="offset points",
                      ha="left", va="center", color=colour, fontsize=8)
        say(f"  {label:10s} {stock[0]:8.1f} -> {stock[-1]:8.1f} PgC  "
            f"({change[-1]:+6.1f})")

    ax_c.set_ylabel("Change in soil carbon since 2024 (PgC)")
    ax_c.set_title("Permafrost soil carbon by depth", loc="left")
    ax_c.legend(loc="lower left", ncol=1)
    ax_c.set_xlim(years[0], years[-1] + 8)
    plotting.despine(ax_c)
    ax_c.annotate(f"peak warming {peak_year}", xy=(peak_year, ax_c.get_ylim()[1]),
                  xytext=(6, -6), textcoords="offset points",
                  ha="left", va="top", color=INK_MUTED, fontsize=8)

    ax_t.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    ax_t.axvline(peak_year, color=INK_MUTED, linewidth=0.8)
    ax_t.plot(years, gmst, color=INK_PRIMARY, linewidth=1.6, label="GMST (FaIR)")
    if land_anom is not None:
        ax_t.plot(years, land_anom, color=INK_SECONDARY, linewidth=0.9,
                  label="Global land mean")
        ax_t.plot(years, pf_anom, color=INK_SECONDARY, linewidth=0.9,
                  linestyle=(0, (5, 2)), label="Permafrost region")
    ax_t.set_ylabel("Temperature above\npre-industrial (°C)")
    ax_t.set_xlabel("Year")
    ax_t.legend(loc="upper right", ncol=1)
    plotting.despine(ax_t)

    tas_note = (f" the other two are {run.model} air temperature (tas)."
                if land_anom is not None else "")
    plotting.caption(
        fig,
        f"{run.model} / {run.scenario_label} / {run.forcing.upper()}. Summed over "
        f"{years[0]} permafrost cells. Carbon is change relative to {years[0]}; "
        f"{years[0]} stocks in the legend. Total = above + below 1 m. "
        f"Temperatures relative to 1850–1900: GMST from the FaIR pathway"
        f"{',' if tas_note else '.'}{tas_note} Peak warming = GMST maximum. "
        f"Annual values.")

    out = run.figure_path("carbon_components")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out}   (peak warming {peak_year})")
    return out