"""Carbon pools and fluxes plotted against global warming (hysteresis).

Notebook:
    fig_carbon_hysteresis.main(run)

x is the FaIR GMST pathway for the run's scenario; time runs along each curve,
out to peak warming and back. Reads cached CSVs only (carbon_series.csv and
flux_series.csv in the run's processed folder).

Layout, 3 x 3:
    soil            vegetation        soil + vegetation
    NPP             RH                NPP - RH
    % change pools  % change fluxes   (empty)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import calc_carbon_fluxes
import calc_carbon_series
import plotting
import tas_baseline
from plotting import INK_MUTED
from wiemip_io import Run

MARK_START, MARK_END = 2024, 2300

# (title, variable, colour, unit)
POOLS = [
    ("Soil", "cSoilTotal", plotting.pool_color("cSoilTotal"), "PgC"),
    ("Vegetation", "cVeg", plotting.pool_color("cVeg"), "PgC"),
    ("Soil + vegetation", "cSoilVeg", plotting.pool_color("cSoilVeg"), "PgC"),
]
FLUXES = [
    ("NPP", "npp", plotting.flux_color("npp"), "PgC/yr"),
    ("RH", "rh", plotting.flux_color("rh"), "PgC/yr"),
    ("NPP − RH", "net", plotting.NEUTRAL, "PgC/yr"),
]


def say(*parts):
    print(*parts, flush=True)


def read_pool(run: Run, variable: str):
    """(years, values_PgC) for one pool from the run's carbon_series.csv."""
    pools = calc_carbon_series.require_cached(run)
    if variable not in pools:
        raise KeyError(
            f"{variable} not in carbon_series.csv for {run.tag}; "
            f"available: {sorted(pools)}")
    return pools[variable]


def read_fluxes(run: Run):
    years, flux = calc_carbon_fluxes.require_cached(run)
    missing = [k for k in ("npp", "rh") if k not in flux]
    if missing:
        raise KeyError(f"flux_series.csv for {run.tag} lacks {missing}")
    out = {k: flux[k] for k in ("npp", "rh")}
    out["net"] = out["npp"] - out["rh"]
    return years, out


def add_direction_arrows(ax, x, y, colour, peak_i, span=10, size=11):
    """Arrowheads along the trajectory. Headings are computed in display space,
    so this must run after the axis limits are final (after fig.canvas.draw())."""
    tail = len(x) - peak_i
    for i in (peak_i // 3, 2 * peak_i // 3,
              peak_i + tail // 4, peak_i + tail // 2, peak_i + 3 * tail // 4):
        j = max(0, i - span)
        (x0, y0), (x1, y1) = (ax.transData.transform((x[j], y[j])),
                              ax.transData.transform((x[i], y[i])))
        if x0 == x1 and y0 == y1:
            continue
        ang = np.degrees(np.arctan2(y1 - y0, x1 - x0)) - 90.0
        ax.plot(x[i], y[i], marker=(3, 0, ang), markersize=size, color=colour,
                markeredgecolor=plotting.SURFACE, markeredgewidth=1.0,
                linestyle="none", zorder=6)


def draw_trajectory(ax, temp, values, years, colour, peak_i, label=None):
    """One trajectory: open circle = start, filled circle = peak, square = end."""
    ax.plot(temp, values, color=colour, linewidth=1.0, label=label)
    i0 = int(np.where(years == MARK_START)[0][0])
    i1 = int(np.where(years == MARK_END)[0][0])
    ax.plot(temp[i0], values[i0], marker="o", markersize=7, markerfacecolor="white",
            markeredgecolor=colour, markeredgewidth=1.6, zorder=5)
    ax.plot(temp[peak_i], values[peak_i], marker="o", markersize=6, color=colour, zorder=5)
    ax.plot(temp[i1], values[i1], marker="s", markersize=6, color=colour, zorder=5)
    return i0, i1


def main(run: Run):
    plotting.apply_style()

    series = {}
    years = None
    for _label, variable, _c, _u in POOLS:
        yrs, series[variable] = read_pool(run, variable)
        years = yrs if years is None else years
        if not np.array_equal(yrs, years):
            raise ValueError(f"{variable}: year axis differs")
    flux_years, fluxes = read_fluxes(run)
    if not np.array_equal(flux_years, years):
        raise ValueError(
            f"flux years ({flux_years[0]}-{flux_years[-1]}) differ from "
            f"pool years ({years[0]}-{years[-1]})")
    series.update(fluxes)

    temp = tas_baseline.gmst_series(years, run.scenario_label)
    if temp is None:
        raise FileNotFoundError("missing data/raw/FaIR_overshoot_temperature.nc")
    peak_i = int(np.argmax(temp))

    fig, axes = plt.subplots(3, 3, figsize=(14.0, 12.5),
                             gridspec_kw={"hspace": 0.38, "wspace": 0.30})
    arrow_jobs = []

    for ax, (label, variable, colour, unit) in zip(axes[:2].ravel(), POOLS + FLUXES):
        values = series[variable]
        i0, i1 = draw_trajectory(ax, temp, values, years, colour, peak_i)
        arrow_jobs.append((ax, values, colour))
        if variable == "net":
            ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=0)
        fmt = "{:.2f}" if unit == "PgC/yr" else "{:.0f}"
        ax.set_title(f"{label}: {fmt.format(values[i0])} → {fmt.format(values[i1])} {unit}",
                     loc="left")
        ax.set_ylabel(unit)
        say(f"  {label:18s} {values[i0]:8.2f} -> {values[i1]:8.2f} {unit}")

    # Relative change from the first year: pools in one panel, fluxes in another.
    # NPP - RH is left out: it is near zero, so a percentage of it is meaningless.
    for ax, group, title in ((axes[2, 0], POOLS, "Pools, change from 2024"),
                             (axes[2, 1], FLUXES[:2], "Fluxes, change from 2024")):
        for label, variable, colour, _unit in group:
            values = series[variable]
            pct = 100.0 * (values - values[0]) / values[0]
            draw_trajectory(ax, temp, pct, years, colour, peak_i, label=label)
            arrow_jobs.append((ax, pct, colour))
        ax.axhline(0.0, color=INK_MUTED, linewidth=0.8, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_ylabel("%")
        ax.legend(loc="upper left", fontsize=8)
    axes[2, 2].axis("off")

    pad = 0.05 * float(temp.max() - temp.min())
    for ax in axes.ravel()[:8]:
        ax.set_xlim(float(temp.min()) - pad, float(temp.max()) + pad)
        ax.axvline(float(temp[peak_i]), color=INK_MUTED, linewidth=0.9,
                   linestyle=(0, (4, 3)), zorder=0)
        ax.set_xlabel("GMST above 1850–1900 (°C)")
        plotting.despine(ax, keep=("left", "bottom"))

    fig.canvas.draw()
    for ax, ys, colour in arrow_jobs:
        add_direction_arrows(ax, temp, ys, colour, peak_i)

    plotting.caption(
        fig,
        f"{run.model} / {run.scenario_label} / {run.forcing.upper()}. Summed over "
        f"{years[0]} permafrost cells, annual values. x = GMST from the FaIR "
        f"pathway. Open circle = {MARK_START}, filled circle = peak warming "
        f"({years[peak_i]}), square = {MARK_END}; triangles show direction of "
        f"travel; dashed line = peak warming. NPP = net primary production, "
        f"RH = heterotrophic respiration.")

    out = run.figure_path("carbon_hysteresis")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out}")
    return out