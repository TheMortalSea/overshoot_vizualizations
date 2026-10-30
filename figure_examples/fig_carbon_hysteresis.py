"""Carbon pools and fluxes plotted against global warming (hysteresis).

    python -u code/fig_carbon_hysteresis.py

x is the FaIR GMST pathway for HL; time runs along each curve, out to peak
warming and back. Reads cached CSVs only.

Layout, 3 x 3:
    soil            vegetation        soil + vegetation
    NPP             RH                NPP - RH
    % change pools  % change fluxes   (empty)
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

import plotting
import tas_baseline
from plotting import INK_MUTED
from wiemip_io import FIGURE_DIR, PROCESSED_DIR

MODEL, SCENARIO, PATTERN = "CLM6", "HL", "GFDL"
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


def read_pool(variable):
    path = PROCESSED_DIR / f"fig01_series_{variable}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} -- build it with:\n"
            f"    python -u code/calc_carbon_series.py {variable}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return (np.array([int(r["year"]) for r in rows]),
            np.array([float(r["carbon_PgC"]) for r in rows]))


def read_fluxes():
    path = PROCESSED_DIR / "flux_series.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path.name} -- build it with:\n"
            f"    python -u code/calc_carbon_fluxes.py --recompute")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    years = np.array([int(r["year"]) for r in rows])
    out = {k: np.array([float(r[k]) for r in rows]) for k in ("npp", "rh")}
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


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plotting.apply_style()

    series = {}
    years = None
    for _label, variable, _c, _u in POOLS:
        yrs, series[variable] = read_pool(variable)
        years = yrs if years is None else years
        if not np.array_equal(yrs, years):
            raise ValueError(f"{variable}: year axis differs")
    flux_years, fluxes = read_fluxes()
    if not np.array_equal(flux_years, years):
        raise ValueError("flux years differ from pool years")
    series.update(fluxes)

    temp = tas_baseline.gmst_series(years, SCENARIO)
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
        f"{MODEL} / {SCENARIO} / {PATTERN}. Summed over {years[0]} permafrost cells, "
        f"annual values. x = GMST from the FaIR pathway. Open circle = {MARK_START}, "
        f"filled circle = peak warming ({years[peak_i]}), square = {MARK_END}; triangles "
        f"show direction of travel; dashed line = peak warming. NPP = net primary "
        f"production, RH = heterotrophic respiration.")

    out = plotting.dated_path(FIGURE_DIR, "carbon_hysteresis")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out.name}")


if __name__ == "__main__":
    main()
