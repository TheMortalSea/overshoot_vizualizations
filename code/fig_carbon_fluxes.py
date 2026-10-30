"""Carbon fluxes (NPP, RH) and their difference over the permafrost cells.

Notebook:
    fig_carbon_fluxes.main(run)

Reads flux_series.csv from the run's processed folder (build with
calc_carbon_fluxes.main(run), which documents the flux definitions and the
npp units).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import calc_carbon_fluxes as calc
import plotting
import tas_baseline
from plotting import INK_MUTED
from wiemip_io import Run

# Recessive warm line for the background temperature axis, matching the extent
# figures. Not a pool colour: temperature is context, not a carbon pool.
GMST_COLOUR = "#c0504d"

# (variable, label, colour). plotting.flux_color is the single source of truth,
# so NPP and RH look the same here as in carbon_hysteresis. They used to borrow
# the vegetation green and the soil brown; that was dropped because a flux
# wearing a pool's hue reads as that pool in any figure showing both.
FLUXES = [
    ("npp", "NPP (input)", plotting.flux_color("npp")),
    ("rh", "RH (output)", plotting.flux_color("rh")),
]
NET_COLOUR = plotting.NEUTRAL


def say(*parts):
    print(*parts, flush=True)


def main(run: Run):
    # Not optional: apply_style sets savefig.dpi=300 and bbox="tight". Without
    # it the figure saves at dpi 100 and the caption, which sits just below the
    # axes, falls outside the saved region entirely.
    plotting.apply_style()

    years, series = calc.require_cached(run)

    missing = [v for v, _l, _c in FLUXES if v not in series]
    if missing:
        say(f"  NOTE: not plotted, not on disk: {', '.join(missing)}")

    gmst = tas_baseline.gmst_series(years, run.scenario_label)
    if gmst is None:
        raise FileNotFoundError(
            f"no GMST pathway for scenario {run.scenario_label!r} -- check "
            "data/raw/FaIR_overshoot_temperature.nc exists and has this scenario")
    peak_year = int(years[int(np.argmax(gmst))])

    # Net gets its OWN panel. NPP and RH are ~8-10 PgC/yr and their difference is
    # ~0.1-0.4, so on a shared axis the net is a flat line on zero -- and the
    # difference is the entire point of the figure.
    has_net = all(v in series for v, _l, _c in FLUXES)
    n_panels = 2 if has_net else 1
    ratios = [1.5, 0.8] if has_net else [1.0]
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(9.5, 6.4 if has_net else 5.0), sharex=True,
        squeeze=False, gridspec_kw={"height_ratios": ratios, "hspace": 0.14})
    axes = axes[:, 0]
    ax_f = axes[0]
    ax_n = axes[1] if has_net else None

    # GMST rides BEHIND the fluxes on a right-hand axis instead of taking a panel
    # of its own. This is a dual-axis chart, normally the thing to avoid: two
    # y-scales manufacture crossings that mean nothing. It is acceptable here
    # only because temperature is context, not a series being compared against
    # the fluxes -- so it is drawn faint, behind, and never labelled in a way
    # that invites reading an intersection.
    ax_g = ax_f.twinx()
    ax_g.plot(years, gmst, color=GMST_COLOUR, linewidth=1.2, alpha=0.55)
    ax_g.set_ylabel("GMST above pre-industrial (°C)", color=GMST_COLOUR)
    ax_g.tick_params(axis="y", colors=GMST_COLOUR)
    ax_g.grid(False)
    for side, spine in ax_g.spines.items():
        spine.set_visible(side == "right")
    # Flux axes on top, patch transparent: matplotlib draws one whole axes over
    # the other, so per-artist zorder cannot lift a flux line above the twin.
    ax_f.set_zorder(ax_g.get_zorder() + 1)
    ax_f.patch.set_visible(False)

    ax_f.axvline(peak_year, color=INK_MUTED, linewidth=0.8)
    # Staggered so the two end labels cannot overprint: they finish 0.16 PgC/yr
    # apart, which is a couple of points at this scale.
    for offset, (variable, label, colour) in zip((7, -7), FLUXES):
        if variable not in series:
            continue
        values = series[variable]
        ax_f.plot(years, values, color=colour, linewidth=0.9,
                  label=f"{label}  ({values[0]:.2f} PgC/yr in {years[0]})")
        ax_f.annotate(f"{values[-1]:.2f}", xy=(years[-1], values[-1]),
                      xytext=(7, offset), textcoords="offset points",
                      ha="left", va="center", color=colour, fontsize=8)
        say(f"  {label:14s} {values[0]:6.2f} -> {values[-1]:6.2f} PgC/yr  "
            f"(peak {values.max():.2f} in {int(years[int(np.argmax(values))])})")

    ax_f.set_ylabel("Carbon flux (PgC/yr)")
    ax_f.set_title(
        f"Permafrost carbon fluxes — {run.model} · {run.forcing.upper()} · "
        f"{run.scenario_label}", loc="left")
    # Upper right: both curves rise from the lower left, so that corner is free.
    ax_f.legend(loc="upper right")
    ax_f.set_xlim(years[0], years[-1] + 10)
    plotting.despine(ax_f)
    ax_f.annotate(f"peak warming {peak_year}", xy=(peak_year, ax_f.get_ylim()[0]),
                  xytext=(6, 6), textcoords="offset points",
                  ha="left", va="bottom", color=INK_MUTED, fontsize=8)

    if has_net:
        net = series["npp"] - series["rh"]
        ax_n.axhline(0.0, color=INK_MUTED, linewidth=0.8)
        ax_n.axvline(peak_year, color=INK_MUTED, linewidth=0.8)
        ax_n.plot(years, net, color=NET_COLOUR, linewidth=0.9)
        ax_n.set_ylabel("Net, NPP − RH\n(PgC/yr)")
        plotting.despine(ax_n)
        say(f"  {'Net':14s} {net[0]:6.2f} -> {net[-1]:6.2f} PgC/yr  "
            f"cumulative {np.sum(net):+.1f} PgC")

    axes[-1].set_xlabel("Year")

    plotting.caption(
        fig,
        f"{run.model} / {run.scenario_label} / {run.forcing.upper()}. Summed over "
        f"{years[0]} permafrost cells, annual values. NPP = net primary "
        f"production, RH = heterotrophic respiration. Red line, right axis: GMST "
        f"from the FaIR pathway. Vertical line: peak warming ({peak_year}).")

    out = run.figure_path("carbon_fluxes")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out}")
    return out