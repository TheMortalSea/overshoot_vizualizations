"""Why the permafrost-region soil carbon anomaly differs from the global one.

Notebook:
    fig_domain_comparison.main(run)

Same model, same scenario, same variable, same pipeline -- the ONLY difference
is the spatial domain. Permafrost region = the 2024 permafrost cells
(masks.permafrost_mask); global land uses every land cell.

Soil carbon here is cSoilAbove1m + cSoilBelow1m, i.e. the full soil column.
Reads the raw model output directly (no processed cache needed).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import aggregate
import masks
import plotting
import wiemip_io
from plotting import INK_MUTED
from wiemip_io import Run

DEPTH_VARS = ("cSoilAbove1m", "cSoilBelow1m")


def say(*parts):
    print(*parts, flush=True)


def total_soil_carbon(run: Run, area_m2, frac, landfrac):
    """Summed depth components over one domain, PgC per year."""
    years, total = None, None
    for variable in DEPTH_VARS:
        da = wiemip_io.load(
            variable=variable,
            model=run.model,
            forcing=run.forcing,
            simulation=run.simulation,
        )
        tot = aggregate.regional_total(da, area_m2, frac, landfrac)
        annual = wiemip_io.to_annual(tot)
        this_years = np.asarray(annual["time"].values, dtype=int)
        this_vals = np.asarray(annual.values, dtype="f8")

        years = this_years if years is None else years
        total = this_vals if total is None else total + this_vals
    return years, total


def main(run: Run):
    plotting.apply_style()

    grid = wiemip_io.load(
        variable=DEPTH_VARS[0],
        model=run.model,
        forcing=run.forcing,
        simulation=run.simulation,
    )
    area_m2 = wiemip_io.get_cell_area(grid)
    landfrac = wiemip_io.get_landfrac(grid)
    pf_frac = masks.permafrost_mask(grid)

    say(f"computing both domains through the same code path ({run.tag}) ...")
    years, pf = total_soil_carbon(run, area_m2, pf_frac, landfrac)
    _, glob = total_soil_carbon(run, area_m2, None, landfrac)
    outside = glob - pf

    for label, series in (
        ("permafrost region", pf),
        ("global land", glob),
        ("non-permafrost", outside),
    ):
        say(
            f"  {label:18s} {series[0]:8.1f} -> {series[-1]:8.1f} PgC "
            f"({series[-1] - series[0]:+7.1f})"
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)
    for label, series, colour in (
        ("Global land", glob, plotting.SOIL_LIGHT),
        ("Permafrost region", pf, plotting.SOIL_DARK),
    ):
        change = series - series[0]
        ax.plot(years, change, color=colour, label=label)
        ax.annotate(
            f"{change[-1]:+.0f} PgC",
            xy=(years[-1], change[-1]),
            xytext=(-6, 6),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=colour,
            fontsize=8,
        )

    ax.set_xlim(years[0], years[-1])
    ax.set_xlabel("Year")
    ax.set_ylabel("Soil carbon anomaly [PgC]")
    ax.legend(loc="lower left")
    plotting.despine(ax)

    plotting.caption(
        fig,
        f"{run.model} / {run.scenario_label} / {run.forcing.upper()}. "
        f"Soil carbon = cSoilAbove1m + cSoilBelow1m, "
        f"anomaly relative to {int(years[0])}.",
    )

    out = run.figure_path("domain_comparison")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out}")
    return out