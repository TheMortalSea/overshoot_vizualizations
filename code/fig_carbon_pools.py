"""Carbon anomaly by pool: soil, vegetation and their total on one set of axes.

Notebook:
    fig_carbon_pools_anomaly.main(run)

Anomalies in PgC relative to the first year, summed over the 2024 permafrost
cells. Reads carbon_series.csv from the run's processed folder (built by
calc_carbon_series.main). Adding a pool means one more entry in POOLS.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import calc_carbon_series
import plotting
from plotting import INK_MUTED
from wiemip_io import Run

# label -> series csv variable -> colour.
# "Total" is soil + vegetation only; it does not include litter, coarse woody
# debris or products, which are deliberately out of scope for this project.
POOLS = [
    ("Soil", "cSoilTotal", plotting.pool_color("cSoilTotal")),
    ("Vegetation", "cVeg", plotting.pool_color("cVeg")),
    ("Total", "cSoilVeg", plotting.pool_color("cSoilVeg")),
]


def say(*parts):
    print(*parts, flush=True)


def main(run: Run):
    plotting.apply_style()

    cached = calc_carbon_series.require_cached(run)
    missing = [v for _l, v, _c in POOLS if v not in cached]
    if missing:
        raise KeyError(
            f"carbon_series.csv for {run.tag} lacks {missing}; "
            f"available: {sorted(cached)}")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.axhline(0.0, color=INK_MUTED, linewidth=0.8)

    years = None
    for label, variable, colour in POOLS:
        years, values = cached[variable]
        anomaly = values - values[0]

        ax.plot(years, anomaly, color=colour, label=label)
        ax.annotate(f"{anomaly[-1]:+.1f}", xy=(years[-1], anomaly[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", color=colour, fontsize=8)
        say(f"  {label:12s} {variable:12s} {anomaly[-1]:+7.1f} PgC by {years[-1]}")

    ax.set_xlim(years[0], years[-1] + 10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Carbon anomaly [PgC]")
    ax.set_title("Permafrost region: carbon anomaly by pool", loc="left")
    ax.legend(loc="lower left")
    plotting.despine(ax)

    plotting.caption(
        fig,
        f"{run.model} / {plotting.scenario_label(run.simulation)} / "
        f"{run.forcing.upper()}. Summed over {years[0]} permafrost cells. "
        f"Anomaly relative to {years[0]}. Total is soil + vegetation. "
        f"Annual values.")

    out = run.figure_path("carbon_pools_anomaly")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out}")
    return out