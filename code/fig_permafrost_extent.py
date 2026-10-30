"""Plot annual permafrost extent time series (2024-2300) for one run."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import calc_permafrost_extent
import plotting
from wiemip_io import Run


def main(run: Run):
    plotting.apply_style()
    years, extent = calc_permafrost_extent.require_cached(run)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(years, extent, color="#2b5c8f", linewidth=2.2,
            label=f"{run.model} ({run.forcing.upper()})")

    idx_min = int(np.argmin(extent))
    yr_min, ext_min = years[idx_min], extent[idx_min]
    ext_start, ext_end = extent[0], extent[-1]

    ax.scatter([years[0], yr_min, years[-1]], [ext_start, ext_min, ext_end],
               color="#d95f02", zorder=5)
    ax.annotate(f"{years[0]}: {ext_start:.1f} M km²", xy=(years[0], ext_start),
                xytext=(10, 5), textcoords="offset points", fontsize=9)
    ax.annotate(f"Min ({yr_min}): {ext_min:.1f} M km²", xy=(yr_min, ext_min),
                xytext=(-30, -15), textcoords="offset points", fontsize=9)
    ax.annotate(f"{years[-1]}: {ext_end:.1f} M km²", xy=(years[-1], ext_end),
                xytext=(-70, 10), textcoords="offset points", fontsize=9)

    ax.set_title(f"Permafrost Extent (Dynamic 9.8 m Soil Temp) — "
                 f"{run.scenario_label} Pathway", fontsize=12, pad=10)
    ax.set_ylabel("Permafrost Extent (Million km²)")
    ax.set_xlabel("Year")
    ax.set_xlim(years[0], years[-1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    out = run.figure_path("permafrost_extent")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    return out