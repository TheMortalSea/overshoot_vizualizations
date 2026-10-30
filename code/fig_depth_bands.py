"""Thaw depth bands over time, by region, for the 2024 permafrost cells.

    python -u code/fig_depth_bands.py
    python -u code/fig_depth_bands.py --recompute    # reads all soil levels, 10-20 min

Thaw depth per cell and year = depth of the deepest soil level (down to 9.795 m)
that reaches 0 °C or above in any month. A cell whose 9.795 m level thaws is no
longer permafrost under the project definition ("thawed through"). Same soil
temperature data as calc_permafrost_extent.py, so the two cannot disagree.

Cell set fixed to the 2024 permafrost cells, so each region's stack total is
constant. Regions by latitude / longitude (no RECCAP2), see region_of().
"""
#THIS SCRIPT HAS BEEN MIGRATED TO WIEMIP REGISTRY

from __future__ import annotations

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Patch

import masks
import plotting
import wiemip_io
from wiemip_io import FIGURE_DIR, PROCESSED_DIR

MODEL = "CLM"
FORCING = "gfdl"
SIMULATION = "hl"
SCENARIO_LABEL = "HL"
PATTERN_LABEL = "GFDL"

FREEZING_K = 273.15
DEEPEST_M = 9.795  # level 21

LOST_COLOUR = "#b0ada2"
BANDS = [  # (label, upper edge in m, colour)
    ("0–1 m", 1.0, plotting.BLUE_RAMP[0]),
    ("1–2 m", 2.0, plotting.BLUE_RAMP[3]),
    ("2–5 m", 5.0, plotting.BLUE_RAMP[6]),
    ("5–9.8 m", DEEPEST_M, plotting.BLUE_RAMP[10]),
    ("thawed through", np.inf, LOST_COLOUR),
]
REGIONS = ["Northern Eurasia", "North America", "Tibetan Plateau"]

CACHE_CSV = PROCESSED_DIR / "permafrost_depth_bands.csv"


def say(*parts):
    print(*parts, flush=True)


def region_of(lat, lon):
    """Region index per cell (lat ascending, lon 0-360), -1 outside all three."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    out = np.zeros(la.shape, dtype=int)                         # Northern Eurasia
    out[(lo >= 190) & (lo < 330)] = 1                           # North America
    out[(la >= 25) & (la < 40) & (lo >= 70) & (lo < 105)] = 2   # Tibetan Plateau
    out[la < 0] = -1
    return out


def compute_thaw_bands(soil_t_da: xr.DataArray, cells):
    """Band index (year, n_cells) from soilT, restricted to `cells`."""
    levels = soil_t_da["levgrnd"].values
    use = np.where(levels <= DEEPEST_M + 1e-6)[0]
    nyr = soil_t_da.sizes["time"] // 12
    deepest = np.full((nyr, cells[0].size), -1.0)

    # Calculate years from standardized coordinate
    annual_proto = wiemip_io.to_annual(soil_t_da.isel(levgrnd=0))
    years = np.asarray(annual_proto["time"].values, dtype=int)[:nyr]

    for k in use:
        level_da = soil_t_da.isel(levgrnd=int(k))
        temp = level_da.values[: nyr * 12, cells[0], cells[1]]
        warmest = temp.reshape(nyr, 12, -1).max(axis=1)
        deepest = np.where(warmest >= FREEZING_K, levels[k], deepest)
        say(f"      level {k + 1:2d} ({levels[k]:.3f} m)")

    edges = [b[1] for b in BANDS[:-2]]  # 1, 2, 5
    band = np.digitize(np.maximum(deepest, 0.0), edges, right=True)
    band[deepest >= DEEPEST_M - 1e-6] = len(BANDS) - 1
    return years, band


def compute():
    grid = wiemip_io.load(
        variable="cSoilAbove1m",
        model=MODEL,
        forcing=FORCING,
        simulation=SIMULATION,
    )
    lat, lon = grid.lat.values, grid.lon.values
    pf = masks.permafrost_mask(grid).values > 0.5
    landfrac = wiemip_io.get_landfrac(grid)
    land = landfrac if landfrac is not None else xr.ones_like(grid)
    weight = (
        (wiemip_io.get_cell_area(grid) * land).fillna(0.0).values / 1.0e12
    )  # M km2
    cells = np.where(pf)
    region = region_of(lat, lon)[cells]
    w = weight[cells]

    say(f"    loading {MODEL} {SIMULATION} soilT via registry...")
    soil_t = wiemip_io.load(
        variable="soilT",
        model=MODEL,
        forcing=FORCING,
        simulation=SIMULATION,
    )
    years, band = compute_thaw_bands(soil_t, cells)

    rows = []
    for i, year in enumerate(years):
        for r, name in enumerate(REGIONS):
            in_r = region == r
            for b, (label, _edge, _c) in enumerate(BANDS):
                rows.append(
                    [
                        int(year),
                        name,
                        label,
                        float(w[in_r & (band[i] == b)].sum()),
                    ]
                )
    return rows


def load(recompute=False):
    if recompute or not CACHE_CSV.exists():
        if not recompute:
            raise FileNotFoundError(
                f"missing {CACHE_CSV.name} -- build it with:\n"
                f"    python -u code/fig_depth_bands.py --recompute"
            )
        rows = compute()
        with open(CACHE_CSV, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(
                [["year", "region", "band", "area_Mkm2"]] + rows
            )
        say(f"  wrote {CACHE_CSV.name}")
    with open(CACHE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    years = np.array(sorted({int(r["year"]) for r in rows}))
    data = {name: np.zeros((len(BANDS), len(years))) for name in REGIONS}
    band_i = {b[0]: i for i, b in enumerate(BANDS)}
    for r in rows:
        data[r["region"]][band_i[r["band"]], int(r["year"]) - years[0]] = float(
            r["area_Mkm2"]
        )
    return years, data


def main(recompute=False):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    years, data = load(recompute)
    plotting.apply_style()

    fig, axes = plt.subplots(
        len(REGIONS),
        1,
        figsize=(9.5, 8.6),
        sharex=True,
        gridspec_kw={"hspace": 0.30},
    )
    for ax, name in zip(axes, REGIONS):
        arr = data[name]
        base = float(arr[:, 0].sum())
        ax.stackplot(years, *arr, colors=[b[2] for b in BANDS], linewidth=0)
        ax.set_xlim(years[0], years[-1])
        ax.set_ylim(0, base * 1.02)
        ax.set_ylabel("Area (M km²)")
        ax.set_title(f"{name}: {base:.2f} M km² permafrost in {years[0]}", loc="left")
        plotting.despine(ax, keep=("left", "bottom"))
        lost = arr[-1]
        say(
            f"  {name:17s} {base:5.2f} M km2 | thawed through: max {lost.max():.2f} "
            f"in {years[int(np.argmax(lost))]}, {years[-1]} {lost[-1]:.2f}"
        )

    axes[-1].set_xlabel("Year")
    axes[-1].legend(
        handles=[Patch(facecolor=c, label=lab) for lab, _e, c in BANDS],
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        ncol=len(BANDS),
        frameon=False,
        fontsize=8,
    )

    plotting.caption(
        fig,
        f"{MODEL} / {SCENARIO_LABEL} / {PATTERN_LABEL}. Fixed set of cells that were permafrost in "
        f"{years[0]}. Band = depth of the deepest soil level reaching 0 °C in any month; "
        f"thawed through = the 9.8 m level thaws. Regions: Tibetan Plateau 25–40 °N, "
        f"70–105 °E; North America 170–30 °W; Northern Eurasia the rest. Y-axes differ "
        f"between panels. Annual values.",
        y=-0.02,
    )

    out = plotting.dated_path(FIGURE_DIR, "depth_bands")
    fig.savefig(out)
    plt.close(fig)
    say(f"\n  saved {out.name}")


if __name__ == "__main__":
    main(recompute="--recompute" in sys.argv)