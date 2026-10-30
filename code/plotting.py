"""Shared figure style.

Colours are the validated reference palette from the dataviz skill. Output is
PNG for a paper, so only the light-mode values are used.

Rules carried over from that skill's anti-pattern list:
  - no dual-axis charts (stack panels on a shared x instead)
  - gridlines and axes are solid hairlines, never dashed
  - a legend whenever there are >= 2 series; direct-label selectively
  - thin marks, recessive chrome
"""
# THIS SCRIPT HAS BEEN MIGRATED TO WIEMIP

from __future__ import annotations

import colorsys
from datetime import datetime
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")  # no display on this machine

from matplotlib.colors import LinearSegmentedColormap, to_rgb

# Import FIGURE_DIR for dated_path default fallback
import wiemip_io


def shade(colour, factor):
    """A lighter (factor > 1) or darker (factor < 1) version of a colour.

    Scales LIGHTNESS only, leaving hue and saturation alone, so the result still
    reads as the same pool. Use it for a secondary distinction WITHIN a pool --
    the warming and cooling branches of one trajectory, say -- never to
    introduce a new meaning. A new meaning needs its own entry in POOL_COLOR.
    """
    hue, light, sat = colorsys.rgb_to_hls(*to_rgb(colour))
    return colorsys.hls_to_rgb(hue, min(1.0, max(0.0, light * factor)), sat)


# --- ink & chrome ---------------------------------------------------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical slots ----------------------------------------------------
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua

# --- sequential (magnitude): one hue, light -> dark ------------------------
BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("wiemip_seq", BLUE_RAMP)

# --- scenarios ordered by peak warming: blue = cooler, red = hotter -------
WARMING = LinearSegmentedColormap.from_list(
    "wiemip_warming",
    ["#104281", "#2a78d6", "#6da7ec", "#eb8b6a", "#e34948", "#a32220"],
)

_COOL = ["#104281", "#2a78d6", "#6da7ec"]   # dark -> light blue
_WARM = ["#eb8b6a", "#e34948", "#a32220"]   # light -> dark red


def warming_colors(n):
    """n colours ordered from least to most peak warming."""
    if n == 1:
        return [_COOL[1]]
    cool, warm = (n + 1) // 2, n // 2
    return _COOL[:cool] + _WARM[len(_WARM) - warm:]


# Supports both hyphen ("hl-cf") and registry underscore ("hl_cf") conventions,
# as well as historical / control runs.
SCENARIO_COLOR = {
    "l": "#104281",       # lowest peak warming
    "m": "#6da7ec",
    "hl": "#e34948",      # highest peak warming
    "hl-cf": "#a32220",   # HL counterfactual: held at peak warming
    "hl_cf": "#a32220",
    "hist": "#52514e",    # historical run (neutral dark)
    "ctrl": "#898781",    # control run (muted grey)
}
SCENARIO_LABEL = {
    "l": "L",
    "m": "M",
    "hl": "HL",
    "hl-cf": "HL-CF",
    "hl_cf": "HL-CF",
    "hist": "Hist",
    "ctrl": "Ctrl",
}


def scenario_color(scenario: str, counterfactual: bool = False):
    """Get colour for a scenario or registry simulation string."""
    key = str(scenario).lower().replace("_", "-")
    if counterfactual and not key.endswith("-cf"):
        key = f"{key}-cf"
    return SCENARIO_COLOR.get(key, SERIES_1)


def scenario_label(scenario: str, counterfactual: bool = False):
    """Get display label for a scenario or registry simulation string."""
    key = str(scenario).lower().replace("_", "-")
    if counterfactual and not key.endswith("-cf"):
        key = f"{key}-cf"
    return SCENARIO_LABEL.get(key, str(scenario).upper())


# --- carbon pool palette --------------------------------------------------
SOIL_DARK = "#7f4f24"    # deep soil / below 1 m
SOIL_MID = "#b5762f"     # soil, single series
SOIL_LIGHT = "#dcab6b"   # shallow soil / above 1 m
VEG_DARK = "#2f6b34"
VEG_MID = "#4f9d47"      # vegetation, single series
TEAL = "#17827a"         # soil + vegetation combined
NEUTRAL = "#52514e"      # totals and sums with no pool of their own

POOL_COLOR = {
    "cSoilAbove1m": SOIL_LIGHT,
    "cSoilBelow1m": SOIL_DARK,
    "cSoilTotal": SOIL_MID,
    "cSoil": SOIL_MID,
    "cVeg": VEG_MID,
    "cSoilVeg": TEAL,
}


def pool_color(variable: str):
    """Colour for a carbon pool variable. Falls back to categorical slot 1."""
    return POOL_COLOR.get(variable, SERIES_1)


# --- carbon flux palette --------------------------------------------------
FLUX_NPP = "#8bc34a"   # leaf green: productivity, vegetation's input
FLUX_RH = "#9a6fb0"    # purple: respiration, no collision with soil brown

FLUX_COLOR = {"npp": FLUX_NPP, "rh": FLUX_RH}


def flux_color(variable: str):
    """Colour for a carbon flux variable. Falls back to categorical slot 1."""
    return FLUX_COLOR.get(variable, SERIES_1)


# Sequential ramps for maps of absolute stocks
SOIL_SEQ = LinearSegmentedColormap.from_list(
    "wiemip_soil", ["#f7ecd9", "#e6c99a", "#cfa05e", "#b5762f", "#8a551f", "#5c3611"]
)
VEG_SEQ = LinearSegmentedColormap.from_list(
    "wiemip_veg", ["#edf5e6", "#c6e0b4", "#93c47d", "#4f9d47", "#2f6b34", "#1c4220"]
)
TEAL_SEQ = LinearSegmentedColormap.from_list(
    "wiemip_teal", ["#e3f2f0", "#b8ddd8", "#7fc3bb", "#42a49a", "#17827a", "#0d5a55"]
)

POOL_CMAP = {
    "cSoilAbove1m": SOIL_SEQ,
    "cSoilBelow1m": SOIL_SEQ,
    "cSoilTotal": SOIL_SEQ,
    "cSoil": SOIL_SEQ,
    "cVeg": VEG_SEQ,
    "cSoilVeg": TEAL_SEQ,
}


def pool_cmap(variable: str):
    """Sequential colormap for a carbon pool variable."""
    return POOL_CMAP.get(variable, SEQUENTIAL)


# --- diverging: two opposing hues, NEUTRAL GRAY midpoint ------------------
DIVERGING = LinearSegmentedColormap.from_list(
    "wiemip_div",
    ["#104281", "#2a78d6", "#86b6ef", "#f0efec", "#f0a1a0", "#e34948", "#a32220"],
)

_LOSS_ARM = ["#4a2a63", "#7a4f96", "#b99ccb"]
_GAIN_ARM = {
    "soil": ["#dcab6b", "#b5762f", "#5c3611"],
    "veg": ["#93c47d", "#4f9d47", "#1c4220"],
    "teal": ["#7fc3bb", "#17827a", "#0d4d48"],
}
_GAIN_FAMILY = {
    "cSoilAbove1m": "soil",
    "cSoilBelow1m": "soil",
    "cSoilTotal": "soil",
    "cSoil": "soil",
    "cVeg": "veg",
    "cSoilVeg": "teal",
}


def carbon_change_cmap(variable: str):
    """Diverging map for a carbon change: purple loss, pool-hue gain."""
    gain = _GAIN_ARM[_GAIN_FAMILY.get(variable, "teal")]
    return LinearSegmentedColormap.from_list(
        f"wiemip_change_{variable}", _LOSS_ARM + ["#f0efec"] + gain
    )


def warming_cmap(vmin: float, vmax: float):
    """Neutral at 0, reds above, blues below, with 0 placed at its true position."""
    zero = -vmin / (vmax - vmin)
    nodes = [
        (0.0, "#2a78d6"),
        (zero * 0.5, "#86b6ef"),
        (zero, "#f0efec"),
        (zero + (1 - zero) * 0.3, "#f0a1a0"),
        (zero + (1 - zero) * 0.6, "#e34948"),
        (zero + (1 - zero) * 0.85, "#a32220"),
        (1.0, "#5e1210"),
    ]
    return LinearSegmentedColormap.from_list("wiemip_warming_map", nodes)


def apply_style() -> None:
    """Global rcParams for every figure in this project."""
    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 1.6,
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "svg.fonttype": "none",
    })


def despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def caption(fig, note: str = "", y: float = -0.01) -> None:
    """Caption below the figure."""
    if not note:
        return
    fig.text(
        0.02,
        y,
        textwrap.fill(note, width=112),
        ha="left",
        va="top",
        fontsize=7.5,
        color=INK_MUTED,
    )


def dated_path(directory: Path | str | None = None, stem: str = "figure", ext: str = "png") -> Path:
    """<stem>_YYYY-MM-DD.<ext> - one file per day, so versions sit side by side.

    If directory is None, defaults to wiemip_io.FIGURE_DIR.
    """
    dest_dir = Path(directory) if directory is not None else wiemip_io.FIGURE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / f"{stem}_{datetime.now():%Y-%m-%d}.{ext}"