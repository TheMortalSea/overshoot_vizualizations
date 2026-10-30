"""Circumpolar (north polar stereographic) maps, drawn without cartopy.

cartopy is not installed in any env here and conda is pinned to `defaults`, so
the projection is done by hand:

    r = tan(pi/4 - lat/2)       x = r sin(lon)      y = -r cos(lon)

Coastlines come from the model's own `landmask`/`landfrac` via `wiemip_io` rather
than a cartographic dataset -- that shows the land the model actually sees, at
its real resolution. The permafrost boundary is contoured from the RECCAP2
fraction at 0.5.
"""
# THIS SCRIPT HAS BEEN MIGRATED TO TEH WIEMIP REGISTRY
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import plotting
from plotting import INK_MUTED, INK_SECONDARY
import wiemip_io

LAT_LIMIT = 40.0  # southern edge of the disc


def _edges(centres: np.ndarray) -> np.ndarray:
    step = np.diff(centres).mean()
    return np.concatenate([centres - step / 2.0, [centres[-1] + step / 2.0]])


def _project(lat, lon):
    """North polar stereographic. lat/lon in degrees, broadcast to 2-D."""
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    r = np.tan(np.pi / 4.0 - lat_r / 2.0)
    return r * np.sin(lon_r), -r * np.cos(lon_r)


def _wrap(arr2d: np.ndarray, lon_centres: np.ndarray):
    """Repeat the first column past 360 so contours close across the seam."""
    return (
        np.concatenate([arr2d, arr2d[:, :1]], axis=1),
        np.concatenate([lon_centres, [lon_centres[0] + 360.0]]),
    )


def _panel(ax, field, lat, lon, landmask, pf_frac, cmap, vmin, vmax, title, label):
    # Coerce DataArrays to raw 2D numpy arrays
    if isinstance(field, xr.DataArray):
        field = field.squeeze().values
    elif hasattr(field, "values"):
        field = field.values

    if isinstance(lat, xr.DataArray):
        lat = lat.values
    if isinstance(lon, xr.DataArray):
        lon = lon.values
    if landmask is not None and hasattr(landmask, "values"):
        landmask = landmask.values
    if pf_frac is not None and hasattr(pf_frac, "values"):
        pf_frac = pf_frac.values

    keep = lat >= LAT_LIMIT
    lat_k, field_k = lat[keep], field[keep, :]

    lat_e, lon_e = _edges(lat_k), _edges(lon)
    mesh_lon_e, mesh_lat_e = np.meshgrid(lon_e, lat_e)
    x_e, y_e = _project(mesh_lat_e, mesh_lon_e)

    img = ax.pcolormesh(
        x_e, y_e, field_k, cmap=cmap, vmin=vmin, vmax=vmax,
        shading="flat", rasterized=True
    )

    mesh_lon_c = mesh_lat_c = None
    if landmask is not None:
        landmask_k = landmask[keep, :]
        land_w, lon_w = _wrap(landmask_k, lon)
        mesh_lon_c, mesh_lat_c = np.meshgrid(lon_w, lat_k)
        x_c, y_c = _project(mesh_lat_c, mesh_lon_c)
        ax.contour(x_c, y_c, land_w, levels=[0.5], colors=[INK_MUTED], linewidths=0.5)

    if pf_frac is not None:
        pf_k = pf_frac[keep, :]
        pf_w, lon_w = _wrap(pf_k, lon)
        if mesh_lon_c is None:
            mesh_lon_c, mesh_lat_c = np.meshgrid(lon_w, lat_k)
        x_c, y_c = _project(mesh_lat_c, mesh_lon_c)
        ax.contour(x_c, y_c, pf_w, levels=[0.5], colors=[INK_SECONDARY], linewidths=1.2)

    radius = float(np.tan(np.pi / 4.0 - np.deg2rad(LAT_LIMIT) / 2.0))
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, color=INK_MUTED, lw=0.8))
    ax.set_xlim(-radius * 1.02, radius * 1.02)
    ax.set_ylim(-radius * 1.02, radius * 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="center", fontsize=9)

    bar = plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.03, shrink=0.82)
    bar.set_label(label, color=INK_SECONDARY, fontsize=8)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7)


def symmetric_limit(*fields):
    """99th percentile of |value| across fields, for a diverging scale centred on 0."""
    arrays = [f.values if hasattr(f, "values") else f for f in fields]
    values = np.abs(np.concatenate([a[np.isfinite(a)] for a in arrays]))
    return float(np.percentile(values, 99)) or 1.0


def row(
    panels,
    lat=None,
    lon=None,
    landmask=None,
    pf_frac=None,
    out_path=None,
    name="",
    note="",
):
    """Panels side by side. Each panel is (field, title, cmap, vmin, vmax, label)."""
    first_field = panels[0][0]

    # Automatically extract lat/lon from DataArray coordinates if not passed
    if lat is None:
        if hasattr(first_field, "lat"):
            lat = first_field.lat.values
        else:
            raise ValueError("`lat` was not provided and could not be inferred from field.")

    if lon is None:
        if hasattr(first_field, "lon"):
            lon = first_field.lon.values
        else:
            raise ValueError("`lon` was not provided and could not be inferred from field.")

    # Automatically infer landmask using wiemip_io helper
    if landmask is None and isinstance(first_field, xr.DataArray):
        landmask = wiemip_io.get_landfrac(first_field)

    # Derive caption metadata from wiemip_io attributes when available
    attrs = getattr(first_field, "attrs", {})
    model = attrs.get("wiemip_model", "CLM6")
    simulation = attrs.get("wiemip_simulation", "HL")
    forcing = attrs.get("wiemip_forcing", "GFDL")

    run_desc = f"{model.upper()} / {simulation.upper()} scenario / {forcing.upper()} pattern."
    pf_desc = "Outline: 2024 permafrost (soil at 9.8 m below 0 °C all year). " if pf_frac is not None else ""
    caption_text = f"{run_desc} {pf_desc}{note}".strip()

    plotting.apply_style()
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.8))
    if n_panels == 1:
        axes = [axes]

    for ax, (field, title, cmap, vmin, vmax, label) in zip(axes, panels):
        _panel(ax, field, lat, lon, landmask, pf_frac, cmap, vmin, vmax, title, label)

    fig.suptitle(name, x=0.02, ha="left", fontsize=10, fontweight="bold")
    plotting.caption(fig, caption_text, y=0.06)

    # Save to FIGURE_DIR if given a relative filename or path
    if out_path is not None:
        dest = Path(out_path)
        if not dest.is_absolute():
            dest = wiemip_io.FIGURE_DIR / dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest)

    plt.close(fig)