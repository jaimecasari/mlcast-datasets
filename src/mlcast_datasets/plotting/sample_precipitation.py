"""Sample precipitation maps from a specific event window."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from ._map_helpers import (
    PLATE_CARREE,
    add_map_features,
    get_precipitation_cmap,
    savefig,
    setup_rcparams,
)
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable


def plot_sample_precipitation(
    ds: xr.Dataset,
    time_slice: slice,
    var_name: str | None = None,
    time_spacing_hours: int = 3,
    max_frames: int = 6,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 9),
    levels: list[float] | None = None,
    colors: list[str] | None = None,
    map_resolution: str = "50m",
    output_path: str | None = None,
) -> Figure:
    """Plot a grid of precipitation maps from a specific event window.

    Args:
        ds: xarray Dataset.
        time_slice: REQUIRED. A slice object, e.g. slice("2023-05-16", "2023-05-17T06:00").
        var_name: Variable to plot (auto-detected if None).
        time_spacing_hours: Minimum spacing between selected frames.
        max_frames: Maximum number of frames to show (determines grid layout).
        title: Figure suptitle (auto-generated from time range if None).
        figsize: Figure size in inches.
        levels: Discrete colorbar levels (default: [0, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]).
        colors: Colors for the discrete bins (default: 9-color scheme).
        map_resolution: NaturalEarth feature resolution.
        output_path: If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    event_window = ds.sel(time=time_slice)
    all_times = event_window.time.values

    # Pick timestamps at least time_spacing_hours apart
    picked = [all_times[0]]
    for t in all_times[1:]:
        if (t - picked[-1]) >= np.timedelta64(time_spacing_hours, "h"):
            picked.append(t)
        if len(picked) == max_frames:
            break
    event = event_window.sel(time=picked)

    data = event[var_name].values
    timestamps = event.time.values

    lat = ds["lat"].values
    lon = ds["lon"].values

    cmap, norm = get_precipitation_cmap(levels=levels, colors=colors)

    if title is None:
        t0 = np.datetime_as_string(timestamps[0], unit="D")
        title = f"Precipitation Event, {t0}"

    n_frames = len(timestamps)
    ncols = min(3, n_frames)
    nrows = (n_frames + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    im = None
    for idx, ax in enumerate(axes.flat):
        if idx < n_frames:
            im = ax.pcolormesh(
                lon,
                lat,
                data[idx],
                transform=PLATE_CARREE,
                cmap=cmap,
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            add_map_features(ax, resolution=map_resolution)
            ax.set_extent(extent, crs=PLATE_CARREE)
            ts = np.datetime_as_string(timestamps[idx], unit="m")
            ax.set_title(ts.replace("T", " ") + " UTC", fontsize=9)
        else:
            ax.set_visible(False)

    if im is not None:
        cbar = fig.colorbar(
            im,
            ax=axes.ravel().tolist(),
            orientation="horizontal",
            shrink=0.6,
            pad=0.03,
            aspect=40,
        )
        cbar.set_label("Precipitation rate (mm h$^{-1}$)")

    fig.suptitle(title, fontsize=13, fontweight="bold")

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
