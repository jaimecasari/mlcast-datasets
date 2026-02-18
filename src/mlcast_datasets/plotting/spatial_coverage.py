"""Spatial coverage map: fraction of valid data per pixel."""

from __future__ import annotations

import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.figure import Figure

from ._map_helpers import PLATE_CARREE, add_map_features, savefig, setup_rcparams
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable
from ._stats import compute_spatial_coverage


def plot_spatial_coverage(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_samples: int = 1000,
    title: str = "Spatial Data Coverage",
    figsize: tuple[float, float] = (6, 8),
    cmap: str = "YlOrRd_r",
    vmin: float = 0,
    vmax: float = 1,
    contour_levels: tuple[float, ...] = (0.5, 0.9),
    contour_colors: tuple[str, ...] = ("#c0392b", "#2c3e50"),
    map_resolution: str = "10m",
    output_path: str | None = None,
) -> Figure:
    """Plot the fraction of valid observations per pixel.

    Args:
        ds: xarray Dataset with a 3D (time, y, x) variable.
        var_name: Variable to analyse (auto-detected if None).
        n_samples: Number of frames to sample uniformly across time.
        title: Figure title.
        figsize: Figure size in inches.
        cmap: Colormap name.
        vmin, vmax: Colorbar range.
        contour_levels: Iso-contour levels to overlay.
        contour_colors: Colors for the contour lines.
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

    frac = compute_spatial_coverage(ds, var_name, n_samples=n_samples).values

    lat = ds["lat"].values
    lon = ds["lon"].values

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": data_crs})

    im = ax.pcolormesh(
        lon,
        lat,
        frac,
        transform=PLATE_CARREE,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
        rasterized=True,
    )

    if contour_levels:
        ax.contour(
            lon,
            lat,
            frac,
            levels=list(contour_levels),
            colors=list(contour_colors),
            linewidths=[0.8 + 0.2 * i for i in range(len(contour_levels))],
            transform=PLATE_CARREE,
        )

    add_map_features(ax, resolution=map_resolution)
    ax.set_extent(extent, crs=PLATE_CARREE)

    cbar = fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.7, pad=0.08)
    cbar.set_label("Fraction of valid observations")
    ax.set_title(title, fontsize=12, fontweight="bold")

    if output_path:
        savefig(fig, output_path, close=False)
    return fig
