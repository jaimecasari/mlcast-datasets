"""Precipitation statistics: mean/max/std maps and value histogram."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from ._map_helpers import PLATE_CARREE, add_map_features, savefig, setup_rcparams
from ._metadata import get_data_crs, get_domain_extent, select_plot_variable
from ._stats import compute_welford_stats


def plot_precipitation_stats(
    ds: xr.Dataset,
    var_name: str | None = None,
    n_samples: int = 2000,
    figsize_maps: tuple[float, float] = (15, 5.5),
    figsize_hist: tuple[float, float] = (6, 4),
    cmap_mean: str = "YlGnBu",
    cmap_max: str = "YlOrRd",
    cmap_std: str = "OrRd",
    max_vmin: float = 1,
    max_vmax: float = 300,
    hist_color: str = "#3182bd",
    hist_xlim: tuple[float, float] = (0.01, 1000),
    map_resolution: str = "50m",
    output_path_maps: str | None = None,
    output_path_hist: str | None = None,
) -> tuple[Figure, Figure]:
    """Plot 3-panel stat maps (mean, max, std) and a precipitation histogram.

    Args:
        ds: xarray Dataset.
        var_name: Variable to analyse (auto-detected if None).
        n_samples: Number of frames to sample.
        figsize_maps: Figure size for the 3-panel map.
        figsize_hist: Figure size for the histogram.
        cmap_mean, cmap_max, cmap_std: Colormaps for each panel.
        max_vmin, max_vmax: Log-scale range for the max panel.
        hist_color: Bar color for the histogram.
        hist_xlim: X-axis limits for the histogram.
        map_resolution: NaturalEarth feature resolution.
        output_path_maps: If given, save the map figure.
        output_path_hist: If given, save the histogram figure.

    Returns:
        (fig_maps, fig_hist) tuple of matplotlib Figures.
    """
    setup_rcparams()
    if var_name is None:
        var_name = select_plot_variable(ds)
    data_crs = get_data_crs(ds, var_name)
    extent = get_domain_extent(ds, var_name)

    stats = compute_welford_stats(ds, var_name, n_samples=n_samples)

    lat = ds["lat"].values
    lon = ds["lon"].values

    # ===================== 3-panel map =====================
    fig_maps, axes = plt.subplots(
        1,
        3,
        figsize=figsize_maps,
        subplot_kw={"projection": data_crs},
        constrained_layout=True,
    )

    # (a) Mean
    im_a = axes[0].pcolormesh(
        lon,
        lat,
        stats["mean"],
        transform=PLATE_CARREE,
        cmap=cmap_mean,
        vmin=0,
        vmax=np.nanpercentile(stats["mean"], 99),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[0], resolution=map_resolution)
    axes[0].set_extent(extent, crs=PLATE_CARREE)
    axes[0].set_title("(a) Mean", fontweight="bold")
    fig_maps.colorbar(
        im_a,
        ax=axes[0],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    # (b) Max
    im_b = axes[1].pcolormesh(
        lon,
        lat,
        stats["max_val"],
        transform=PLATE_CARREE,
        cmap=cmap_max,
        norm=LogNorm(vmin=max_vmin, vmax=max_vmax),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[1], resolution=map_resolution)
    axes[1].set_extent(extent, crs=PLATE_CARREE)
    axes[1].set_title("(b) Maximum", fontweight="bold")
    fig_maps.colorbar(
        im_b,
        ax=axes[1],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    # (c) Std
    im_c = axes[2].pcolormesh(
        lon,
        lat,
        stats["std"],
        transform=PLATE_CARREE,
        cmap=cmap_std,
        vmin=0,
        vmax=np.nanpercentile(stats["std"], 99),
        shading="auto",
        rasterized=True,
    )
    add_map_features(axes[2], resolution=map_resolution)
    axes[2].set_extent(extent, crs=PLATE_CARREE)
    axes[2].set_title("(c) Std. dev.", fontweight="bold")
    fig_maps.colorbar(
        im_c,
        ax=axes[2],
        orientation="horizontal",
        shrink=0.8,
        pad=0.04,
        label="mm h$^{-1}$",
    )

    if output_path_maps:
        savefig(fig_maps, output_path_maps, close=False)

    # ===================== Histogram =====================
    fig_hist, ax = plt.subplots(figsize=figsize_hist)

    bins = stats["hist_bins"]
    bin_centers = np.sqrt(bins[:-1] * bins[1:])
    widths = np.diff(bins)
    ax.bar(
        bin_centers,
        stats["hist_counts"],
        width=widths,
        align="center",
        color=hist_color,
        edgecolor="none",
        alpha=0.8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Precipitation rate (mm h$^{-1}$)")
    ax.set_ylabel("Count (from sampled timesteps)")
    ax.set_title("Distribution of Non-Zero Precipitation Values", fontweight="bold")
    ax.set_xlim(*hist_xlim)

    if output_path_hist:
        savefig(fig_hist, output_path_hist, close=False)

    return fig_maps, fig_hist
