"""Dataset introspection utilities for automatic CRS, variable, and extent detection."""

from __future__ import annotations

import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import xarray as xr


def select_plot_variable(ds: xr.Dataset) -> str:
    """Find the first 3D (time, y, x) data variable in the dataset."""
    for var_name, da in ds.data_vars.items():
        if "time" in da.dims and da.ndim >= 3:
            return var_name
    raise ValueError("Could not find a 2D spatial variable with a time dimension.")


def get_data_crs(ds: xr.Dataset, var_name: str | None = None) -> ccrs.CRS:
    """Extract a cartopy CRS from the grid_mapping attribute of a variable.

    Falls back to PlateCarree if no grid_mapping is found.
    """
    if var_name is None:
        var_name = select_plot_variable(ds)
    grid_mapping_name = ds[var_name].attrs.get("grid_mapping")
    if grid_mapping_name and grid_mapping_name in ds:
        crs_wkt = ds[grid_mapping_name].attrs.get("crs_wkt")
        if crs_wkt:
            return ccrs.Projection(crs_wkt)
    return ccrs.PlateCarree()


def get_domain_extent(ds: xr.Dataset, var_name: str | None = None) -> list[float]:
    """Compute [lon_min, lon_max, lat_min, lat_max] in PlateCarree degrees.

    Uses 2D lat/lon coordinates if present, otherwise reprojects from native CRS.
    """
    if "lat" in ds and "lon" in ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        if lat.ndim == 2:
            return [
                float(np.nanmin(lon)),
                float(np.nanmax(lon)),
                float(np.nanmin(lat)),
                float(np.nanmax(lat)),
            ]
        elif lat.ndim == 1:
            return [
                float(lon.min()),
                float(lon.max()),
                float(lat.min()),
                float(lat.max()),
            ]

    # Fallback: use 1D spatial coordinates with CRS reprojection
    if var_name is None:
        var_name = select_plot_variable(ds)
    da = ds[var_name]
    spatial_dims = [d for d in da.dims if d != "time"]
    if len(spatial_dims) != 2:
        raise ValueError(f"Expected two spatial dims, got {spatial_dims}")
    y_dim, x_dim = spatial_dims

    x = ds[x_dim].values
    y = ds[y_dim].values
    data_crs = get_data_crs(ds, var_name)
    plate_carree = ccrs.PlateCarree()

    corners = np.array(
        [
            [x.min(), y.min()],
            [x.max(), y.min()],
            [x.min(), y.max()],
            [x.max(), y.max()],
        ]
    )
    transformed = plate_carree.transform_points(data_crs, corners[:, 0], corners[:, 1])
    return [
        float(transformed[:, 0].min()),
        float(transformed[:, 0].max()),
        float(transformed[:, 1].min()),
        float(transformed[:, 1].max()),
    ]


def infer_time_step_minutes(ds: xr.Dataset, sample_size: int = 128) -> float:
    """Infer the dominant time step in minutes from the first sample_size timestamps."""
    sample = pd.DatetimeIndex(ds["time"].isel(time=slice(0, sample_size)).values)
    if len(sample) < 2:
        return np.nan
    deltas = sample.to_series().diff().dropna()
    if deltas.empty:
        return np.nan
    return deltas.median().total_seconds() / 60.0


def get_variable_label(ds: xr.Dataset, var_name: str | None = None) -> str:
    """Return a human-readable label like 'Precipitation rate (mm h⁻¹)' from attrs."""
    if var_name is None:
        var_name = select_plot_variable(ds)
    attrs = ds[var_name].attrs
    long_name = attrs.get("long_name", var_name)
    units = attrs.get("units", "")
    if units:
        return f"{long_name} ({units})"
    return long_name
