"""Statistical computation helpers decoupled from plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ._metadata import select_plot_variable


def _uniform_indices(n_total: int, n_samples: int) -> np.ndarray:
    """Return n_samples evenly-spaced integer indices in [0, n_total-1]."""
    return np.linspace(0, n_total - 1, min(n_samples, n_total), dtype=int)


def compute_welford_stats(
    ds: xr.Dataset, var_name: str | None = None, n_samples: int = 2000
) -> dict[str, np.ndarray]:
    """Mean/std/max maps and value histogram over sampled frames.

    Computes all five quantities in a single dask scheduler pass so
    the data is read from zarr only once.

    Returns a dict with keys:
        mean, std, max_val  -- 2D arrays (ny, nx)
        n_valid             -- 2D int64 array
        hist_counts         -- 1D int64 array (199 bins)
        hist_bins           -- 1D float64 array (200 edges, log-spaced 0.01..1000)
    """
    if var_name is None:
        var_name = select_plot_variable(ds)

    N = ds.sizes["time"]
    indices = _uniform_indices(N, n_samples)

    sampled = ds[var_name].isel(time=indices)
    hist_bins = np.logspace(-2, 3, 200)

    # Build lazy computations for all pixel-level stats
    mean_da = sampled.mean(dim="time", skipna=True)
    std_da = sampled.std(dim="time", skipna=True, ddof=1)
    max_da = sampled.max(dim="time", skipna=True)
    n_valid_da = sampled.count(dim="time")

    try:
        import dask
        import dask.array as da
        from dask.diagnostics import ProgressBar as DaskProgressBar

        # NaN / zeros / negatives fall outside the log bins and are ignored.
        hist_da, _ = da.histogram(sampled.data.ravel(), bins=hist_bins)

        with DaskProgressBar(dt=0.5):
            c_mean, c_std, c_max, c_n, hist_counts = dask.compute(
                mean_da, std_da, max_da, n_valid_da, hist_da
            )
    except (ImportError, AttributeError):
        # Fallback for non-dask-backed datasets
        c_mean = mean_da.values
        c_std = std_da.values
        c_max = max_da.values
        c_n = n_valid_da.values
        hist_counts, _ = np.histogram(np.asarray(sampled).ravel(), bins=hist_bins)

    mean_arr = np.asarray(c_mean)
    std_arr = np.asarray(c_std)
    max_arr = np.asarray(c_max)
    n_arr = np.asarray(c_n).astype(np.int64)

    return {
        "mean": np.where(n_arr > 0, mean_arr, np.nan),
        "std": np.where(n_arr > 1, std_arr, np.nan),
        "max_val": np.where(n_arr > 0, max_arr, np.nan),
        "n_valid": n_arr,
        "hist_counts": np.asarray(hist_counts).astype(np.int64),
        "hist_bins": hist_bins,
    }


def compute_spatial_coverage(
    ds: xr.Dataset, var_name: str | None = None, n_samples: int = 1000
) -> xr.DataArray:
    """Fraction of valid observations per pixel, in [0, 1]."""
    if var_name is None:
        var_name = select_plot_variable(ds)
    N = ds[var_name].shape[0]
    sample_indices = np.linspace(0, N - 1, n_samples, dtype=int)
    ds_filtered = ds[var_name].isel(time=sample_indices)
    return ds_filtered.map_blocks(lambda b: ~np.isnan(b)).sum(dim="time") / len(
        sample_indices
    )


def compute_monthly_stats(
    ds: xr.Dataset, var_name: str | None = None, n_samples: int = 3000
) -> dict[int, list[float]]:
    """Domain-mean values grouped by month.

    Returns a dict mapping month number (1-12) to a list of float spatial-mean values.
    """
    if var_name is None:
        var_name = select_plot_variable(ds)

    N = ds.sizes["time"]
    indices = _uniform_indices(N, n_samples)

    sampled = ds[var_name].isel(time=indices)
    spatial_dims = [d for d in sampled.dims if d != "time"]
    lazy_means = sampled.mean(dim=spatial_dims, skipna=True)

    try:
        from dask.diagnostics import ProgressBar as DaskProgressBar

        with DaskProgressBar(dt=0.5):
            spatial_means = lazy_means.values
    except ImportError:
        spatial_means = lazy_means.values

    months = pd.DatetimeIndex(ds.time.values[indices]).month
    valid = np.isfinite(spatial_means)

    monthly_data: dict[int, list[float]] = {m: [] for m in range(1, 13)}
    for m in range(1, 13):
        monthly_data[m] = spatial_means[(months == m) & valid].tolist()

    return monthly_data
