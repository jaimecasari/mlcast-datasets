import importlib
import sys

import pytest
from loguru import logger

import mlcast_datasets

VALIDATOR_SPECS = {
    "precipitation": ("source_data", "radar_precipitation"),
}


@pytest.fixture
def catalog():
    return mlcast_datasets.open_catalog()


def all_entries():
    catalog = mlcast_datasets.open_catalog()
    return list(catalog.walk(depth=10))


@pytest.mark.parametrize("dataset_name", all_entries())
def test_get_intake_source(catalog, dataset_name):
    item = catalog[dataset_name]
    if item.container == "catalog":
        item.reload()
    else:
        logger.debug(f"Testing {dataset_name}")
        plugin = item.cat.describe()["plugin"][0]
        if plugin in ["opendap", "zarr", "netcdf"]:
            _ = item.to_dask()
        elif plugin in ["intake_esm.esm_datastore", "parquet"]:
            _ = item.get()
        elif plugin in ["json"]:
            _ = item.read()
        elif plugin == "yaml_file_cat":
            pass
        else:
            raise Exception(plugin)


def _infer_validator_spec(dataset_name: str):
    parts = dataset_name.replace("/", ".").split(".")
    if not parts:
        return None
    return VALIDATOR_SPECS.get(parts[0])


def _load_validator(spec):
    data_stage, product = spec
    module = importlib.import_module(
        f"mlcast_dataset_validator.specs.{data_stage}.{product}"
    )
    return module.validate_dataset


@pytest.mark.parametrize("dataset_name", all_entries())
def test_dataset_passes_validator(catalog, dataset_name):
    item = catalog[dataset_name]
    if item.container == "catalog":
        pytest.skip("Catalog entry; validator applies to datasets only.")

    spec = _infer_validator_spec(dataset_name)
    if spec is None:
        pytest.fail(f"No validator spec mapping for dataset '{dataset_name}'.")

    validate_dataset = _load_validator(spec)
    if not hasattr(item, "to_dask"):
        pytest.fail(f"Dataset '{dataset_name}' does not support to_dask().")
    ds = item.to_dask()

    # set storage_options explicitly on ds.attrs so that it is available to the
    # validator, which needs these when working out the zarr store path for the
    # dataset (looking for consolidated meta data), the storage options
    ds.encoding["storage_options"] = item.reader.data.storage_options
    report, _ = validate_dataset(ds)
    report.console_print(file=sys.stderr)

    if report.has_fails():
        pytest.fail(report.summarize())


@pytest.mark.modified_on_branch
def test_make_ci_happy_if_no_test_is_selected():
    """pytest returns exit code 5 if no test is selected"""
    pass
