import importlib
import io
import sys
from contextlib import redirect_stdout

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

    if hasattr(item, "describe"):
        description = item.describe()
    else:
        description = item._entry.describe()
    args = description.get("args", {})
    dataset_path = args.get("urlpath") or args.get("path")
    if not dataset_path:
        pytest.fail(f"No dataset path found for '{dataset_name}'.")

    storage_options = args.get("storage_options")
    validate_dataset = _load_validator(spec)
    report = validate_dataset(dataset_path, storage_options=storage_options)
    buffer = io.StringIO()
    # Rich writes tables to stdout; capture and replay to stderr so CI logs include the report.
    with redirect_stdout(buffer):
        report.console_print()
    print(buffer.getvalue(), file=sys.stderr)

    if report.has_fails():
        pytest.fail(report.summarize())


@pytest.mark.modified_on_branch
def test_make_ci_happy_if_no_test_is_selected():
    """pytest returns exit code 5 if no test is selected"""
    pass
