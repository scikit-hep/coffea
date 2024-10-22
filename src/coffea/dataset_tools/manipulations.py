from __future__ import annotations

import copy
from typing import Any, Callable

import awkward
import numpy

from coffea.dataset_tools.preprocess import CoffeaFileSpec, DatasetSpec, FilesetSpec


def max_chunks(fileset: FilesetSpec, maxchunks: int | None = None) -> FilesetSpec:
    """
    Modify the input dataset so that only the first "maxchunks" chunks of each file will be processed.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets reduce to max-chunks row-ranges.
        maxchunks: int | None, default None
            How many chunks to keep for each file.

    Returns
    -------
        out : FilesetSpec
            The reduced fileset with only the first maxchunks event ranges left in.
    """
    return slice_chunks(fileset, slice(maxchunks))


def slice_chunks(fileset: FilesetSpec, theslice: Any = slice(None)) -> FilesetSpec:
    """
    Modify the input dataset so that only the chunks of each file specified by the input slice are processed.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets to be sliced.
        theslice: Any, default slice(None)
            How to slice the array of row-ranges (steps) in the input fileset.

    Returns
    -------
        out : FilesetSpec
            The reduce fileset with only the row-ranges specific by theslice left.
    """
    if not isinstance(theslice, slice):
        theslice = slice(theslice)

    out = copy.deepcopy(fileset)
    for name, entry in fileset.items():
        for fname, finfo in entry["files"].items():
            out[name]["files"][fname]["steps"] = finfo["steps"][theslice]

    return out


def max_files(fileset: FilesetSpec, maxfiles: int | None = None) -> FilesetSpec:
    """
    Modify the input dataset so that only the first "maxfiles" files of each dataset will be processed.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets reduce to max-files files per dataset.
        maxfiles: int | None, default None
            How many files to keep for each dataset.

    Returns
    -------
        out : FilesetSpec
            The reduced fileset with only the first maxfiles files left in.
    """
    return slice_files(fileset, slice(maxfiles))


def slice_files(fileset: FilesetSpec, theslice: Any = slice(None)) -> FilesetSpec:
    """
    Modify the input dataset so that only the files of each dataset specified by the input slice are processed.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets to be sliced.
        theslice: Any, default slice(None)
            How to slice the array of files in the input datasets. We slice in key-order.

    Returns
    -------
        out : FilesetSpec
            The reduce fileset with only the files specified by theslice left.
    """
    if not isinstance(theslice, slice):
        theslice = slice(theslice)

    out = copy.deepcopy(fileset)
    for name, entry in fileset.items():
        fnames = list(entry["files"].keys())[theslice]
        finfos = list(entry["files"].values())[theslice]

        out[name]["files"] = {fname: finfo for fname, finfo in zip(fnames, finfos)}

    return out


def _default_filter(name_and_spec):
    name, spec = name_and_spec
    num_entries = spec["num_entries"]
    return num_entries is not None and num_entries > 0


def filter_files(
    fileset: FilesetSpec,
    thefilter: Callable[[tuple[str, CoffeaFileSpec]], bool] = _default_filter,
) -> FilesetSpec:
    """
    Modify the input dataset so that only the files of each dataset that pass the filter remain.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets to be sliced.
        thefilter: Callable[[tuple[str, CoffeaFileSpec]], bool], default filters empty files
            How to filter the files in the each dataset.

    Returns
    -------
        out : FilesetSpec
            The reduce fileset with only the files specified by thefilter left.
    """
    out = copy.deepcopy(fileset)
    for name, entry in fileset.items():
        out[name]["files"] = dict(filter(thefilter, out[name]["files"].items()))
    return out


def get_failed_steps_for_dataset(
    dataset: DatasetSpec, report: awkward.Array
) -> DatasetSpec:
    """
    Modify an input dataset to only contain the files and row-ranges for *failed* processing jobs as specified in the supplied report.

    Parameters
    ----------
        dataset: DatasetSpec
            The dataset to be reduced to only contain files and row-ranges that have previously encountered failed file access.
        report: awkward.Array
            The computed file-access error report from dask-awkward.

    Returns
    -------
        out : DatasetSpec
            The reduced dataset with only the row-ranges and files that failed processing, according to the input report.
    """
    failed_dataset = copy.deepcopy(dataset)
    failed_dataset["files"] = {}
    failures = report[~awkward.is_none(report.exception)]

    if not awkward.all(report.args[:, 4] == "True"):
        raise RuntimeError(
            "step specification is not completely in starts/stops form, failed-step extraction is not available for steps_per_file."
        )

    for fname, fdesc in dataset["files"].items():
        if "steps" not in fdesc:
            raise RuntimeError(
                f"steps specification not found in file description for {fname}, "
                "please specify steps consistently in input dataset."
            )

    fnames = set(dataset["files"].keys())
    rnames = (
        set(numpy.unique(failures.args[:, 0][:, 1:-1:])) if len(failures) > 0 else set()
    )
    if not rnames.issubset(fnames):
        raise RuntimeError(
            f"Files: {rnames - fnames} are not in input dataset, please ensure report corresponds to input dataset!"
        )

    for failure in failures:
        args_as_types = tuple(eval(arg) for arg in failure.args)

        fname, object_path, start, stop, is_step = args_as_types

        if fname in failed_dataset["files"]:
            failed_dataset["files"][fname]["steps"].append([start, stop])
        else:
            failed_dataset["files"][fname] = copy.deepcopy(dataset["files"][fname])
            failed_dataset["files"][fname]["steps"] = [[start, stop]]

    return failed_dataset


def get_failed_steps_for_fileset(
    fileset: FilesetSpec, report_dict: dict[str, awkward.Array]
):
    """
    Modify an input dataset to only contain the files and row-ranges for *failed* processing jobs as specified in the supplied report.

    Parameters
    ----------
        fileset: FilesetSpec
            The set of datasets to be reduced to only contain files and row-ranges that have previously encountered failed file access.
        report_dict: dict[str, awkward.Array]
            The computed file-access error reports from dask-awkward, indexed by dataset name.

    Returns
    -------
        out : FilesetSpec
            The reduced dataset with only the row-ranges and files that failed processing, according to the input report.
    """
    failed_fileset = {}
    for name, dataset in fileset.items():
        failed_dataset = get_failed_steps_for_dataset(dataset, report_dict[name])
        if len(failed_dataset["files"]) > 0:
            failed_fileset[name] = failed_dataset
    return failed_fileset
