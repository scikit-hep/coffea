from __future__ import annotations

import copy
import sys
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import awkward
import numpy

from coffea.dataset_tools.filespec import (
    CoffeaParquetFileSpec,
    CoffeaROOTFileSpec,
    DataGroupSpec,
    DatasetSpec,
    InputFiles,
    PreprocessedFiles,
)


@runtime_checkable
class LimitFilesProtocol(Protocol):
    # handle both limit_files with max_files and max_files + per_dataset
    def limit_files(self, max_files: int | slice, per_dataset: bool = True) -> Self: ...


@runtime_checkable
class LimitStepsProtocol(Protocol):
    def limit_steps(
        self, max_steps: int | slice, per_file: bool = False, per_dataset: bool = True
    ) -> Self: ...


def max_chunks(
    fileset: LimitStepsProtocol | DataGroupSpec, maxchunks: int | None = None
) -> DataGroupSpec:
    """
    Modify the input fileset so that only the first "maxchunks" chunks of each dataset will be processed.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets reduce to max-chunks row-ranges.
        maxchunks : int or None, default None
            How many chunks to keep for each file.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the first maxchunks event ranges left in.
    """
    return slice_chunks(fileset, slice(maxchunks))


def max_chunks_per_file(
    fileset: LimitStepsProtocol | DataGroupSpec, maxchunks: int | None = None
) -> DataGroupSpec:
    """
    Modify the input fileset so that only the first "maxchunks" chunks of each file will be processed.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets reduce to max-chunks row-ranges.
        maxchunks : int or None, default None
            How many chunks to keep for each file.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the first maxchunks event ranges left in.
    """
    return slice_chunks(fileset, slice(maxchunks), bydataset=False)


def _concatenated_step_slice(
    stepdict: dict[str, Any], theslice: int | slice
) -> dict[str, Any]:
    """
    Modify the input step description to only contain the steps specified by the input slice.

    Parameters
    ----------
        stepdict : dict[str, Any]
            The step description to be sliced.
        theslice : int | slice
            How to slice the array of row-ranges (steps) in the input step description.

    Returns
    -------
        out : dict[str, Any]
            The reduced step description with only the row-ranges specified by theslice left.
    """
    if isinstance(theslice, int):
        theslice = slice(theslice)
    out = {key: [] for key in stepdict}

    # 1) build a flat list of (key, step)
    flat: list[tuple[str, Any]] = []
    for key, steps in stepdict.items():
        for step in steps:
            flat.append((key, step))

    # 2) slice that flat list
    kept = flat[theslice]

    # 3) repopulate in order, up to maxchunks total
    for key, step in kept:
        out[key].append(step)
    return out  # {key: steps for key, steps in out.items() if steps}


def slice_chunks(
    fileset: LimitStepsProtocol | DataGroupSpec,
    theslice: Any = slice(None),
    bydataset: bool = True,
) -> DataGroupSpec:
    """
    Modify the input fileset so that only the chunks of each file or each dataset specified by the input slice are processed.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets to be sliced.
        theslice : Any, default slice(None)
            How to slice the array of row-ranges (steps) in the input fileset.
        bydataset : bool, default True
            If True, slices across all steps in all files in each dataset, otherwise slices each file individually.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the row-ranges specified by theslice left.
    """
    if isinstance(fileset, LimitStepsProtocol):
        return fileset.limit_steps(theslice, per_file=not bydataset)

    if not isinstance(theslice, slice):
        theslice = slice(theslice)

    out = copy.deepcopy(fileset)

    if not bydataset:
        for dname, d in fileset.items():
            for fname, finfo in d["files"].items():
                out[dname]["files"][fname]["steps"] = finfo["steps"][theslice]
        return out

    for dname, d in fileset.items():
        # 1) build a flat list of (fname, step)
        flat: list[tuple[str, Any]] = []
        for fname, finfo in d["files"].items():
            for step in finfo["steps"]:
                flat.append((fname, step))

        # 2) slice that flat list
        kept = flat[theslice]

        # 3) zero-out all steps in the output
        for fname in out[dname]["files"]:
            out[dname]["files"][fname]["steps"] = []

        # 4) repopulate in order, up to maxchunks total
        for fname, step in kept:
            out[dname]["files"][fname]["steps"].append(step)

        # 5) drop files with no steps
        out[dname]["files"] = {
            fname: finfo
            for fname, finfo in out[dname]["files"].items()
            if finfo["steps"]
        }
    return out


def max_files(fileset: DataGroupSpec, maxfiles: int | None = None) -> DataGroupSpec:
    """
    Modify the input fileset so that only the first "maxfiles" files of each dataset will be processed.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets reduce to max-files files per dataset.
        maxfiles : int or None, default None
            How many files to keep for each dataset.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the first maxfiles files left in.
    """
    return slice_files(fileset, slice(maxfiles))


def slice_files(fileset: DataGroupSpec, theslice: Any = slice(None)) -> DataGroupSpec:
    """
    Modify the input fileset so that only the files of each dataset specified by the input slice are processed.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets to be sliced.
        theslice : Any, default slice(None)
            How to slice the array of files in the input datasets. We slice in key-order.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the files specified by theslice left.
    """
    if isinstance(fileset, LimitFilesProtocol):
        return fileset.limit_files(theslice)

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
    num_entries = (
        spec.num_entries if hasattr(spec, "num_entries") else spec["num_entries"]
    )
    return num_entries is not None and num_entries > 0


def filter_files(
    fileset: DataGroupSpec,
    thefilter: Callable[
        [
            tuple[str, CoffeaROOTFileSpec | CoffeaParquetFileSpec]
            | InputFiles
            | PreprocessedFiles
        ],
        bool,
    ] = _default_filter,
) -> DataGroupSpec:
    """
    Modify the input fileset so that only the files of each dataset that pass the filter remain.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets to be sliced.
        thefilter: Callable[[tuple[str, CoffeaROOTFileSpec | CoffeaParquetFileSpec] | InputFiles | PreprocessedFiles], bool], default filters empty files
            How to filter the files in the each dataset.

    Returns
    -------
        out : DataGroupSpec
            The reduced fileset with only the files specified by thefilter left.
    """
    out = copy.deepcopy(fileset)
    for name, entry in fileset.items():
        is_datasetspec = isinstance(entry, DatasetSpec)
        to_apply_to = getattr(entry, "files") if is_datasetspec else entry["files"]
        updated = dict(filter(thefilter, to_apply_to.items()))
        if is_datasetspec:
            out[name].files = InputFiles(updated)
        else:
            out[name]["files"] = updated
    return out


def get_failed_steps_for_dataset(
    dataset: dict | DatasetSpec, report: awkward.Array
) -> dict | DatasetSpec:
    """
    Modify the input dataset to only contain the files and row-ranges for *failed* processing jobs as specified in the supplied report.

    Parameters
    ----------
        dataset: DatasetSpec | dict
            The dataset to be reduced to only contain files and row-ranges that have previously encountered failed file access.
        report : awkward.Array
            The computed file-access error report from dask-awkward.

    Returns
    -------
        out : DatasetSpec | dict
            The reduced dataset with only the row-ranges and files that failed processing, according to the input report.
    """
    is_datasetspec = isinstance(dataset, DatasetSpec)
    dataset = dataset.model_dump() if is_datasetspec else dataset
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

    return DatasetSpec(**failed_dataset) if is_datasetspec else failed_dataset


def get_failed_steps_for_fileset(
    fileset: DataGroupSpec, report_dict: dict[str, awkward.Array]
):
    """
    Modify the input fileset to only contain the files and row-ranges for *failed* processing jobs as specified in the supplied report.

    Parameters
    ----------
        fileset : DataGroupSpec
            The set of datasets to be reduced to only contain files and row-ranges that have previously encountered failed file access.
        report_dict : dict[str, awkward.Array]
            The computed file-access error reports from dask-awkward, indexed by dataset name.

    Returns
    -------
        out : DataGroupSpec
            The reduced dataset with only the row-ranges and files that failed processing, according to the input report.
    """
    failed_fileset = {}
    if isinstance(fileset, DataGroupSpec):
        for name, dataset in fileset.items():
            failed_dataset = get_failed_steps_for_dataset(dataset, report_dict[name])
            if len(failed_dataset.files) > 0:
                failed_fileset[name] = failed_dataset
        return DataGroupSpec(failed_fileset)
    else:
        for name, dataset in fileset.items():
            failed_dataset = get_failed_steps_for_dataset(dataset, report_dict[name])
            if len(failed_dataset["files"]) > 0:
                failed_fileset[name] = failed_dataset
        return failed_fileset
