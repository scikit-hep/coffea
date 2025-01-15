from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Hashable, List, Set, Tuple, Union

import awkward
import dask.base
import dask_awkward

from coffea.dataset_tools.preprocess import (
    DatasetSpec,
    DatasetSpecOptional,
    FilesetSpec,
    FilesetSpecOptional,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC
from coffea.util import decompress_form

DaskOutputBaseType = Union[
    dask.base.DaskMethodsMixin,
    Dict[Hashable, dask.base.DaskMethodsMixin],
    Set[dask.base.DaskMethodsMixin],
    List[dask.base.DaskMethodsMixin],
    Tuple[dask.base.DaskMethodsMixin],
]

# NOTE TO USERS: You can use nested python containers as arguments to dask.compute!
DaskOutputType = Union[DaskOutputBaseType, Tuple[DaskOutputBaseType, ...]]

GenericHEPAnalysis = Callable[[dask_awkward.Array], DaskOutputType]


def apply_to_dataset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    dataset: DatasetSpec | DatasetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    metadata: dict[Hashable, Any] = {},
    uproot_options: dict[str, Any] = {},
) -> DaskOutputType | tuple[DaskOutputType, dask_awkward.Array]:
    """
    Apply the supplied function or processor to the supplied dataset.

    Parameters
    ----------
        data_manipulation : ProcessorABC or GenericHEPAnalysis
            The user analysis code to run on the input dataset
        dataset: DatasetSpec | DatasetSpecOptional
            The data to be acted upon by the data manipulation passed in.
        schemaclass: BaseSchema, default NanoAODSchema
            The nanoevents schema to interpret the input dataset with.
        metadata: dict[Hashable, Any], default {}
            Metadata for the dataset that is accessible by the input analysis. Should also be dask-serializable.
        uproot_options: dict[str, Any], default {}
            Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.

    Returns
    -------
        out : DaskOutputType
            The output of the analysis workflow applied to the dataset
        report : dask_awkward.Array, optional
            The file access report for running the analysis on the input dataset. Needs to be computed in simultaneously with the analysis to be accurate.
    """
    maybe_base_form = dataset.get("form", None)
    if maybe_base_form is not None:
        maybe_base_form = awkward.forms.from_json(decompress_form(maybe_base_form))
    files = dataset["files"]
    events = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
        known_base_form=maybe_base_form,
        uproot_options=uproot_options,
    ).events()

    report = None
    if isinstance(events, tuple):
        events, report = events

    out = None
    if isinstance(data_manipulation, ProcessorABC):
        out = data_manipulation.process(events)
    elif isinstance(data_manipulation, Callable):
        out = data_manipulation(events)
    else:
        raise ValueError("data_manipulation must either be a ProcessorABC or Callable")

    if report is not None:
        return out, report
    return (out,)


def apply_to_fileset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    fileset: FilesetSpec | FilesetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    uproot_options: dict[str, Any] = {},
) -> dict[str, DaskOutputType] | tuple[dict[str, DaskOutputType], dask_awkward.Array]:
    """
    Apply the supplied function or processor to the supplied fileset (set of datasets).

    Parameters
    ----------
        data_manipulation : ProcessorABC or GenericHEPAnalysis
            The user analysis code to run on the input dataset
        fileset: FilesetSpec | FilesetSpecOptional
            The data to be acted upon by the data manipulation passed in. Metadata within the fileset should be dask-serializable.
        schemaclass: BaseSchema, default NanoAODSchema
            The nanoevents schema to interpret the input dataset with.
        uproot_options: dict[str, Any], default {}
            Options to pass to uproot. Pass at least {"allow_read_errors_with_report": True} to turn on file access reports.

    Returns
    -------
        out : dict[str, DaskOutputType]
            The output of the analysis workflow applied to the datasets, keyed by dataset name.
        report : dask_awkward.Array, optional
            The file access report for running the analysis on the input dataset. Needs to be computed in simultaneously with the analysis to be accurate.
    """
    out = {}
    report = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        if metadata is None:
            metadata = {}
        metadata.setdefault("dataset", name)
        dataset_out = apply_to_dataset(
            data_manipulation, dataset, schemaclass, metadata, uproot_options
        )
        if isinstance(dataset_out, tuple) and len(dataset_out) > 1:
            out[name], report[name] = dataset_out
        else:
            out[name] = dataset_out[0]
    if len(report) > 0:
        return out, report
    return out
