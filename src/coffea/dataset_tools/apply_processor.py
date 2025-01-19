from __future__ import annotations

import copy
from functools import partial
from typing import Any, Callable, Dict, Hashable, List, Set, Tuple, Union

import awkward
import cloudpickle
import dask.base
import dask.delayed
import dask_awkward
import lz4.frame

from coffea.dataset_tools.preprocess import (
    DatasetSpec,
    DatasetSpecOptional,
    FilesetSpec,
    FilesetSpecOptional,
)
from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.processor import ProcessorABC
from coffea.util import decompress_form, load, save

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


def _pack_meta_to_wire(*collections):
    unpacked, repacker = dask.base.unpack_collections(*collections)

    output = []
    for i in range(len(unpacked)):
        output.append(unpacked[i])
        if isinstance(
            unpacked[i], (dask_awkward.Array, dask_awkward.Record, dask_awkward.Scalar)
        ):
            output[-1]._meta = awkward.Array(
                unpacked[i]._meta.layout.form.length_zero_array(),
                behavior=unpacked[i]._meta.behavior,
                attrs=unpacked[i]._meta.attrs,
            )
    packed_out = repacker(output)
    return packed_out


def _unpack_meta_from_wire(*collections):
    unpacked, repacker = dask.base.unpack_collections(*collections)

    output = []
    for i in range(len(unpacked)):
        output.append(unpacked[i])
        if isinstance(
            unpacked[i], (dask_awkward.Array, dask_awkward.Record, dask_awkward.Scalar)
        ):
            output[-1]._meta = awkward.Array(
                unpacked[i]._meta.layout.to_typetracer(forget_length=True),
                behavior=unpacked[i]._meta.behavior,
                attrs=unpacked[i]._meta.attrs,
            )
    packed_out = repacker(output)
    return packed_out


def _apply_analysis_wire(analysis, events_wire):
    (events,) = _unpack_meta_from_wire(events_wire)
    events._meta.attrs["@original_array"] = events
    out = analysis(events)
    out_wire = _pack_meta_to_wire(out)
    return out_wire


def apply_to_dataset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    dataset: DatasetSpec | DatasetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    metadata: dict[Hashable, Any] = {},
    uproot_options: dict[str, Any] = {},
    parallelize_with_dask: bool = False,
) -> (
    DaskOutputType
    | tuple[DaskOutputType, dask_awkward.Array]
    | tuple[dask_awkward.Array, DaskOutputType, dask_awkward.Array]
):
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
        parallelize_with_dask: bool, default False
            Create dask.delayed objects that will return the the computable dask collections for the analysis when computed.
        return_events: bool, default True
            Return the created events object, or not.

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
    events_and_maybe_report = NanoEventsFactory.from_root(
        files,
        metadata=metadata,
        schemaclass=schemaclass,
        known_base_form=maybe_base_form,
        uproot_options=uproot_options,
    ).events()

    events = events_and_maybe_report
    report = None
    if isinstance(events, tuple):
        events, report = events

    analysis = None
    if isinstance(data_manipulation, ProcessorABC):
        analysis = data_manipulation.process
    elif isinstance(data_manipulation, Callable):
        analysis = data_manipulation
    else:
        raise ValueError("data_manipulation must either be a ProcessorABC or Callable")

    out = None
    if parallelize_with_dask:
        (wired_events,) = _pack_meta_to_wire(events)
        out = dask.delayed(
            lambda: lz4.frame.compress(
                cloudpickle.dumps(
                    partial(_apply_analysis_wire, analysis, wired_events)()
                ),
                compression_level=6,
            )
        )()
        if hasattr(dask.base, "function_cache"):
            dask.base.function_cache.clear()
    else:
        out = analysis(events)
        if not isinstance(out, tuple):
            out = (out,)

    if report is not None:
        return events, out, report
    return events, out


def apply_to_fileset(
    data_manipulation: ProcessorABC | GenericHEPAnalysis,
    fileset: FilesetSpec | FilesetSpecOptional,
    schemaclass: BaseSchema = NanoAODSchema,
    uproot_options: dict[str, Any] = {},
    parallelize_with_dask: bool = False,
    scheduler: Callable | str | None = None,
    return_events: bool = False,
) -> (
    dict[str, DaskOutputType]
    | tuple[dict[str, DaskOutputType], dict[str, dask_awkward.Array]]
    | tuple[
        dict[str, dask_awkward.Array],
        dict[str, DaskOutputType],
        dict[str, dask_awkward.Array],
    ]
):
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
        parallelize_with_dask: bool, default False
            Create dask.delayed objects that will return the the computable dask collections for the analysis when computed.
        scheduler: Callable | str | None, default None
            If parallelize_with_dask is True, this specifies the dask scheduler used to calculate task graphs in parallel.

    Returns
    -------
        events: dict[str, dask_awkward.Array]
            The NanoEvents objects the analysis function was applied to.
        out : dict[str, DaskOutputType]
            The output of the analysis workflow applied to the datasets, keyed by dataset name.
        report : dask_awkward.Array, optional
            The file access report for running the analysis on the input dataset. Needs to be computed in simultaneously with the analysis to be accurate.
    """
    events = {}
    out = {}
    analyses_to_compute = {}
    report = {}
    for name, dataset in fileset.items():
        metadata = copy.deepcopy(dataset.get("metadata", {}))
        if metadata is None:
            metadata = {}
        metadata.setdefault("dataset", name)
        dataset_out = apply_to_dataset(
            data_manipulation,
            dataset,
            schemaclass,
            metadata,
            uproot_options,
            parallelize_with_dask,
        )

        if parallelize_with_dask:
            if len(dataset_out) == 3:
                events[name], analyses_to_compute[name], report[name] = dataset_out
            elif len(dataset_out) == 2:
                events[name], analyses_to_compute[name] = dataset_out
                print(dataset_out)
            else:
                raise ValueError(
                    "apply_to_dataset only returns (events, outputs) or (events, outputs, reports)"
                )
        elif isinstance(dataset_out, tuple) and len(dataset_out) == 3:
            events[name], out[name], report[name] = dataset_out
        elif isinstance(dataset_out, tuple) and len(dataset_out) == 2:
            events[name], out[name] = dataset_out
        else:
            raise ValueError(
                "apply_to_dataset only returns (events, outputs) or (events, outputs, reports)"
            )

    if parallelize_with_dask:
        (calculated_graphs,) = dask.compute(analyses_to_compute, scheduler=scheduler)
        for name, compressed_taskgraph in calculated_graphs.items():
            dataset_out_wire = cloudpickle.loads(
                lz4.frame.decompress(compressed_taskgraph)
            )
            (out[name],) = _unpack_meta_from_wire(*dataset_out_wire)

    for name in out:
        if isinstance(out[name], tuple) and len(out[name]) == 1:
            out[name] = out[name][0]

    if len(report) > 0:
        return (events, out, report) if return_events else (out, report)
    return (events, out) if return_events else out


def save_taskgraph(filename, events, *data_products, optimize_graph=False):
    """
    Save a task graph and its originating nanoevents to a file
    Parameters
    ----------
        filename: str
            Where to save the resulting serialized taskgraph and nanoevents.
            Suggested postfix ".hlg", after dask's HighLevelGraph object.
        events: dict[str, dask_awkward.Array]
            A dictionary of nanoevents objects.
        data_products: dict[str, DaskOutputBaseType]
            The data products resulting from applying an analysis to
            a NanoEvents object. This may include report objects.
        optimize_graph: bool, default False
            Whether or not to save the task graph in its optimized form.

    Returns
    -------
    """
    (events_wire,) = _pack_meta_to_wire(events)

    if len(data_products) == 0:
        raise ValueError(
            "You must supply at least one analysis data product to save a task graph!"
        )

    data_products_out = data_products
    if optimize_graph:
        data_products_out = dask.optimize(data_products)

    data_products_wire = _pack_meta_to_wire(*data_products_out)

    save(
        {
            "events": events_wire,
            "data_products": data_products_wire,
            "optimized": optimize_graph,
        },
        filename,
    )


def load_taskgraph(filename):
    """
    Load a task graph and its originating nanoevents from a file.
    Parameters
    ----------
        filename: str
            The file from which to load the task graph.
    Returns
    _______
    """
    graph_information_wire = load(filename)

    (events,) = _unpack_meta_from_wire(graph_information_wire["events"])
    (data_products,) = _unpack_meta_from_wire(*graph_information_wire["data_products"])
    optimized = graph_information_wire["optimized"]

    for dataset_name in events:
        events[dataset_name]._meta.attrs["@original_array"] = events[dataset_name]

    return events, data_products, optimized
