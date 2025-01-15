from __future__ import annotations

import copy
import hashlib
import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Hashable

import awkward
import dask
import dask.base
import dask_awkward
import numpy
import uproot
from uproot._util import no_filter

from coffea.util import _remove_not_interpretable, compress_form, decompress_form


def get_steps(
    normed_files: awkward.Array | dask_awkward.Array,
    step_size: int | None = None,
    align_clusters: bool = False,
    recalculate_steps: bool = False,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = False,
    step_size_safety_factor: float = 0.5,
    uproot_options: dict = {},
) -> awkward.Array | dask_awkward.Array:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    Parameters
    ----------
        normed_files: awkward.Array | dask_awkward.Array
            The list of normalized file descriptions to process for steps.
        step_size: int | None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters: bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps: bool, default False
            If steps are present in the input normed files, force the recalculation of those steps, instead
            of only recalculating the steps if the uuid has changed.
        skip_bad_files: bool, False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions: Exception | Warning | tuple[Exception | Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form: bool, default False
            Extract the form of the TTree from the file so we can skip opening files later.
        step_size_safety_factor: float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.

    Returns
    -------
        array : awkward.Array | dask_awkward.Array
            The normalized file descriptions, appended with the calculated steps for those files.
    """
    nf_backend = awkward.backend(normed_files)
    lz_or_nf = awkward.typetracer.length_zero_if_typetracer(normed_files)

    array = [] if nf_backend != "typetracer" else lz_or_nf
    for arg in lz_or_nf:
        try:
            the_file = uproot.open({arg.file: None}, **uproot_options)
            tree = the_file[arg.object_path]
        except file_exceptions as e:
            if skip_bad_files:
                array.append(None)
                continue
            else:
                raise e

        num_entries = tree.num_entries

        form_json = None
        form_hash = None
        if save_form:
            form_str = uproot.dask(
                tree,
                ak_add_doc=True,
                filter_name=no_filter,
                filter_typename=no_filter,
                filter_branch=partial(_remove_not_interpretable, emit_warning=False),
            ).layout.form.to_json()
            # the function cache needs to be popped if present to prevent memory growth
            if hasattr(dask.base, "function_cache"):
                dask.base.function_cache.popitem()

            form_hash = hashlib.md5(form_str.encode("utf-8")).hexdigest()
            form_json = compress_form(form_str)

        target_step_size = num_entries if step_size is None else step_size

        file_uuid = str(the_file.file.uuid)

        out_uuid = arg.uuid
        out_steps = arg.steps

        if out_uuid != file_uuid or recalculate_steps:
            if align_clusters:
                clusters = tree.common_entry_offsets()
                out = [0]
                for c in clusters:
                    if c >= out[-1] + target_step_size:
                        out.append(c)
                if clusters[-1] != out[-1]:
                    out.append(clusters[-1])
                out = numpy.array(out, dtype="int64")
                out = numpy.stack((out[:-1], out[1:]), axis=1)

                step_mask = (
                    out[:, 1] - out[:, 0]
                    > (1 + step_size_safety_factor) * target_step_size
                )
                if numpy.any(step_mask):
                    warnings.warn(
                        f"In file {arg.file}, steps: {out[step_mask]} with align_clusters=True are "
                        f"{step_size_safety_factor*100:.0f}% larger than target "
                        f"step size: {target_step_size}!"
                    )

            else:
                n_steps_target = max(round(num_entries / target_step_size), 1)
                actual_step_size = math.ceil(num_entries / n_steps_target)
                out = numpy.array(
                    [
                        [
                            i * actual_step_size,
                            min((i + 1) * actual_step_size, num_entries),
                        ]
                        for i in range(n_steps_target)
                    ],
                    dtype="int64",
                )

            out_uuid = file_uuid
            out_steps = out.tolist()

        if out_steps is not None and len(out_steps) == 0:
            out_steps = [[0, 0]]

        array.append(
            {
                "file": arg.file,
                "object_path": arg.object_path,
                "steps": out_steps,
                "num_entries": num_entries,
                "uuid": out_uuid,
                "form": form_json,
                "form_hash_md5": form_hash,
            }
        )

    if len(array) == 0:
        array = awkward.Array(
            [
                {
                    "file": "junk",
                    "object_path": "junk",
                    "steps": [[0, 0]],
                    "num_entries": 0,
                    "uuid": "junk",
                    "form": "junk",
                    "form_hash_md5": "junk",
                },
                None,
            ]
        )
        array = awkward.Array(array.layout.form.length_zero_array(highlevel=False))
    else:
        array = awkward.Array(array)

    if nf_backend == "typetracer":
        array = awkward.Array(
            array.layout.to_typetracer(forget_length=True),
        )

    return array


@dataclass
class UprootFileSpec:
    object_path: str
    steps: list[list[int]] | list[int] | None


@dataclass
class CoffeaFileSpec(UprootFileSpec):
    steps: list[list[int]]
    num_entries: int
    uuid: str


@dataclass
class CoffeaFileSpecOptional(CoffeaFileSpec):
    steps: list[list[int]] | None
    num_entriees: int | None
    uuid: str | None


@dataclass
class DatasetSpec:
    files: dict[str, CoffeaFileSpec]
    metadata: dict[Hashable, Any] | None
    form: str | None


@dataclass
class DatasetSpecOptional(DatasetSpec):
    files: (
        dict[str, str] | list[str] | dict[str, UprootFileSpec | CoffeaFileSpecOptional]
    )


FilesetSpecOptional = Dict[str, DatasetSpecOptional]
FilesetSpec = Dict[str, DatasetSpec]


def _normalize_file_info(file_info):
    normed_files = None
    if isinstance(file_info, list) or (
        isinstance(file_info, dict) and "files" not in file_info
    ):
        normed_files = uproot._util.regularize_files(file_info, steps_allowed=True)
    elif isinstance(file_info, dict) and "files" in file_info:
        normed_files = uproot._util.regularize_files(
            file_info["files"], steps_allowed=True
        )

    for ifile in range(len(normed_files)):
        maybe_finfo = None
        if isinstance(file_info, dict) and "files" not in file_info:
            maybe_finfo = file_info.get(normed_files[ifile][0], None)
        elif isinstance(file_info, dict) and "files" in file_info:
            maybe_finfo = file_info["files"].get(normed_files[ifile][0], None)
        maybe_uuid = (
            None if not isinstance(maybe_finfo, dict) else maybe_finfo.get("uuid", None)
        )
        this_file = normed_files[ifile]
        this_file += (4 - len(this_file)) * (None,) + (maybe_uuid,)
        normed_files[ifile] = this_file
    return normed_files


_trivial_file_fields = {"run", "luminosityBlock", "event"}


def preprocess(
    fileset: FilesetSpecOptional,
    step_size: None | int = None,
    align_clusters: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = False,
    scheduler: None | Callable | str = None,
    uproot_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
) -> tuple[FilesetSpec, FilesetSpecOptional]:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    Parameters
    ----------
        fileset: FilesetSpecOptional
            The set of datasets whose files will be preprocessed.
        step_size: int | None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters: bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps: bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        skip_bad_files: bool, False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions: Exception | Warning | tuple[Exception | Warning], default (FileNotFoundError, OSError)
            What exceptions to catch when skipping bad files.
        save_form: bool, default False
            Extract the form of the TTree from each file in each dataset, creating the union of the forms over the dataset.
        scheduler: None | Callable | str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        uproot_options: dict, default {}
            Options to pass to get_steps for opening files with uproot.
        step_size_safety_factor: float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets: bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
    Returns
    -------
        out_available : FilesetSpec
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : FilesetSpecOptional
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    out_updated = copy.deepcopy(fileset)
    out_available = copy.deepcopy(fileset)

    all_ak_norm_files = {}
    files_to_preprocess = {}
    for name, info in fileset.items():
        norm_files = _normalize_file_info(info)
        fields = ["file", "object_path", "steps", "num_entries", "uuid"]
        ak_norm_files = awkward.from_iter(norm_files)
        ak_norm_files = awkward.Array(
            {field: ak_norm_files[str(ifield)] for ifield, field in enumerate(fields)}
        )
        all_ak_norm_files[name] = ak_norm_files

        dak_norm_files = dask_awkward.from_awkward(
            ak_norm_files, math.ceil(len(ak_norm_files) / files_per_batch)
        )

        concat_fn = partial(
            awkward.concatenate,
            axis=0,
        )

        split_every = 8

        files_trl_label = f"preprocess-{name}"
        files_trl_token = dask.base.tokenize(dak_norm_files, concat_fn, split_every)
        files_trl_name = f"{files_trl_label}-{files_trl_token}"
        files_trl_tree_node_name = f"{files_trl_label}-tree-node-{files_trl_token}"

        files_part = dask_awkward.map_partitions(
            get_steps,
            dak_norm_files,
            step_size=step_size,
            align_clusters=align_clusters,
            recalculate_steps=recalculate_steps,
            skip_bad_files=skip_bad_files,
            file_exceptions=file_exceptions,
            save_form=save_form,
            step_size_safety_factor=step_size_safety_factor,
            uproot_options=uproot_options,
            meta=dask_awkward.lib.core.empty_typetracer(),
        )

        files_trl = dask_awkward.layers.layers.AwkwardTreeReductionLayer(
            name=files_trl_name,
            name_input=files_part.name,
            npartitions_input=files_part.npartitions,
            concat_func=concat_fn,
            tree_node_func=lambda x: x,
            finalize_func=lambda x: x,
            split_every=split_every,
            tree_node_name=files_trl_tree_node_name,
        )

        files_graph = dask.highlevelgraph.HighLevelGraph.from_collections(
            files_trl_name, files_trl, dependencies=[files_part]
        )

        files_to_preprocess[name] = dask_awkward.lib.core.new_array_object(
            files_graph,
            files_trl_name,
            meta=dask_awkward.lib.core.empty_typetracer(),
            npartitions=len(files_trl.output_partitions),
        )

    (all_processed_files,) = dask.compute(files_to_preprocess, scheduler=scheduler)

    for name, processed_files in all_processed_files.items():

        if len(awkward.drop_none(processed_files, axis=0)) == 0:
            ds_empty_msg = (
                "There was no populated list of files returned from querying your input dataset."
                "\nPlease check your xrootd endpoints, and avoid redirectors."
                f"\nInput dataset: {name}"
                f"\nAs parsed for querying: {awkward.to_list(all_ak_norm_files[name])}"
            )

            if not allow_empty_datasets:
                raise Exception(ds_empty_msg)

            warnings.warn(ds_empty_msg)
            del out_available[name]
            continue

        processed_files_without_forms = processed_files[
            ["file", "object_path", "steps", "num_entries", "uuid"]
        ]

        forms = processed_files[["file", "form", "form_hash_md5", "num_entries"]][
            ~awkward.is_none(processed_files.form_hash_md5)
        ]

        _, unique_forms_idx = numpy.unique(
            forms.form_hash_md5.to_numpy(), return_index=True
        )

        dataset_forms = []
        unique_forms = forms[unique_forms_idx]
        for thefile, formstr, num_entries in zip(
            unique_forms.file, unique_forms.form, unique_forms.num_entries
        ):
            # skip trivially filled or empty files
            form = awkward.forms.from_json(decompress_form(formstr))
            if num_entries >= 0 and set(form.fields) != _trivial_file_fields:
                dataset_forms.append(form)
            else:
                warnings.warn(
                    f"{thefile} has fields {form.fields} and num_entries={num_entries} "
                    "and has been skipped during form-union determination. You will need "
                    "to skip this file when processing. You can either manually remove it "
                    "or, if it is an empty file, dynamically remove it with the function "
                    "dataset_tools.filter_files which takes the output of preprocess and "
                    ", by default, removes empty files each dataset in a fileset."
                )

        union_array = None
        union_form_jsonstr = None
        while len(dataset_forms):
            new_array = awkward.Array(dataset_forms.pop().length_zero_array())
            if union_array is None:
                union_array = new_array
            else:
                union_array = awkward.to_packed(
                    awkward.merge_union_of_records(
                        awkward.concatenate([union_array, new_array]), axis=0
                    )
                )
                union_array.layout.parameters.update(new_array.layout.parameters)
        if union_array is not None:
            union_form = union_array.layout.form

            for icontent, content in enumerate(union_form.contents):
                if isinstance(content, awkward.forms.IndexedOptionForm):
                    if (
                        not isinstance(content.content, awkward.forms.NumpyForm)
                        or content.content.primitive != "bool"
                    ):
                        raise ValueError(
                            "IndexedOptionArrays can only contain NumpyArrays of "
                            "bools in mergers of flat-tuple-like schemas!"
                        )
                    parameters = (
                        content.content.parameters.copy()
                        if content.content.parameters is not None
                        else {}
                    )
                    # re-create IndexOptionForm with parameters of lower level array
                    union_form.contents[icontent] = awkward.forms.IndexedOptionForm(
                        content.index,
                        content.content,
                        parameters=parameters,
                        form_key=content.form_key,
                    )

            union_form_jsonstr = union_form.to_json()

        files_available = {
            item["file"]: {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "num_entries": item["num_entries"],
                "uuid": item["uuid"],
            }
            for item in awkward.drop_none(processed_files_without_forms).to_list()
        }

        files_out = {}
        for proc_item, orig_item in zip(
            processed_files_without_forms.to_list(), all_ak_norm_files[name].to_list()
        ):
            item = orig_item if proc_item is None else proc_item
            files_out[item["file"]] = {
                "object_path": item["object_path"],
                "steps": item["steps"],
                "num_entries": item["num_entries"],
                "uuid": item["uuid"],
            }

        if "files" in out_updated[name]:
            out_updated[name]["files"] = files_out
            out_available[name]["files"] = files_available
        else:
            out_updated[name] = {"files": files_out, "metadata": None, "form": None}
            out_available[name] = {
                "files": files_available,
                "metadata": None,
                "form": None,
            }

        compressed_union_form = None
        if union_form_jsonstr is not None:
            compressed_union_form = compress_form(union_form_jsonstr)
            out_updated[name]["form"] = compressed_union_form
            out_available[name]["form"] = compressed_union_form
        else:
            out_updated[name]["form"] = None
            out_available[name]["form"] = None

        if "metadata" not in out_updated[name]:
            out_updated[name]["metadata"] = None
            out_available[name]["metadata"] = None

    return out_available, out_updated
