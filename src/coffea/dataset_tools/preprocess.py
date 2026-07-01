from __future__ import annotations

import copy
import hashlib
import math
import warnings
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dask_awkward

import awkward
import numpy
import uproot
from uproot._util import no_filter

try:
    # Private uproot helper that builds a TTree's awkward form without dask. It is exactly what
    # uproot.dask() uses internally, so the result is byte-identical to
    # uproot.dask(tree).layout.form (verified in the test suite). Guarded so a uproot release
    # that moves or renames it degrades gracefully to the dask-based path below.
    from uproot._dask import _get_ttree_form as _uproot_get_ttree_form
except Exception:  # pragma: no cover - depends on uproot internals
    _uproot_get_ttree_form = None

from coffea.dataset_tools.backends import (
    DaskBackend,
    PreprocessBackend,
    PreprocessJob,
    print_dask_backend_fallback_hint,
    resolve_backend,
)
from coffea.dataset_tools.filespec import (
    DataGroupSpec,
    DatasetSpec,
    ModelFactory,
)
from coffea.util import (
    _import_dask,
    _import_dask_awkward,
    _is_interpretable,
    compress_form,
    decompress_form,
)


def _even_steps(num_entries: int, target_step_size: int) -> numpy.ndarray:
    """Split ``num_entries`` into as-even-as-possible ``[start, stop]`` steps of ~target size."""
    n_steps_target = max(round(num_entries / target_step_size), 1)
    actual_step_size = math.ceil(num_entries / n_steps_target)
    return numpy.array(
        [
            [i * actual_step_size, min((i + 1) * actual_step_size, num_entries)]
            for i in range(n_steps_target)
        ],
        dtype="int64",
    )


def _aligned_steps(
    boundaries,
    target_step_size: int,
    step_size_safety_factor: float,
    file_label: str,
    mode_label: str,
) -> numpy.ndarray:
    """Build ``[start, stop]`` steps that snap to natural boundaries (TTree clusters, RNTuple
    cluster summaries, or parquet row groups).

    ``boundaries`` is the increasing sequence of absolute entry offsets at which a step is
    allowed to end, with the final element equal to ``num_entries``. Steps accumulate
    boundaries until at least ``target_step_size`` entries are covered. ``mode_label`` is the
    name of the user-facing option (``align_clusters`` or ``use_row_groups``) used in the
    over-size warning.
    """
    out = [0]
    for c in boundaries:
        if c >= out[-1] + target_step_size:
            out.append(c)
    if boundaries[-1] != out[-1]:
        out.append(boundaries[-1])
    out = numpy.array(out, dtype="int64")
    out = numpy.stack((out[:-1], out[1:]), axis=1)

    step_mask = out[:, 1] - out[:, 0] > (1 + step_size_safety_factor) * target_step_size
    if numpy.any(step_mask):
        warnings.warn(
            f"In file {file_label}, steps: {out[step_mask]} with {mode_label}=True are "
            f"{step_size_safety_factor*100:.0f}% larger than target "
            f"step size: {target_step_size}!"
        )
    return out


def _rntuple_cluster_boundaries(rntuple, num_entries: int) -> list[int]:
    """Absolute entry offsets at RNTuple cluster boundaries, terminating at ``num_entries``."""
    boundaries = [cluster.num_first_entry for cluster in rntuple.cluster_summaries]
    boundaries.append(num_entries)
    return boundaries


def _union_form_jsonstr(forms: list) -> str | None:
    """Compute the union form (as a JSON string) over a list of awkward forms.

    The input list is consumed. Returns None if the list is empty. Mirrors the merging of
    flat-tuple-like schemas used when building a dataset's union form across files.
    """
    union_array = None
    while len(forms):
        new_array = awkward.Array(forms.pop().length_zero_array())
        if union_array is None:
            union_array = new_array
        else:
            union_array = awkward.to_packed(
                awkward.merge_union_of_records(
                    awkward.concatenate([union_array, new_array]), axis=0
                )
            )
            union_array.layout.parameters.update(new_array.layout.parameters)
    if union_array is None:
        return None

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
    return union_form.to_json()


_FORM_AK_ADD_DOC = {"__doc__": "title", "typename": "typename"}


def _ttree_form_json(tree) -> str:
    """Build a TTree's awkward form JSON without importing dask.

    Reuses uproot's own form builder (``uproot._dask._get_ttree_form``), so the output is
    byte-identical to ``uproot.dask(tree).layout.form.to_json()``. This is what lets the
    ``iterative``/``futures`` backends extract TTree forms in a dask-free environment.
    """
    common_keys = tree.keys(
        recursive=True,
        filter_name=no_filter,
        filter_typename=no_filter,
        filter_branch=partial(_is_interpretable, emit_warning=False),
        ignore_duplicates=True,
    )
    base_form = _uproot_get_ttree_form(awkward, tree, common_keys, _FORM_AK_ADD_DOC)
    return base_form.to_json()


def _dask_form_json(uproot_target, is_rntuple: bool) -> str:
    """Build a form JSON via ``uproot.dask`` (requires dask).

    Used for RNTuples -- whose form the dask path derives with transforms (dropping
    ``_collection*`` subfields, nulling form keys, applying ``ak_add_doc``) that are not cheaply
    reproducible without dask -- and as a fallback for TTrees when the private uproot form
    builder is unavailable. ``uproot_target`` is an already-open TTree object, or a
    ``{file: object_path}`` mapping (required for RNTuples, which cannot build a form from an
    already-open object).
    """
    if is_rntuple:
        form_dask = uproot.dask(
            uproot_target,
            open_files=False,
            full_paths=True,
            ak_add_doc=_FORM_AK_ADD_DOC,
            filter_name=no_filter,
            filter_typename=no_filter,
            filter_branch=partial(_is_interpretable, emit_warning=False),
        )
    else:
        form_dask = uproot.dask(
            uproot_target,
            ak_add_doc=_FORM_AK_ADD_DOC,
            filter_name=no_filter,
            filter_typename=no_filter,
            filter_branch=partial(_is_interpretable, emit_warning=False),
        )
    form_str = form_dask.layout.form.to_json()
    # the function cache needs to be popped if present to prevent memory growth
    dask = _import_dask()
    if getattr(dask.base, "function_cache", None):
        dask.base.function_cache.popitem()
    return form_str


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
    legacy_form_key: bool = True,
    require_rntuple: bool = False,
) -> awkward.Array | dask_awkward.Array:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    Parameters
    ----------
        normed_files : awkward.Array or dask_awkward.Array
            The list of normalized file descriptions to process for steps.
        step_size : int or None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters : bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps, instead
            of only recalculating the steps if the uuid has changed.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception or Warning or tuple[Exception or Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default False
            Extract the form of the TTree from the file so we can skip opening files later.
        step_size_safety_factor : float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        legacy_form_key : bool, default True
            Use "form" for the compressed form key in the output for backwards compatibility.
            Set to False to use "compressed_form" instead.
        require_rntuple : bool, default False
            If True, require every object to be an RNTuple and raise a ValueError otherwise.
            If False, TTree and RNTuple objects are auto-detected and handled transparently.

    Returns
    -------
        array : awkward.Array or dask_awkward.Array
            The normalized file descriptions, appended with the calculated steps for those files.
    """
    nf_backend = awkward.backend(normed_files)
    lz_or_nf = awkward.typetracer.length_zero_if_typetracer(normed_files)
    output_form_key = "form" if legacy_form_key else "compressed_form"

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

        is_rntuple = isinstance(tree, uproot.behaviors.RNTuple.HasFields)
        if require_rntuple and not is_rntuple:
            raise ValueError(
                f"require_rntuple=True but {arg.object_path!r} in {arg.file!r} is a "
                f"{type(tree).__name__}, not an RNTuple."
            )

        num_entries = tree.num_entries

        form_json = None
        form_hash = None
        if save_form:
            if not is_rntuple and _uproot_get_ttree_form is not None:
                # dask-free TTree form extraction; byte-identical to the uproot.dask form
                form_str = _ttree_form_json(tree)
            elif is_rntuple:
                # RNTuple form extraction still goes through uproot.dask (requires dask)
                form_str = _dask_form_json({arg.file: arg.object_path}, is_rntuple=True)
            else:
                # uproot without the private form builder: fall back to the dask-based path
                form_str = _dask_form_json(tree, is_rntuple=False)

            form_hash = hashlib.md5(form_str.encode("utf-8")).hexdigest()
            form_json = compress_form(form_str)

        target_step_size = num_entries if step_size is None else step_size

        file_uuid = str(the_file.file.uuid)

        out_uuid = arg.uuid
        out_steps = arg.steps

        if num_entries == 0:
            array.append(
                {
                    "file": arg.file,
                    "object_path": arg.object_path,
                    "steps": [[0, 0]],
                    "num_entries": num_entries,
                    "uuid": file_uuid,
                    output_form_key: form_json,
                    "form_hash_md5": form_hash,
                }
            )
            continue

        if out_uuid != file_uuid or recalculate_steps:
            if align_clusters:
                if is_rntuple:
                    boundaries = _rntuple_cluster_boundaries(tree, num_entries)
                else:
                    boundaries = tree.common_entry_offsets()
                out = _aligned_steps(
                    boundaries,
                    target_step_size,
                    step_size_safety_factor,
                    arg.file,
                    "align_clusters",
                )
            else:
                out = _even_steps(num_entries, target_step_size)

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
                output_form_key: form_json,
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
                    output_form_key: "junk",
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


def _normalize_file_info(file_info):
    normed_files = None
    if isinstance(file_info, DatasetSpec):
        normed_files = uproot._util.regularize_files(
            ModelFactory.datasetspec_to_dict(file_info, coerce_filespec_to_dict=True)[
                "files"
            ],
            steps_allowed=True,
        )
    elif isinstance(file_info, list) or (
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


def preprocess_legacy(
    fileset: dict,
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
) -> tuple[dict, dict]:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    Parameters
    ----------
        fileset : dict
            The set of datasets whose files will be preprocessed.
        step_size : int | None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters : bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        files_per_batch : int, default 1
            The number of files to preprocess in a single batch.
            Large values will result in fewer dask tasks but each task will have to do more work.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception | Warning | tuple[Exception | Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default False
            Extract the form of the TTree from each file in each dataset, creating the union of the forms over the dataset.
        scheduler : None | Callable | str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        uproot_options : dict, default {}
            Options to pass to get_steps for opening files with uproot.
        step_size_safety_factor : float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets : bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
    Returns
    -------
        out_available : dict
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : dict
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    dask = _import_dask()
    dask_awkward = _import_dask_awkward()

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
            legacy_form_key=True,  # for backwards compatibility, the output form key is always "form" in this legacy preprocess function
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
            if set(form.fields) != _trivial_file_fields:
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

        union_form_jsonstr = _union_form_jsonstr(dataset_forms)

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


def _normalize_pydantic_file_info(datasetspec: DatasetSpec):
    """
    Structure file info akin to _normalize_file_info for uproot files, which returns a list of (filename, object_path, steps, num_entries, uuid) tuples.
    """
    if not isinstance(datasetspec, DatasetSpec):
        raise ValueError(
            f"_normalize_pydantic_file_info expects a DatasetSpec, got {type(datasetspec)}"
        )
    normed_files = []
    for filename, fileinfo in datasetspec.files.items():
        normed_files.append(
            (
                filename,
                fileinfo.object_path,
                fileinfo.steps,
                fileinfo.num_entries,
                fileinfo.uuid,
            )
        )
    return normed_files


def get_parquet_form_uuid_steps(
    normed_files: awkward.Array | dask_awkward.Array,
    step_size: int | None = None,
    use_row_groups: bool = False,
    recalculate_steps: bool = False,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = False,
    step_size_safety_factor: float = 0.5,
    parquet_options: dict = {},
) -> awkward.Array | dask_awkward.Array:
    """
    Given a list of normalized file and object paths, determine the form, steps, uuid for each file according to the supplied processing options.

    Parameters
    ----------
        normed_files : awkward.Array | dask_awkward.Array
            The list of normalized file descriptions to process for steps.
        step_size : int | None, default None
            If specified, the size of the steps to make when analyzing the input files.
        use_row_groups : bool, default False
            Calculate steps according to the row_groups in the parquet files.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps, instead
            of only recalculating the steps if the uuid has changed.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception | Warning | tuple[Exception | Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default False
            Extract the form from the parquet metadata so we can skip opening files later.
        step_size_safety_factor : float, default 0.5
            When using use_row_groups, if a resulting step is larger than step_size by this factor
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
            the_file = awkward.metadata_from_parquet(arg.file, **parquet_options)
        except file_exceptions as e:
            if skip_bad_files:
                array.append(None)
                continue
            else:
                raise e

        num_entries = the_file["num_rows"]

        form_json = None
        form_hash = None
        if save_form:
            # parquet metadata already carries the form; reading it builds no dask graph,
            # so (unlike the TTree/RNTuple path) there is no function cache to pop here.
            form = the_file["form"]
            form_str = form.to_json()

            form_hash = hashlib.md5(form_str.encode("utf-8")).hexdigest()
            form_json = compress_form(form_str)

        target_step_size = num_entries if step_size is None else step_size

        file_uuid = the_file.get("uuid", None)

        # Mirror the ROOT get_steps num_entries==0 guard: a 0-row file would otherwise reach
        # _even_steps(0, 0) (division by zero) or _aligned_steps on empty row-group boundaries.
        if num_entries == 0:
            array.append(
                {
                    "file": arg.file,
                    "object_path": arg.object_path,
                    "steps": [[0, 0]],
                    "num_entries": num_entries,
                    "uuid": file_uuid,
                    "compressed_form": form_json,
                    "form_hash_md5": form_hash,
                }
            )
            continue

        out_uuid = arg.uuid
        out_steps = arg.steps

        if num_entries == 0:
            array.append(
                {
                    "file": arg.file,
                    "object_path": arg.object_path,
                    "steps": [[0, 0]],
                    "num_entries": num_entries,
                    "uuid": file_uuid,
                    "compressed_form": form_json,
                    "form_hash_md5": form_hash,
                }
            )
            continue

        if out_uuid != file_uuid or recalculate_steps:
            if use_row_groups:
                # cumulative row counts give the absolute offset at each row-group boundary
                boundaries = numpy.cumsum(the_file["col_counts"]).tolist()
                out = _aligned_steps(
                    boundaries,
                    target_step_size,
                    step_size_safety_factor,
                    arg.file,
                    "use_row_groups",
                )
            else:
                out = _even_steps(num_entries, target_step_size)

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
                "compressed_form": form_json,
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
                    "compressed_form": "junk",
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


def preprocess_root(
    datagroupspec: DataGroupSpec,
    step_size: None | int = None,
    align_clusters: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = True,
    scheduler: None | Callable | str = None,
    uproot_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
    backend: str | PreprocessBackend = "dask",
) -> tuple[DataGroupSpec, DataGroupSpec]:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    Both TTree and RNTuple objects are auto-detected and handled; use :func:`preprocess_rntuple`
    if you want to require that every object is an RNTuple.

    Parameters
    ----------
        datagroupspec : DataGroupSpec
            The set of datasets whose files will be preprocessed.
        step_size : int or None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters : bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        files_per_batch : int, default 1
            The number of files to preprocess in a single batch.
            Large values will result in fewer dask tasks but each task will have to do more work.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception or Warning or tuple[Exception or Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default True
            Extract the form of the TTree from each file in each dataset, creating the union of the forms over the dataset.
        scheduler : None or Callable or str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        uproot_options : dict, default {}
            Options to pass to get_steps for opening files with uproot.
        step_size_safety_factor : float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets : bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
        backend : str or PreprocessBackend, default "dask"
            Execution backend for preprocessing: "dask" (default), "iterative" (immediate,
            synchronous, dask-free), "futures" (dask-free concurrent.futures thread pool), or a
            PreprocessBackend instance. The ``scheduler`` argument only affects the dask backend.
    Returns
    -------
        out_available : DataGroupSpec
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : DataGroupSpec
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    return _preprocess_pydantic(
        datagroupspec=datagroupspec,
        step_size=step_size,
        use_alignment_boundaries=align_clusters,
        recalculate_steps=recalculate_steps,
        files_per_batch=files_per_batch,
        skip_bad_files=skip_bad_files,
        file_exceptions=file_exceptions,
        save_form=save_form,
        scheduler=scheduler,
        filetype_options=uproot_options,
        step_size_safety_factor=step_size_safety_factor,
        allow_empty_datasets=allow_empty_datasets,
        backend=backend,
    )


def preprocess_rntuple(
    datagroupspec: DataGroupSpec,
    step_size: None | int = None,
    align_clusters: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = True,
    scheduler: None | Callable | str = None,
    uproot_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
    backend: str | PreprocessBackend = "dask",
) -> tuple[DataGroupSpec, DataGroupSpec]:
    """
    Preprocess datasets of ROOT files containing RNTuples, determining the steps for each file.

    This is the RNTuple-specific counterpart to :func:`preprocess_root` (which is implicitly
    TTree-oriented) and :func:`preprocess_parquet`. It requires every object to be an RNTuple
    and raises a ValueError if a TTree is encountered, making it useful to assert the intended
    file type. ``preprocess`` and ``preprocess_root`` already auto-detect and handle RNTuples
    transparently, so this function is primarily for RNTuple-only workflows that want the
    stricter contract.

    Parameters
    ----------
        datagroupspec : DataGroupSpec
            The set of datasets whose files will be preprocessed.
        step_size : int or None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters : bool, default False
            Round to the RNTuple cluster boundaries when chunks are specified. Reduces data
            transfer in analysis.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        files_per_batch : int, default 1
            The number of files to preprocess in a single batch.
            Large values will result in fewer dask tasks but each task will have to do more work.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception or Warning or tuple[Exception or Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default True
            Extract the form of the RNTuple from each file in each dataset, creating the union of the forms over the dataset.
        scheduler : None or Callable or str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        uproot_options : dict, default {}
            Options to pass to get_steps for opening files with uproot.
        step_size_safety_factor : float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets : bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
        backend : str or PreprocessBackend, default "dask"
            Execution backend for preprocessing: "dask" (default), "iterative" (immediate,
            synchronous, dask-free), "futures" (dask-free concurrent.futures thread pool), or a
            PreprocessBackend instance. The ``scheduler`` argument only affects the dask backend.
    Returns
    -------
        out_available : DataGroupSpec
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : DataGroupSpec
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    return _preprocess_pydantic(
        datagroupspec=datagroupspec,
        step_size=step_size,
        use_alignment_boundaries=align_clusters,
        recalculate_steps=recalculate_steps,
        files_per_batch=files_per_batch,
        skip_bad_files=skip_bad_files,
        file_exceptions=file_exceptions,
        save_form=save_form,
        scheduler=scheduler,
        filetype_options=uproot_options,
        step_size_safety_factor=step_size_safety_factor,
        allow_empty_datasets=allow_empty_datasets,
        require_rntuple=True,
        backend=backend,
    )


def preprocess_parquet(
    datagroupspec: DataGroupSpec,
    step_size: None | int = None,
    use_row_groups: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = True,
    scheduler: None | Callable | str = None,
    parquet_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
    backend: str | PreprocessBackend = "dask",
) -> tuple[DataGroupSpec, DataGroupSpec]:
    """
    Given a list of normalized files, determine the form, steps, and add the metadata for each file according to the supplied processing options.

    Parameters
    ----------
        datagroupspec : DataGroupSpec
            The set of datasets whose files will be preprocessed.
        step_size : int | None, default None
            If specified, the size of the steps to make when analyzing the input files.
        use_row_groups : bool, default False
            Use the row groups in the parquet files to determine the steps.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception | Warning | tuple[Exception | Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool, default True
            Extract the form of each file in each dataset, creating the union of the forms over the dataset.
        scheduler : None | Callable | str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        parquet_options : dict, default {}
            Options to pass to get_parquet_form_uuid_steps for opening files
        step_size_safety_factor : float, default 0.5
            When using use_row_groups, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets : bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
        backend : str or PreprocessBackend, default "dask"
            Execution backend for preprocessing: "dask" (default), "iterative" (immediate,
            synchronous, dask-free), "futures" (dask-free concurrent.futures thread pool), or a
            PreprocessBackend instance. The ``scheduler`` argument only affects the dask backend.
    Returns
    -------
        out_available : DataGroupSpec
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : DataGroupSpec
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    return _preprocess_pydantic(
        datagroupspec=datagroupspec,
        step_size=step_size,
        use_alignment_boundaries=use_row_groups,
        recalculate_steps=recalculate_steps,
        files_per_batch=files_per_batch,
        skip_bad_files=skip_bad_files,
        file_exceptions=file_exceptions,
        save_form=save_form,
        scheduler=scheduler,
        filetype_options=parquet_options,
        step_size_safety_factor=step_size_safety_factor,
        allow_empty_datasets=allow_empty_datasets,
        backend=backend,
    )


def _preprocess_pydantic(
    datagroupspec: DataGroupSpec,
    step_size: None | int = None,
    use_alignment_boundaries: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: bool = True,
    scheduler: None | Callable | str = None,
    filetype_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
    require_rntuple: bool = False,
    backend: str | PreprocessBackend = "dask",
) -> tuple[DataGroupSpec, DataGroupSpec]:
    """
    Internal function to preprocess either ROOT or parquet DatasetSpecs in a DataGroupSpec.

    This function dispatches to format-specific processing (ROOT TTrees or parquet files)
    based on the format of each dataset. It extracts file metadata including steps, UUIDs,
    entry counts, and optionally computes the union form across files.

    Parameters
    ----------
        datagroupspec : DataGroupSpec
            The set of datasets whose files will be preprocessed.
        step_size : int or None, default None
            If specified, the size of the steps to make when analyzing the input files.
        use_alignment_boundaries : bool, default False
            For ROOT: align to cluster boundaries. For parquet: align to row groups.
        recalculate_steps : bool, default False
            Force recalculation of steps even if UUID hasn't changed.
        files_per_batch : int, default 1
            Number of files to preprocess in a single dask task.
        skip_bad_files : bool, default False
            Catch file_exceptions and return null data instead of failing.
        file_exceptions : Exception or Warning or tuple, default (OSError,)
            Exceptions to catch when skip_bad_files is True.
        save_form : bool, default True
            Extract and compute the union form across files in each dataset.
        scheduler : None or Callable or str, default None
            Dask scheduler to use for preprocessing.
        filetype_options : dict, default {}
            Options passed to uproot (for ROOT) or awkward (for parquet) when opening files.
        step_size_safety_factor : float, default 0.5
            Warn if aligned steps exceed target by this factor.
        allow_empty_datasets : bool, default False
            If True, warn instead of raising when a dataset has no accessible files.
        backend : str or PreprocessBackend, default "dask"
            Execution backend for the per-dataset map-reduce. One of "dask" (default),
            "iterative" (immediate, synchronous, dask-free), "futures" (a dask-free
            concurrent.futures thread pool), or a PreprocessBackend instance for full control.
            ``scheduler`` only affects the dask backend.

    Returns
    -------
        out_available : DataGroupSpec
            Datasets containing only successfully preprocessed files.
        out_updated : DataGroupSpec
            Original datasets updated with preprocessing results where available.

    Raises
    ------
        ValueError
            If datagroupspec is not a DataGroupSpec or contains unsupported formats.
        Exception
            If a dataset has no accessible files and allow_empty_datasets is False.
    """
    if not isinstance(datagroupspec, DataGroupSpec):
        raise ValueError(
            f"_preprocess_pydantic expects a DataGroupSpec, got {type(datagroupspec)}"
        )
    if len(datagroupspec) == 0:
        return DataGroupSpec({}), DataGroupSpec({})

    out_updated = datagroupspec.model_dump()
    out_available = datagroupspec.model_dump()

    # Build one map-reduce job per dataset. The map worker (get_steps /
    # get_parquet_form_uuid_steps) and the concatenating reduce are backend-agnostic; only the
    # execution strategy (dask graph vs. futures vs. synchronous) is selected via `backend`.
    all_ak_norm_files = {}
    jobs = {}
    for name, info in datagroupspec.items():
        norm_files = _normalize_pydantic_file_info(info)
        fields = ["file", "object_path", "steps", "num_entries", "uuid"]
        ak_norm_files = awkward.from_iter(norm_files)
        ak_norm_files = awkward.Array(
            {field: ak_norm_files[str(ifield)] for ifield, field in enumerate(fields)}
        )
        all_ak_norm_files[name] = ak_norm_files

        if info.format == "root":
            map_fn = partial(
                get_steps,
                step_size=step_size,
                align_clusters=use_alignment_boundaries,
                recalculate_steps=recalculate_steps,
                skip_bad_files=skip_bad_files,
                file_exceptions=file_exceptions,
                save_form=save_form,
                step_size_safety_factor=step_size_safety_factor,
                legacy_form_key=False,  # in the pydantic preprocess function, the output form key is always "compressed_form", "form" is a method to extract the uncompressed form
                uproot_options=filetype_options,
                require_rntuple=require_rntuple,
            )
        elif info.format == "parquet":
            map_fn = partial(
                get_parquet_form_uuid_steps,
                step_size=step_size,
                use_row_groups=use_alignment_boundaries,
                recalculate_steps=recalculate_steps,
                skip_bad_files=skip_bad_files,
                file_exceptions=file_exceptions,
                save_form=save_form,
                step_size_safety_factor=step_size_safety_factor,
                parquet_options=filetype_options,
            )
        else:
            raise ValueError(
                f"Dataset {name} has unsupported format {info.format}, supported formats are 'root' and 'parquet'."
            )

        jobs[name] = PreprocessJob(
            array=ak_norm_files, map_fn=map_fn, files_per_batch=files_per_batch
        )

    backend_obj = resolve_backend(backend, scheduler)
    # Only submit() imports dask (for the dask backend); a ModuleNotFoundError here means the
    # dask stack itself is missing, so we can point the user at the dask-free backends. Worker
    # ImportErrors (e.g. a missing codec) surface later in result() and must NOT trigger the
    # dask hint, so result() is called outside this guard.
    try:
        preprocess_task = backend_obj.submit(jobs)
    except ModuleNotFoundError:
        if isinstance(backend_obj, DaskBackend):
            print_dask_backend_fallback_hint()
        raise
    all_processed_files = preprocess_task.result()

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

        compressed_forms = processed_files[
            ["file", "compressed_form", "form_hash_md5", "num_entries"]
        ][~awkward.is_none(processed_files.form_hash_md5)]

        _, unique_forms_idx = numpy.unique(
            compressed_forms.form_hash_md5.to_numpy(), return_index=True
        )

        dataset_forms = []
        unique_forms = compressed_forms[unique_forms_idx]
        for thefile, formstr, num_entries in zip(
            unique_forms.file, unique_forms.compressed_form, unique_forms.num_entries
        ):
            # skip trivially filled or empty files
            form = awkward.forms.from_json(decompress_form(formstr))
            if set(form.fields) != _trivial_file_fields:
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

        union_form_jsonstr = _union_form_jsonstr(dataset_forms)

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

        out_updated[name]["files"] = files_out
        out_available[name]["files"] = files_available

        compressed_union_form = (
            compress_form(union_form_jsonstr) if union_form_jsonstr else None
        )
        out_updated[name]["compressed_form"] = compressed_union_form
        out_available[name]["compressed_form"] = compressed_union_form

    return DataGroupSpec.model_validate(out_available), DataGroupSpec.model_validate(
        out_updated
    )


def _advertise_datagroupspec() -> None:
    """Print a friendly advertisement for the pydantic DataGroupSpec API to the coffea console."""
    from coffea.util import coffea_console

    coffea_console.print(
        "[bold cyan]coffea.dataset_tools.preprocess[/]: you passed a plain dict fileset. "
        "Dict-in / dict-out is still fully supported, but the pydantic "
        "[bold]DataGroupSpec[/] API offers input validation, mixed ROOT+parquet handling, "
        "form management, and richer fileset manipulation. Consider building a "
        "[bold]coffea.dataset_tools.DataGroupSpec[/] for new code.",
        style="dim",
    )


def _datagroupspec_to_dict(datagroupspec: DataGroupSpec) -> dict:
    """Convert a DataGroupSpec back to a plain (JSON-serializable) dict fileset."""
    return datagroupspec.model_dump()


def preprocess(
    fileset: DataGroupSpec | dict,
    step_size: None | int = None,
    align_clusters: bool = False,
    recalculate_steps: bool = False,
    files_per_batch: int = 1,
    skip_bad_files: bool = False,
    file_exceptions: Exception | Warning | tuple[Exception | Warning] = (OSError,),
    save_form: None | bool = None,
    scheduler: None | Callable | str = None,
    uproot_options: dict = {},
    step_size_safety_factor: float = 0.5,
    allow_empty_datasets: bool = False,
    preprocess_legacy_root: bool = False,
    use_row_groups: bool = False,
    parquet_options: dict = {},
    backend: str | PreprocessBackend = "dask",
) -> tuple[DataGroupSpec, DataGroupSpec] | tuple[dict, dict]:
    """
    Given a list of normalized file and object paths (defined in uproot), determine the steps for each file according to the supplied processing options.

    The return type matches the input type: passing a ``DataGroupSpec`` returns a
    tuple of ``DataGroupSpec``; passing a plain dict returns a tuple of dicts.

    Parameters
    ----------
        fileset : DataGroupSpec | dict
            The set of datasets whose files will be preprocessed.
        step_size : int or None, default None
            If specified, the size of the steps to make when analyzing the input files.
        align_clusters : bool, default False
            Round to the cluster size in a root file, when chunks are specified. Reduces data transfer in
            analysis.
        recalculate_steps : bool, default False
            If steps are present in the input normed files, force the recalculation of those steps,
            instead of only recalculating the steps if the uuid has changed.
        files_per_batch : int, default 1
            The number of files to preprocess in a single batch.
            Large values will result in fewer dask tasks but each task will have to do more work.
        skip_bad_files : bool, default False
            Instead of failing, catch exceptions specified by file_exceptions and return null data.
        file_exceptions : Exception or Warning or tuple[Exception or Warning], default (OSError,)
            What exceptions to catch when skipping bad files.
        save_form : bool or None, default None
            Extract the form of each file in each dataset, creating the union of the forms over the dataset.
            When None (the default), the pydantic preprocessing path uses True and the legacy path
            (preprocess_legacy_root=True) uses False, preserving each path's historical default.
        scheduler : None or Callable or str, default None
            Specifies the scheduler that dask should use to execute the preprocessing task graph.
        uproot_options : dict, default {}
            Options to pass to get_steps for opening files with uproot.
        step_size_safety_factor : float, default 0.5
            When using align_clusters, if a resulting step is larger than step_size by this factor
            warn the user that the resulting steps may be highly irregular.
        allow_empty_datasets : bool, default False
            When a dataset query comes back completely empty, this is normally considered a processing error.
            Toggle this argument to True to change this to warnings and allow incomplete returned filesets.
        preprocess_legacy_root : bool, default False
            Use the legacy root preprocessing function for all files, even if the fileset is a DataGroupSpec.
            Not compatible with parquet files.
        use_row_groups : bool, default False
            Calculate steps according to the row_groups in the parquet files (only applies to DataGroupSpec datasets with parquet files).
        parquet_options : dict, default {}
            Options to pass to get_parquet_form_uuid_steps for opening parquet files (only applies to DataGroupSpec datasets with parquet files).
        backend : str or PreprocessBackend, default "dask"
            Execution backend for preprocessing: "dask" (default), "iterative" (immediate,
            synchronous, dask-free), "futures" (dask-free concurrent.futures thread pool), or a
            PreprocessBackend instance. The ``scheduler`` argument only affects the dask backend.
            Ignored when ``preprocess_legacy_root=True`` (the legacy path is always dask-based).
    Returns
    -------
        out_available : DataGroupSpec | dict
            The subset of files in each dataset that were successfully preprocessed, organized by dataset.
        out_updated : DataGroupSpec | dict
            The original set of datasets including files that were not accessible, updated to include the result of preprocessing where available.
    """
    input_is_dict = not isinstance(fileset, DataGroupSpec)

    if preprocess_legacy_root:
        # use the legacy root TTree preprocessing function if requested;
        # the legacy path historically defaulted to save_form=False
        legacy_save_form = False if save_form is None else save_form
        if isinstance(fileset, DataGroupSpec):
            fileset_input = fileset.model_dump()
            for k in fileset_input.keys():
                form = fileset_input[k].pop("compressed_form", None)
                if "form" not in fileset_input[k] or fileset_input[k]["form"] is None:
                    fileset_input[k]["form"] = form
        else:
            fileset_input = fileset
        return preprocess_legacy(
            fileset_input,
            step_size=step_size,
            align_clusters=align_clusters,
            recalculate_steps=recalculate_steps,
            files_per_batch=files_per_batch,
            skip_bad_files=skip_bad_files,
            file_exceptions=file_exceptions,
            save_form=legacy_save_form,
            scheduler=scheduler,
            uproot_options=uproot_options,
            step_size_safety_factor=step_size_safety_factor,
            allow_empty_datasets=allow_empty_datasets,
        )
    else:
        # the pydantic path historically defaulted to save_form=True
        pydantic_save_form = True if save_form is None else save_form
        if isinstance(fileset, DataGroupSpec):
            datasetspecs = fileset
        else:
            warnings.warn(
                "Passing a dict to preprocess is deprecated. Dict input still returns "
                "dict output for backwards compatibility, but the pydantic DataGroupSpec "
                "API is recommended. To use the legacy preprocessing function, set "
                "preprocess_legacy_root=True or utilize preprocess_legacy directly.",
                DeprecationWarning,
                stacklevel=2,
            )
            _advertise_datagroupspec()
            datasetspecs = DataGroupSpec.model_validate(fileset)
        # Entries assigned via DataGroupSpec item assignment are not validated; guard
        # against raw dicts sneaking in so we raise a clear error instead of an obscure
        # AttributeError deep in the dispatcher.
        for name, dss in datasetspecs.items():
            if not isinstance(dss, DatasetSpec):
                raise TypeError(
                    f"Dataset {name!r} in the DataGroupSpec is a {type(dss).__name__}, not a DatasetSpec. "
                    "Entries assigned via item assignment are not validated; rebuild or re-validate the "
                    "DataGroupSpec (e.g. DataGroupSpec.model_validate(...)) so every entry is a DatasetSpec."
                )
        # split datasetspecs into uproot and parquet files, keeping track of original order
        original_order = list(datasetspecs.keys())
        formats = [dss.format for dss in datasetspecs.values()]
        if len(set(formats)) > 1 and align_clusters != use_row_groups:
            warnings.warn(
                "When preprocessing a mixed fileset, align_clusters and use_row_groups serve a similar function. If you didn't intend to treat root and parquet files' boundary alignments differently, set both to the same value."
            )
        out_available_uproot, out_updated_uproot = preprocess_root(
            datasetspecs.filter_datasets(
                filter_callable=lambda ds: ds.format == "root"
            ),
            step_size=step_size,
            align_clusters=align_clusters,
            recalculate_steps=recalculate_steps,
            files_per_batch=files_per_batch,
            skip_bad_files=skip_bad_files,
            file_exceptions=file_exceptions,
            save_form=pydantic_save_form,
            scheduler=scheduler,
            uproot_options=uproot_options,
            step_size_safety_factor=step_size_safety_factor,
            allow_empty_datasets=allow_empty_datasets,
            backend=backend,
        )
        out_available_parquet, out_updated_parquet = preprocess_parquet(
            datasetspecs.filter_datasets(
                filter_callable=lambda ds: ds.format == "parquet"
            ),
            step_size=step_size,
            use_row_groups=use_row_groups,
            recalculate_steps=recalculate_steps,
            files_per_batch=files_per_batch,
            skip_bad_files=skip_bad_files,
            file_exceptions=file_exceptions,
            save_form=pydantic_save_form,
            scheduler=scheduler,
            parquet_options=parquet_options,
            step_size_safety_factor=step_size_safety_factor,
            allow_empty_datasets=allow_empty_datasets,
            backend=backend,
        )
        # recombine outputs in original order, skipping datasets removed due to allow_empty_datasets.
        # The sub-results are already-validated DatasetSpec instances, so use model_construct to
        # avoid a redundant full re-validation/deepcopy pass over the fileset.
        out_available = DataGroupSpec.model_construct(
            root={
                k: (
                    out_available_uproot[k]
                    if k in out_available_uproot
                    else out_available_parquet[k]
                )
                for k in original_order
                if k in out_available_uproot or k in out_available_parquet
            }
        )
        out_updated = DataGroupSpec.model_construct(
            root={
                k: (
                    out_updated_uproot[k]
                    if k in out_updated_uproot
                    else out_updated_parquet[k]
                )
                for k in original_order
                if k in out_updated_uproot or k in out_updated_parquet
            }
        )
        if input_is_dict:
            # honor the dict-in / dict-out contract
            return _datagroupspec_to_dict(out_available), _datagroupspec_to_dict(
                out_updated
            )
        return out_available, out_updated
