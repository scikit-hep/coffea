"""Utility functions"""

import base64
import gzip
import hashlib
import warnings
from functools import partial
from typing import Any

import awkward
import cloudpickle
import dask_awkward
import fsspec
import hist
import math
import json
import numba
import numpy
import uproot
from dask.base import unpack_collections
from rich.console import Console
from rich.progress import (
    BarColumn,
    Column,
    Progress,
    ProgressColumn,
    Text,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

ak = awkward
dak = dask_awkward
np = numpy
nb = numba

__all__ = [
    "load",
    "save",
    "rich_bar",
    "deprecate",
    "awkward_rewrap",
    "maybe_map_partitions",
    "compress_form",
    "decompress_form",
    "coffea_console",
    "split_fileset",
    "hash_fileset",
]


def hash_fileset(chunk):
    """
    Return a stable SHA-256 hash for a fileset chunk.
    The hash considers dataset names, file paths in sorted order.

    Input
    chunk: fileset dict  {dataset: {"files": {path: treename, ...}, ...}, ...}

    Output
    hex string uniquely identifying this chunk's file content
    """
    canonical = {
        dataset: dict(sorted(files.get("files", {}).items()))
        for dataset, files in chunk.items()
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized).hexdigest()


def split_fileset(fileset, strategy=None, datasets=None, percentage=None):
    """
    Split fileset into chunks to enable getting partial result if one of the filesets(
    one partial fileset is one chunk here, these are not usual coffea chunks)
    is failed to produce a result while being processed.
    
    Input
    fileset:    {dataset: {"files": {path: treename, ...}}}
    strategy:   "by_dataset" — one dataset is one chunk; None — all datasets together
    percentage: integer that divides 100 evenly (20, 25, 50...).
                Each chunk gets this percentage of each dataset's files.
    
    Output
    List of fileset dicts
    If strategy only:
        chunks = _split_fileset(fileset, "by_dataset") - one chunk per dataset
    If percentage only:
        chunks = _split_fileset(fileset, percentage=50) - 2 chunks (50 of each dataset in 1st chunk and 2nd)
        chunks = _split_fileset(fileset, "by_dataset", percentage=50) - N_datasets * 2 chunks
    """
    if strategy is not None and strategy != "by_dataset":
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'by_dataset' or None."
        )
    if percentage is not None:
        if not isinstance(percentage, int) or not (1 <= percentage <= 100) or 100 % percentage != 0:
            raise ValueError(
                "'percentage' must be an int that divides 100 evenly (e.g. 10, 20, 25, 50)."
            )
    
    if datasets is None:
        pass
    elif callable(datasets):
        fileset = {k: v for k, v in fileset.items() if datasets(k)}
    else:
        fileset = {k: fileset[k] for k in datasets if k in fileset}
    
    
  
    if strategy == "by_dataset":
        groups = [{name: data} for name, data in fileset.items()]
    else:
        groups = [fileset]

    if percentage is None:
        return groups

    n_chunks = 100 // percentage
    result = []
    for group in groups:
        for bin_idx in range(n_chunks):
            chunk = {}
            for dataset, data in group.items():
                files = data.get("files", {})
                if not files:
                    continue
                file_items = list(files.items())
                n = len(file_items)
                chunk_size = max(1, math.ceil(n / n_chunks))
                start = bin_idx * chunk_size
                end = min(start + chunk_size, n)
                if start >= n:
                    continue
                chunk[dataset] = {**data, "files": dict(file_items[start:end])}
            if chunk:
                result.append(chunk)
    return result

def load(filename, compression="lz4"):
    """Load a coffea file from disk

    ``compression`` specified the algorithm to use to decompress the file.
    It must be one of the ``fsspec`` supported compression string names.
    These compression algorithms may have dependencies that need to be installed separately.
    If it is ``None``, it means no compression.
    """
    with fsspec.open(filename, "rb", compression=compression) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename, compression="lz4"):
    """Save a coffea object or collection thereof to disk.

    This function can accept any picklable object.  Suggested suffix: ``.coffea``

    ``compression`` can be one of the ``fsspec`` supported compression string names.
    These compression algorithms may have dependencies that need to be installed separately.
    If it is ``None``, it means no compression.
    """
    with fsspec.open(filename, "wb", compression=compression) as fout:
        cloudpickle.dump(output, fout)


def _hex(string):
    try:
        return string.hex()
    except AttributeError:
        return "".join(f"{ord(c):02x}" for c in string)


def _ascii(maybebytes):
    try:
        return maybebytes.decode("ascii")
    except AttributeError:
        return maybebytes


def _hash(items):
    # python 3.3 salts hash(), we want it to persist across processes
    x = hashlib.md5(bytes(";".join(str(x) for x in items), "ascii"))
    return int(x.hexdigest()[:16], base=16)
    
def _ensure_flat(array, allow_missing=False):
    """Normalize an array to a flat numpy array, or ensure it is a flat dask-awkward array, or raise ValueError"""
    if not isinstance(array, (dak.Array, ak.Array, numpy.ndarray)):
        raise ValueError("Expected a numpy or awkward array, received: %r" % array)

    aktype = (
        ak.type(array) if not isinstance(array, dak.Array) else ak.type(array._meta)
    )
    if not isinstance(aktype, ak.types.ArrayType):
        raise ValueError("Expected an array type, received: %r" % aktype)
    isprimitive = isinstance(aktype.content, ak.types.NumpyType)
    isoptionprimitive = isinstance(aktype.content, ak.types.OptionType) and isinstance(
        aktype.content.content, ak.types.NumpyType
    )
    if allow_missing and not (isprimitive or isoptionprimitive):
        raise ValueError(
            "Expected an array of type N * primitive or N * ?primitive, received: %r"
            % aktype
        )
    if not (allow_missing or isprimitive):
        raise ValueError(
            "Expected an array of type N * primitive, received: %r" % aktype
        )
    if isinstance(array, ak.Array):
        array = ak.to_numpy(array, allow_missing=allow_missing)
    return array


def _gethistogramaxis(name, var, bins, start, stop, edges, transform, delayed_mode):
    "Get a hist axis for plot_vars in PackedSelection"

    if edges is not None:
        return hist.axis.Variable(edges=edges, name=name)

    if not delayed_mode:
        start = ak.min(var) - 1e-6 if start is None else start
        stop = ak.max(var) + 1e-6 if stop is None else stop
    elif delayed_mode:
        start = dak.min(var).compute() - 1e-6 if start is None else start
        stop = dak.max(var).compute() + 1e-6 if stop is None else stop
    bins = 20 if bins is None else bins

    return hist.axis.Regular(
        bins=bins, start=start, stop=stop, name=name, transform=transform
    )


def _exception_chain(exc: BaseException) -> list[BaseException]:
    """Retrieves the entire exception chain as a list."""
    ret = []
    while isinstance(exc, BaseException):
        ret.append(exc)
        exc = exc.__cause__
    return ret


class SpeedColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, fmt: str = ".1f", table_column: Column | None = None):
        self.fmt = fmt
        super().__init__(table_column=table_column)

    def render(self, task: Any) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:{self.fmt}}", style="progress.data.speed")


coffea_console = Console()
coffea_console.__doc__ += """
\nA `rich.console.Console` for coffea. Used through-out coffea for consistent logging and
progress bars. May be used by users for their own logging. Using the same console
ensures that output is nicely integrated with coffea's progress bars.
"""


def rich_bar():
    return Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        "[progress.percentage]{task.percentage:>3.0f}%",
        BarColumn(bar_width=None),
        TextColumn(
            "[bold blue][progress.completed]{task.completed}/{task.total}",
            justify="right",
        ),
        "[",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        "|",
        SpeedColumn(".1f"),
        TextColumn("[progress.data.speed]{task.fields[unit]}/s", justify="right"),
        "]",
        auto_refresh=False,
        console=coffea_console,
    )


# lifted from awkward - https://github.com/scikit-hep/awkward/blob/2b80da6b60bd5f0437b66f266387f1ab4bf98fe1/src/awkward/_errors.py#L421 # noqa
# drive our deprecations-as-errors as with awkward
def deprecate(
    message,
    version,
    date=None,
    will_be="an error",
    category=DeprecationWarning,
    stacklevel=2,
):
    if date is None:
        date = ""
    else:
        date = " (target date: " + date + ")"
    warning = f"""In version {version}{date}, this will be {will_be}.
To raise these warnings as errors (and get stack traces to find out where they're called), run
    import warnings
    warnings.filterwarnings("error", module="coffea.*")
after the first `import coffea` or use `@pytest.mark.filterwarnings("error:::coffea.*")` in pytest.
Issue: {message}."""
    warnings.warn(warning, category, stacklevel=stacklevel + 1)


# re-nest a record array into a ListArray
def awkward_rewrap(arr, like_what, gfunc):
    func = partial(gfunc, data=arr.layout)
    return awkward.transform(func, like_what, behavior=like_what.behavior)


# we're gonna assume that the first record array we encounter is the flattened data
def rewrap_recordarray(layout, depth, data, **kwargs):
    if isinstance(layout, awkward.contents.RecordArray):
        return data
    return None


def maybe_map_partitions(func, *args, **kwargs):
    _MP_ONLY = {
        "label",
        "token",
        "meta",
        "output_divisions",
        "traverse",
        "opt_touch_all",
    }
    traverse = kwargs.pop("traverse", True)

    func_kwargs = {k: v for k, v in kwargs.items() if k not in _MP_ONLY}
    deps, _ = unpack_collections(*args, *func_kwargs.values(), traverse=traverse)

    if len(deps) > 0:
        return dask_awkward.map_partitions(func, *args, traverse=traverse, **kwargs)

    return func(*args, **func_kwargs)


# shorthand for compressing forms
def compress_form(formjson):
    return base64.b64encode(gzip.compress(formjson.encode("utf-8"))).decode("ascii")


# shorthand for decompressing forms
def decompress_form(form_compressedb64):
    return gzip.decompress(base64.b64decode(form_compressedb64)).decode("utf-8")


def _is_interpretable(branch, emit_warning=True):
    if isinstance(branch, uproot.behaviors.RNTuple.HasFields):
        # These are collections made by the RNTuple Importer
        # Once "real" (i.e. non-converted) RNTuples start to be written,
        # these should not be here and this check can be removed
        if branch.path.startswith("_collection"):
            return False
        # Subfields should be accessed via the parent branch since
        # the way forms are set up for subfields
        if "." in branch.path:
            return False
        return True
    if isinstance(
        branch.interpretation, uproot.interpretation.identify.uproot.AsGrouped
    ):
        for name, interpretation in branch.interpretation.subbranches.items():
            if isinstance(
                interpretation, uproot.interpretation.identify.UnknownInterpretation
            ):
                if emit_warning:
                    warnings.warn(
                        f"Skipping {branch.name} as it is not interpretable by Uproot"
                    )
                return False
    if isinstance(
        branch.interpretation, uproot.interpretation.identify.UnknownInterpretation
    ):
        if emit_warning:
            warnings.warn(
                f"Skipping {branch.name} as it is not interpretable by Uproot"
            )
        return False

    try:
        _ = branch.interpretation.awkward_form(None)
    except uproot.interpretation.objects.CannotBeAwkward:
        if emit_warning:
            warnings.warn(
                f"Skipping {branch.name} as it is it cannot be represented as an Awkward array"
            )
        return False
    else:
        return True
