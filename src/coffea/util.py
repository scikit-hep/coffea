"""Utility functions"""

import base64
import gzip
import hashlib
import warnings
from functools import partial
from typing import Any

import awkward
import cloudpickle
import fsspec
import hist
import numba
import numpy
import uproot
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
    "dask_method",
    "dask_property",
    "_import_dask",
    "_import_distributed",
    "_import_dask_awkward",
]


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


def _isinstance(arg: Any, *class_prefixes: str) -> bool:
    """Return True if arg is an instance of a class with any given prefix."""
    for cls in type(arg).__mro__:
        class_name = f"{cls.__module__}.{cls.__qualname__}"
        if any(class_name.startswith(prefix) for prefix in class_prefixes):
            return True
    return False


def _import_dask():
    try:
        import dask
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""to use this feature, you must install dask:

pip install dask

or

conda install -c conda-forge dask""") from err

    return dask


def _import_distributed():
    try:
        import distributed
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""to use this feature, you must install distributed:

pip install distributed

or

conda install -c conda-forge distributed""") from err

    return distributed


def _import_dask_awkward():
    try:
        import dask_awkward
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError("""to use this feature, you must install dask-awkward:

pip install dask-awkward

or

conda install -c conda-forge dask-awkward""") from err

    return dask_awkward


def _ensure_flat(array, allow_missing=False):
    """Normalize an array to a flat numpy array, or ensure it is a flat dask-awkward array, or raise ValueError"""
    if not _isinstance(
        array,
        "dask_awkward.lib.core.Array",
        "awkward.highlevel.Array",
        "numpy.ndarray",
    ):
        raise ValueError("Expected a numpy or awkward array, received: %r" % array)

    aktype = (
        ak.type(array._meta)
        if _isinstance(array, "dask_awkward.lib.core.Array")
        else ak.type(array)
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
        dask_awkward = _import_dask_awkward()

        start = dask_awkward.min(var).compute() - 1e-6 if start is None else start
        stop = dask_awkward.max(var).compute() + 1e-6 if stop is None else stop
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
\nA `rich.console.Console` for coffea. Used throughout coffea for consistent logging and
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


def _make_dask_descriptor(func):
    def descriptor(instance, owner, dask_array):
        impl = func.__get__(instance, owner)
        return impl(dask_array)

    return descriptor


def _make_dask_method(func):
    def descriptor(instance, owner, dask_array):
        def impl(*args, **kwargs):
            impl = func.__get__(instance, owner)
            return impl(dask_array, *args, **kwargs)

        return impl

    return descriptor


class _DaskProperty(property):
    _dask_get = None

    def dask(self, func):
        assert self._dask_get is None
        self._dask_get = _make_dask_descriptor(func)
        return self


def _adapt_naive_dask_get(func):
    def wrapper(self, dask_array, *args, **kwargs):
        return func(self, *args, **kwargs)

    return wrapper


def dask_property(maybe_func=None, *, no_dispatch=False):
    def dask_property_wrapper(func):
        prop = _DaskProperty(func)
        if no_dispatch:
            return prop.dask(_adapt_naive_dask_get(func))
        else:
            return prop

    if maybe_func is None:
        return dask_property_wrapper
    else:
        return dask_property_wrapper(maybe_func)


class _DaskMethod:
    _dask_get = None

    def __init__(self, impl):
        self._impl = impl

    def __get__(self, instance, owner=None):
        if instance is None:
            return self

        return self._impl.__get__(instance, owner)

    def dask(self, func):
        self._dask_get = _make_dask_method(func)
        return self


def dask_method(maybe_func=None, *, no_dispatch=False):
    def dask_method_wrapper(func):
        method = _DaskMethod(func)

        if no_dispatch:
            return method.dask(_adapt_naive_dask_get(func))
        else:
            return method

    if maybe_func is None:
        return dask_method_wrapper
    else:
        return dask_method_wrapper(maybe_func)


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

    try:
        dask = _import_dask()
    except ModuleNotFoundError:
        return func(*args, **func_kwargs)

    deps, _ = dask.base.unpack_collections(
        *args, *func_kwargs.values(), traverse=traverse
    )

    if len(deps) > 0:
        dask_awkward = _import_dask_awkward()

        return dask_awkward.map_partitions(func, *args, traverse=traverse, **kwargs)

    return func(*args, **func_kwargs)
