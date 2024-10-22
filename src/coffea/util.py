"""Utility functions

"""

import base64
import gzip
import hashlib
from typing import Any, List, Optional

import awkward
import dask_awkward
import hist
import numba
import numpy
import uproot
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

import warnings
from functools import partial

import cloudpickle
import lz4.frame


def load(filename):
    """Load a coffea file from disk"""
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    """Save a coffea object or collection thereof to disk

    This function can accept any picklable object.  Suggested suffix: ``.coffea``
    """
    with lz4.frame.open(filename, "wb") as fout:
        thepickle = cloudpickle.dumps(output)
        fout.write(thepickle)


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


def _exception_chain(exc: BaseException) -> List[BaseException]:
    """Retrieves the entire exception chain as a list."""
    ret = []
    while isinstance(exc, BaseException):
        ret.append(exc)
        exc = exc.__cause__
    return ret


class SpeedColumn(ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, fmt: str = ".1f", table_column: Optional[Column] = None):
        self.fmt = fmt
        super().__init__(table_column=table_column)

    def render(self, task: Any) -> Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return Text("?", style="progress.data.speed")
        return Text(f"{speed:{self.fmt}}", style="progress.data.speed")


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
    behavior = awkward._util.behaviorof(like_what)
    func = partial(gfunc, data=arr.layout)
    layout = awkward.operations.convert.to_layout(like_what)
    newlayout = awkward._util.recursively_apply(layout, func)
    return awkward._util.wrap(newlayout, behavior=behavior)


# we're gonna assume that the first record array we encounter is the flattened data
def rewrap_recordarray(layout, depth, data):
    if isinstance(layout, awkward.layout.RecordArray):
        return lambda: data
    return None


# shorthand for compressing forms
def compress_form(formjson):
    return base64.b64encode(gzip.compress(formjson.encode("utf-8"))).decode("ascii")


# shorthand for decompressing forms
def decompress_form(form_compressedb64):
    return gzip.decompress(base64.b64decode(form_compressedb64)).decode("utf-8")


def _remove_not_interpretable(branch, emit_warning=True):
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
