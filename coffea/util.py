"""Utility functions

"""
from typing import List, Optional, Any
import awkward
import hashlib
import numpy
import numba
import coffea
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    Column,
    ProgressColumn,
    Text,
)

ak = awkward
np = numpy
nb = numba

import fsspec
import cloudpickle
import warnings
from functools import partial


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
        return "".join("{:02x}".format(ord(c)) for c in string)


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
    """Normalize an array to a flat numpy array or raise ValueError"""
    if not isinstance(array, (ak.Array, numpy.ndarray)):
        raise ValueError("Expected a numpy or awkward array, received: %r" % array)

    aktype = ak.type(array)
    if not isinstance(aktype, ak.types.ArrayType):
        raise ValueError("Expected an array type, received: %r" % aktype)
    isprimitive = isinstance(aktype.type, ak.types.PrimitiveType)
    isoptionprimitive = isinstance(aktype.type, ak.types.OptionType) and isinstance(
        aktype.type.type, ak.types.PrimitiveType
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


# lifted from awkward - https://github.com/scikit-hep/awkward-1.0/blob/5fe31a916bf30df6c2ea10d4094f6f1aefcf3d0c/src/awkward/_util.py#L47-L61 # noqa
# drive our deprecations-as-errors as with awkward
def deprecate(exception, version, date=None):
    if coffea.deprecations_as_errors:
        raise exception
    else:
        if date is None:
            date = ""
        else:
            date = " (target date: " + date + ")"
        message = """In coffea version {0}{1}, this will be an error.
(Set coffea.deprecations_as_errors = True to get a stack trace now.)
{2}: {3}""".format(
            version, date, type(exception).__name__, str(exception)
        )
        warnings.warn(message, FutureWarning)


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
