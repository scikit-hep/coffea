from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial

import awkward as ak
from awkward._nplikes.shape import unknown_length
from awkward.forms.form import regularize_buffer_key

from coffea.nanoevents.util import unquote

__all__ = [
    "trace_with_typetracer",
    "trace_with_length_zero_array",
    "trace_with_length_one_array",
    "trace",
]


def _make_typetracer(
    events: ak.Array,
) -> tuple[ak.Array, ak._nplikes.typetracer.TypeTracerReport]:
    tracer, report = ak.typetracer.typetracer_with_report(
        form=events.attrs["@form"],
        buffer_key=events.attrs["@buffer_key"],
        behavior=events.behavior,
        attrs=events.attrs.copy(),
        highlevel=True,
    )
    tracer.attrs["@original_array"] = tracer

    return tracer, report


def _make_length_zero_one_tracer(
    events: ak.Array, length: int
) -> tuple[ak.Array, list]:
    form = ak.forms.from_dict(events.attrs["@form"])
    buffer_key = events.attrs["@buffer_key"]
    buffer_keys = form.expected_from_buffers(buffer_key=buffer_key).keys()

    report = []

    def generate(buffer, report, buffer_key):
        report.append(buffer_key)
        return buffer

    if length == 0:
        buffers = {key: b"\x00\x00\x00\x00\x00\x00\x00\x00" for key in buffer_keys}

        container = {}
        for key, buffer in buffers.items():
            container[key] = partial(
                generate, buffer=buffer, report=report, buffer_key=key
            )

    elif length == 1:
        buffers = {}
        getkey = regularize_buffer_key(buffer_key)

        def prepare_empty(form):
            form_key = form.form_key

            if isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm)):
                buffers[getkey(form, "mask")] = b""
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.IndexedOptionForm):
                buffers[getkey(form, "index")] = b""
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.EmptyForm):
                return form

            elif isinstance(form, ak.forms.UnmaskedForm):
                return form.copy(content=prepare_empty(form.content))

            elif isinstance(form, (ak.forms.IndexedForm, ak.forms.ListForm)):
                buffers[getkey(form, "index")] = b""
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.ListOffsetForm):
                buffers[getkey(form, "offsets")] = b""
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.RegularForm):
                return form.copy(content=prepare_empty(form.content))

            elif isinstance(form, ak.forms.NumpyForm):
                buffers[getkey(form, "data")] = b""
                return form.copy(form_key=form_key)

            elif isinstance(form, ak.forms.RecordForm):
                return form.copy(contents=[prepare_empty(x) for x in form.contents])

            elif isinstance(form, ak.forms.UnionForm):
                # both tags and index will get this buffer
                buffers[getkey(form, "tags")] = b""
                buffers[getkey(form, "index")] = b""
                return form.copy(
                    contents=[prepare_empty(x) for x in form.contents],
                    form_key=form_key,
                )

            else:
                raise AssertionError(f"not a Form: {form!r}")

        def prepare(form, multiplier):
            form_key = form.form_key

            if isinstance(form, (ak.forms.BitMaskedForm, ak.forms.ByteMaskedForm)):
                if form.valid_when:
                    buffers[getkey(form, "mask")] = b"\x00" * multiplier
                else:
                    buffers[getkey(form, "mask")] = b"\xff" * multiplier
                # switch from recursing down `prepare` to `prepare_empty`
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.IndexedOptionForm):
                buffers[getkey(form, "index")] = (
                    b"\xff\xff\xff\xff\xff\xff\xff\xff"  # -1
                )
                # switch from recursing down `prepare` to `prepare_empty`
                return form.copy(content=prepare_empty(form.content), form_key=form_key)

            elif isinstance(form, ak.forms.EmptyForm):
                # no error if protected by non-recursing node type
                raise TypeError(
                    "cannot generate a length_one_array from a Form with an "
                    "unknowntype that cannot be hidden (EmptyForm not within "
                    "BitMaskedForm, ByteMaskedForm, or IndexedOptionForm)"
                )

            elif isinstance(form, ak.forms.UnmaskedForm):
                return form.copy(content=prepare(form.content, multiplier))

            elif isinstance(form, (ak.forms.IndexedForm, ak.forms.ListForm)):
                buffers[getkey(form, "index")] = b"\x00" * (8 * multiplier)
                buffers[getkey(form, "starts")] = b"\x00" * (8 * multiplier)
                buffers[getkey(form, "stops")] = b"\x00" * (8 * multiplier)
                return form.copy(
                    content=prepare(form.content, multiplier), form_key=form_key
                )

            elif isinstance(form, ak.forms.ListOffsetForm):
                # offsets length == array length + 1
                buffers[getkey(form, "offsets")] = b"\x00" * (8 * (multiplier + 1))
                return form.copy(
                    content=prepare(form.content, multiplier), form_key=form_key
                )

            elif isinstance(form, ak.forms.RegularForm):
                size = form.size

                # https://github.com/scikit-hep/awkward/pull/2499#discussion_r1220503454
                if size is unknown_length:
                    size = 1

                return form.copy(content=prepare(form.content, multiplier * size))

            elif isinstance(form, ak.forms.NumpyForm):
                dtype = ak.types.numpytype.primitive_to_dtype(form._primitive)
                size = multiplier * dtype.itemsize
                for x in form.inner_shape:
                    if x is not unknown_length:
                        size *= x

                buffers[getkey(form, "data")] = b"\x00" * size
                return form.copy(form_key=form_key)

            elif isinstance(form, ak.forms.RecordForm):
                return form.copy(
                    # recurse down all contents
                    contents=[prepare(x, multiplier) for x in form.contents]
                )

            elif isinstance(form, ak.forms.UnionForm):
                # both tags and index will get this buffer, but index is 8 bytes
                buffers[getkey(form, "tags")] = b"\x00" * (8 * multiplier)
                buffers[getkey(form, "index")] = b"\x00" * (8 * multiplier)
                # recurse down contents[0] with `prepare`, but others with `prepare_empty`
                contents = [prepare(form.contents[0], multiplier)]
                for x in form.contents[1:]:
                    contents.append(prepare_empty(x))
                return form.copy(contents=contents, form_key=form_key)

            else:
                raise AssertionError(f"not a Form: {form!r}")

        form = prepare(form, 1)
        container = {}
        for key, buffer in buffers.items():
            container[key] = partial(
                generate, buffer=buffer, report=report, buffer_key=key
            )

    else:
        raise ValueError("length must be 0 or 1")

    array = ak.from_buffers(
        form=form,
        length=length,
        container=container,
        buffer_key=buffer_key,
        backend=ak.backend(events),
        byteorder=ak._util.native_byteorder,
        allow_noncanonical_form=False,
        highlevel=True,
        behavior=events.behavior,
        attrs=events.attrs.copy(),
    )
    array.attrs["@original_array"] = array

    return array, report


def _form_keys_to_columns(touched: list) -> frozenset[str]:
    # translate the touched buffer keys to branch names
    keys = set()
    # each buffer key encodes the necessary branches through a "!load" instruction in the coffea DSL
    for _buffer_key in touched:
        elements = unquote(_buffer_key.split("/")[-1]).split(",")
        keys |= {
            elements[idx - 1] for idx, instr in enumerate(elements) if instr == "!load"
        }
    return frozenset(keys)


def _check_inputs(fun: Callable, events: ak.Array) -> None:
    if not callable(fun):
        raise TypeError(
            "fun must be a callable function that accepts a single ak.Array argument"
        )
    if not isinstance(events, ak.Array):
        raise TypeError("events must be an instance of ak.Array")
    if "@form" not in events.attrs or "@buffer_key" not in events.attrs:
        raise ValueError(
            "events must have '@form' and '@buffer_key' attributes set; it is automatically set when using `NanoEventsFactory.from_*(...).events()`"
        )


def _attempt_tracing(fun: Callable, tracer: ak.Array, throw: bool) -> None:
    try:
        _ = fun(tracer)
    except Exception as e:
        if throw:
            raise e
        else:
            warnings.warn(
                f"Exception during function tracing: {e}",
                RuntimeWarning,
                stacklevel=3,
            )


def trace_with_typetracer(
    fun: Callable, events: ak.Array, throw: bool = True
) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents using Awkward's typetracer to determine which buffers are touched.

    Parameters
    ----------
    fun : Callable
        The function to trace. It should accept a single argument, which is an ak.Array.
    events : ak.Array
        The ak.Array instance to use for tracing.
    throw : bool, optional
        If True, exceptions during function execution will be raised; otherwise, they will be caught and
        a warning will be issued. Default is True.

    Returns
    -------
    frozenset[str]
        A set of branch names that were touched during the execution of the function.
    """
    _check_inputs(fun, events)
    tracer, report = _make_typetracer(events)
    _attempt_tracing(fun, tracer, throw)

    return _form_keys_to_columns(report.data_touched)


def trace_with_length_zero_array(
    fun: Callable, events: ak.Array, throw: bool = True
) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents using a length-zero array to determine which buffers are touched.

    Parameters
    ----------
    fun : Callable
        The function to trace. It should accept a single argument, which is an ak.Array.
    events : ak.Array
        The ak.Array instance to use for tracing.
    throw : bool, optional
        If True, exceptions during function execution will be raised; otherwise, they will be caught and
        a warning will be issued. Default is True.

    Returns
    -------
    frozenset[str]
        A set of branch names that were touched during the execution of the function.
    """
    _check_inputs(fun, events)
    tracer, report = _make_length_zero_one_tracer(events, length=0)
    _attempt_tracing(fun, tracer, throw)

    return _form_keys_to_columns(report)


def trace_with_length_one_array(
    fun: Callable, events: ak.Array, throw: bool = True
) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents using a length-one array to determine which buffers are touched.

    Parameters
    ----------
    fun : Callable
        The function to trace. It should accept a single argument, which is an ak.Array.
    events : ak.Array
        The ak.Array instance to use for tracing.
    throw : bool, optional
        If True, exceptions during function execution will be raised; otherwise, they will be caught and
        a warning will be issued. Default is True.

    Returns
    -------
    frozenset[str]
        A set of branch names that were touched during the execution of the function.
    """
    _check_inputs(fun, events)
    tracer, report = _make_length_zero_one_tracer(events, length=1)
    _attempt_tracing(fun, tracer, throw)

    return _form_keys_to_columns(report)


def trace(fun: Callable, events: ak.Array) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents to determine which buffers are touched.

    This function first attempts to use Awkward's typetracer for tracing. If that fails,
    it attempts tracing with a length-zero array. If that also fails, it finally attempts
    tracing with a length-one array.
    Eventually, it reports the set union of all branches touched during the attempts.

    Parameters
    ----------
    fun : Callable
        The function to trace. It should accept a single argument, which is an ak.Array.
    events : ak.Array
        The ak.Array instance to use for tracing.

    Returns
    -------
    frozenset[str]
        A set of branch names that were touched during the execution of the function.
    """
    _check_inputs(fun, events)
    touched = set()

    try:
        touched |= trace_with_typetracer(fun, events)
        return frozenset(touched)
    except Exception as e1:
        warnings.warn(
            f"Exception during typetracer tracing: {e1}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        touched |= trace_with_length_zero_array(fun, events)
        return frozenset(touched)
    except Exception as e2:
        warnings.warn(
            f"Exception during length-zero array tracing: {e2}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        touched |= trace_with_length_one_array(fun, events)
        return frozenset(touched)
    except Exception as e3:
        warnings.warn(
            f"Exception during length-one array tracing: {e3}",
            RuntimeWarning,
            stacklevel=2,
        )

    return frozenset(touched)
