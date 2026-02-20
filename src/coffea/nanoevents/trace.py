from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial

import awkward as ak

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
    expected_buffer_keys = form.expected_from_buffers(buffer_key=buffer_key).keys()

    if length == 0:
        tmp_array = form.length_zero_array()
    elif length == 1:
        tmp_array = form.length_one_array()
    else:
        raise ValueError("length must be 0 or 1")

    def getkey(layout, form, attribute):
        return buffer_key(
            form_key=form.form_key,
            attribute=attribute,
            form=form,
        )

    container = {}
    tmp_array._to_buffers(
        form, getkey, container, events._layout.backend, ak._util.native_byteorder
    )

    report = []

    def generate(buffer, report, buffer_key):
        report.append(buffer_key)
        return buffer

    for key, buffer in container.items():
        container[key] = partial(generate, buffer=buffer, report=report, buffer_key=key)

    assert list(container.keys()) == list(
        expected_buffer_keys
    ), "length zero/one array buffer keys do not match the expected ones"
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
