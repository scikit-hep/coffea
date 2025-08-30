from __future__ import annotations

from functools import partial
from typing import Callable

import awkward as ak

from coffea.nanoevents.util import unquote


def _make_typetracer(events):
    tracer, report = ak.typetracer.typetracer_with_report(
        form=events.attrs["@form"],
        buffer_key=events.attrs["@buffer_key"],
        behavior=events.behavior,
        attrs=events.attrs,
        highlevel=True,
    )
    tracer.attrs["@original_array"] = tracer

    return tracer, report


def _make_length_zero_one_tracer(events, length):
    form = ak.forms.from_dict(events.attrs["@form"])
    buffer_key = events.attrs["@buffer_key"]
    buffer_keys = form.expected_from_buffers(buffer_key=buffer_key).keys()

    if length == 0:
        buffers = ak.to_buffers(
            form.length_zero_array(),
            byteorder=ak._util.native_byteorder,
        )[2].values()
    elif length == 1:
        buffers = ak.to_buffers(
            form.length_one_array(),
            byteorder=ak._util.native_byteorder,
        )[2].values()
    else:
        raise ValueError("length must be 0 or 1")

    report = []

    def generate(buffer, report, buffer_key):
        report.append(buffer_key)
        return buffer

    container = {}
    for key, buffer in zip(buffer_keys, buffers):
        container[key] = partial(generate, buffer=buffer, report=report, buffer_key=key)
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
        attrs=events.attrs,
    )
    array.attrs["@original_array"] = array

    return array, report


def trace(
    fun: Callable, events: ak.Array, mode: str = "typetracer", throw: bool = True
) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents to determine which buffers are touched.

    Parameters
    ----------
    fun : Callable
        The function to trace.
    events : ak.Array
        The ak.Array instance to use for tracing.
    mode : str, optional
        Can be 'typetracer', 'length_zero_array', or 'length_one_array'.
        The tracing mode to use. 'typetracer' uses Awkward's typetracer, while 'length_zero_array' and
        'length_one_array' create arrays of length-zero or length-one, respectively, to trace actual data access.
        Default is 'typetracer'.
    throw : bool, optional
        If True, exceptions during function execution will be raised; otherwise, they will be caught and
        printed. Default is True.

    Returns
    -------
    frozenset[str]
        A set of branch names that were touched during the execution of the function.
    """
    if not isinstance(events, ak.Array):
        raise TypeError("events must be an instance of ak.Array")

    if "@form" not in events.attrs or "@buffer_key" not in events.attrs:
        raise ValueError(
            "events must have '@form' and '@buffer_key' attributes set; it is automatically set when using `NanoEventsFactory.from_*(...).events()`"
        )

    if mode == "typetracer":
        tracer, report = _make_typetracer(events)
    elif mode == "length_zero_array":
        tracer, report = _make_length_zero_one_tracer(events, length=0)
    elif mode == "length_one_array":
        tracer, report = _make_length_zero_one_tracer(events, length=1)
    else:
        raise ValueError(
            "mode must be one of 'typetracer', 'length_zero_array', or 'length_one_array'"
        )

    try:
        _ = fun(tracer)
    except Exception as e:
        if throw:
            raise e
        else:
            print(f"Exception during function tracing: {e}")

    touched = report.data_touched if mode == "typetracer" else report
    # translate the touched buffer keys to branch names
    keys = set()
    # each buffer key encodes the necessary branches through a "!load" instruction in the coffea DSL
    for _buffer_key in touched:
        elements = unquote(_buffer_key.split("/")[-1]).split(",")
        keys |= {
            elements[idx - 1] for idx, instr in enumerate(elements) if instr == "!load"
        }
    return frozenset(keys)
