from __future__ import annotations

from typing import Callable

import awkward as ak

from coffea.nanoevents.util import unquote


def trace(fun: Callable, events: ak.Array, throw: bool = True) -> frozenset[str]:
    """
    Trace the execution of a function on NanoEvents to determine which buffers are touched.

    Parameters
    ----------
    fun : Callable
        The function to trace.
    events : ak.Array
        The ak.Array instance to use for tracing.
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

    tracer, report = ak.typetracer.typetracer_with_report(
        form=events.attrs["@form"],
        buffer_key=events.attrs["@buffer_key"],
        behavior=events.behavior,
        attrs=events.attrs,
        highlevel=True,
    )

    try:
        _ = fun(tracer)
    except Exception as e:
        if throw:
            raise e
        else:
            print(f"Exception during function tracing: {e}")

    # translate the touched buffer keys to branch names
    keys = set()
    # each buffer key encodes the necessary branches through a "!load" instruction in the coffea DSL
    for _buffer_key in report.data_touched:
        elements = unquote(_buffer_key.split("/")[-1]).split(",")
        keys |= {
            elements[idx - 1] for idx, instr in enumerate(elements) if instr == "!load"
        }
    return frozenset(keys)
