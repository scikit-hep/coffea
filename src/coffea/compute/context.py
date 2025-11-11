from collections.abc import Iterator
from dataclasses import dataclass
from itertools import repeat
from typing import Any, Callable, Generic, TypeVar

from coffea.compute.protocol import DataElement, InputT


@dataclass(frozen=True, kw_only=True)
class Context:
    """Base class for context metadata associated with data elements."""

    pass


Ctx = TypeVar("Ctx", bound=Context)
"""Invariant context type."""

Ctx_co = TypeVar("Ctx_co", bound=Context, covariant=True)
"""Covariant context type, to allow user functions to accept supertypes of the provided context."""


@dataclass(frozen=True)
class ContextInput(Generic[InputT, Ctx_co]):
    """Input data to a computation, paired with some context about what it is

    For example, this could be an EventsArray along with metadata about which file
    and which dataset it came from.
    """

    data: InputT
    # mypy does not recognize frozen fields as read-only, which is required for covariant types
    context: Ctx_co  # type: ignore[misc]


@dataclass(frozen=True)
class ContextDataElement(Generic[InputT, Ctx_co]):
    """A DataElement paired with some context about what it is.

    For example, this could be a DataElement that loads an EventsArray along with
    metadata about which file and which dataset it came from.
    """

    data: DataElement[InputT]
    # mypy does not recognize frozen fields as read-only, which is required for covariant types
    context: Ctx_co  # type: ignore[misc]

    def load(self) -> ContextInput[InputT, Ctx_co]:
        return ContextInput(self.data.load(), self.context)

    def replace_context(self, context: Ctx) -> "ContextDataElement[InputT, Ctx]":
        return ContextDataElement(self.data, context)

    def update_context(
        self, fn: Callable[[Ctx_co], Ctx]
    ) -> "ContextDataElement[InputT, Ctx]":
        return ContextDataElement(self.data, fn(self.context))


def with_context(
    it: Iterator[DataElement[InputT]], context: Ctx
) -> Iterator[ContextDataElement[InputT, Ctx]]:
    """Attach a context to each DataElement in an iterator."""
    return map(ContextDataElement, it, repeat(context))


def replace_context(
    it: Iterator[ContextDataElement[InputT, Any]], context: Ctx
) -> Iterator[ContextDataElement[InputT, Ctx]]:
    """Replace the context of each ContextDataElement in an iterator."""
    return map(ContextDataElement.replace_context, it, repeat(context))


def update_context(
    it: Iterator[ContextDataElement[InputT, Ctx_co]], fn: Callable[[Ctx_co], Ctx]
) -> Iterator[ContextDataElement[InputT, Ctx]]:
    """Update the context of each ContextDataElement in an iterator using a function."""
    return map(ContextDataElement.update_context, it, repeat(fn))
