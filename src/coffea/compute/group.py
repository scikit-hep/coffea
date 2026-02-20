import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

from coffea.compute.context import ContextInput, Ctx_co
from coffea.compute.protocol import InputT, ResultT


@dataclass(frozen=True)
class AggResult(Generic[ResultT]):
    """Aggregated result, with counts of processed items.

    Used to track completion of an item in a GroupedResult."""

    # mypy does not recognize frozen fields as read-only, which is required for covariant types
    result: ResultT  # type: ignore[misc]
    items_processed: int
    # TODO: items_expected? Need to get that info into the context

    def __add__(self, other: Self) -> Self:
        return type(self)(
            result=self.result + other.result,
            items_processed=self.items_processed + other.items_processed,
        )


@dataclass
class GroupedResult(Generic[ResultT]):
    """Result of a grouped computation, mapping group keys to aggregated results."""

    # TODO: make key a generic Hashable type?
    children: dict[str, AggResult[ResultT]]

    def __add__(self, other: Self) -> Self:
        children = {
            name: self.children[name] + other.children[name]
            for name in self.children
            if name in other.children
        }
        children.update(
            {
                name: child
                for name, child in self.children.items()
                if name not in other.children
            }
        )
        children.update(
            {
                name: child
                for name, child in other.children.items()
                if name not in self.children
            }
        )
        return type(self)(children=children)


@dataclass
class GroupedFunction(Generic[InputT, Ctx_co, ResultT]):
    func: Callable[[ContextInput[InputT, Ctx_co]], ResultT]
    """The wrapped user function."""
    key_func: Callable[[Ctx_co], str]
    """Function to extract grouping key from context."""

    def __call__(self, item: ContextInput[InputT, Ctx_co]) -> GroupedResult[ResultT]:
        key = self.key_func(item.context)
        result = self.func(item)
        return GroupedResult(
            children={key: AggResult(result=result, items_processed=1)}
        )
