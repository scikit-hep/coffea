"""Generic work element and computable implementations for data processing."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from itertools import repeat
from typing import Generic

from coffea.compute.protocol import DataElement, InputT, ResultT


@dataclass(frozen=True)
class DataWorkElement(Generic[InputT, ResultT]):
    """Concrete WorkElement, applies func to loaded item.

    TODO: do we really need a protocol and a concrete class for this or can it just be the concrete class?
    """

    func: Callable[[InputT], ResultT]
    item: DataElement[InputT]

    def __call__(self) -> ResultT:
        return self.func(self.item.load())


@dataclass(frozen=True)
class MapData(Generic[InputT, ResultT]):
    """Concrete Computable, generates DataWorkElements by calling iter_gen and applying func."""

    func: Callable[[InputT], ResultT]
    "Function to apply to each loaded DataElement."
    make_iter: Callable[[], Iterator[DataElement[InputT]]]
    "Callable to make an iterator over DataElements. Must be pure to allow re-iteration."

    def __iter__(self) -> Iterator[DataWorkElement[InputT, ResultT]]:
        return map(DataWorkElement, repeat(self.func), self.make_iter())


__all__ = [
    "DataWorkElement",
    "MapData",
]
