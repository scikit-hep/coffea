"""Generic work element and computable implementations for data processing."""

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Protocol

from coffea.compute.protocol import (
    AbstractInput,
    Computable,
    DataT,
    InputT,
    ResultT,
    WorkElement,
)


class DataElement(Protocol[DataT]):
    """Shim for older code that presents a DataElement class"""

    def __len__(self) -> int: ...
    def load(self) -> DataT:
        """Load the data for this element. Must be implemented by subclasses."""
        ...


@dataclass(frozen=True)
class InputWithIndex(Generic[DataT]):
    data: DataElement[DataT]
    start: int
    stop: int

    def __len__(self) -> int:
        return self.stop - self.start

    def load(self) -> DataT:
        return self.data.load()


if TYPE_CHECKING:
    _x: type[AbstractInput] = InputWithIndex


@dataclass(frozen=True)
class MapData(Generic[InputT, ResultT]):
    """Concrete Computable, generates DataWorkElements by calling iter_gen and applying func."""

    func: Callable[[InputT], ResultT]
    "Function to apply to each loaded DataElement."
    make_iter: Callable[[], Iterator[DataElement[InputT]]]
    "Callable to make an iterator over DataElements. Must be pure to allow re-iteration."

    def __len__(self) -> int:
        return sum(len(element) for element in self.make_iter())

    @property
    def key(self) -> str:
        # TODO: make actually unique by hashing the function and the data
        return f"{self.func!r}({self.make_iter.__name__})"

    def gen_steps(self) -> Generator[WorkElement[ResultT], int | None, None]:
        i = 0
        for element in self.make_iter():
            idxel = InputWithIndex(element, i, i + len(element))
            yield WorkElement(self.func, idxel)
            i += len(element)


if TYPE_CHECKING:
    _y: type[Computable] = MapData

__all__ = [
    "MapData",
]
