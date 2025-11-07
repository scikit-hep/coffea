from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Generic, Literal

import uproot
from more_itertools import roundrobin
from uproot import ReadOnlyDirectory

from coffea.compute.func import DirectoryFunc, EventsArray, EventsFunc, Processor
from coffea.compute.protocol import Addable, ResultT


@dataclass(slots=True)
class Step:
    path: str
    entry_range: tuple[int, int]

    def __len__(self) -> int:
        return self.entry_range[1] - self.entry_range[0]

    def load(self) -> EventsArray:
        # Dummy implementation of event loading
        info = str(self.entry_range)
        return info + "A" * (len(self) - len(info))  # Dummy events


class StepIterable(ABC):
    @abstractmethod
    def iter_steps(self) -> Iterator[Step]:
        """Return an iterator over steps in the computation."""
        raise NotImplementedError

    def map_steps(
        self, func: EventsFunc[ResultT] | Processor[ResultT]
    ) -> "StepwiseComputable[ResultT]":
        """Apply a function or Processor to each step in this data."""
        if not callable(func):
            func = func.process
        return StepwiseComputable(func=func, iterable=self)


@dataclass(frozen=True)
class StepWorkElement(Generic[ResultT]):
    func: EventsFunc[ResultT]
    item: Step

    def __call__(self) -> ResultT:
        # TODO: where should we attach the metadata about the step parentship?
        # Here or in Step itself? Maybe in Step itself so that WorkElement can be generic?
        # Or maybe it should be here in StepWorkElement so that Step can remain lightweight
        # and not duplicate metadata contained inside the parent File or Dataset?
        return self.func(self.item.load())


@dataclass(frozen=True)
class StepwiseComputable(Generic[ResultT]):
    func: EventsFunc[ResultT]
    iterable: StepIterable

    def __iter__(self) -> Iterator[StepWorkElement[ResultT]]:
        return map(StepWorkElement, repeat(self.func), self.iterable.iter_steps())


@dataclass
class File(StepIterable):
    path: str
    steps: list[tuple[int, int]]

    def iter_steps(self) -> Iterator[Step]:
        return map(Step, repeat(self.path), self.steps)

    def load(self) -> ReadOnlyDirectory:
        # Dummy implementation of file loading
        file = uproot.open(self.path)
        assert isinstance(file, ReadOnlyDirectory)
        return file


class FileIterable(ABC):
    @abstractmethod
    def iter_files(self) -> Iterator[File]:
        """Return an iterator over files in the computation."""
        raise NotImplementedError

    def map_files(self, func: DirectoryFunc) -> "FilewiseComputable":
        return FilewiseComputable(func=func, iterable=self)


@dataclass
class FileWorkElement:
    func: DirectoryFunc
    item: File

    def __call__(self) -> Addable:
        return self.func(self.item.load())


@dataclass
class FilewiseComputable:
    func: DirectoryFunc
    iterable: FileIterable

    def __iter__(self) -> Iterator[FileWorkElement]:
        return map(FileWorkElement, repeat(self.func), self.iterable.iter_files())


@dataclass
class Dataset(StepIterable, FileIterable):
    files: list[File]
    traversal: Literal["depth", "breadth"] = "depth"
    """The traversal strategy for iterating over files in the dataset.

    "depth" means to process all steps in one file before moving to the next file.
    "breadth" means to process the first step in all files, then the second step in all files, etc.
    """

    def iter_steps(self) -> Iterator[Step]:
        if self.traversal == "breadth":
            return roundrobin(*map(File.iter_steps, self.files))
        return chain.from_iterable(map(File.iter_steps, self.files))

    def iter_files(self) -> Iterator[File]:
        return iter(self.files)


@dataclass
class DataGroup(StepIterable, FileIterable):
    datasets: dict[str, Dataset]
    traversal: Literal["depth", "breadth"] = "depth"
    """The traversal strategy for iterating over datasets in the group."""

    def iter_steps(self) -> Iterator[Step]:
        if self.traversal == "breadth":
            return roundrobin(*map(Dataset.iter_steps, self.datasets.values()))
        return chain.from_iterable(map(Dataset.iter_steps, self.datasets.values()))

    def iter_files(self) -> Iterator[File]:
        if self.traversal == "breadth":
            return roundrobin(*map(Dataset.iter_files, self.datasets.values()))
        return chain.from_iterable(map(Dataset.iter_files, self.datasets.values()))
