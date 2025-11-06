from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain, repeat
from typing import Callable, Literal

import uproot
from more_itertools import roundrobin
from uproot import ReadOnlyDirectory

from coffea.compute.func import EventsArray, ProcessorABC
from coffea.compute.protocol import ResultType

EventsFunc = Callable[[EventsArray], ResultType]
"Function that processes an EventsArray and returns a ResultType"
DirectoryFunc = Callable[[ReadOnlyDirectory], ResultType]
"Function that processes a uproot directory"


@dataclass(slots=True)
class Step:
    path: str
    entry_range: tuple[int, int]

    def __len__(self) -> int:
        return self.entry_range[1] - self.entry_range[0]


@dataclass(slots=True)
class StepWorkElement:
    func: EventsFunc
    step: Step

    def __call__(self) -> ResultType:
        # Dummy implementation of event loading
        info = str(self.step.entry_range)
        events = info + "A" * (len(self.step) - len(info))  # Dummy events
        return self.func(events)


class StepIterable(ABC):
    @abstractmethod
    def iter_steps(self) -> Iterator[Step]:
        """Return an iterator over steps in the computation."""
        raise NotImplementedError

    def map_steps(self, func: EventsFunc | ProcessorABC) -> "StepwiseComputable":
        """Apply a function or Processor to each step in this data."""
        if isinstance(func, ProcessorABC):
            func = func.process
        return StepwiseComputable(func=func, iterable=self)


@dataclass
class StepwiseComputable:
    func: EventsFunc
    iterable: StepIterable

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        return map(StepWorkElement, repeat(self.func), self.iterable.iter_steps())


@dataclass
class File(StepIterable):
    path: str
    steps: list[tuple[int, int]]

    def iter_steps(self) -> Iterator[Step]:
        return map(Step, repeat(self.path), self.steps)


@dataclass
class FileWorkElement:
    func: DirectoryFunc
    file: File

    def __call__(self) -> ResultType:
        # Dummy implementation of file loading
        file = uproot.open(self.file.path)
        assert isinstance(file, ReadOnlyDirectory)
        return self.func(file)


class FileIterable(ABC):
    @abstractmethod
    def iter_files(self) -> Iterator[File]:
        """Return an iterator over files in the computation."""
        raise NotImplementedError

    def map_files(self, func: DirectoryFunc) -> "FileComputable":
        return FileComputable(func=func, iterable=self)


@dataclass
class FileComputable:
    func: DirectoryFunc
    iterable: FileIterable

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
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
