from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import chain, repeat
from typing import Literal

from more_itertools import roundrobin

from coffea.compute.context import (
    Context,
    ContextDataElement,
    update_context,
    with_context,
)
from coffea.compute.data.rootfile import FileIterable, OpenROOTFile, ROOTFileElement
from coffea.compute.data.step import ChunkElement, StepElement, StepIterable


@dataclass(frozen=True)
class ContextFile(Context):
    """Based on the File properties we'd like to pass as context to each step."""

    file_path: str
    uuid: str


@dataclass
class File(StepIterable[ContextFile]):
    path: str
    steps: list[tuple[int, int]]
    # TODO: object path within the file, file type, etc.
    uuid: str = ""

    def iter_steps(self) -> Iterator[ChunkElement[ContextFile]]:
        steps = map(StepElement, self.steps, repeat(self.path))
        context = ContextFile(file_path=self.path, uuid=self.uuid)
        return with_context(steps, context)


@dataclass(frozen=True)
class ContextDataset(Context):
    dataset_name: str
    """Name of the dataset."""
    cross_section: float | None
    """Cross section in pb."""


# Note: field order is from right to left when inheriting from multiple dataclasses


@dataclass(frozen=True)
class StepContextDataset(ContextDataset, ContextFile):
    """Metadata associated with a dataset."""

    @classmethod
    def update_from_file(
        cls,
        mixin: ContextDataset,
        base: ContextFile,
    ) -> "StepContextDataset":
        return StepContextDataset(**base.__dict__, **mixin.__dict__)


@dataclass
class Dataset(StepIterable[StepContextDataset], FileIterable[ContextDataset]):
    files: list[File]
    metadata: ContextDataset
    traversal: Literal["depth", "breadth"] = "depth"
    """The traversal strategy for iterating over files in the dataset.

    "depth" means to process all steps in one file before moving to the next file.
    "breadth" means to process the first step in all files, then the second step in all files, etc.
    """

    def iter_steps(self) -> Iterator[ChunkElement[StepContextDataset]]:
        if self.traversal == "breadth":
            iterable = roundrobin(*map(File.iter_steps, self.files))
        else:
            iterable = chain.from_iterable(map(File.iter_steps, self.files))
        return update_context(
            iterable, partial(StepContextDataset.update_from_file, self.metadata)
        )

    def iter_files(self) -> Iterator[ContextDataElement[OpenROOTFile, ContextDataset]]:
        files = map(ROOTFileElement, (f.path for f in self.files))
        return with_context(files, self.metadata)


@dataclass(frozen=True)
class ContextDataGroup:
    """Metadata associated with a data group."""

    group_name: str
    """Name of the data group."""


@dataclass(frozen=True)
class StepContextDataGroup(ContextDataGroup, StepContextDataset):
    @classmethod
    def update_from_dataset(
        cls,
        mixin: ContextDataGroup,
        base: StepContextDataset,
    ) -> "StepContextDataGroup":
        return StepContextDataGroup(**base.__dict__, **mixin.__dict__)


@dataclass(frozen=True)
class FileContextDataGroup(ContextDataGroup, ContextDataset):
    """Metadata associated with a data group file."""

    @classmethod
    def update_from_dataset(
        cls,
        mixin: ContextDataGroup,
        base: ContextDataset,
    ) -> "FileContextDataGroup":
        return FileContextDataGroup(**base.__dict__, **mixin.__dict__)


@dataclass
class DataGroup(StepIterable[StepContextDataGroup], FileIterable[FileContextDataGroup]):
    datasets: list[Dataset]
    metadata: ContextDataGroup
    traversal: Literal["depth", "breadth"] = "depth"
    """The traversal strategy for iterating over datasets in the group."""

    def iter_steps(self) -> Iterator[ChunkElement[StepContextDataGroup]]:
        if self.traversal == "breadth":
            iterable = roundrobin(*map(Dataset.iter_steps, self.datasets))
        else:
            iterable = chain.from_iterable(map(Dataset.iter_steps, self.datasets))
        return update_context(
            iterable,
            partial(
                StepContextDataGroup.update_from_dataset,
                self.metadata,
            ),
        )

    def iter_files(
        self,
    ) -> Iterator[ContextDataElement[OpenROOTFile, FileContextDataGroup]]:
        if self.traversal == "breadth":
            iterable = roundrobin(*map(Dataset.iter_files, self.datasets))
        else:
            iterable = chain.from_iterable(map(Dataset.iter_files, self.datasets))
        return update_context(
            iterable,
            partial(
                FileContextDataGroup.update_from_dataset,
                self.metadata,
            ),
        )


__all__ = [
    "ContextFile",
    "File",
    "ContextDataset",
    "StepContextDataset",
    "Dataset",
    "ContextDataGroup",
    "StepContextDataGroup",
    "FileContextDataGroup",
    "DataGroup",
]
