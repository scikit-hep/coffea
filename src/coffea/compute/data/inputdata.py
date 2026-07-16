from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from itertools import chain

from coffea.compute.context import ContextDataElement, update_context, with_context
from coffea.compute.data.processable_data import (
    ContextDataGroup,
    ContextDataset,
    FileContextDataGroup,
)
from coffea.compute.data.rootfile import FileIterable, OpenROOTFile, ROOTFileElement


@dataclass
class InputDataset(FileIterable[ContextDataset]):
    files: list[str]
    metadata: ContextDataset

    def iter_files(self) -> Iterator[ContextDataElement[OpenROOTFile, ContextDataset]]:
        files = map(ROOTFileElement, self.files)
        return with_context(files, self.metadata)


@dataclass
class InputDataGroup(FileIterable[FileContextDataGroup]):
    datasets: list[InputDataset]
    metadata: ContextDataGroup

    def iter_files(
        self,
    ) -> Iterator[ContextDataElement[OpenROOTFile, FileContextDataGroup]]:
        iterable = chain.from_iterable(map(InputDataset.iter_files, self.datasets))
        return update_context(
            iterable,
            partial(
                FileContextDataGroup.update_from_dataset,
                self.metadata,
            ),
        )


__all__ = [
    "InputDataset",
    "InputDataGroup",
]
