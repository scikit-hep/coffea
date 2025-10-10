from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cloudpickle
import fsspec
from rich import print

if TYPE_CHECKING:
    from coffea.processor import Accumulatable, ProcessorABC


class CheckpointerABC(metaclass=ABCMeta):
    """ABC for a generalized checkpointer

    Checkpointers are used to save chunk outputs to disk, and reload them if the same chunk is processed again.
    This is useful for long-running jobs that may be interrupted (resumable processing).

    Examples
    --------

    >>> from datetime import datetime
    >>> from coffea import processor
    >>> from coffea.processor import SimpleCheckpointer

    # create a checkpointer that stores checkpoints in a directory with the current date/time
    # (you may want to use a more specific directory in practice)
    >>> datestring = datetime.now().strftime("%Y%m%d%H")
    >>> checkpointer = SimpleCheckpointer(checkpoint_dir=f"checkpoints/{datestring}", verbose=True)

    # pass the checkpointer to a Runner
    >>> run = processor.Runner(..., checkpointer=checkpointer)
    >>> output = run(...)

    After the run, the checkpoints will be stored in the directory ``checkpoints/{datestring}``. On a subsequent run,
    if the same chunks are processed (and the same checkpointer, or rather ``checkpoint_dir`` is used),
    the results will be loaded from disk instead of being recomputed.
    """

    @abstractmethod
    def load(
        self, metadata: Any, processor_instance: ProcessorABC
    ) -> Accumulatable | None: ...

    @abstractmethod
    def save(
        self, output: Accumulatable, metadata: Any, processor_instance: ProcessorABC
    ) -> None: ...


class SimpleCheckpointer(CheckpointerABC):
    def __init__(
        self,
        checkpoint_dir: str,
        verbose: bool = False,
        overwrite: bool = True,
    ) -> None:
        fs, path = fsspec.url_to_fs(checkpoint_dir)
        self.fs = fs
        self.checkpoint_dir = path
        self.verbose = verbose
        self.overwrite = overwrite

    def filepath(self, metadata: Any, processor_instance: ProcessorABC) -> str:
        del processor_instance  # not used here, but could be in subclasses

        # build a path from metadata, how to include 'metadata["filename"]'? Is it needed?
        path = Path(self.checkpoint_dir)
        path /= metadata["dataset"]
        path /= metadata["fileuuid"]
        path /= metadata["treename"]
        path /= f"{metadata['entrystart']}-{metadata['entrystop']}.coffea"
        return str(path)

    def load(
        self, metadata: Any, processor_instance: ProcessorABC
    ) -> Accumulatable | None:
        fs = self.fs
        fpath = self.filepath(metadata, processor_instance)
        if not fs.exists(fpath):
            if self.verbose:
                print(
                    f"Checkpoint file {fpath} does not exist. May be the first run..."
                )
            return None
        # else:
        try:
            with fs.open(fpath, "rb", compression="lz4") as fin:
                output = cloudpickle.load(fin)
            return output

        except Exception as e:
            if self.verbose:
                print(f"Could not load checkpoint: {e}.")
            return None

    def save(
        self, output: Accumulatable, metadata: Any, processor_instance: ProcessorABC
    ) -> None:
        fs = self.fs
        fpath = self.filepath(metadata, processor_instance)
        # ensure directory exists
        fs.mkdirs(str(Path(fpath).parent), exist_ok=True)
        if fs.exists(fpath) and not self.overwrite:
            if self.verbose:
                print(f"Checkpoint file {fpath} already exists. Not overwriting...")
            return None
        # else:
        try:
            with fs.open(fpath, "wb", compression="lz4") as fout:
                output = cloudpickle.dump(output, fout)
        except Exception as e:
            if self.verbose:
                print(
                    f"Could not save checkpoint: {e}. Continuing without checkpointing..."
                )
        return None
