from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from coffea.util import load, save

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
    >>> from coffea.processor import LocalCheckpointer

    # create a checkpointer that stores checkpoints in a directory with the current date/time
    # (you may want to use a more specific directory in practice)
    >>> datestring = datetime.now().strftime("%Y%m%d%H")
    >>> checkpointer = LocalCheckpointer(checkpoint_dir=f"checkpoints/{datestring}", verbose=True)

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


class LocalCheckpointer(CheckpointerABC):
    def __init__(
        self,
        checkpoint_dir: str | Path,
        verbose: bool = False,
        overwrite: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.verbose = verbose
        self.overwrite = overwrite

    def filepath(self, metadata: Any, processor_instance: ProcessorABC) -> str:
        del processor_instance  # not used here, but could be in subclasses

        # build a path from metadata, how to include 'metadata["filename"]'? Is it needed?
        path = self.checkpoint_dir
        path /= metadata["dataset"]
        path /= metadata["fileuuid"]
        path /= metadata["treename"]
        path /= f"{metadata['entrystart']}-{metadata['entrystop']}.coffea"
        return path

    def load(
        self, metadata: Any, processor_instance: ProcessorABC
    ) -> Accumulatable | None:
        fpath = self.filepath(metadata, processor_instance)
        if not fpath.exists():
            if self.verbose:
                print(
                    f"Checkpoint file {fpath} does not exist. May be the first run..."
                )
            return None
        # else:
        try:
            return load(fpath)
        except Exception as e:
            if self.verbose:
                print(f"Could not load checkpoint: {e}.")
            return None

    def save(
        self, output: Accumulatable, metadata: Any, processor_instance: ProcessorABC
    ) -> None:
        fpath = self.filepath(metadata, processor_instance)
        # ensure directory exists
        fpath.parent.mkdir(parents=True, exist_ok=True)
        if fpath.exists() and not self.overwrite:
            if self.verbose:
                print(f"Checkpoint file {fpath} already exists. Not overwriting...")
            return None
        # else:
        try:
            save(output, fpath)
        except Exception as e:
            if self.verbose:
                print(
                    f"Could not save checkpoint: {e}. Continuing without checkpointing..."
                )
        return None
