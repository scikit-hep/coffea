"""Preprocessing in graphed mode: imported only once graphed is known to be installed."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import awkward
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, Task


@dataclass(frozen=True)
class _StepsOfBatch:
    """A ``Plan`` process: a task's partition names a dataset (``uri``) and the half-open range of
    its file rows (``entry_start``/``entry_stop``) to run that dataset's steps function on.
    """

    norm_files: dict[str, awkward.Array]
    steps_fns: dict[str, Callable]

    def __call__(
        self, partition, resources
    ) -> dict[str, list[tuple[int, awkward.Array]]]:
        name = partition.uri
        batch = self.norm_files[name][partition.entry_start : partition.entry_stop]
        return {name: [(partition.entry_start, self.steps_fns[name](batch))]}


def _concat_batches(a: dict, b: dict) -> dict:
    return {k: a.get(k, []) + b.get(k, []) for k in {**a, **b}}


def compute_steps(
    norm_files: dict[str, awkward.Array],
    steps_fns: dict[str, Callable],
    files_per_batch: int,
    scheduler,
) -> dict[str, awkward.Array]:
    runner = SequentialRunner() if scheduler is None else scheduler
    if not callable(getattr(runner, "run", None)):
        raise TypeError(
            "with mode='graphed', scheduler is a graphed executor (an object with run(plan)), "
            f"not {scheduler!r}"
        )
    partitions = [
        Partition(name, "", lo, lo + files_per_batch)
        for name, files in norm_files.items()
        for lo in range(0, len(files), files_per_batch)
    ]
    plan = Plan(
        process=_StepsOfBatch(norm_files, steps_fns),
        combine=_concat_batches,
        empty=dict,
        tasks=[Task(i, p) for i, p in enumerate(partitions)],
    )
    batches = runner.run(plan).value
    # Plan.combine may run in any order; put each dataset's batches in file order
    return {
        name: awkward.concatenate([part for _, part in sorted(batches[name])], axis=0)
        for name in norm_files
    }
