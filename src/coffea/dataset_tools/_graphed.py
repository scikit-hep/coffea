"""Preprocessing with the graphed backend: imported only once graphed is known to be installed."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import awkward
from graphed.core import Partition
from graphed.core.execution import Plan, SequentialRunner, Task

from coffea.dataset_tools.preprocess_backends import (
    PreprocessJob,
    PreprocessTask,
    _CompletedTask,
)


@dataclass(frozen=True)
class _StepsOfBatch:
    """A ``Plan`` process: a task's partition names a dataset (``uri``) and the half-open range of
    its file rows (``entry_start``/``entry_stop``) to run that dataset's job on.
    """

    jobs: dict[str, PreprocessJob]

    def __call__(
        self, partition, resources
    ) -> dict[str, list[tuple[int, awkward.Array]]]:
        name = partition.uri
        job = self.jobs[name]
        batch = job.array[partition.entry_start : partition.entry_stop]
        return {name: [(partition.entry_start, job.map_fn(batch))]}


def _concat_batches(a: dict, b: dict) -> dict:
    return {k: a.get(k, []) + b.get(k, []) for k in {**a, **b}}


def _in_file_order(batches: dict, names) -> dict[str, awkward.Array]:
    # Plan.combine may run in any order; put each dataset's batches in file order
    return {
        name: awkward.concatenate([part for _, part in sorted(batches[name])], axis=0)
        for name in names
    }


class _GraphedFutureTask:
    """A ``PreprocessTask`` over the future of a graphed executor's ``submit(plan)``."""

    def __init__(self, future, names):
        self._future = future
        self._names = list(names)

    def result(self) -> dict[str, awkward.Array]:
        return _in_file_order(self._future.result().value, self._names)

    def partial_result(self) -> dict[str, awkward.Array]:
        # a graphed run reports only its combined value, so a partial result is the full one
        return self.result()

    def wait(self) -> None:
        self._future.result()


def submit(jobs: Mapping[str, PreprocessJob], executor) -> PreprocessTask:
    """Run ``jobs`` as one ``Plan`` with a task per ``files_per_batch`` file rows of each dataset.

    An empty dataset gets one task, so its result is a typed empty array as on the other backends.
    """
    runner = SequentialRunner() if executor is None else executor
    if not callable(getattr(runner, "run", None)):
        raise TypeError(
            "with backend='graphed', scheduler is a graphed executor (an object with run(plan)), "
            f"not {executor!r}"
        )
    tasks = []
    for name, job in jobs.items():
        step = max(1, job.files_per_batch)
        for lo in range(0, max(1, len(job.array)), step):
            tasks.append(Task(len(tasks), Partition(name, "", lo, lo + step)))
    plan = Plan(
        process=_StepsOfBatch(dict(jobs)),
        combine=_concat_batches,
        empty=dict,
        tasks=tasks,
    )
    if callable(getattr(runner, "submit", None)):
        return _GraphedFutureTask(runner.submit(plan), jobs)
    return _CompletedTask(_in_file_order(runner.run(plan).value, jobs))
