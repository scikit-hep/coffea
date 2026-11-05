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
    _batch_ranges,
    _CompletedTask,
    _ordered_concat,
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
        name: _ordered_concat(part for _, part in sorted(batches[name]))
        for name in names
    }


class _GraphedFutureTask:
    def __init__(self, future, names):
        self._future = future
        self._names = list(names)

    def result(self) -> dict[str, awkward.Array]:
        return _in_file_order(self._future.result().value, self._names)

    def partial_result(self) -> dict[str, awkward.Array]:
        if not self._future.done() or self._future.exception() is not None:
            return {}
        return self.result()

    def wait(self) -> None:
        self._future.result()


def submit(jobs: Mapping[str, PreprocessJob], executor) -> PreprocessTask:
    """Run ``jobs`` as one ``Plan`` with a task per batch on ``executor``, or on a
    ``SequentialRunner`` when it is None."""
    runner = SequentialRunner() if executor is None else executor
    tasks = [
        Task(i, Partition(name, "", start, stop))
        for i, (name, start, stop) in enumerate(
            (name, start, stop)
            for name, job in jobs.items()
            for start, stop in _batch_ranges(len(job.array), job.files_per_batch)
        )
    ]
    plan = Plan(
        process=_StepsOfBatch(dict(jobs)),
        combine=_concat_batches,
        empty=dict,
        tasks=tasks,
    )
    if callable(getattr(runner, "submit", None)):
        return _GraphedFutureTask(runner.submit(plan), jobs)
    try:
        value = runner.run(plan).value
    except Exception as err:
        return _CompletedTask({}, err)
    return _CompletedTask(_in_file_order(value, jobs))
