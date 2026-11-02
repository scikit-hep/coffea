"""Preprocessing with the graphed backend: imported only once graphed is known to be installed."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import awkward
from graphed.core import Partition
from graphed.core.execution import ExecResult, Plan, Task

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


class _ThreadRunner:
    """Runs a plan's tasks on a thread pool and combines their results in task-key order.

    Only for the plans this module builds, whose tasks need no resources and which set no stop,
    services or finalize. ``submit`` runs the plan on a thread of its own and returns a future of
    its result. The first task error is raised, and tasks not yet started are cancelled.
    """

    def __init__(self, max_workers: int | None = None):
        self.max_workers = max_workers

    def run(self, plan: Plan) -> ExecResult:
        tasks = sorted(plan.tasks, key=lambda t: t.key)
        pool = ThreadPoolExecutor(max_workers=self.max_workers)
        try:
            futures = [pool.submit(plan.process, t.partition, None) for t in tasks]
            value = plan.empty()
            for future in futures:
                value = plan.combine(value, future.result())
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
        return ExecResult(value=value, n_partitions=len(tasks), n_combines=len(tasks))

    def submit(self, plan: Plan) -> Future:
        future: Future = Future()

        def run():
            try:
                future.set_result(self.run(plan))
            except BaseException as err:
                future.set_exception(err)

        threading.Thread(target=run, daemon=True).start()
        return future


class _GraphedFutureTask:
    """A ``PreprocessTask`` over the future of a graphed executor's ``submit(plan)``."""

    def __init__(self, future, names):
        self._future = future
        self._names = list(names)

    def result(self) -> dict[str, awkward.Array]:
        return _in_file_order(self._future.result().value, self._names)

    def partial_result(self) -> dict[str, awkward.Array]:
        # a graphed run reports only its combined value: everything once done, nothing before
        if not self._future.done() or self._future.exception() is not None:
            return {}
        return self.result()

    def wait(self) -> None:
        self._future.result()


def submit(
    jobs: Mapping[str, PreprocessJob], executor, workers: int | None
) -> PreprocessTask:
    """Run ``jobs`` as one ``Plan`` with a task per batch of each dataset (:func:`_batch_ranges`),
    on ``executor``, or on a :class:`_ThreadRunner` of ``workers`` threads when it is None.
    """
    runner = _ThreadRunner(workers) if executor is None else executor
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
