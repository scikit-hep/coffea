import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from coffea.compute.backends.common import (
    Accumulator,
    PoolTask,
    Reducer,
    ResultHolder,
    Submitter,
    WorkWrapper,
)
from coffea.compute.errors import ErrorPolicy
from coffea.compute.protocol import (
    Backend,
    Computable,
    EmptyResult,
    ResultT,
    ResultWrapper,
    Task,
)


class RunningThreadedBackend:
    def __init__(
        self, pool: ThreadPoolExecutor, tasks: set[weakref.ReferenceType[PoolTask]]
    ) -> None:
        self.pool = pool
        self.tasks = tasks

    def compute(
        self,
        item: Computable[ResultT],
        /,
        error_policy: ErrorPolicy = ErrorPolicy(),
        merge_every: int = 4,
    ) -> PoolTask[ResultT]:
        """Start a computation using the thread pool and return a Task object to track it.

        Args:
            item: The Computable item to compute.
            error_policy: Policy for handling errors during computation.
            merge_every: How many partial results to merge together at a time
                (affects the granularity of intermediate results and memory usage)
        """
        max_inflight = self.pool._max_workers + 1  # TODO: tune this parameter
        work = WorkWrapper.gen_steps(item)
        submitter = Submitter(
            self.pool, error_policy=error_policy, max_inflight=max_inflight
        )
        reducer = Reducer(self.pool, merge_every=merge_every)
        accumulator = Accumulator(
            self.pool,
            reducer(submitter(work)),
            ResultHolder(ResultWrapper[ResultT](EmptyResult(), item.key, [], [])),
        )
        future = self.pool.submit(accumulator)
        task = PoolTask(future, accumulator.holder, submitter)
        self.tasks.add(weakref.ref(task))
        return task


class ThreadedBackend:
    """A compute backend that uses a thread pool to execute work items in parallel.

    It submits the work items, reduction, and accumulation tasks all to the same thread pool.

    Args:
        pool_factory: A callable that returns a new ThreadPoolExecutor when called. This allows
            for custom configuration of the thread pool (e.g. number of workers, thread name prefix, etc.).
        cancel_on_exit: If True, any tasks that are running when the context manager is exited will be cancelled.
            If False (default), the backend will wait for all tasks to complete before exiting.
    """

    def __init__(
        self,
        pool_factory: Callable[[], ThreadPoolExecutor] = ThreadPoolExecutor,
        cancel_on_exit: bool = False,
    ):
        self.pool_factory = pool_factory
        self.cancel_on_exit = cancel_on_exit

    def __enter__(self) -> RunningThreadedBackend:
        self.pool = self.pool_factory()
        self.tasks: set[weakref.ReferenceType[PoolTask]] = set()
        self.pool.__enter__()
        return RunningThreadedBackend(pool=self.pool, tasks=self.tasks)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for ref in self.tasks:
            task = ref()
            if task is None:
                continue
            if self.cancel_on_exit:
                task.cancel()
            else:
                task.wait()
        del self.tasks
        self.pool.__exit__(exc_type, exc_value, traceback)
        del self.pool

    def compute(
        self,
        item: Computable[ResultT],
        /,
        error_policy: ErrorPolicy = ErrorPolicy(),
        merge_every: int = 4,
    ) -> ResultWrapper[ResultT]:
        """Blocking computation. See RunningThreadedBackend.compute for details."""
        with self as backend:
            task = backend.compute(
                item,
                error_policy=error_policy,
                merge_every=merge_every,
            )
            task.wait()
            return task.partial_result()


if TYPE_CHECKING:
    _t: type[Task] = PoolTask
    _x: type[Backend] = ThreadedBackend
