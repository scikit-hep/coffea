"""Common code for pool-based backends.

Any execution backend that conforms to the Pool protocol defined in this module should be
able to use the common code in this module to implement the compute Backend protocol.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import dataclass
from functools import partial, reduce
from typing import (
    Generic,
    Protocol,
    TypeVar,
)

from coffea.compute.errors import ErrorAction, ErrorPolicy, FailedWorkElemement
from coffea.compute.protocol import (
    Computable,
    EmptyResult,
    ResultT,
    ResultWrapper,
    TaskStatus,
    WorkElement,
)

T = TypeVar("T")


class Pool(Protocol):
    def submit(self, item: Callable[[], T], /) -> Future[T]: ...


@dataclass
class WorkWrapper(Generic[ResultT]):
    """Run work and capture any exceptions, returning a ResultWrapper with either the result or the exception info."""

    work: WorkElement[ResultT]
    producer_key: str

    def preload(self) -> None:
        self.work.preload()

    def _range(self) -> tuple[int, int]:
        return (self.work.start, self.work.stop)

    def __call__(self) -> ResultWrapper[ResultT]:
        try:
            res = self.work()
        except Exception as e:
            failed = [(self._range(), e)]
            return ResultWrapper(EmptyResult(), self.producer_key, [], failed)
        else:
            completed = [self._range()]
            return ResultWrapper(res, self.producer_key, completed, [])

    @staticmethod
    def gen_steps(
        item: Computable[ResultT],
    ) -> Generator[WorkWrapper[ResultT], int | None, None]:
        gen = item.gen_steps()
        send = None
        while True:
            try:
                step = gen.send(send)
            except StopIteration:
                break
            send = yield WorkWrapper(step, producer_key=item.key)


class Submitter:
    """Submit work items to the pool, yielding futures as they complete.

    Arguments:
        pool: the pool to submit work to
        error_policy: policy for handling errors encountered during work execution
        max_inflight: maximum number of work items to have inflight at once
        poll_interval: how often to poll for cancel calls (in seconds)

    Note: max_inflight should be tuned according to the expected number of
    workers in the pool.
    """

    exception: Exception | None

    def __init__(
        self,
        pool: Pool,
        *,
        error_policy: ErrorPolicy,
        max_inflight: int,
        poll_interval: float = 0.1,
    ) -> None:
        self.pool = pool
        self.error_policy = error_policy
        self.max_inflight = max_inflight
        self.poll_interval = poll_interval
        self.exception = None

    def cancel(self, exception: Exception) -> None:
        """Cancel the entire computation with the given exception.

        Since the submitter is the source of new work items, we use
        it as the point to trigger cancellation of the entire computation
        when a fatal error is encountered.
        """
        self.exception = exception

    def __call__(
        self, work: Iterator[WorkWrapper[ResultT]]
    ) -> Generator[Future[ResultWrapper[ResultT]]]:
        inflight: dict[
            Future[ResultWrapper[ResultT]],
            WorkWrapper[ResultT] | FailedWorkElemement[ResultWrapper[ResultT]],
        ] = {}
        while (element := next(work, None)) or inflight:
            if element:
                fut = self.pool.submit(element)
                inflight[fut] = element
            if len(inflight) >= self.max_inflight or element is None:
                done, _ = wait(
                    set(inflight),
                    timeout=self.poll_interval,
                    return_when=FIRST_COMPLETED,
                )
                for fut in done:
                    old_element = inflight.pop(fut)
                    assert (
                        fut.exception() is None
                    ), "WorkWrapper futures should not raise exceptions"
                    exception = fut.result().first_failure()
                    if exception is None:
                        yield fut
                        continue
                    if isinstance(old_element, WorkWrapper):
                        new_element, action = self.error_policy.first_action(
                            old_element, exception
                        )
                    else:
                        new_element, action = self.error_policy.retry_action(
                            old_element, exception
                        )
                    if action == ErrorAction.CANCEL:
                        self.cancel(exception)
                    elif action == ErrorAction.RETRY:
                        # TODO: wait some time since new_element.last_attempt before retrying?
                        fut = self.pool.submit(new_element)
                        inflight[fut] = new_element
                    else:
                        yield fut
            if self.exception is not None:
                for fut in inflight:
                    fut.cancel()
                break


def add(xs: Iterable[Future[ResultT]]) -> ResultT:
    return reduce(operator.add, (x.result() for x in xs))


class Reducer:
    """Merge futures into larger sums, yielding futures for the merged results.

    Arguments:
        pool: the thread pool to submit merge work to
        merge_every: how many futures to merge at once
    """

    def __init__(self, pool: Pool, *, merge_every: int):
        self.pool = pool
        self.merge_every = merge_every

    def __call__(self, work: Iterable[Future[ResultT]]) -> Generator[Future[ResultT]]:
        workiter = iter(work)
        ready: list[Future[ResultT]] = []
        while (item := next(workiter, None)) or ready:
            if item:
                ready.append(item)
            if len(ready) >= self.merge_every or item is None:
                # print(f"Submitting merge of {len(ready)} elements")
                yield self.pool.submit(partial(add, ready))
                ready = []


class ResultHolder(Generic[ResultT]):
    """Box to hold the (partial) result of an accumulation.

    The data is set via callbacks from futures as partial results become ready.
    """

    _error: Exception | None

    def __init__(self, initial: ResultT) -> None:
        self._result = initial
        self._error = None

    @property
    def result(self) -> ResultT:
        if self._error is not None:
            msg = "Computation failed with an internal error; check the chained exception for details."
            raise RuntimeError(msg) from self._error
        return self._result

    @property
    def failed(self) -> bool:
        """Whether the computation failed with an exception during merging or accumulation."""
        return self._error is not None

    def set_result(self, fut: Future[ResultT]) -> None:
        try:
            self._result = fut.result()
        except Exception as e:
            self._error = e


class Accumulator(Generic[ResultT]):
    """Accumulate futures into a single final result.

    Places partial sums into a ResultHolder as they become available.

    Arguments:
        pool: the thread pool to submit merge work to
        work: an iterable of futures to accumulate
    """

    def __init__(
        self,
        pool: Pool,
        work: Generator[Future[ResultT]],
        holder: ResultHolder[ResultT],
    ):
        self.pool = pool
        self.work = work
        self.holder = holder

    def __call__(self) -> ResultT:
        partial_sum = next(self.work, None)
        if partial_sum is None:
            raise RuntimeError("No work to accumulate")
        partial_sum.add_done_callback(self.holder.set_result)
        while (fut := next(self.work, None)) and not self.holder.failed:
            partial_sum = self.pool.submit(partial(add, (partial_sum, fut)))
            partial_sum.add_done_callback(self.holder.set_result)
        self.holder.set_result(partial_sum)
        return partial_sum.result()


class PoolTask(Generic[ResultT]):
    _future: Future[ResultWrapper[ResultT]]
    _holder: ResultHolder[ResultWrapper[ResultT]]
    _submitter: Submitter

    def __init__(
        self,
        future: Future[ResultWrapper[ResultT]],
        holder: ResultHolder[ResultWrapper[ResultT]],
        submitter: Submitter,
    ) -> None:
        self._future = future
        self._holder = holder
        self._submitter = submitter

    def result(self) -> ResultT | EmptyResult:
        if self._submitter.exception is not None:
            msg = (
                "Computation failed with a fatal error; check the chained exception for details.\n"
                " Use .partial_result() to access any partial results that were produced before\n"
                " the failure. Note that you can adjust the ErrorPolicy to mark some exceptions\n"
                " as non-fatal and allow the computation to continue."
            )
            raise RuntimeError(msg) from self._submitter.exception
        res_wrapper = self._future.result()
        if res_wrapper.failed:
            msg = (
                f"Computation failed with {len(res_wrapper.failed)} non-fatal errors;\n"
                " the first error is shown in the chained exception above."
                f" Use {str(type(self))}.partial_result() to access partial results and\n"
                "  the remaining work to be done.\n"
            )
            raise RuntimeError(msg) from res_wrapper.first_failure()
        return res_wrapper.result

    def partial_result(self) -> ResultWrapper[ResultT]:
        return self._holder.result

    def wait(self) -> None:
        wait((self._future,))

    def status(self) -> TaskStatus:
        if self._future.running():
            return TaskStatus.RUNNING
        elif self._future.done():
            if self._submitter.exception is not None:
                return TaskStatus.CANCELLED
            res_wrapper = self._future.result()
            if res_wrapper.failed:
                return TaskStatus.INCOMPLETE
            else:
                return TaskStatus.COMPLETE
        else:
            return TaskStatus.PENDING

    def done(self) -> bool:
        return self.status().done()

    def cancel(self) -> None:
        self._submitter.cancel(RuntimeError("Task cancelled by user"))
