from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import starmap
from queue import Queue
from threading import Condition, Thread
from types import TracebackType
from typing import TYPE_CHECKING, Generic

from coffea.compute.errors import (
    ErrorAction,
    ErrorPolicy,
    FailedTaskElement,
    TaskElement,
)
from coffea.compute.protocol import (
    Backend,
    Computable,
    EmptyResult,
    InputT,
    ResultT,
    Task,
    TaskStatus,
    WorkElement,
)


@dataclass
class Continuation(Generic[InputT, ResultT]):
    original: Computable[InputT, ResultT]
    "The original computable item."
    status: TaskStatus
    "The status of the original computation task."
    failed_indices: frozenset[int]
    "Indices of task elements that failed in the original computation."
    continue_at: int
    "Index to continue processing from, in the case where the original task was cancelled."

    def __iter__(self) -> Iterator[WorkElement[InputT, ResultT]]:
        for i, task_element in enumerate(self.original):
            if i in self.failed_indices or i >= self.continue_at:
                yield task_element


@dataclass
class _TaskState(Generic[InputT, ResultT]):
    output: ResultT | EmptyResult = EmptyResult()
    next_index: int = 0
    failures: list[FailedTaskElement[InputT, ResultT]] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING

    def get_continuation(
        self, original: Computable[InputT, ResultT]
    ) -> Continuation[InputT, ResultT]:
        return Continuation(
            original=original,
            status=self.status,
            failed_indices=frozenset(element.index for element in self.failures),
            continue_at=self.next_index,
        )


def _try_advance(
    state: _TaskState[InputT, ResultT],
    element: TaskElement[InputT, ResultT],
    error_policy: ErrorPolicy,
) -> _TaskState[InputT, ResultT]:
    try:
        result = element()
    except Exception as ex:
        new_element, action = error_policy.first_action(element, ex)
        if action == ErrorAction.CANCEL:
            return _TaskState(
                output=state.output,
                next_index=state.next_index + 1,
                failures=state.failures + [new_element],
                status=TaskStatus.CANCELLED,
            )
        elif action == ErrorAction.CONTINUE:
            return _TaskState(
                output=state.output,
                next_index=state.next_index + 1,
                failures=state.failures + [new_element],
                status=TaskStatus.RUNNING,
            )
    else:
        # This could use a more sophisticated merging strategy
        return _TaskState(
            output=state.output + result,
            next_index=state.next_index + 1,
            failures=state.failures,
            status=TaskStatus.RUNNING,
        )
    # Now handle retries
    assert action == ErrorAction.RETRY  # (proven by control flow)
    while True:
        try:
            result = new_element()
        except Exception as ex:
            new_element, action = error_policy.retry_action(new_element, ex)
            if action == ErrorAction.CANCEL:
                return _TaskState(
                    output=state.output,
                    next_index=state.next_index + 1,
                    failures=state.failures + [new_element],
                    status=TaskStatus.CANCELLED,
                )
            elif action == ErrorAction.CONTINUE:
                return _TaskState(
                    output=state.output,
                    next_index=state.next_index + 1,
                    failures=state.failures + [new_element],
                    status=TaskStatus.RUNNING,
                )
        else:
            # This could use a more sophisticated merging strategy
            return _TaskState(
                output=state.output + result,
                next_index=state.next_index + 1,
                failures=state.failures,
                status=TaskStatus.RUNNING,
            )
        assert action == ErrorAction.RETRY


class ThreadedTask(Generic[InputT, ResultT]):
    item: Computable[InputT, ResultT]
    error_policy: ErrorPolicy
    _iter: Iterator[TaskElement[InputT, ResultT]]
    _state: _TaskState[InputT, ResultT]
    "To be modified only under _cv lock"
    _cv: Condition

    def __init__(
        self, item: Computable[InputT, ResultT], error_policy: ErrorPolicy
    ) -> None:
        self.item = item
        self.error_policy = error_policy
        self._iter = starmap(TaskElement, enumerate(item))
        self._state = _TaskState()
        self._cv = Condition()

    def result(self) -> ResultT | EmptyResult:
        self.wait()
        if self._state.failures:
            # Reraise the first error encountered
            msg = (
                f"Computation failed with {len(self._state.failures)} errors;\n"
                " use Task.partial_result() to access partial results and a\n"
                " continuation Computable for the remaining work.\n"
                " You can also adjust the ErrorPolicy to continue on certain errors.\n"
                " The first error is shown in the chained exception above."
            )
            raise RuntimeError(msg) from self._state.failures[0].exception
        return self._state.output

    def partial_result(
        self,
    ) -> tuple[ResultT | EmptyResult, Continuation[InputT, ResultT]]:
        # Hold lock so we get a consistent snapshot of state
        with self._cv:
            return self._state.output, self._state.get_continuation(self.item)

    def wait(self) -> None:
        with self._cv:
            self._cv.wait_for(self.done)

    def status(self) -> TaskStatus:
        return self._state.status

    def done(self) -> bool:
        return self._state.status.done()

    def cancel(self) -> None:
        with self._cv:
            self._state.status = TaskStatus.CANCELLED
            self._cv.notify_all()

    def _run(self) -> None:
        """Run the task to completion.
        This is intended to be called by a single worker thread.
        """
        for task_element in self._iter:
            next_state = _try_advance(self._state, task_element, self.error_policy)
            with self._cv:
                # First check if we were aborted while working
                if self._state.status == TaskStatus.CANCELLED:
                    self._cv.notify_all()
                    return
                self._state = next_state
                if self._state.status.done():
                    self._cv.notify_all()
                    return
        with self._cv:
            assert self._state.status == TaskStatus.RUNNING
            if self._state.failures:
                self._state.status = TaskStatus.INCOMPLETE
            else:
                self._state.status = TaskStatus.COMPLETE
            self._cv.notify_all()


class _Shutdown:
    """A sentinel to signal worker threads to exit.

    In python 3.13+, we can use queue.ShutDown directly.
    """

    pass


def _work(task_queue: Queue[ThreadedTask]) -> None:  # type: ignore[type-arg]
    while True:
        task = task_queue.get()
        if isinstance(task, _Shutdown):
            task_queue.task_done()
            break
        try:
            task._run()
        except Exception:
            # Any exceptions not caught by the task itself are bugs in the backend
            # TODO: find a way to report these in the user thread
            task.cancel()
        task_queue.task_done()


class SingleThreadedBackend:
    _task_queue: Queue[ThreadedTask | _Shutdown] | None  # type: ignore[type-arg]
    _thread: Thread | None

    def __init__(self) -> None:
        self._task_queue = None
        self._thread = None

    def __enter__(self) -> "RunningSingleThreadedBackend":
        self._task_queue = Queue()
        self._thread = Thread(
            target=_work,
            name="SingleThreadedBackend",
            args=(self._task_queue,),
        )
        self._thread.start()
        return RunningSingleThreadedBackend(self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._thread and self._task_queue
        self._task_queue.put(_Shutdown())
        self._thread.join()
        self._task_queue = None
        self._thread = None


class RunningSingleThreadedBackend:
    _backend: SingleThreadedBackend

    def __init__(self, backend: SingleThreadedBackend):  # type: ignore[type-arg]
        self._backend = backend

    def compute(
        self,
        item: Computable[InputT, ResultT],
        /,
        error_policy: ErrorPolicy = ErrorPolicy(),
    ) -> ThreadedTask[InputT, ResultT]:
        if self._backend._task_queue is None:
            raise RuntimeError(
                "Cannot compute on a backend that has been exited from its context manager"
            )
        if hasattr(item, "__next__"):
            raise TypeError("Computable items must be iterables, not iterators")
        task = ThreadedTask(item, error_policy)
        self._backend._task_queue.put(task)
        return task

    def wait_all(self, progress: bool = False) -> None:
        """Wait for all tasks in the backend to complete.

        Parameters
        ----------
        progress : bool, optional
            If True, display a progress bar while waiting, by default False.
        """
        if progress:
            raise NotImplementedError("Progress bars are not yet implemented")
        else:
            if self._backend._task_queue:
                self._backend._task_queue.join()


if TYPE_CHECKING:
    # TODO: is this the best way to do this?
    check1: type[Task] = ThreadedTask
    check2: type[Backend] = SingleThreadedBackend
