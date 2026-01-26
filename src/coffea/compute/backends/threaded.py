from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import starmap
from queue import Queue
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Generic

from typing_extensions import Self

from coffea.compute.backends._genericbkg import ConcurrentTask, TaskState, _Shutdown
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
class _ThreadedTaskState(TaskState[InputT, ResultT]):
    _iter: Iterator[TaskElement[InputT, ResultT]]
    "Iterator over the original computable's task elements input"
    _output: ResultT | EmptyResult = EmptyResult()
    next_index: int = 0
    failures: list[FailedTaskElement[InputT, ResultT]] = field(default_factory=list)
    _status: TaskStatus = TaskStatus.PENDING

    @property
    def status(self) -> TaskStatus:
        return self._status

    @classmethod
    def from_computable(cls, item: Computable[InputT, ResultT]) -> Self:
        return cls(_iter=starmap(TaskElement, enumerate(item)))

    def first_failure(self) -> FailedTaskElement[InputT, ResultT] | None:
        if self.failures:
            return self.failures[0]
        return None

    def num_failures(self) -> int:
        return len(self.failures)

    def output(self) -> ResultT | EmptyResult:
        return self._output

    def cancel(self) -> Self:
        self._status = TaskStatus.CANCELLED
        return self

    def add_failure_and_cancel(
        self, element: FailedTaskElement[InputT, ResultT]
    ) -> Self:
        self.next_index = self.next_index + 1
        self.failures = self.failures + [element]
        self._status = TaskStatus.CANCELLED
        return self

    def add_failure_and_continue(
        self, element: FailedTaskElement[InputT, ResultT]
    ) -> Self:
        self.next_index = self.next_index + 1
        self.failures = self.failures + [element]
        self._status = TaskStatus.RUNNING
        return self

    # TODO: would like to have add_success_and_continue
    # but it doesn't like ResultT being covariant
    # def add_success_and_continue(self, result: ResultT) -> Self:
    #     self._output = self._output + result
    #     self.next_index = self.next_index + 1
    #     self._status = TaskStatus.RUNNING
    #     return self

    def get_continuation(
        self, original: Computable[InputT, ResultT]
    ) -> Continuation[InputT, ResultT]:
        return Continuation(
            original=original,
            status=self.status,
            failed_indices=frozenset(element.index for element in self.failures),
            continue_at=self.next_index,
        )

    def advance(self, error_policy: ErrorPolicy) -> Self:
        try:
            element = next(self._iter)
        except StopIteration:
            if self.num_failures() > 0:
                self._status = TaskStatus.INCOMPLETE
            else:
                self._status = TaskStatus.COMPLETE
            return self

        try:
            result = element()
        except Exception as ex:
            new_element, action = error_policy.first_action(element, ex)
            if action == ErrorAction.CANCEL:
                return self.add_failure_and_cancel(new_element)
            elif action == ErrorAction.CONTINUE:
                return self.add_failure_and_continue(new_element)
        else:
            self._output = self._output + result
            self.next_index = self.next_index + 1
            self._status = TaskStatus.RUNNING
            return self

        # Now handle retries
        assert action == ErrorAction.RETRY  # (proven by control flow)
        while True:
            try:
                result = new_element()
            except Exception as ex:
                new_element, action = error_policy.retry_action(new_element, ex)
                if action == ErrorAction.CANCEL:
                    return self.add_failure_and_cancel(new_element)
                elif action == ErrorAction.CONTINUE:
                    return self.add_failure_and_continue(new_element)
            else:
                self._output = self._output + result
                self.next_index = self.next_index + 1
                self._status = TaskStatus.RUNNING
                return self
            assert action == ErrorAction.RETRY


def _work(task_queue: Queue[ConcurrentTask]) -> None:  # type: ignore[type-arg]
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


class ThreadedTask(ConcurrentTask[InputT, ResultT]):
    @classmethod
    def from_computable(
        cls, item: Computable[InputT, ResultT], error_policy: ErrorPolicy
    ) -> Self:
        state = _ThreadedTaskState.from_computable(item)
        return cls(item, error_policy, state)


class SingleThreadedBackend:
    _task_queue: Queue[ConcurrentTask | _Shutdown] | None  # type: ignore[type-arg]
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
        task = ThreadedTask.from_computable(item, error_policy)
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
