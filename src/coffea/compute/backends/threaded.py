from dataclasses import dataclass, field
from itertools import starmap
from queue import Queue, ShutDown
from threading import Condition, Thread
from typing import Callable
from collections.abc import Iterator

from coffea.compute.errors import (
    ErrorAction,
    ErrorPolicy,
    FailedTaskElement,
    TaskElement,
)
from coffea.compute.protocol import Computable, EmptyResult, ResultType, TaskStatus


@dataclass
class Continuation:
    original: Computable
    "The original computable item."
    status: TaskStatus
    "The status of the original computation task."
    failed_indices: frozenset[int]
    "Indices of task elements that failed in the original computation."
    continue_at: int
    "Index to continue processing from, in the case where the original task was cancelled."

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        for i, task_element in enumerate(self.original):
            if i in self.failed_indices or i >= self.continue_at:
                yield task_element


@dataclass
class _TaskState:
    output: ResultType = EmptyResult()
    next_index: int = 0
    failures: list[FailedTaskElement] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING

    def get_continuation(self, original: Computable) -> Continuation:
        return Continuation(
            original=original,
            status=self.status,
            failed_indices=frozenset(element.index for element in self.failures),
            continue_at=self.next_index,
        )


def _try_advance(
    state: _TaskState, element: TaskElement, error_policy: ErrorPolicy
) -> _TaskState:
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


class ThreadedTask:
    item: Computable
    error_policy: ErrorPolicy
    _iter: Iterator[TaskElement]
    _state: _TaskState
    "To be modified only under _cv lock"
    _cv: Condition

    def __init__(self, item: Computable, error_policy: ErrorPolicy) -> None:
        self.item = item
        self.error_policy = error_policy
        self._iter = starmap(TaskElement, enumerate(item))
        self._state = _TaskState()
        self._cv = Condition()

    def result(self) -> ResultType:
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

    def partial_result(self) -> tuple[ResultType, Continuation]:
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


def _work(task_queue: Queue[ThreadedTask]) -> None:
    while True:
        try:
            task = task_queue.get()
        except ShutDown:
            break
        task._run()
        task_queue.task_done()


class SingleThreadedBackend:
    _task_queue: Queue[ThreadedTask] | None
    _thread: Thread | None

    def __init__(self) -> None:
        self._task_queue = None
        self._thread = None

    def __enter__(self) -> "SingleThreadedBackend":
        self._task_queue = Queue()
        self._thread = Thread(
            target=_work,
            name="SingleThreadedBackend",
            args=(self._task_queue,),
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._thread and self._task_queue
        self._task_queue.shutdown()
        self._thread.join()
        self._task_queue = None
        self._thread = None

    def compute(
        self, item: Computable, *, error_policy: ErrorPolicy = ErrorPolicy()
    ) -> ThreadedTask:
        if not self._task_queue:
            raise RuntimeError("Backend compute called outside of context manager")
        if hasattr(item, "__next__"):
            raise TypeError("Computable items must be iterables, not iterators")
        task = ThreadedTask(item, error_policy)
        self._task_queue.put(task)
        return task
