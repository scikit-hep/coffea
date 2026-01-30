"""A generic background thread implementation for use with compute backends.

This allows for a threadsafe way to return a future compatible with the compute.Task protocol.

This is a private module as it is only intended for use by compute backends.
"""

from abc import ABC, abstractmethod
from queue import Queue
from threading import Condition, Thread
from types import TracebackType
from typing import Generic, TypeVar

from typing_extensions import Self

from coffea.compute.errors import (
    ErrorPolicy,
    FailedTaskElement,
)
from coffea.compute.protocol import Computable, EmptyResult, ResultT, TaskStatus


class _Shutdown:
    """A sentinel to signal worker threads to exit.

    In python 3.13+, we can use queue.ShutDown directly.
    """

    pass


class TaskState(ABC, Generic[ResultT]):
    """A base implementation for internal task state to be used by backend ConcurrentTask implementations."""

    @property
    @abstractmethod
    def status(self) -> TaskStatus:
        """The current status of the task."""
        ...

    @classmethod
    @abstractmethod
    def from_computable(cls, item: Computable[ResultT]) -> Self:
        """Create an initial TaskState from a Computable.

        Args:
            item: The Computable to create the TaskState for.

        Returns:
            An initial TaskState for the Computable.
            Its status should be TaskStatus.PENDING.
        """
        ...

    @abstractmethod
    def first_failure(self) -> FailedTaskElement[ResultT] | None:
        """Return the first failure encountered, or None if no failures occurred.

        If not None, the ConcurrentTask.result() will raise the exception from this failure.
        """
        ...

    @abstractmethod
    def num_failures(self) -> int:
        """Return the number of failures encountered so far."""
        ...

    @abstractmethod
    def output(self) -> ResultT | EmptyResult:
        """Return the current output of the task."""
        ...

    @abstractmethod
    def cancel(self) -> Self:
        """Mark the task as cancelled.

        Returns:
            The TaskState after being marked as cancelled.
            Its status should be TaskStatus.CANCELLED.
        """
        ...

    @abstractmethod
    def get_continuation(self, original: Computable[ResultT]) -> Computable[ResultT]:
        """Get a continuation Computable for the remaining work.

        Args:
            original: The original Computable that was being processed.

        Returns:
            A Computable representing the remaining work to be done.
        """
        ...

    @abstractmethod
    def advance(self, error_policy: ErrorPolicy) -> Self:
        """Advance the task state by processing a single TaskElement.

        This should block until some meaningful progress has been made (either a success or a failure).
        It will be called repeatedly by the ConcurrentTask until all work is complete or the task is cancelled.

        Args:
            error_policy: The ErrorPolicy to use when handling errors.

        Returns:
            The updated TaskState after processing the TaskElement.
        """
        ...


class ConcurrentTask(ABC, Generic[ResultT]):
    """A generic concurrent task implementation. It is to be used by compute backends.

    The backend can derive from this class and a corresponding TaskState implementation
    to manage the state of the computation in a threadsafe manner, while providing
    the non-blocking compute.Task protocol to users.
    """

    item: Computable[ResultT]
    error_policy: ErrorPolicy
    _state: TaskState[ResultT]
    "To be modified only under _cv lock."
    _cv: Condition

    def __init__(
        self,
        item: Computable[ResultT],
        error_policy: ErrorPolicy,
        _state: TaskState[ResultT],
    ) -> None:
        self.item = item
        self.error_policy = error_policy
        self._state = _state
        self._cv = Condition()

    @classmethod
    @abstractmethod
    def from_computable(
        cls, item: Computable[ResultT], error_policy: ErrorPolicy
    ) -> Self:
        """Create a ConcurrentTask from a Computable and ErrorPolicy."""
        ...

    def result(self) -> ResultT:
        # TODO: if backend is shutdown without waiting on all tasks, raise an error here
        self.wait()
        # no lock needed after this, state is final
        if failed := self._state.first_failure():
            # Reraise the first error encountered
            msg = (
                f"Computation failed with {self._state.num_failures} errors;\n"
                f" use {str(type(self))}.partial_result() to access partial results and a\n"
                " continuation Computable for the remaining work.\n"
                " You can also adjust the ErrorPolicy to continue on certain errors.\n"
                " The first error is shown in the chained exception above."
            )
            raise RuntimeError(msg) from failed.exception
        out = self._state.output()
        assert not isinstance(out, EmptyResult)
        return out

    def partial_result(
        self,
    ) -> tuple[ResultT | EmptyResult, Computable[ResultT]]:
        # Hold lock so we get a consistent snapshot of state
        with self._cv:
            return self._state.output(), self._state.get_continuation(self.item)

    def wait(self) -> None:
        with self._cv:
            self._cv.wait_for(self.done)

    def status(self) -> TaskStatus:
        return self._state.status

    def done(self) -> bool:
        return self._state.status.done()

    def cancel(self) -> None:
        with self._cv:
            self._state.cancel()
            self._cv.notify_all()

    def _run(self) -> None:
        """Run the task to completion.

        This is intended to be called by a single worker thread.
        """
        while True:
            with self._cv:
                self._state.advance(self.error_policy)
                if self._state.status.done():
                    self._cv.notify_all()
                    return
                # let other threads have a chance to access the state
                # TODO: this is not reliable, we should use some sort of channel
                self._cv.wait(timeout=0)


TaskType = TypeVar("TaskType", bound=ConcurrentTask)


class GenericBackend(ABC, Generic[TaskType]):
    """A generic backend that runs tasks in a single background thread.

    This is intended as a base class for backends that want a background thread
    to manage task execution.
    """

    _task_queue: Queue[TaskType | _Shutdown] | None  # type: ignore[type-arg]
    _thread: Thread | None

    @classmethod
    @abstractmethod
    def _work(cls, task_queue: Queue[TaskType | _Shutdown]) -> None:
        """The worker function to run tasks from the task queue.

        This is intended to be run in a background thread.
        """
        ...

    @abstractmethod
    def _create_task(
        self, item: Computable[ResultT], error_policy: ErrorPolicy
    ) -> TaskType:
        """Create a ConcurrentTask for the given Computable and ErrorPolicy.

        To be used by the running backend to create tasks.
        It should also enqueue the task for execution.

        Args:
            item: The Computable to create the task for.
            error_policy: The ErrorPolicy to use when handling errors.

        Returns:
            A ConcurrentTask representing the computation.
        """
        ...

    def __init__(self) -> None:
        self._task_queue = None
        self._thread = None

    def __enter__(self) -> Self:
        self._task_queue = Queue()
        self._thread = Thread(
            target=self._work,
            name=str(self.__class__.__name__) + "-worker",
            args=(self._task_queue,),
        )
        self._thread.start()
        return self

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

    def compute(
        self,
        item: Computable[ResultT],
        /,
        error_policy: ErrorPolicy = ErrorPolicy(),
    ) -> TaskType:
        if hasattr(item, "__next__"):
            raise TypeError("Computable items must be iterables, not iterators")
        return self._create_task(item, error_policy)

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
            if self._task_queue:
                self._task_queue.join()
