"""A generic background thread implementation for use with compute backends.

This allows for a threadsafe way to return a future compatible with the compute.Task protocol.

This is a private module as it is only intended for use by compute backends.
"""

from abc import ABC, abstractmethod
from threading import Condition
from typing import Generic

from typing_extensions import Self

from coffea.compute.errors import (
    ErrorPolicy,
    FailedTaskElement,
)
from coffea.compute.protocol import Computable, EmptyResult, InputT, ResultT, TaskStatus


class _Shutdown:
    """A sentinel to signal worker threads to exit.

    In python 3.13+, we can use queue.ShutDown directly.
    """

    pass


class TaskState(ABC, Generic[InputT, ResultT]):
    """A base implementation for internal task state to be used by backend ConcurrentTask implementations."""

    @property
    @abstractmethod
    def status(self) -> TaskStatus:
        """The current status of the task."""
        ...

    @classmethod
    @abstractmethod
    def from_computable(cls, item: Computable[InputT, ResultT]) -> Self:
        """Create an initial TaskState from a Computable.

        Args:
            item: The Computable to create the TaskState for.

        Returns:
            An initial TaskState for the Computable.
            Its status should be TaskStatus.PENDING.
        """
        ...

    @abstractmethod
    def first_failure(self) -> FailedTaskElement[InputT, ResultT] | None:
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
    def get_continuation(
        self, original: Computable[InputT, ResultT]
    ) -> Computable[InputT, ResultT]:
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


class ConcurrentTask(ABC, Generic[InputT, ResultT]):
    """A generic concurrent task implementation. It is to be used by compute backends.

    The backend can derive from this class and a corresponding TaskState implementation
    to manage the state of the computation in a threadsafe manner, while providing
    the non-blocking compute.Task protocol to users.
    """

    item: Computable[InputT, ResultT]
    error_policy: ErrorPolicy
    _state: TaskState[InputT, ResultT]
    "To be modified only under _cv lock."
    _cv: Condition

    def __init__(
        self,
        item: Computable[InputT, ResultT],
        error_policy: ErrorPolicy,
        _state: TaskState[InputT, ResultT],
    ) -> None:
        self.item = item
        self.error_policy = error_policy
        self._state = _state
        self._cv = Condition()

    @classmethod
    @abstractmethod
    def from_computable(
        cls, item: Computable[InputT, ResultT], error_policy: ErrorPolicy
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
    ) -> tuple[ResultT | EmptyResult, Computable[InputT, ResultT]]:
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
