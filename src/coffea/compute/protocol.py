from enum import Enum
from typing import Callable, Protocol, Self, TypeVar
from collections.abc import Iterator

T = TypeVar("T")


class ResultType(Protocol):
    def __add__(self: T, other: T, /) -> T:
        """Merge two results together."""
        ...


class EmptyResult:
    """A convenience class representing an empty result that can be added to anything."""

    def __add__(self, other: T, /) -> T:
        return other

    def __radd__(self, other: T, /) -> T:
        return other

    def __repr__(self) -> str:
        return "EmptyResult()"


class Computable(Protocol):
    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        """Return an iterator over callables that each compute a part of the result.

        This establishes a global index over the computation. We can use filters
        and slicing on this basis to create resumable and retryable computations.

        A potential optimization is to yield a tuple of (index, callable) so that
        the index can be smarter than just the integer position in the iterator.
        Then Computable could be a Container type.

        BIG TODO: self must be an iterable, not an iterator, so that it can be
        re-iterated over for resumptions. How to enforce that in the Protocol?
        """
        ...


class TaskStatus(Enum):
    PENDING = "pending"
    """The computation has not yet started."""
    RUNNING = "running"
    """The computation is in progress, partial results may be available."""
    COMPLETE = "complete"
    """The computation has finished successfully."""
    INCOMPLETE = "incomplete"
    """The computation has finished with some (non-abortable) failures."""
    CANCELLED = "cancelled"
    """The computation was cancelled before completion, or a failure caused it to stop early."""

    def done(self) -> bool:
        return self in (
            TaskStatus.COMPLETE,
            TaskStatus.INCOMPLETE,
            TaskStatus.CANCELLED,
        )


class Task(Protocol):
    """Task represents an ongoing or completed computation.

    A Task is created by a Backend when a Computable is submitted for execution.
    """

    def result(self) -> ResultType:
        """Get the full final result of the computation. Blocking."""
        ...

    def partial_result(self) -> tuple[ResultType, Computable]:
        """Get a partial result and the corresponding continuation computation. Non-blocking.

        The partial result may either be because the task is not yet complete,
        or because the computation failed on a subset of the data.

        TODO: add cancel parameter to indicate whether to stop ongoing computation?
        """
        ...

    def wait(self) -> None:
        """Block until the computation is complete.

        TODO: add timeout parameter and return_when options?
        c.f. https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.wait
        FIRST_COMPLETED, FIRST_EXCEPTION, ALL_COMPLETED
        """
        ...

    def status(self) -> TaskStatus:
        """Return the current status of the computation."""
        ...

    def done(self) -> bool:
        """Return True if the computation is complete, False otherwise. Non-blocking."""
        ...

    def cancel(self) -> None:
        """Attempt to cancel the computation. Non-blocking."""
        ...


class Backend(Protocol):
    def compute(self, item: Computable) -> Task:
        """Launch a computation and return a Task representing it.

        The backend holds any resources needed to perform the computation,
        such as a thread pool, process pool, or cluster connection. It should
        manage a queue of tasks and execute them in FIFO order.

        Probably this should have a context manager interface to clean up resources.
        """
        ...

    def __enter__(self) -> Self:
        """Enter the backend context manager, allocating resources.

        TODO: should we have a separate type for the context manager rather than return Self here?
        One may also want to choose the behavior of exiting the context manager (wait for tasks to finish, cancel them, etc).
        """
        ...

    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
