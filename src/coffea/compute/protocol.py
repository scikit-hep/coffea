from collections.abc import Iterator
from enum import Enum
from types import TracebackType
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class Addable(Protocol):
    def __add__(self: T, other: T, /) -> T:
        """Merge two results together."""
        ...

    # TODO: past experience shows that we'll need __iadd__ as well for performance
    # which means we need to be careful about which results are mutable vs immutable


ResultT = TypeVar("ResultT", bound=Addable, covariant=True)
"""The result type produced by a computation.

It must support addition to allow merging/reduction."""


class EmptyResult:
    """A convenience class representing an empty result that can be added to anything."""

    def __add__(self, other: T, /) -> T:
        return other

    def __radd__(self, other: T, /) -> T:
        return other

    def __repr__(self) -> str:
        return "EmptyResult()"

    def __bool__(self) -> bool:
        return False


DataT = TypeVar("DataT", covariant=True)


class DataElement(Protocol[DataT]):
    """A data element has the necessary information to load a piece of data.

    It should be lightweight and serializable. By using this protocol rather than
    DataT/InputT directly, backends can manage data loading more flexibly.
    """

    def load(self) -> DataT:
        """Load and return the data represented by this element.

        It is assumed to be IO-bound.
        """
        ...


InputT = TypeVar("InputT")
"""The input type consumed by a computation.

This is the same as the DataT of DataElement, but it needs to be invariant.
"""


class WorkElement(Protocol[InputT, ResultT]):
    """A work element pairs a function with a data element to be processed.

    We enforce that this is used over, say partial(func, item.load()) so that:
    - We can type the input and output types separately
    - We can avoid capturing item.load() in a closure
    - We can later add metadata to this class if needed
    - We can isolate the data element as it is generally more serializable than a partial function

    Concrete implementations should be in the data module where DataElement is defined.
    They will generally only be generic in ResultT, as InputT is tied to the DataElement.
    """

    @property
    def func(self) -> Callable[[InputT], ResultT]: ...

    @property
    def item(self) -> DataElement[InputT]: ...

    def __call__(self) -> ResultT:
        """Execute the work element by loading the data and applying the function.

        Concrete implementations should define this method rather than inheriting the protocol.
        They MUST implement the method exactly as specified here.

        Backends MAY call `item.load()` to preload data while working on another element,
        as DataElement.load is assumed to be IO-bound.
        """
        return self.func(self.item.load())


class Computable(Protocol[InputT, ResultT]):
    def __iter__(self) -> Iterator[WorkElement[InputT, ResultT]]:
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

    # TODO: dependencies method to declare other Computables that must be run first?


class TaskStatus(Enum):
    """Enumeration of possible task statuses.

    This is a concrete type so we are clear about definitions across backends.
    """

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


class Task(Protocol[InputT, ResultT]):
    """Task represents an ongoing or completed computation.

    A Task is created by a Backend when a Computable is submitted for execution.
    """

    def result(self) -> ResultT | EmptyResult:
        """Get the full final result of the computation. Blocking."""
        ...

    def partial_result(
        self,
    ) -> tuple[ResultT | EmptyResult, Computable[InputT, ResultT]]:
        """Get a partial result and the corresponding continuation computation. Non-blocking.

        The partial result may either be because the task is not yet complete,
        or because the computation failed on a subset of the data.

        TODO: add cancel parameter to indicate whether to stop ongoing computation?
        """
        ...

    # TODO: retrieve exceptions (this is managed in errors module, should errors define a Protocol?)

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


class RunningBackend(Protocol):
    """A RunningBackend represents an active backend context.

    This is the type returned by Backend.__enter__.
    It may be distinct from Backend to separate resource management
    from computation submission.
    """

    def compute(self, item: Computable[InputT, ResultT], /) -> Task[InputT, ResultT]:
        """Launch a computation and return a Task representing it.

        Note: a trivial implementation of this is `sum((f() for f in item), EmptyResult())`
        """
        ...


class Backend(Protocol):
    """A backend manages the execution of computations.

    The backend holds any resources needed to perform the computation,
    such as a thread pool, process pool, or cluster connection. It should
    manage a queue of tasks and execute them in FIFO order.

    An example configuration option may be the behavior of exiting the context
    manager (wait for tasks to finish, cancel them, etc).
    """

    # Implementation note: mypy was not happy with having Backend inherit from
    # ContextManager[RunningBackend], so we define the methods explicitly here.

    def __enter__(self) -> RunningBackend: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None: ...
