from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Generic, Protocol, TypeAlias, TypeVar

from coffea.compute.util import merge_ranges

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


Range: TypeAlias = tuple[int, int]


@dataclass
class ResultWrapper(Generic[ResultT]):
    """A wrapper for results that also tracks which input ranges were completed or failed.

    The ranges are interpretable only in the context of the original Computable object that
    was passed to the Backend to create this ResultWrapper, and are not necessarily globally meaningful.
    """

    result: ResultT | EmptyResult
    """A result, which may be partial/incomplete"""
    producer_key: str
    """The key of the Computable that produced this result"""
    completed: list[Range]
    """The ranges of input data that were successfully processed"""
    failed: list[tuple[Range, Exception]]
    """The ranges of input data that failed, with their exceptions"""

    def first_failure(self) -> Exception | None:
        return self.failed[0][1] if self.failed else None

    def __add__(self, other: ResultWrapper[ResultT]) -> ResultWrapper[ResultT]:
        if self.producer_key != other.producer_key:
            msg = f"Cannot merge results from different producers: {self.producer_key} vs. {other.producer_key}"
            raise ValueError(msg)
        return ResultWrapper(
            self.result + other.result,
            self.producer_key,
            merge_ranges(self.completed, other.completed),
            self.failed + other.failed,
        )


DataT = TypeVar("DataT", covariant=True)


class AbstractInput(Protocol[DataT]):
    """A range of data to be processed in a computation.

    It should be lightweight and serializable. By using this protocol rather than
    DataT/InputT directly, backends can preload the data.

    It has a start, stop, and implied length, but the actual data is only loaded when
    load() is called.
    """

    @property
    def start(self) -> int: ...

    @property
    def stop(self) -> int: ...

    def __len__(self) -> int: ...

    def load(self) -> DataT:
        """Load and return the data represented by this range.

        This should be a blocking but low-cpu operation, suitable for running
        in a thread pool to hide IO latency.
        """
        ...


class AbstractWorkElement(Protocol[ResultT]):
    """Minimal abstraction of WorkElement

    Use this wherever you don't need to know the input range, so
    that composite types are a bit simpler to implement.
    """

    def preload(self) -> None: ...

    def __call__(self) -> ResultT: ...


InputT = TypeVar("InputT")
"""The input type consumed by a computation.

This is the same as the DataT of AbstractInput, but it needs to be invariant.
"""


class WorkElement(Generic[ResultT]):
    """A work element pairs a function with a chunk of data to be processed.

    We enforce that this is used over, say partial(func, item.load()) so that
    we can optionally preload data in advance, and have more control over
    serialization. This is a concrete type, it can be extended via composition.
    See FailedWorkElement in errors.py for an example of this.

    The input type is erased by this wrapper, so we are only generic in the output type.
    """

    def __init__(
        self,
        func: Callable[[InputT], ResultT],
        item: AbstractInput[InputT],
    ):
        self._func = func
        self._item = item

    @property
    def start(self) -> int:
        return self._item.start

    @property
    def stop(self) -> int:
        return self._item.stop

    def __len__(self) -> int:
        return len(self._item)

    def __reduce__(self) -> tuple[Any, ...]:
        return (self.__class__, (self._func, self._item))

    def preload(self):
        """Preload any data needed for the computation."""
        self._loaded = self._item.load()

    def __call__(self) -> ResultT:
        """Execute the work element and return the result."""
        if hasattr(self, "_loaded"):
            return self._func(self._loaded)
        return self._func(self._item.load())


class Computable(Protocol[ResultT]):
    """Abstraction for something that can be computed by a backend.

    It represents the entire computation, and is responsible for defining how to break itself
    into WorkElements that can be executed in parallel. Optionally, it can support dynamic
    adjustment of the chunk sizes by implementing gen_steps instead of iter_steps.
    """

    def __len__(self) -> int:
        """Total size of the input to be processed

        This should respect the invariant that
        `len(computable) == sum(len(work) for work in computable.iter_steps())`
        """
        ...

    @property
    def key(self) -> str:
        """A string key that identifies this computation

        This should be the same for computations that will produce the same result.
        It should not include any information about how the computation is broken into
        steps, as that is up to the backend and should not affect caching.

        This may be used for caching and for matching ResultWrappers to Computables
        when merging results from multiple computations. A sha256 hash of inputs
        and function code may be a good choice for this.
        """
        ...

    def gen_steps(self, /) -> Generator[WorkElement[ResultT], int | None, None]:
        """Step through the input data in chunks, yielding WorkElements that can be executed to produce partial results.

        The Send parameter is the new chunk size requested by the backend, which may
        be ignored. This allows for dynamic adjustment of chunk sizes based on runtime
        information about processing times, resource availability, etc.
        """
        ...


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
    """The computation has finished with some (non-cancelable) failures."""
    CANCELLED = "cancelled"
    """The computation was cancelled before completion, or a failure caused it to stop early."""

    def done(self) -> bool:
        return self in (
            TaskStatus.COMPLETE,
            TaskStatus.INCOMPLETE,
            TaskStatus.CANCELLED,
        )


class Task(Protocol[ResultT]):
    """Task represents an ongoing or completed computation.

    A Task is created by a Backend when a Computable is submitted for execution.
    It is effectively a Future object for the entire computation, but it provides
    additional methods for retrieving partial results, checking status, and cancelling
    the whole task (even if already running) that would not necessarily be supported by
    a plain Future. The exact behavior of cancellation may depend on the backend and the
    state of the task.
    """

    def result(self) -> ResultT | EmptyResult:
        """Get the full final result of the computation. Blocking.

        Either the complete result is returned or an exception is raised if the
        computation is incomplete or failed. Partial results are not accessible through this method.
        """
        ...

    def partial_result(
        self,
    ) -> ResultWrapper[ResultT]:
        """Get a partial result. Non-blocking.

        The partial result may either be because the task is not yet complete,
        or because the computation failed on a subset of the data. The ResultWrapper
        will indicate which input ranges were completed successfully and which failed,
        as well as any exception information for the failed work elements.
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
        """Cancel the computation. Non-blocking."""
        ...


class RunningBackend(Protocol):
    """A RunningBackend represents an active backend context.

    This is the type returned by Backend.__enter__.
    It may be distinct from Backend to separate resource management
    from computation submission.
    """

    def compute(self, item: Computable[ResultT], /) -> Task[ResultT]:
        """Launch a computation

        Returns a Task object that can be used to track the computation and retrieve results.
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

    def compute(self, item: Computable[ResultT], /) -> ResultWrapper[ResultT]:
        """Launch a computation and block until it finishes

        Returns a ResultWrapper object that can be used to access the final
        result as well as information about which input ranges completed or failed.
        """
        ...
