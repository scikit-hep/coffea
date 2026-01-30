from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Generic

from coffea.compute.protocol import ResultT, WorkElement


@dataclass(slots=True, frozen=True)
class TaskElement(Generic[ResultT]):
    """A wrapper of WorkElement with an index for tracking."""

    index: int
    work: WorkElement[ResultT]

    def __call__(self) -> ResultT:
        return self.work()


@dataclass(slots=True, frozen=True)
class FailedTaskElement(TaskElement[ResultT]):
    exception: Exception
    retries: int
    last_attempt: datetime = field(default_factory=datetime.now)


class ErrorAction(Enum):
    CANCEL = "cancel"
    CONTINUE = "continue"
    RETRY = "retry"


@dataclass(frozen=True)
class ErrorPolicy:
    _: KW_ONLY
    cancel_on: tuple[type[Exception], ...] = ()
    """Exception types that should cause immediate cancellation."""
    continue_on: tuple[type[Exception], ...] = ()
    """Exception types that should be ignored and continue without retries."""
    max_retries: int = 3
    """Maximum number of retries for failed task elements."""
    continue_after_retries_on: tuple[type[Exception], ...] = (OSError,)
    """Exception types that should continue after max retries are exhausted.

    All other exceptions will cause task cancellation after max retries."""

    def first_action(
        self, element: TaskElement[ResultT], exception: Exception
    ) -> tuple[FailedTaskElement[ResultT], ErrorAction]:
        """Determine action to take on first failure of a task element."""
        new_element = FailedTaskElement(
            index=element.index,
            work=element.work,
            exception=exception,
            retries=0,
        )
        if isinstance(exception, self.cancel_on):
            return new_element, ErrorAction.CANCEL
        if isinstance(exception, self.continue_on):
            return new_element, ErrorAction.CONTINUE
        return new_element, ErrorAction.RETRY

    def retry_action(
        self, element: FailedTaskElement[ResultT], exception: Exception
    ) -> tuple[FailedTaskElement[ResultT], ErrorAction]:
        """Determine action to take on retry failure of a task element."""
        new_element = FailedTaskElement(
            index=element.index,
            work=element.work,
            exception=exception,  # update to latest exception
            retries=element.retries + 1,
        )
        if isinstance(exception, self.cancel_on):
            return new_element, ErrorAction.CANCEL
        if isinstance(exception, self.continue_on):
            return new_element, ErrorAction.CONTINUE
        if new_element.retries >= self.max_retries:
            if isinstance(exception, self.continue_after_retries_on):
                return new_element, ErrorAction.CONTINUE
            else:
                return new_element, ErrorAction.CANCEL
        return new_element, ErrorAction.RETRY
