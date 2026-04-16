"""
Rust-style Result type for coffea Runner return values.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Result(Generic[T]):
    """Result type — either Ok or Err."""

    def is_ok(self) -> bool:
        raise NotImplementedError

    def is_err(self) -> bool:
        return not self.is_ok()

    def unwrap(self) -> T:
        """Either returns value (accumulator) for Ok Result or an exception if Err result."""
        raise NotImplementedError


class Ok(Result[T]):
    """A successful result containing a value."""

    def __init__(self, value: T) -> None:
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    def is_ok(self) -> bool:
        return True

    def unwrap(self) -> T:
        return self._value

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


class Err(Result):
    """A failed result containing an exception."""

    def __init__(self, exception: BaseException) -> None:
        self._exception = exception

    @property
    def exception(self) -> BaseException:
        return self._exception

    def is_ok(self) -> bool:
        return False

    def unwrap(self):
        raise self._exception

    def __repr__(self) -> str:
        return f"Err({self._exception!r})"
