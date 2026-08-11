import time
from collections.abc import Callable, Generator

import pytest

from coffea.compute.backends.threaded import ThreadedBackend
from coffea.compute.errors import ErrorPolicy
from coffea.compute.protocol import Backend, TaskStatus, WorkElement

BACKENDS: list[type[Backend]] = [ThreadedBackend]


def func(x: int):
    time.sleep(0.002)
    return x * x


def buggy_func(x: int):
    if x == 42:
        raise ValueError("The answer to life, universe and everything caused an error")
    return x * x


class IntLoader:
    def __init__(self, item: int):
        self.item = item
        self.start = item
        self.stop = item + 1

    def __len__(self) -> int:
        return 1

    def load(self) -> int:
        return self.item


class IntComputable:
    def __init__(self, func: Callable[[int], int], max: int):
        self.func = func
        self.max = max

    def __len__(self) -> int:
        return self.max

    @property
    def key(self) -> str:
        return f"IntComputable({self.func.__name__}, {self.max})"

    def gen_steps(self) -> Generator[WorkElement[int], int | None, None]:
        for i in range(self.max):
            yield WorkElement(self.func, IntLoader(i))


class RerunComputable:
    """Pass last run's completed and skip the next run

    TODO: generalize and put in src/coffea/compute somewhere
    """

    def __init__(
        self, func: Callable[[int], int], max: int, completed: list[tuple[int, int]]
    ):
        self.func = func
        self.max = max
        self.completed = completed

    def __len__(self) -> int:
        return self.max - sum(end - start for start, end in self.completed)

    @property
    def key(self) -> str:
        return f"RerunComputable({self.func.__name__}, {self.max}, completed={self.completed})"

    def gen_steps(self) -> Generator[WorkElement[int], int | None, None]:
        for i in range(self.max):
            if any(start <= i < end for start, end in self.completed):
                continue
            yield WorkElement(self.func, IntLoader(i))


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_compute(backend_class):
    computable = IntComputable(func, 100)

    thing = backend_class()

    with thing as backend:
        task = backend.compute(computable)
        result = task.result()
        assert task.status() == TaskStatus.COMPLETE
        assert result == sum(x * x for x in range(100))

    with pytest.raises(RuntimeError):
        backend.compute(computable)

    with thing as backend:
        task = backend.compute(computable)

    rw = thing.compute(computable)
    assert rw.result == sum(x * x for x in range(100))
    assert rw.completed == [(0, 100)]
    assert rw.failed == []

    # TODO: test that the task was cancelled if cancel_on_exit=True, and that it completed if cancel_on_exit=False
    assert task.status() == TaskStatus.COMPLETE


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_partial_result(backend_class):
    with backend_class() as backend:
        computable = IntComputable(func, 100)
        task = backend.compute(computable)

        assert task.status() in (TaskStatus.PENDING, TaskStatus.RUNNING)

        t0 = time.monotonic()
        while task.status() != TaskStatus.RUNNING and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        if task.status() != TaskStatus.RUNNING:
            raise TimeoutError("Task did not reach desired state in time")

        t0 = time.monotonic()
        part = task.partial_result()
        while (not part.completed) and time.monotonic() - t0 < 5:
            time.sleep(0.01)
            part = task.partial_result()
        if not part.completed:
            raise TimeoutError("Task did not produce partial result in time")

        nprocessed = sum(end - start for start, end in part.completed)
        assert nprocessed < 100
        assert nprocessed > 0

        task.wait()
        assert task.status() == TaskStatus.COMPLETE

        rerun = RerunComputable(func, 100, completed=part.completed)
        resumed_task = backend.compute(rerun)
        resumed_task.wait()
        assert resumed_task.status() == TaskStatus.COMPLETE
        final_result = part.result + resumed_task.result()
        assert final_result == sum(x * x for x in range(100))


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_error_handling(backend_class):
    with backend_class() as backend:
        computable = IntComputable(buggy_func, 100)

        # With default error policy, the task should cancel on error after 3 retries
        task = backend.compute(computable)
        task.wait()
        assert task.status() == TaskStatus.CANCELLED
        with pytest.raises(RuntimeError):
            task.result()
        part = task.partial_result()
        assert part.result > 0
        assert part.failed == []
        assert not any(start <= 42 < end for start, end in part.completed)

        # With continue_on ValueError, the task should complete, skipping the error
        task = backend.compute(
            computable,
            error_policy=ErrorPolicy(continue_on=(ValueError,)),
        )
        task.wait()
        assert task.status() == TaskStatus.INCOMPLETE
        with pytest.raises(RuntimeError):
            task.result()
        part = task.partial_result()
        assert part.result == sum(x * x for x in range(100) if x != 42)
        assert len(part.failed) == 1
        assert part.failed[0][0] == (42, 43)
        assert isinstance(part.failed[0][1], ValueError)
        assert part.completed == [(0, 42), (43, 100)]
