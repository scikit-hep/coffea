import time
from functools import partial
from itertools import repeat

import pytest

from coffea.compute.backends.threaded import SingleThreadedBackend
from coffea.compute.errors import ErrorPolicy
from coffea.compute.protocol import Backend, TaskStatus

BACKENDS: list[type[Backend]] = [SingleThreadedBackend]


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_compute(backend_class):
    def func(x: int):
        time.sleep(0.001)
        return x * x

    computable = list(map(partial, repeat(func), range(100)))

    thing = backend_class()
    # Test that using compute outside of context manager raises
    with pytest.raises(RuntimeError, match="context manager"):
        thing.compute(computable)

    with thing as backend:
        task = backend.compute(computable)
        result = task.result()
        assert task.status() == TaskStatus.COMPLETE
        assert result == sum(x * x for x in range(100))

    with pytest.raises(RuntimeError, match="context manager"):
        backend.compute(computable)

    with thing as backend:
        task = backend.compute(computable)

    # TODO: should we cancel tasks when exiting context manager?
    assert task.status() == TaskStatus.COMPLETE

    # detect iterators (stateful)
    computable = map(partial, repeat(func), range(100))
    with thing as backend:
        with pytest.raises(TypeError, match="iterables, not iterators"):
            backend.compute(computable)


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_partial_result(backend_class):
    def func(x: int):
        time.sleep(0.001)
        return x * x

    with backend_class() as backend:
        computable = list(map(partial, repeat(func), range(100)))
        task = backend.compute(computable)

        time.sleep(0.01)
        assert task.status() == TaskStatus.RUNNING
        part, resumable = task.partial_result()
        assert part > 0
        assert len(list(resumable)) < 100

        task.wait()
        assert task.status() == TaskStatus.COMPLETE

        resumed_task = backend.compute(resumable)
        resumed_task.wait()
        assert resumed_task.status() == TaskStatus.COMPLETE
        final_result = part + resumed_task.result()
        assert final_result == sum(x * x for x in range(100))


@pytest.mark.parametrize("backend_class", BACKENDS)
def test_backend_error_handling(backend_class):
    def func(x: int):
        if x == 42:
            raise ValueError(
                "The answer to life, universe and everything caused an error"
            )
        return x * x

    with backend_class() as backend:
        computable = list(map(partial, repeat(func), range(100)))

        # With default error policy, the task should cancel on error after 3 retries
        task = backend.compute(computable)
        task.wait()
        assert task.status() == TaskStatus.CANCELLED
        with pytest.raises(RuntimeError):
            task.result()
        res, cont = task.partial_result()
        assert res == sum(x * x for x in range(42))
        assert len(list(cont)) == 58  # steps 42 to 99 remain

        # With continue_on ValueError, the task should complete, skipping the error
        task = backend.compute(
            computable,
            error_policy=ErrorPolicy(continue_on=(ValueError,)),
        )
        task.wait()
        assert task.status() == TaskStatus.INCOMPLETE
        res, cont = task.partial_result()
        assert res == sum(x * x for x in range(100) if x != 42)
        assert len(list(cont)) == 1  # only one failed step
