"""Execution backends for preprocessing (experimental).

Preprocessing maps a worker over batches of each dataset's file records and concatenates the
results in file order. A :class:`PreprocessBackend` runs this as a dask graph
(:class:`DaskBackend`), a graphed plan (:class:`GraphedBackend`), in the current thread
(:class:`IterativeBackend`) or on a :mod:`concurrent.futures` pool (:class:`FuturesBackend`).
"""

from __future__ import annotations

import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, replace
from typing import ClassVar, Protocol, runtime_checkable

import awkward

from coffea.util import _import_dask, _import_graphed

__all__ = [
    "PreprocessJob",
    "PreprocessTask",
    "PreprocessBackend",
    "IterativeBackend",
    "FuturesBackend",
    "DaskBackend",
    "GraphedBackend",
]


def _ordered_concat(parts: Iterable[awkward.Array | None]) -> awkward.Array | None:
    """Concatenate ``parts`` in order, dropping ``None``; ``None`` if nothing remains."""
    parts = [part for part in parts if part is not None]
    if len(parts) == 0:
        return None
    if len(parts) == 1:
        return parts[0]
    return awkward.concatenate(parts, axis=0)


def _batch_ranges(n: int, files_per_batch: int) -> list[tuple[int, int]]:
    """``[start, stop)`` ranges of ``files_per_batch`` rows (at least 1) covering ``n`` rows,
    or one empty range when ``n`` is 0."""
    step = max(1, files_per_batch)
    return [(start, min(start + step, n)) for start in range(0, max(n, 1), step)]


def _iter_batches(array: awkward.Array, files_per_batch: int):
    for start, stop in _batch_ranges(len(array), files_per_batch):
        yield array[start:stop]


@dataclass
class PreprocessJob:
    """One dataset's preprocessing: ``map_fn`` mapped over batches of ``array``.

    Parameters
    ----------
        array : awkward.Array
            Normalized file records (``file``, ``object_path``, ``steps``, ``num_entries``,
            ``uuid``) of one dataset.
        map_fn : Callable[[awkward.Array], awkward.Array]
            The worker run on each batch, e.g. ``functools.partial(get_steps, **options)``.
            Must be picklable to run in other processes.
        files_per_batch : int, default 1
            Number of file records per batch.
    """

    array: awkward.Array
    map_fn: Callable[[awkward.Array], awkward.Array]
    files_per_batch: int = 1


@runtime_checkable
class PreprocessTask(Protocol):
    """Handle to submitted preprocessing work."""

    def result(self) -> dict[str, awkward.Array]:
        """Wait for the work and return ``{dataset_name: records}``.

        Raises the first error the worker raised.
        """

    def partial_result(self) -> dict[str, awkward.Array]:
        """Return the results finished so far, without waiting.

        Each dataset with a finished batch appears with its finished batches only. Dask and
        graphed report all results once the work is done and ``{}`` before.
        """

    def wait(self) -> None:
        """Wait for the work to finish."""


class PreprocessBackend(ABC):
    """Base class of the preprocessing backends.

    ``scheduler_field`` names the field that receives the ``scheduler`` argument of the
    ``preprocess*`` functions, or is ``None`` when the backend takes none.
    """

    scheduler_field: ClassVar[str | None] = None

    @abstractmethod
    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        """Start ``jobs`` and return their :class:`PreprocessTask`.

        The futures backend, and graphed executors with ``submit(plan)``, run the work in the
        background; the iterative backend, and graphed executors with only ``run(plan)``, run it
        here; the dask backend runs it in the task's ``result()`` or ``wait()``. Errors the worker
        raises are raised by ``result()``.
        """

    def compute(self, jobs: Mapping[str, PreprocessJob]) -> dict[str, awkward.Array]:
        """``submit(jobs).result()``."""
        return self.submit(jobs).result()


@dataclass
class _CompletedTask:
    _results: dict[str, awkward.Array]
    _error: Exception | None = None

    def result(self) -> dict[str, awkward.Array]:
        if self._error is not None:
            raise self._error
        return self._results

    def partial_result(self) -> dict[str, awkward.Array]:
        return self._results

    def wait(self) -> None:
        return None


@dataclass
class IterativeBackend(PreprocessBackend):
    """Run preprocessing in the current thread, one batch at a time."""

    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        results: dict[str, awkward.Array] = {}
        try:
            for name, job in jobs.items():
                parts = [
                    job.map_fn(batch)
                    for batch in _iter_batches(job.array, job.files_per_batch)
                ]
                results[name] = _ordered_concat(parts)
        except Exception as err:
            return _CompletedTask(results, err)
        return _CompletedTask(results)


class _FuturesTask:
    def __init__(
        self,
        name_to_futures: dict[str, list[Future]],
        pool: Executor,
        owns_pool: bool,
    ):
        self._name_to_futures = name_to_futures
        self._pool = pool
        self._owns_pool = owns_pool

    def _all_futures(self) -> list[Future]:
        return [fut for futs in self._name_to_futures.values() for fut in futs]

    def wait(self) -> None:
        futures_wait(self._all_futures())

    def result(self) -> dict[str, awkward.Array]:
        try:
            results: dict[str, awkward.Array] = {}
            for name, futs in self._name_to_futures.items():
                results[name] = _ordered_concat(fut.result() for fut in futs)
        except BaseException:
            for fut in self._all_futures():
                fut.cancel()
            if self._owns_pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
            raise
        if self._owns_pool:
            self._pool.shutdown()
        return results

    def partial_result(self) -> dict[str, awkward.Array]:
        results: dict[str, awkward.Array] = {}
        for name, futs in self._name_to_futures.items():
            merged = _ordered_concat(
                fut.result()
                for fut in futs
                if fut.done() and not fut.cancelled() and fut.exception() is None
            )
            if merged is not None:
                results[name] = merged
        return results

    def __del__(self):
        if getattr(self, "_owns_pool", False):
            self._pool.shutdown(wait=False)


@dataclass
class FuturesBackend(PreprocessBackend):
    """Run preprocessing on a :mod:`concurrent.futures` pool, threads by default.

    Parameters
    ----------
        workers : int or None, default None
            Number of workers of a pool this backend creates; None uses the executor's default.
        use_processes : bool, default False
            Create a :class:`~concurrent.futures.ProcessPoolExecutor` instead of a
            :class:`~concurrent.futures.ThreadPoolExecutor`.
        pool : concurrent.futures.Executor or Callable, optional
            An executor to use, which is left open, or a callable creating one as
            ``pool(max_workers=workers)``. Takes precedence over ``use_processes``. Set by the
            ``scheduler`` argument of the ``preprocess*`` functions.
    """

    scheduler_field: ClassVar[str | None] = "pool"

    workers: int | None = None
    use_processes: bool = False
    pool: Executor | Callable[..., Executor] | None = None

    def __post_init__(self):
        if self.pool is not None and not (
            isinstance(self.pool, Executor) or callable(self.pool)
        ):
            raise TypeError(
                "FuturesBackend.pool must be a concurrent.futures.Executor or a callable "
                f"returning one, not {self.pool!r}"
            )

    def _make_pool(self) -> tuple[Executor, bool]:
        if isinstance(self.pool, Executor):
            return self.pool, False
        if self.pool is not None:
            return self.pool(max_workers=self.workers), True
        if self.use_processes:
            return ProcessPoolExecutor(max_workers=self.workers), True
        return ThreadPoolExecutor(max_workers=self.workers), True

    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        pool, owns_pool = self._make_pool()
        try:
            name_to_futures: dict[str, list[Future]] = {}
            for name, job in jobs.items():
                name_to_futures[name] = [
                    pool.submit(job.map_fn, batch)
                    for batch in _iter_batches(job.array, job.files_per_batch)
                ]
        except BaseException:
            if owns_pool:
                pool.shutdown()
            raise
        return _FuturesTask(name_to_futures, pool, owns_pool)


class _DaskTask:
    def __init__(self, collections: dict, scheduler):
        self._collections = collections
        self._scheduler = scheduler
        self._computed: dict[str, awkward.Array] | None = None

    def wait(self) -> None:
        self.result()

    def result(self) -> dict[str, awkward.Array]:
        if self._computed is None:
            dask = _import_dask()
            (self._computed,) = dask.compute(
                self._collections, scheduler=self._scheduler
            )
        return self._computed

    def partial_result(self) -> dict[str, awkward.Array]:
        return {} if self._computed is None else self._computed


@dataclass
class DaskBackend(PreprocessBackend):
    """Run preprocessing as a dask graph: a task per batch and a concatenation per dataset.

    Parameters
    ----------
        scheduler : None or Callable or str, default None
            The scheduler of ``dask.compute``; None uses dask's current one. Set by the
            ``scheduler`` argument of the ``preprocess*`` functions.
    """

    scheduler_field: ClassVar[str | None] = "scheduler"

    scheduler: None | Callable | str = None

    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        dask = _import_dask()
        collections = {
            name: dask.delayed(_ordered_concat)(
                [
                    dask.delayed(job.map_fn)(batch)
                    for batch in _iter_batches(job.array, job.files_per_batch)
                ]
            )
            for name, job in jobs.items()
        }
        return _DaskTask(collections, self.scheduler)


@dataclass
class GraphedBackend(PreprocessBackend):
    """Run preprocessing as one graphed ``Plan`` with a task per batch.

    Parameters
    ----------
        executor : graphed executor or None, default None
            An object with ``run(plan)``, and optionally ``submit(plan)`` to run without
            blocking; None uses ``graphed.core.execution.SequentialRunner``. Set by the
            ``scheduler`` argument of the ``preprocess*`` functions.
    """

    scheduler_field: ClassVar[str | None] = "executor"

    executor: object = None

    def __post_init__(self):
        if self.executor is not None and not callable(
            getattr(self.executor, "run", None)
        ):
            raise TypeError(
                "with backend='graphed', scheduler is a graphed executor (an object with "
                f"run(plan)), not {self.executor!r}"
            )

    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        _import_graphed()
        from coffea.dataset_tools import _graphed

        return _graphed.submit(jobs, self.executor)


_BACKENDS: dict[str, type[PreprocessBackend]] = {
    "dask": DaskBackend,
    "graphed": GraphedBackend,
    "iterative": IterativeBackend,
    "futures": FuturesBackend,
}


def _scheduler_backend_name(scheduler: object) -> str:
    """The backend a ``scheduler`` argument is for: futures for a ``concurrent.futures``
    executor or executor class, graphed for an object with ``run(plan)`` from outside dask and
    distributed (a ``distributed.Client`` has a ``run`` method), dask otherwise."""
    if isinstance(scheduler, Executor) or (
        isinstance(scheduler, type) and issubclass(scheduler, Executor)
    ):
        return "futures"
    module = getattr(scheduler, "__module__", None) or type(scheduler).__module__
    if (
        not isinstance(scheduler, str)
        and module.split(".")[0] not in ("dask", "distributed")
        and callable(getattr(scheduler, "run", None))
    ):
        return "graphed"
    return "dask"


def _graphed_has_session() -> bool:
    """Whether an imported graphed has an open ``with session:`` scope or a default Session."""
    session = sys.modules.get("graphed.session")
    scopes = getattr(session, "_scopes", None)
    defaults = getattr(session, "_defaults", None)
    return bool(scopes is not None and scopes.get(())) or (
        getattr(defaults, "session", None) is not None
    )


def _dask_has_client() -> bool:
    """Whether an imported distributed has a current ``Client``."""
    distributed = sys.modules.get("distributed")
    if distributed is None:
        return False
    try:
        distributed.Client.current(allow_global=True)
    except ValueError:
        return False
    return True


def _default_backend_name(scheduler: object = None) -> str:
    """The backend for ``backend=None``: the one ``scheduler`` is for when given, else graphed
    with a graphed Session, else dask with a current ``distributed.Client``, else iterative.
    """
    if scheduler is not None:
        return _scheduler_backend_name(scheduler)
    if _graphed_has_session():
        return "graphed"
    if _dask_has_client():
        return "dask"
    return "iterative"


def _external_stacklevel() -> int:
    """The ``stacklevel`` of a warning issued here that points at the first frame outside
    ``coffea.dataset_tools``."""
    frame = sys._getframe(1)
    level = 1
    while frame is not None and frame.f_globals.get("__name__", "").startswith(
        "coffea.dataset_tools"
    ):
        frame = frame.f_back
        level += 1
    return level


def _resolve_backend(
    backend: str | PreprocessBackend | None,
    scheduler: object = None,
) -> PreprocessBackend:
    """The :class:`PreprocessBackend` for a ``backend`` name, instance or ``None``, with
    ``scheduler`` set on a copy's ``scheduler_field``. The scheduler is ignored, with a warning,
    when the backend has no such field or already sets it."""
    if backend is None:
        backend = _default_backend_name(scheduler)
    if isinstance(backend, str):
        if backend not in _BACKENDS:
            raise ValueError(
                f"Unknown preprocessing backend {backend!r}; expected one of "
                f"{sorted(_BACKENDS)} or a PreprocessBackend instance."
            )
        backend = _BACKENDS[backend]()
    elif not isinstance(backend, PreprocessBackend):
        raise TypeError(
            f"backend must be a string, a PreprocessBackend or None, not {backend!r}"
        )

    if scheduler is None:
        return backend
    field = type(backend).scheduler_field
    if field is None:
        warnings.warn(
            f"The 'scheduler' argument is ignored by {type(backend).__name__}.",
            stacklevel=_external_stacklevel(),
        )
        return backend
    if getattr(backend, field) is not None:
        warnings.warn(
            f"The 'scheduler' argument is ignored because this {type(backend).__name__} "
            f"already sets {field}={getattr(backend, field)!r}.",
            stacklevel=_external_stacklevel(),
        )
        return backend
    return replace(backend, **{field: scheduler})


_DASK_BACKEND_HINT = (
    "The dask preprocessing backend needs dask. Preprocessing runs without it for parquet and "
    "ROOT input with backend='graphed' (with graphed installed), backend='iterative' (single "
    "process) or backend='futures' (thread pool)."
)
