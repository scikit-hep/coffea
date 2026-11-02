"""Switchable execution backends for preprocessing (experimental; the interface may change).

Preprocessing a fileset is, per dataset, an ordered map-reduce: a worker (``get_steps`` /
``get_parquet_form_uuid_steps``) is mapped over batches of normalized file records and the
resulting awkward arrays are concatenated. This module factors that map-reduce out of
:func:`coffea.dataset_tools.preprocess._preprocess_pydantic` and
:func:`coffea.dataset_tools.preprocess.preprocess_legacy` behind a small backend interface so the
same worker can run on:

- :class:`DaskBackend` -- a dask graph with a task per batch and a concatenation per dataset,
  run by any dask scheduler. Needs dask, not dask-awkward.
- :class:`GraphedBackend` -- one graphed ``Plan`` with a task per batch, run by a graphed
  executor (a thread pool by default). Needs graphed, not dask.
- :class:`IterativeBackend` -- immediate (synchronous, single process) execution, mirroring
  coffea's ``IterativeExecutor``.
- :class:`FuturesBackend` -- a :mod:`concurrent.futures` pool, threads by default (preprocessing
  is IO-bound) with an opt-in process pool (mirrors ``FuturesExecutor``).

Every backend cuts the same batches (:func:`_batch_ranges`), concatenates them with
:func:`_ordered_concat` in file order, hands the worker concrete arrays only, raises the worker's
errors from ``result()``, and returns a :class:`PreprocessTask` whose ``partial_result()`` never
blocks.

Without an explicit backend, the backend follows the ``scheduler`` argument when one is given (a
``concurrent.futures`` executor selects futures, a graphed executor selects graphed, anything
else selects dask), and otherwise is the first of: dask, when dask has an active or configured
scheduler; graphed, when installed; dask, when installed; iterative. The ``scheduler`` is handed
to the backend field named by its class's ``scheduler_field`` (``DaskBackend.scheduler``,
``FuturesBackend.pool``, ``GraphedBackend.executor``); backends without one ignore it with a
warning.

ROOT form extraction (``save_form=True``) calls the private uproot helpers behind
``uproot.dask``'s base form, for both TTree and RNTuple, producing the same form without dask; on
uproot older than 5.7.7, which lacks those helpers, it falls back to ``uproot.dask`` (and thus
dask and dask-awkward).

Ordering is not load-bearing for correctness: the downstream post-processing
(``_preprocess_pydantic`` and ``preprocess_legacy``) matches its outputs to the input files *by
filename*. The backends return file order so that their outputs are identical.
"""

from __future__ import annotations

import importlib.util
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
    """Concatenate mapped outputs, preserving input order.

    ``None`` parts are dropped. Returns ``None`` if nothing remains. This is the reduction step
    for preprocessing. Order is preserved for tidiness/determinism, but is not required for
    correctness: the downstream consumer assembles its outputs by filename, so the reduce may run
    in any order.
    """
    parts = [part for part in parts if part is not None]
    if len(parts) == 0:
        return None
    if len(parts) == 1:
        return parts[0]
    return awkward.concatenate(parts, axis=0)


def _batch_ranges(n: int, files_per_batch: int) -> list[tuple[int, int]]:
    """The ``[start, stop)`` row ranges of the batches of ``n`` file records, in order.

    Each batch holds ``files_per_batch`` records (at least 1), the last one the remainder. No
    records give one empty batch, so the worker still produces a correctly typed empty result.
    """
    step = max(1, files_per_batch)
    return [(start, min(start + step, n)) for start in range(0, max(n, 1), step)]


def _iter_batches(array: awkward.Array, files_per_batch: int):
    """Yield the batches of ``array`` given by :func:`_batch_ranges`, in order."""
    for start, stop in _batch_ranges(len(array), files_per_batch):
        yield array[start:stop]


@dataclass
class PreprocessJob:
    """One dataset's preprocessing work: map ``map_fn`` over batches of ``array`` and concat.

    Parameters
    ----------
        array : awkward.Array
            Normalized file records (fields ``file``, ``object_path``, ``steps``,
            ``num_entries``, ``uuid``) for a single dataset.
        map_fn : Callable[[awkward.Array], awkward.Array]
            The per-batch worker, e.g. ``functools.partial(get_steps, **options)``. Must be a
            top-level (picklable) callable so it can run under a process pool.
        files_per_batch : int, default 1
            Number of files handled per unit of work. Larger values mean fewer, heavier tasks.
    """

    array: awkward.Array
    map_fn: Callable[[awkward.Array], awkward.Array]
    files_per_batch: int = 1


@runtime_checkable
class PreprocessTask(Protocol):
    """Future-like handle to a submitted preprocessing computable."""

    def result(self) -> dict[str, awkward.Array]:
        """Block until done and return ``{dataset_name: concatenated awkward.Array}``.

        Raises the first error the worker raised.
        """

    def partial_result(self) -> dict[str, awkward.Array]:
        """Return the results finished so far, without blocking.

        A dataset appears once at least one of its batches has finished, with only the finished
        batches; ``{}`` means nothing has finished. Backends that report work only as a whole
        (dask, graphed) return everything once the work is done and ``{}`` before.
        """

    def wait(self) -> None:
        """Block until the computation has finished, without returning the result."""


class PreprocessBackend(ABC):
    """Base class for preprocessing execution backends.

    A subclass that accepts the ``scheduler`` argument of the ``preprocess*`` functions names the
    dataclass field that receives it in ``scheduler_field``; ``None`` means it takes none.
    """

    scheduler_field: ClassVar[str | None] = None

    @abstractmethod
    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        """Hand ``jobs`` to the backend and return a :class:`PreprocessTask` for them.

        The work runs in the background (futures; graphed on an executor with ``submit(plan)``),
        inside ``submit`` (iterative; graphed on an executor with only ``run(plan)``), or when
        the task's ``result()`` or ``wait()`` is called (dask). Errors the worker raises surface
        from ``result()``, never from ``submit``.
        """

    def compute(self, jobs: Mapping[str, PreprocessJob]) -> dict[str, awkward.Array]:
        """Convenience blocking call: ``submit(jobs).result()``."""
        return self.submit(jobs).result()


@dataclass
class _CompletedTask:
    """A :class:`PreprocessTask` for work that ran inside ``submit``: its results, and the error
    that stopped it, if any, which ``result()`` raises."""

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
    """Execute preprocessing immediately -- synchronously, in the current thread.

    Named to mirror coffea's ``IterativeExecutor``; the execution is "immediate" in that each
    batch is mapped and reduced inline with no deferral or task graph. No dask required.
    """

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
    """A :class:`PreprocessTask` backed by :mod:`concurrent.futures` futures.

    Futures are kept grouped and ordered per dataset so results reassemble in the original file
    order regardless of completion order. A pool created by the backend is shut down once the
    result has been gathered; an externally-supplied pool is left open.
    """

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
                # .result() re-raises worker exceptions here; ordering follows submission
                parts = [fut.result() for fut in futs]
                results[name] = _ordered_concat(parts)
        except BaseException:
            # fail fast: cancel batches that have not started so a fatal error does not wait
            # for the rest of the fileset to be processed
            for fut in self._all_futures():
                fut.cancel()
            if self._owns_pool:
                self._pool.shutdown(wait=False, cancel_futures=True)
            raise
        if self._owns_pool:
            self._pool.shutdown()
        return results

    def partial_result(self) -> dict[str, awkward.Array]:
        # Gather only the batches that have completed successfully, per dataset, without blocking.
        # The owned pool is not shut down here; unfinished work may still be running.
        results: dict[str, awkward.Array] = {}
        for name, futs in self._name_to_futures.items():
            done_parts = [
                fut.result()
                for fut in futs
                if fut.done() and not fut.cancelled() and fut.exception() is None
            ]
            merged = _ordered_concat(done_parts)
            if merged is not None:
                results[name] = merged
        return results

    def __del__(self):
        # Safety net: an owned pool is normally shut down in result(); if the caller only
        # wait()s or drops the task without calling result(), avoid leaking the pool.
        # Executor.shutdown is idempotent, so a double call after result() is harmless.
        if getattr(self, "_owns_pool", False):
            self._pool.shutdown(wait=False)


@dataclass
class FuturesBackend(PreprocessBackend):
    """Execute preprocessing over a :mod:`concurrent.futures` pool. No dask required.

    Batches from every dataset are submitted to a single shared pool for good cross-dataset
    load balancing, then reassembled per dataset in order.

    Parameters
    ----------
        workers : int or None, default None
            Number of workers when this backend creates the pool. ``None`` uses the executor's
            default sizing (for :class:`~concurrent.futures.ThreadPoolExecutor`,
            ``min(32, os.cpu_count() + 4)``; for
            :class:`~concurrent.futures.ProcessPoolExecutor`, ``os.cpu_count()``).
        use_processes : bool, default False
            Create a :class:`~concurrent.futures.ProcessPoolExecutor` instead of the default
            :class:`~concurrent.futures.ThreadPoolExecutor`. Threads are preferred because
            preprocessing is IO-bound (opening files / reading metadata) and avoids pickling
            the awkward arrays; use processes when form extraction is CPU-heavy.
        pool : concurrent.futures.Executor or Callable, optional
            An existing executor instance to reuse (left open by this backend), or a callable
            (e.g. an ``Executor`` subclass) invoked as ``pool(max_workers=workers)``. Overrides
            ``use_processes`` when given. This is the field the ``scheduler`` argument of the
            ``preprocess*`` functions fills.
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
    """A :class:`PreprocessTask` over dask collections, computed on ``result()`` or ``wait()``."""

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
    """Execute preprocessing as a dask graph: a task per batch and a concatenation per dataset,
    computed by one ``dask.compute`` over all datasets. Needs dask, not dask-awkward.

    Parameters
    ----------
        scheduler : None or Callable or str, default None
            Passed through to ``dask.compute``; None uses dask's active or configured scheduler,
            else its threaded scheduler. This is the field the ``scheduler`` argument of the
            ``preprocess*`` functions fills.
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
    """Execute preprocessing as one graphed ``Plan``: a task per batch of ``files_per_batch``
    files of each dataset. Needs graphed, not dask.

    Parameters
    ----------
        executor : graphed executor or None, default None
            Anything with ``run(plan)``, e.g. ``graphed_executors.local.ThreadExecutor`` or
            ``graphed.core.execution.SequentialRunner``; None runs the tasks on a thread pool of
            ``workers`` threads. Executors that also have ``submit(plan)`` run without blocking
            until ``result()``. This is the field the ``scheduler`` argument of the
            ``preprocess*`` functions fills.
        workers : int or None, default None
            Threads of the default thread pool; None uses
            :class:`~concurrent.futures.ThreadPoolExecutor`'s default. Unused with an
            ``executor``.
    """

    scheduler_field: ClassVar[str | None] = "executor"

    executor: object = None
    workers: int | None = None

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

        return _graphed.submit(jobs, self.executor, self.workers)


_BACKENDS: dict[str, type[PreprocessBackend]] = {
    "dask": DaskBackend,
    "graphed": GraphedBackend,
    "iterative": IterativeBackend,
    "futures": FuturesBackend,
}


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _scheduler_backend_name(scheduler: object) -> str:
    """The backend a ``scheduler`` argument belongs to.

    A ``concurrent.futures`` executor (or executor class) belongs to futures; an object with
    ``run(plan)`` that does not come from dask or distributed (a ``distributed.Client`` has a
    ``run`` method) is a graphed executor; anything else is a dask scheduler.
    """
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


def _dask_has_scheduler() -> bool:
    """Whether dask has an active ``distributed.Client`` or a configured scheduler."""
    if not _installed("dask"):
        return False
    dask = _import_dask()
    return dask.base.get_scheduler() is not None


def _default_backend_name(scheduler: object = None) -> str:
    """The backend for ``backend=None``: the one ``scheduler`` belongs to when given, else dask
    when dask has an active or configured scheduler, else graphed when installed, else dask when
    installed, else iterative."""
    if scheduler is not None:
        return _scheduler_backend_name(scheduler)
    if _dask_has_scheduler():
        return "dask"
    if _installed("graphed"):
        return "graphed"
    if _installed("dask"):
        return "dask"
    return "iterative"


def _external_stacklevel() -> int:
    """The ``stacklevel`` that attributes a warning to the first frame outside dataset_tools.

    Counts from the frame that calls this helper (``stacklevel=1`` for a warning issued there).
    """
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
    """Turn a ``backend`` selector into a :class:`PreprocessBackend` instance.

    ``backend`` is a :class:`PreprocessBackend` instance, a name registered in ``_BACKENDS``, or
    ``None`` for :func:`_default_backend_name` of ``scheduler``. A ``scheduler`` is set on the backend's
    ``scheduler_field``; a passed instance is copied rather than mutated. It is ignored, with a
    warning, when the backend has no such field or a passed instance already sets it.
    """
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
