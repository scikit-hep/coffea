"""Switchable execution backends for preprocessing (experimental; the interface may change).

Preprocessing a fileset is, per dataset, an ordered map-reduce: a worker (``get_steps`` /
``get_parquet_form_uuid_steps``) is mapped over batches of normalized file records and the
resulting awkward arrays are concatenated. This module factors that map-reduce out of
:func:`coffea.dataset_tools.preprocess._preprocess_pydantic` and
:func:`coffea.dataset_tools.preprocess.preprocess_legacy` behind a small backend interface so the
same worker can run on:

- :class:`DaskBackend` -- ``dask_awkward.map_partitions`` + an ``AwkwardTreeReductionLayer`` +
  ``dask.compute``. This backend builds and executes a dask task graph to orchestrate the
  map-reduce.
- :class:`IterativeBackend` -- immediate (synchronous, single process) execution, mirroring
  coffea's ``IterativeExecutor``.
- :class:`FuturesBackend` -- a :mod:`concurrent.futures` pool, threads by default (preprocessing
  is IO-bound) with an opt-in process pool (mirrors ``FuturesExecutor``).

Without an explicit backend, ``"dask"`` is used when dask and dask-awkward are importable and
``"iterative"`` otherwise. A ``scheduler`` argument is handed to the backend field named by its
class's ``scheduler_field`` (``DaskBackend.scheduler``, ``FuturesBackend.pool``); backends without
one ignore it with a warning.

The iterative and futures backends do not build a dask *task graph* for orchestration, and they
are dask-free for parquet input and for ROOT input of either flavor. ROOT form extraction
(``save_form=True``) calls the private uproot helpers behind ``uproot.dask``'s base form, for both
TTree and RNTuple, producing the same form without dask; it only falls back to ``uproot.dask``
(and thus dask) if those helpers are unavailable on an unexpected uproot version.

A ``Backend`` turns a "computable" into a future-like ``Task`` (``result`` / ``partial_result``
/ ``wait``) whose ``result()`` blocks. Here the computable is a mapping
``{dataset_name: PreprocessJob}`` and the result is ``{dataset_name: awkward.Array}``.

Ordering is not load-bearing for correctness: the reduction concatenates per-batch results, and
the concrete backends happen to preserve submission order, but the downstream post-processing
(``_preprocess_pydantic`` and ``preprocess_legacy``) matches its outputs to the input files *by
filename*, so a backend is free to reduce in any order.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, replace
from functools import partial
from typing import ClassVar, Protocol, runtime_checkable

import awkward

from coffea.util import _import_dask, _import_dask_awkward

__all__ = [
    "PreprocessJob",
    "PreprocessTask",
    "PreprocessBackend",
    "IterativeBackend",
    "FuturesBackend",
    "DaskBackend",
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


def _iter_batches(array: awkward.Array, files_per_batch: int):
    """Yield contiguous slices of ``array`` of at most ``files_per_batch`` rows, in order.

    An empty array yields itself once so the worker still produces a correctly-typed empty
    result (matching the single-partition dask behaviour).
    """
    n = len(array)
    if n == 0:
        yield array
        return
    step = max(1, files_per_batch)
    for start in range(0, n, step):
        yield array[start : start + step]


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
        """Block until done and return ``{dataset_name: concatenated awkward.Array}``."""

    def partial_result(self) -> dict[str, awkward.Array]:
        """Return results for the work finished so far, without blocking on the rest.

        Datasets/batches still running are omitted; a dataset appears only once at least one of
        its batches has completed. Intended for resumable/progress use. Backends that cannot
        produce a cheap partial (e.g. a fused dask graph) may return the full :meth:`result`.
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
        """Start executing ``jobs`` and return a non-blocking :class:`PreprocessTask`."""

    def compute(self, jobs: Mapping[str, PreprocessJob]) -> dict[str, awkward.Array]:
        """Convenience blocking call: ``submit(jobs).result()``."""
        return self.submit(jobs).result()


@dataclass
class _CompletedTask:
    """A :class:`PreprocessTask` whose work is already done (used by the iterative backend)."""

    _results: dict[str, awkward.Array]

    def result(self) -> dict[str, awkward.Array]:
        return self._results

    def partial_result(self) -> dict[str, awkward.Array]:
        # work already finished in submit(), so the partial result is the full result
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
        for name, job in jobs.items():
            parts = [
                job.map_fn(batch)
                for batch in _iter_batches(job.array, job.files_per_batch)
            ]
            results[name] = _ordered_concat(parts)
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
    """A :class:`PreprocessTask` wrapping unmaterialized dask-awkward collections."""

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
        # A fused dask graph has no cheap partial: fall back to the full (blocking) result.
        return self.result()


@dataclass
class DaskBackend(PreprocessBackend):
    """Execute preprocessing as a dask-awkward task graph.

    Per dataset this builds ``from_awkward`` -> ``map_partitions`` -> ``AwkwardTreeReductionLayer``
    and lets a single ``dask.compute`` materialize all datasets together.

    Parameters
    ----------
        scheduler : None or Callable or str, default None
            Passed through to ``dask.compute``. This is the field the ``scheduler`` argument of
            the ``preprocess*`` functions fills.
        split_every : int, default 8
            Fan-in of the tree reduction that concatenates per-batch results.
    """

    scheduler_field: ClassVar[str | None] = "scheduler"

    scheduler: None | Callable | str = None
    split_every: int = 8

    def _build_collection(self, name: str, job: PreprocessJob, dask, dask_awkward):
        ak_norm_files = job.array
        # files_per_batch below 1 is treated as 1, as in _iter_batches
        files_per_batch = max(1, job.files_per_batch)
        dak_norm_files = dask_awkward.from_awkward(
            ak_norm_files, math.ceil(len(ak_norm_files) / files_per_batch)
        )

        concat_fn = partial(awkward.concatenate, axis=0)
        split_every = self.split_every

        files_trl_label = f"preprocess-{name}"
        files_trl_token = dask.base.tokenize(dak_norm_files, concat_fn, split_every)
        files_trl_name = f"{files_trl_label}-{files_trl_token}"
        files_trl_tree_node_name = f"{files_trl_label}-tree-node-{files_trl_token}"

        files_part = dask_awkward.map_partitions(
            job.map_fn,
            dak_norm_files,
            meta=dask_awkward.lib.core.empty_typetracer(),
        )

        files_trl = dask_awkward.layers.layers.AwkwardTreeReductionLayer(
            name=files_trl_name,
            name_input=files_part.name,
            npartitions_input=files_part.npartitions,
            concat_func=concat_fn,
            tree_node_func=lambda x: x,
            finalize_func=lambda x: x,
            split_every=split_every,
            tree_node_name=files_trl_tree_node_name,
        )

        files_graph = dask.highlevelgraph.HighLevelGraph.from_collections(
            files_trl_name, files_trl, dependencies=[files_part]
        )

        return dask_awkward.lib.core.new_array_object(
            files_graph,
            files_trl_name,
            meta=dask_awkward.lib.core.empty_typetracer(),
            npartitions=len(files_trl.output_partitions),
        )

    def submit(self, jobs: Mapping[str, PreprocessJob]) -> PreprocessTask:
        dask = _import_dask()
        dask_awkward = _import_dask_awkward()
        collections = {
            name: self._build_collection(name, job, dask, dask_awkward)
            for name, job in jobs.items()
        }
        return _DaskTask(collections, self.scheduler)


_BACKENDS: dict[str, type[PreprocessBackend]] = {
    "dask": DaskBackend,
    "iterative": IterativeBackend,
    "futures": FuturesBackend,
}


def _default_backend_name() -> str:
    """``"dask"`` when dask and dask-awkward are importable, ``"iterative"`` otherwise."""
    if all(importlib.util.find_spec(m) is not None for m in ("dask", "dask_awkward")):
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
    ``None`` for :func:`_default_backend_name`. A ``scheduler`` is set on the backend's
    ``scheduler_field``; a passed instance is copied rather than mutated. It is ignored, with a
    warning, when the backend has no such field or a passed instance already sets it.
    """
    if backend is None:
        backend = _default_backend_name()
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
    "The dask preprocessing backend needs dask and dask-awkward. Preprocessing runs without "
    "them for parquet and ROOT input with backend='iterative' (single process) or "
    "backend='futures' (thread pool)."
)
