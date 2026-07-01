"""Switchable execution backends for pydantic preprocessing.

Preprocessing a :class:`~coffea.dataset_tools.filespec.DataGroupSpec` is, per dataset, an
ordered map-reduce: a worker (``get_steps`` / ``get_parquet_form_uuid_steps``) is mapped over
batches of normalized file records and the resulting awkward arrays are concatenated. This
module factors that map-reduce out from :func:`coffea.dataset_tools.preprocess._preprocess_pydantic`
behind a small backend interface so the same worker can run on:

- :class:`DaskBackend` -- the historical path (``dask_awkward.map_partitions`` + an
  ``AwkwardTreeReductionLayer`` + ``dask.compute``). This backend builds and executes a dask
  task graph to orchestrate the map-reduce.
- :class:`IterativeBackend` -- immediate (synchronous, single process) execution, mirroring
  coffea's ``IterativeExecutor``.
- :class:`FuturesBackend` -- a :mod:`concurrent.futures` pool, threads by default (preprocessing
  is IO-bound) with an opt-in process pool (mirrors ``FuturesExecutor``).

The iterative and futures backends do not build a dask *task graph* for orchestration, and they
are dask-free for parquet input and for ROOT input of either flavor. ROOT form extraction
(``save_form=True``) uses uproot's own non-dask form builder for both TTree and RNTuple, producing
a form byte-identical to the ``uproot.dask`` form; it only falls back to ``uproot.dask`` (and thus
dask) if that internal uproot helper is unavailable on an unexpected uproot version.

The interface intentionally echoes the ``coffea.compute`` refactor (PR #1470): a ``Backend``
turns a "computable" into a future-like ``Task`` whose ``result()`` blocks. Here the computable
is a mapping ``{dataset_name: PreprocessJob}`` and the result is ``{dataset_name: awkward.Array}``.
When ``coffea.compute`` lands, these backends should adapt to its ``RunningBackend``/``Task``
protocols with little change.

**Ordering is load-bearing.** Downstream post-processing zips the concatenated result against the
original per-file order, so the reduction here is an *ordered* concatenation -- deliberately not
the order-agnostic ``accumulate``/merging used by the event-processing executors.
"""

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass
from functools import partial
from typing import Protocol, runtime_checkable

import awkward

from coffea.util import _import_dask, _import_dask_awkward, coffea_console

__all__ = [
    "PreprocessJob",
    "PreprocessTask",
    "PreprocessBackend",
    "IterativeBackend",
    "FuturesBackend",
    "DaskBackend",
    "ordered_concat",
    "resolve_backend",
]


def ordered_concat(parts: Iterable[awkward.Array | None]) -> awkward.Array | None:
    """Concatenate mapped outputs while preserving input order.

    ``None`` parts are dropped. Returns ``None`` if nothing remains. This is the reduction step
    for preprocessing and, unlike the order-agnostic accumulation used for processor outputs,
    must preserve order because the caller re-zips the result against the original file order.
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
    """Future-like handle to a submitted preprocessing computable (mirrors ``compute.Task``)."""

    def result(self) -> dict[str, awkward.Array]:
        """Block until done and return ``{dataset_name: concatenated awkward.Array}``."""

    def wait(self) -> None:
        """Block until the computation has finished, without returning the result."""


class PreprocessBackend(ABC):
    """Base class for preprocessing execution backends (mirrors ``compute.RunningBackend``)."""

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
            results[name] = ordered_concat(parts)
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
                results[name] = ordered_concat(parts)
            return results
        finally:
            if self._owns_pool:
                self._pool.shutdown()

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
        workers : int, default 1
            Number of workers when this backend creates the pool.
        use_processes : bool, default False
            Create a :class:`~concurrent.futures.ProcessPoolExecutor` instead of the default
            :class:`~concurrent.futures.ThreadPoolExecutor`. Threads are preferred because
            preprocessing is IO-bound (opening files / reading metadata) and avoids pickling
            the awkward arrays; use processes when form extraction is CPU-heavy.
        pool : concurrent.futures.Executor or Callable, optional
            An existing executor instance to reuse (left open by this backend), or a callable
            (e.g. an ``Executor`` subclass) invoked as ``pool(max_workers=workers)``. Overrides
            ``use_processes`` when given.
    """

    workers: int = 1
    use_processes: bool = False
    pool: Executor | Callable[..., Executor] | None = None

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


@dataclass
class DaskBackend(PreprocessBackend):
    """Execute preprocessing as a dask-awkward task graph (the historical behaviour).

    Per dataset this builds ``from_awkward`` -> ``map_partitions`` -> ``AwkwardTreeReductionLayer``
    and lets a single ``dask.compute`` materialize all datasets together.

    Parameters
    ----------
        scheduler : None or Callable or str, default None
            Passed through to ``dask.compute``.
        split_every : int, default 8
            Fan-in of the tree reduction that concatenates per-batch results.
    """

    scheduler: None | Callable | str = None
    split_every: int = 8

    def _build_collection(self, name: str, job: PreprocessJob, dask, dask_awkward):
        ak_norm_files = job.array
        # guard files_per_batch<1 so the dask backend matches the iterative/futures batching
        # (which clamp via max(1, files_per_batch)) instead of dividing by zero
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


def resolve_backend(
    backend: str | PreprocessBackend | None,
    scheduler: None | Callable | str = None,
) -> PreprocessBackend:
    """Turn a ``backend`` selector into a :class:`PreprocessBackend` instance.

    ``backend`` may be an existing :class:`PreprocessBackend` (returned as-is), or one of the
    strings ``"dask"`` (default), ``"iterative"``, or ``"futures"``. ``scheduler`` is forwarded
    to a default-constructed :class:`DaskBackend`; if it is set while a non-dask backend is
    selected, a warning is issued because it has no effect there.
    """
    if isinstance(backend, PreprocessBackend):
        # A pre-built instance is used as-is; we cannot inject `scheduler` into it (even a
        # DaskBackend instance keeps its own scheduler), so warn rather than silently drop it.
        if scheduler is not None:
            warnings.warn(
                "The 'scheduler' argument is ignored when a PreprocessBackend instance is "
                "passed; set it on the instance, e.g. DaskBackend(scheduler=...).",
                stacklevel=2,
            )
        return backend

    if backend is None or backend == "dask":
        return DaskBackend(scheduler=scheduler)

    if scheduler is not None:
        warnings.warn(
            "The 'scheduler' argument only affects the dask backend and is ignored here.",
            stacklevel=2,
        )
    if backend == "iterative":
        return IterativeBackend()
    if backend == "futures":
        return FuturesBackend()
    raise ValueError(
        f"Unknown preprocessing backend {backend!r}; expected 'dask', 'iterative', "
        "'futures', or a PreprocessBackend instance."
    )


def print_dask_backend_fallback_hint() -> None:
    """Print (via ``coffea_console``) a hint that dask-free backends exist.

    Called when the default dask backend fails because dask / dask-awkward are not importable.
    """
    coffea_console.print(
        "[bold red]The dask preprocessing backend is unavailable because dask / dask-awkward "
        "could not be imported.[/bold red]\n"
        "Preprocessing can run without dask for parquet and ROOT input: pass "
        "[bold]backend='iterative'[/bold] (single process) or [bold]backend='futures'[/bold] "
        "(thread pool) to preprocess() and the format-specific preprocess_* functions."
    )
