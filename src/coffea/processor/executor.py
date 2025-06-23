import concurrent.futures
import json
import math
import os
import pickle
import time
import traceback
import uuid
import warnings
from collections import defaultdict
from collections.abc import Awaitable, Generator, Iterable, Mapping, MutableMapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from itertools import repeat
from typing import (
    Callable,
    Optional,
    Union,
)

import cloudpickle
import lz4.frame as lz4f
import toml
import uproot
from cachetools import LRUCache

from ..nanoevents import NanoEventsFactory, schemas
from ..util import _exception_chain, _hash, rich_bar
from .accumulator import Accumulatable, accumulate, set_accumulator
from .processor import ProcessorABC

_PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL
DEFAULT_METADATA_CACHE: MutableMapping = LRUCache(100000)

_PROTECTED_NAMES = {
    "dataset",
    "filename",
    "treename",
    "metadata",
    "entrystart",
    "entrystop",
    "fileuuid",
    "numentries",
    "uuid",
    "clusters",
}


class UprootMissTreeError(uproot.exceptions.KeyInFileError):
    pass


class FileMeta:
    __slots__ = ["dataset", "filename", "treename", "metadata"]

    def __init__(self, dataset, filename, treename, metadata=None):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.metadata = metadata

    def __str__(self):
        return f"FileMeta({self.filename}:{self.treename})"

    def __hash__(self):
        # As used to lookup metadata, no need for dataset
        return _hash((self.filename, self.treename))

    def __eq__(self, other):
        # In case of hash collisions
        return self.filename == other.filename and self.treename == other.treename

    def maybe_populate(self, cache):
        if cache and self in cache:
            self.metadata = cache[self]

    def populated(self, clusters=False):
        """Return true if metadata is populated

        By default, only require bare minimum metadata (numentries, uuid)
        If clusters is True, then require cluster metadata to be populated
        """
        if self.metadata is None:
            return False
        elif "numentries" not in self.metadata or "uuid" not in self.metadata:
            return False
        elif clusters and "clusters" not in self.metadata:
            return False
        return True

    def chunks(self, target_chunksize, align_clusters):
        if not self.populated(clusters=align_clusters):
            raise RuntimeError
        user_keys = set(self.metadata.keys()) - _PROTECTED_NAMES
        user_meta = {k: self.metadata[k] for k in user_keys}
        if align_clusters:
            chunks = [0]
            for c in self.metadata["clusters"]:
                if c >= chunks[-1] + target_chunksize:
                    chunks.append(c)
            if self.metadata["clusters"][-1] != chunks[-1]:
                chunks.append(self.metadata["clusters"][-1])
            for start, stop in zip(chunks[:-1], chunks[1:]):
                yield WorkItem(
                    self.dataset,
                    self.filename,
                    self.treename,
                    start,
                    stop,
                    self.metadata["uuid"],
                    user_meta,
                )
            return target_chunksize
        else:
            numentries = self.metadata["numentries"]
            update = True
            start = 0
            while start < numentries:
                if update:
                    n = max(round((numentries - start) / target_chunksize), 1)
                    actual_chunksize = math.ceil((numentries - start) / n)
                stop = min(numentries, start + actual_chunksize)
                next_chunksize = yield WorkItem(
                    self.dataset,
                    self.filename,
                    self.treename,
                    start,
                    stop,
                    self.metadata["uuid"],
                    user_meta,
                )
                start = stop
                if next_chunksize and next_chunksize != target_chunksize:
                    target_chunksize = next_chunksize
                    update = True
                else:
                    update = False
            return target_chunksize


@dataclass(unsafe_hash=True, frozen=True)
class WorkItem:
    dataset: str
    filename: str
    treename: str
    entrystart: int
    entrystop: int
    fileuuid: str
    usermeta: Optional[dict] = field(default=None, compare=False)

    def __len__(self) -> int:
        return self.entrystop - self.entrystart


def _compress(item, compression):
    if item is None or compression is None:
        return item
    else:
        with BytesIO() as bf:
            with lz4f.open(bf, mode="wb", compression_level=compression) as f:
                pickle.dump(item, f, protocol=_PICKLE_PROTOCOL)
            result = bf.getvalue()
        return result


def _decompress(item):
    if isinstance(item, bytes):
        # warning: if item is not exactly of type bytes, BytesIO(item) will
        # make a copy of it, increasing the memory usage.
        with BytesIO(item) as bf:
            with lz4f.open(bf, mode="rb") as f:
                return pickle.load(f)
    else:
        return item


class _compression_wrapper:
    def __init__(self, level, function, name=None):
        self.level = level
        self.function = function
        self.name = name

    def __str__(self):
        if self.name is not None:
            return self.name
        try:
            name = self.function.__name__
            if name == "<lambda>":
                return "lambda"
            return name
        except AttributeError:
            return str(self.function)

    # no @wraps due to pickle
    def __call__(self, *args, **kwargs):
        out = self.function(*args, **kwargs)
        return _compress(out, self.level)


class _reduce:
    def __init__(self, compression):
        self.compression = compression

    def __str__(self):
        return "reduce"

    def __call__(self, items):
        items = list(it for it in items if it is not None)
        if len(items) == 0:
            raise ValueError("Empty list provided to reduction")
        if self.compression is not None:
            out = _decompress(items.pop())
            out = accumulate(map(_decompress, items), out)
            return _compress(out, self.compression)
        return accumulate(items)


class _FuturesHolder:
    def __init__(self, futures: set[Awaitable], refresh=2):
        self.futures = set(futures)
        self.merges = set()
        self.completed = set()
        self.done = {"futures": 0, "merges": 0}
        self.running = len(self.futures)
        self.refresh = refresh

    def update(self, refresh: int = None):
        if refresh is None:
            refresh = self.refresh
        if self.futures:
            completed, self.futures = concurrent.futures.wait(
                self.futures,
                timeout=refresh,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            self.completed.update(completed)
            self.done["futures"] += len(completed)

        if self.merges:
            completed, self.merges = concurrent.futures.wait(
                self.merges,
                timeout=refresh,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            self.completed.update(completed)
            self.done["merges"] += len(completed)
        self.running = len(self.futures) + len(self.merges)

    def add_merge(self, merges: Awaitable[Accumulatable]):
        self.merges.add(merges)
        self.running = len(self.futures) + len(self.merges)

    def fetch(self, N: int) -> list[Accumulatable]:
        _completed = [self.completed.pop() for _ in range(min(N, len(self.completed)))]
        if all(_good_future(future) for future in _completed):
            return [future.result() for future in _completed if _good_future(future)]
        else:  # Make recoverable
            good_futures = [future for future in _completed if _good_future(future)]
            bad_futures = [future for future in _completed if not _good_future(future)]
            self.completed.update(good_futures)
            raise bad_futures[0].exception()


def _good_future(future: Awaitable) -> bool:
    return future.done() and not future.cancelled() and future.exception() is None


def _futures_handler(futures, timeout):
    """Essentially the same as concurrent.futures.as_completed
    but makes sure not to hold references to futures any longer than strictly necessary,
    which is important if the future holds a large result.
    """
    futures = set(futures)
    try:
        while futures:
            try:
                done, futures = concurrent.futures.wait(
                    futures,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if len(done) == 0:
                    warnings.warn(
                        f"No finished jobs after {timeout}s, stopping remaining {len(futures)} jobs early"
                    )
                    break
                while done:
                    try:
                        yield done.pop().result()
                    except concurrent.futures.CancelledError:
                        pass
            except KeyboardInterrupt as e:
                for job in futures:
                    try:
                        job.cancel()
                        # this is not implemented with parsl AppFutures
                    except NotImplementedError:
                        raise e from None
                running = sum(job.running() for job in futures)
                warnings.warn(
                    f"Early stop: cancelled {len(futures) - running} jobs, will wait for {running} running jobs to complete"
                )
    finally:
        running = sum(job.running() for job in futures)
        if running:
            warnings.warn(
                f"Cancelling {running} running jobs (likely due to an exception)"
            )
        try:
            while futures:
                futures.pop().cancel()
        except NotImplementedError:
            pass


@dataclass
class ExecutorBase:
    # shared by all executors
    status: bool = True
    unit: str = "items"
    desc: str = "Processing"
    compression: Optional[int] = 1
    function_name: Optional[str] = None

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        raise NotImplementedError(
            "This class serves as a base class for executors, do not instantiate it!"
        )

    def copy(self, **kwargs):
        tmp = self.__dict__.copy()
        tmp.update(kwargs)
        return type(self)(**tmp)


def _watcher(
    FH: _FuturesHolder,
    executor: ExecutorBase,
    merge_fcn: Callable,
    pool: Optional[Callable] = None,
) -> Accumulatable:
    with rich_bar() as progress:
        p_id = progress.add_task(executor.desc, total=FH.running, unit=executor.unit)
        desc_m = "Merging" if executor.merging else "Merging (local)"
        p_idm = progress.add_task(desc_m, total=0, unit="merges")

        merged = None
        while FH.running > 0:
            FH.update()
            progress.update(p_id, completed=FH.done["futures"], refresh=True)

            if executor.merging:  # Merge jobs
                merge_size = executor._merge_size(len(FH.completed))
                progress.update(p_idm, completed=FH.done["merges"])
                while len(FH.completed) > 1:
                    if FH.running > 0 and len(FH.completed) < executor.merging[1]:
                        break
                    batch = FH.fetch(merge_size)
                    # Add debug for batch mem size? TODO with logging?
                    if isinstance(executor, FuturesExecutor) and pool is not None:
                        FH.add_merge(pool.submit(merge_fcn, batch))
                    elif isinstance(executor, ParslExecutor):
                        FH.add_merge(merge_fcn(batch))
                    else:
                        raise RuntimeError("Invalid executor")
                    progress.update(
                        p_idm,
                        total=progress._tasks[p_idm].total + 1,
                        refresh=True,
                    )
            else:  # Merge within process
                batch = FH.fetch(len(FH.completed))
                merged = _compress(
                    accumulate(
                        progress.track(
                            map(_decompress, (c for c in batch)),
                            task_id=p_idm,
                            total=progress._tasks[p_idm].total + len(batch),
                        ),
                        _decompress(merged),
                    ),
                    executor.compression,
                )
        # Add checkpointing

        if executor.merging:
            progress.refresh()
            merged = FH.completed.pop().result()
        if len(FH.completed) > 0 or len(FH.futures) > 0 or len(FH.merges) > 0:
            raise RuntimeError("Not all futures are added.")
        return merged


def _wait_for_merges(FH: _FuturesHolder, executor: ExecutorBase) -> Accumulatable:
    with rich_bar() as progress:
        if executor.merging:
            to_finish = len(FH.merges)
            p_id_w = progress.add_task(
                "Waiting for merge jobs",
                total=to_finish,
                unit=executor.unit,
            )
            while len(FH.merges) > 0:
                FH.update()
                progress.update(
                    p_id_w,
                    completed=(to_finish - len(FH.merges)),
                    refresh=True,
                )

        FH.update()
        recovered = [future.result() for future in FH.completed if _good_future(future)]
        p_id_m = progress.add_task("Merging finished jobs", unit="merges")
        return _compress(
            accumulate(
                progress.track(
                    map(_decompress, (c for c in recovered)),
                    task_id=p_id_m,
                    total=len(recovered),
                )
            ),
            executor.compression,
        )


@dataclass
class IterativeExecutor(ExecutorBase):
    """Execute in one thread iteratively

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Ignored for iterative executor
    """

    workers: int = 1

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        with rich_bar() as progress:
            p_id = progress.add_task(
                self.desc, total=len(items), unit=self.unit, disable=not self.status
            )
            return (
                accumulate(
                    progress.track(
                        map(function, (c for c in items)),
                        total=len(items),
                        task_id=p_id,
                    ),
                    accumulator,
                ),
                0,
            )


@dataclass
class FuturesExecutor(ExecutorBase):
    """Execute using multiple local cores using python futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        pool : concurrent.futures.Executor class or instance, optional
            The type of futures executor to use, defaults to ProcessPoolExecutor.
            You can pass an instance instead of a class to reuse an executor
        workers : int, optional
            Number of parallel processes for futures (default 1)
        status : bool, optional
            If true (default), enable progress bar
        desc : str, optional
            Label of progress description (default: 'Processing')
        unit : str, optional
            Label of progress bar bar unit (default: 'items')
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        recoverable : bool, optional
            Instead of raising Exception right away, the exception is captured and returned
            up for custom parsing. Already completed items will be returned as well.
        checkpoints : bool
            To do
        merging : bool | tuple(int, int, int), optional
            Enables submitting intermediate merge jobs to the executor. Format is
            (n_batches, min_batch_size, max_batch_size). Passing ``True`` will use default: (5, 4, 100),
            aka as they are returned try to split completed jobs into 5 batches, but of at least 4 and at most 100 items.
            Default is ``False`` - results get merged as they finish in the main process.
        nparts : int, optional
            Number of merge jobs to create at a time. Also pass via ``merging(X, ..., ...)''
        minred : int, optional
            Minimum number of items to merge in one job. Also pass via ``merging(..., X, ...)''
        maxred : int, optional
            maximum number of items to merge in one job. Also pass via ``merging(..., ..., X)''
        mergepool : concurrent.futures.Executor class or instance | int, optional
            Supply an additional executor to process merge jobs independently.
            An ``int`` will be interpreted as ``ProcessPoolExecutor(max_workers=int)``.
        tailtimeout : int, optional
            Timeout requirement on job tails. Cancel all remaining jobs if none have finished
            in the timeout window.
    """

    pool: Union[
        Callable[..., concurrent.futures.Executor], concurrent.futures.Executor
    ] = concurrent.futures.ProcessPoolExecutor  # fmt: skip
    mergepool: Optional[
        Union[
            Callable[..., concurrent.futures.Executor],
            concurrent.futures.Executor,
            bool,
        ]
    ] = None
    recoverable: bool = False
    merging: Union[bool, tuple[int, int, int]] = False
    workers: int = 1
    tailtimeout: Optional[int] = None

    def __post_init__(self):
        if not (
            isinstance(self.merging, bool)
            or (isinstance(self.merging, tuple) and len(self.merging) == 3)
        ):
            raise ValueError(
                f"merging={self.merging} not understood. Required format is "
                "(n_batches, min_batch_size, max_batch_size)"
            )
        elif self.merging is True:
            self.merging = (5, 4, 100)

    def _merge_size(self, size: int):
        return min(self.merging[2], max(size // self.merging[0] + 1, self.merging[1]))

    def __getstate__(self):
        return dict(self.__dict__, pool=None)

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        if self.compression is not None:
            function = _compression_wrapper(self.compression, function)
        reducer = _reduce(self.compression)

        def _processwith(pool, mergepool):
            FH = _FuturesHolder(
                {pool.submit(function, item) for item in items}, refresh=2
            )

            try:
                if mergepool is None:
                    merged = _watcher(FH, self, reducer, pool)
                else:
                    merged = _watcher(FH, self, reducer, mergepool)
                return accumulate([_decompress(merged), accumulator]), 0

            except Exception as e:
                traceback.print_exc()
                if self.recoverable:
                    print("Exception occurred, recovering progress...")
                    for job in FH.futures:
                        job.cancel()

                    merged = _wait_for_merges(FH, self)
                    return accumulate([_decompress(merged), accumulator]), e
                else:
                    raise e from None

        if isinstance(self.pool, concurrent.futures.Executor):
            return _processwith(pool=self.pool, mergepool=self.mergepool)
        else:
            # assume its a class then
            with ExitStack() as stack:
                poolinstance = stack.enter_context(self.pool(max_workers=self.workers))
                if self.mergepool is not None:
                    if isinstance(self.mergepool, int):
                        self.mergepool = concurrent.futures.ProcessPoolExecutor(
                            max_workers=self.mergepool
                        )
                    mergepoolinstance = stack.enter_context(self.mergepool)
                else:
                    mergepoolinstance = None
                return _processwith(pool=poolinstance, mergepool=mergepoolinstance)


@dataclass
class DaskExecutor(ExecutorBase):
    """Execute using dask futures

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        client : distributed.client.Client
            A dask distributed client instance
        treereduction : int, optional
            Tree reduction factor for output accumulators (default: 20)
        status : bool, optional
            If true (default), enable progress bar
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        priority : int, optional
            Task priority, default 0
        retries : int, optional
            Number of retries for failed tasks (default: 3)
        heavy_input : serializable, optional
            Any value placed here will be broadcast to workers and joined to input
            items in a tuple (item, heavy_input) that is passed to function.
        function_name : str, optional
            Name of the function being passed
        use_dataframes: bool, optional
            Retrieve output as a distributed Dask DataFrame (default: False).
            The outputs of individual tasks must be Pandas DataFrames.

            .. note:: If ``heavy_input`` is set, ``function`` is assumed to be pure.
    """

    client: Optional["dask.distributed.Client"] = None  # noqa
    treereduction: int = 20
    priority: int = 0
    retries: int = 3
    heavy_input: Optional[bytes] = None
    use_dataframes: bool = False
    # secret options
    worker_affinity: bool = False

    def __getstate__(self):
        return dict(self.__dict__, client=None)

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator

        import dask.dataframe as dd
        from dask.distributed import Client
        from distributed.scheduler import KilledWorker

        if self.client is None:
            self.client = Client(threads_per_worker=1)

        if self.use_dataframes:
            self.compression = None

        reducer = _reduce(self.compression)
        if self.compression is not None:
            function = _compression_wrapper(
                self.compression, function, name=self.function_name
            )

        if self.heavy_input is not None:
            # client.scatter is not robust against adaptive clusters
            # https://github.com/CoffeaTeam/coffea/issues/465
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Large object of size")
                items = list(
                    zip(
                        items, repeat(self.client.submit(lambda x: x, self.heavy_input))
                    )
                )

        work = []
        key_to_item = {}
        if self.worker_affinity:
            workers = list(self.client.run(lambda: 0))

            def belongsto(heavy_input, workerindex, item):
                if heavy_input is not None:
                    item = item[0]
                hashed = _hash(
                    (item.fileuuid, item.treename, item.entrystart, item.entrystop)
                )
                return hashed % len(workers) == workerindex

            for workerindex, worker in enumerate(workers):
                items_worker = [
                    item
                    for item in items
                    if belongsto(self.heavy_input, workerindex, item)
                ]
                work_worker = self.client.map(
                    function,
                    items_worker,
                    pure=(self.heavy_input is not None),
                    priority=self.priority,
                    retries=self.retries,
                    workers={worker},
                    allow_other_workers=False,
                )
                work.extend(work_worker)
                key_to_item.update(
                    {
                        future.key: item
                        for future, item in zip(work_worker, items_worker)
                    }
                )
        else:
            work = self.client.map(
                function,
                items,
                pure=(self.heavy_input is not None),
                priority=self.priority,
                retries=self.retries,
            )
            key_to_item.update({future.key: item for future, item in zip(work, items)})
        if (self.function_name == "get_metadata") or not self.use_dataframes:
            while len(work) > 1:
                work = self.client.map(
                    reducer,
                    [
                        work[i : i + self.treereduction]
                        for i in range(0, len(work), self.treereduction)
                    ],
                    pure=True,
                    priority=self.priority,
                    retries=self.retries,
                )
                key_to_item.update({future.key: "(output reducer)" for future in work})
            work = work[0]
            try:
                if self.status:
                    from distributed import progress

                    # FIXME: fancy widget doesn't appear, have to live with boring pbar
                    progress(work, multi=True, notebook=False)
                return (
                    accumulate(
                        [
                            (
                                work.result()
                                if self.compression is None
                                else _decompress(work.result())
                            )
                        ],
                        accumulator,
                    ),
                    0,
                )
            except KilledWorker as ex:
                baditem = key_to_item[ex.task]
                if self.heavy_input is not None and isinstance(baditem, tuple):
                    baditem = baditem[0]
                raise RuntimeError(
                    f"Work item {baditem} caused a KilledWorker exception (likely a segfault or out-of-memory issue)"
                )
        else:
            if self.status:
                from distributed import progress

                progress(work, multi=True, notebook=False)
            return {"out": dd.from_delayed(work)}, 0


@dataclass
class ParslExecutor(ExecutorBase):
    """Execute using parsl pyapp wrapper

    Parameters
    ----------
        items : list
            List of input arguments
        function : callable
            A function to be called on each input, which returns an accumulator instance
        accumulator : Accumulatable
            An accumulator to collect the output of the function
        config : parsl.config.Config, optional
            A parsl DataFlow configuration object. Necessary if there is no active kernel

            .. note:: In general, it is safer to construct the DFK with ``parsl.load(config)`` prior to calling this function
        status : bool
            If true (default), enable progress bar
        unit : str
            Label of progress bar unit
        desc : str
            Label of progress bar description
        compression : int, optional
            Compress accumulator outputs in flight with LZ4, at level specified (default 1)
            Set to ``None`` for no compression.
        recoverable : bool, optional
            Instead of raising Exception right away, the exception is captured and returned
            up for custom parsing. Already completed items will be returned as well.
        merging : bool | tuple(int, int, int), optional
            Enables submitting intermediate merge jobs to the executor. Format is
            (n_batches, min_batch_size, max_batch_size). Passing ``True`` will use default: (5, 4, 100),
            aka as they are returned try to split completed jobs into 5 batches, but of at least 4 and at most 100 items.
            Default is ``False`` - results get merged as they finish in the main process.
        jobs_executors : list | "all" optional
            Labels of the executors (from dfk.config.executors) that will process main jobs.
            Default is 'all'. Recommended is ``['jobs']``, while passing ``label='jobs'`` to the primary executor.
        merges_executors : list | "all" optional
            Labels of the executors (from dfk.config.executors) that will process main jobs.
            Default is 'all'. Recommended is ``['merges']``, while passing ``label='merges'`` to the executor dedicated towards merge jobs.
        tailtimeout : int, optional
            Timeout requirement on job tails. Cancel all remaining jobs if none have finished
            in the timeout window.
    """

    tailtimeout: Optional[int] = None
    config: Optional["parsl.config.Config"] = None  # noqa
    recoverable: bool = False
    merging: Optional[Union[bool, tuple[int, int, int]]] = False
    jobs_executors: Union[str, list] = "all"
    merges_executors: Union[str, list] = "all"

    def __post_init__(self):
        if not (
            isinstance(self.merging, bool)
            or (isinstance(self.merging, tuple) and len(self.merging) == 3)
        ):
            raise ValueError(
                f"merging={self.merging} not understood. Required format is "
                "(n_batches, min_batch_size, max_batch_size)"
            )
        elif self.merging is True:
            self.merging = (5, 4, 100)

    def _merge_size(self, size: int):
        return min(self.merging[2], max(size // self.merging[0] + 1, self.merging[1]))

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        if len(items) == 0:
            return accumulator
        import parsl
        from parsl.app.app import python_app

        from .parsl.timeout import timeout

        if self.compression is not None:
            function = _compression_wrapper(self.compression, function)

        # Parse config if passed
        cleanup = False
        try:
            parsl.dfk()
        except RuntimeError:
            cleanup = True
            pass
        if cleanup and self.config is None:
            raise RuntimeError(
                "No active parsl DataFlowKernel, must specify a config to construct one"
            )
        elif not cleanup and self.config is not None:
            raise RuntimeError("An active parsl DataFlowKernel already exists")
        elif self.config is not None:
            parsl.clear()
            parsl.load(self.config)

        # Check config/executors
        _exec_avail = [exe.label for exe in parsl.dfk().config.executors]
        _execs_tried = (
            [] if self.jobs_executors == "all" else [e for e in self.jobs_executors]
        )
        _execs_tried += (
            [] if self.merges_executors == "all" else [e for e in self.merges_executors]
        )
        if not all([_e in _exec_avail for _e in _execs_tried]):
            raise RuntimeError(
                f"Executors: [{','.join(_e for _e in _execs_tried if _e not in _exec_avail)}] not available in the config."
            )

        # Apps
        app = timeout(python_app(function, executors=self.jobs_executors))
        reducer = timeout(
            python_app(_reduce(self.compression), executors=self.merges_executors)
        )

        FH = _FuturesHolder(set(map(app, items)), refresh=2)
        try:
            merged = _watcher(FH, self, reducer)
            return accumulate([_decompress(merged), accumulator]), 0

        except Exception as e:
            traceback.print_exc()
            if self.recoverable:
                print("Exception occurred, recovering progress...")
                # for job in FH.futures:  # NotImplemented in parsl
                #     job.cancel()

                merged = _wait_for_merges(FH, self)
                return accumulate([_decompress(merged), accumulator]), e
            else:
                raise e from None
        finally:
            if cleanup:
                parsl.dfk().cleanup()
                parsl.clear()


class ParquetFileUprootShim:
    def __init__(self, table, name):
        self.table = table
        self.name = name

    def array(self, **kwargs):
        import awkward

        return awkward.Array(self.table[self.name])


class ParquetFileContext:
    def __init__(self, filename):
        self.filename = filename
        self.filehandle = None
        self.branchnames = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

    def _get_handle(self):
        import pyarrow.parquet as pq

        if self.filehandle is None:
            self.filehandle = pq.ParquetFile(self.filename)
            self.branchnames = {
                item.path.split(".")[0] for item in self.filehandle.schema
            }

    @property
    def num_entries(self):
        self._get_handle()
        return self.filehandle.metadata.num_rows

    def keys(self):
        self._get_handle()
        return self.branchnames

    def __iter__(self):
        self._get_handle()
        return iter(self.branchnames)

    def __getitem__(self, name):
        self._get_handle()
        if name in self.branchnames:
            return ParquetFileUprootShim(
                self.filehandle.read([name], use_threads=False), name
            )
        else:
            return KeyError(name)

    def __contains__(self, name):
        self._get_handle()
        return name in self.branchnames


@dataclass
class Runner:
    """A tool to run a processor using uproot for data delivery

    A convenience wrapper to submit jobs for a file set, which is a
    dictionary of dataset: [file list] entries.  Supports only uproot TTree
    reading, via NanoEvents.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        executor : ExecutorBase instance
            Executor, which implements a callable with inputs: items, function, accumulator
            and performs some action equivalent to:
            ``for item in items: accumulator += function(item)``
        pre_executor : ExecutorBase instance
            Executor, used to calculate fileset metadata
            Defaults to executor
        chunksize : int, optional
            Maximum number of entries to process at a time in the data frame, default: 100k
        maxchunks : int, optional
            Maximum number of chunks to process per dataset
            Defaults to processing the whole dataset
        metadata_cache : mapping, optional
            A dict-like object to use as a cache for (file, tree) metadata that is used to
            determine chunking.  Defaults to a in-memory LRU cache that holds 100k entries
            (about 1MB depending on the length of filenames, etc.)  If you edit an input file
            (please don't) during a session, the session can be restarted to clear the cache.
    """

    executor: ExecutorBase
    pre_executor: Optional[ExecutorBase] = None
    chunksize: int = 100000
    maxchunks: Optional[int] = None
    metadata_cache: Optional[MutableMapping] = None
    skipbadfiles: bool = False
    xrootdtimeout: Optional[int] = 60
    align_clusters: bool = False
    savemetrics: bool = False
    schema: Optional[schemas.BaseSchema] = schemas.NanoAODSchema
    processor_compression: int = 1
    use_skyhook: Optional[bool] = False
    skyhook_options: Optional[dict] = field(default_factory=dict)
    format: str = "root"

    @staticmethod
    def read_coffea_config():
        config_path = None
        if "HOME" in os.environ:
            config_path = os.path.join(os.environ["HOME"], ".coffea.toml")
        elif "_CONDOR_SCRATCH_DIR" in os.environ:
            config_path = os.path.join(
                os.environ["_CONDOR_SCRATCH_DIR"], ".coffea.toml"
            )

        if config_path is not None and os.path.exists(config_path):
            with open(config_path) as f:
                return toml.loads(f.read())
        else:
            return dict()

    def __post_init__(self):
        if self.pre_executor is None:
            self.pre_executor = self.executor

        assert isinstance(
            self.executor, ExecutorBase
        ), "Expected executor to derive from ExecutorBase"
        assert isinstance(
            self.pre_executor, ExecutorBase
        ), "Expected pre_executor to derive from ExecutorBase"

        if self.metadata_cache is None:
            self.metadata_cache = DEFAULT_METADATA_CACHE

        assert self.format in ("root", "parquet")

    @property
    def retries(self):
        if isinstance(self.executor, DaskExecutor):
            retries = 0
        else:
            retries = getattr(self.executor, "retries", 0)
        assert retries >= 0
        return retries

    @property
    def use_dataframes(self):
        if isinstance(self.executor, DaskExecutor):
            return self.executor.use_dataframes
        else:
            return False

    @staticmethod
    def automatic_retries(retries: int, skipbadfiles: bool, func, *args, **kwargs):
        """This should probably defined on Executor-level."""
        import warnings

        retry_count = 0
        while retry_count <= retries:
            try:
                return func(*args, **kwargs)
            # catch xrootd errors and optionally skip
            # or retry to read the file
            except Exception as e:
                chain = _exception_chain(e)
                if skipbadfiles and any(
                    isinstance(c, (OSError, UprootMissTreeError)) for c in chain
                ):
                    warnings.warn(str(e))
                    break
                if (
                    skipbadfiles
                    and (retries == retry_count)
                    and any(
                        e in str(c)
                        for c in chain
                        for e in [
                            "Invalid redirect URL",
                            "Operation expired",
                            "Socket timeout",
                        ]
                    )
                ):
                    warnings.warn(str(e))
                    break
                if (
                    not skipbadfiles
                    or any("Auth failed" in str(c) for c in chain)
                    or retries == retry_count
                ):
                    raise e
                warnings.warn("Attempt %d of %d." % (retry_count + 1, retries + 1))
            retry_count += 1

    @staticmethod
    def _normalize_fileset(
        fileset: dict,
        treename: str,
    ) -> Generator[FileMeta, None, None]:
        if isinstance(fileset, str):
            with open(fileset) as fin:
                fileset = json.load(fin)
        elif not isinstance(fileset, Mapping):
            raise ValueError("Expected fileset to be a path string or mapping")
        reserved_metakeys = _PROTECTED_NAMES
        for dataset, filelist in fileset.items():
            user_meta = None
            if isinstance(filelist, dict):
                user_meta = filelist["metadata"] if "metadata" in filelist else None
                if user_meta is not None:
                    for rkey in reserved_metakeys:
                        if rkey in user_meta.keys():
                            raise ValueError(
                                f'Reserved word "{rkey}" in metadata section of fileset dictionary, please rename this entry!'
                            )
                if "treename" not in filelist and treename is None:
                    if not isinstance(filelist["files"], dict):
                        raise ValueError(
                            "treename must be specified if the fileset does not contain tree names"
                        )
                local_treename = (
                    filelist["treename"] if "treename" in filelist else treename
                )
                filelist = filelist["files"]
            elif isinstance(filelist, list):
                if treename is None:
                    raise ValueError(
                        "treename must be specified if the fileset does not contain tree names"
                    )
                local_treename = treename
            else:
                raise ValueError(
                    "list of filenames in fileset must be a list or a dict"
                )
            if local_treename is None:
                for filename, local_treename in filelist.items():
                    yield FileMeta(dataset, filename, local_treename, user_meta)
            else:
                for filename in filelist:
                    yield FileMeta(dataset, filename, local_treename, user_meta)

    @staticmethod
    def metadata_fetcher_root(
        xrootdtimeout: int, align_clusters: bool, item: FileMeta
    ) -> Accumulatable:
        with uproot.open({item.filename: None}, timeout=xrootdtimeout) as file:
            try:
                tree = file[item.treename]
            except uproot.exceptions.KeyInFileError as e:
                raise UprootMissTreeError(str(e)) from e

            metadata = {}
            if item.metadata:
                metadata.update(item.metadata)
            metadata.update({"numentries": tree.num_entries, "uuid": file.file.fUUID})
            if align_clusters:
                metadata["clusters"] = tree.common_entry_offsets()
            out = set_accumulator(
                [FileMeta(item.dataset, item.filename, item.treename, metadata)]
            )
        return out

    @staticmethod
    def metadata_fetcher_parquet(item: FileMeta):
        with ParquetFileContext(item.filename) as file:
            metadata = {}
            if item.metadata:
                metadata.update(item.metadata)
            metadata.update(
                {"numentries": file.num_entries, "uuid": b"NO_UUID_0000_000"}
            )
            out = set_accumulator(
                [FileMeta(item.dataset, item.filename, item.treename, metadata)]
            )
        return out

    def _preprocess_fileset_root(self, fileset: dict) -> None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = {
            filemeta
            for filemeta in fileset
            if not filemeta.populated(clusters=self.align_clusters)
        }
        if len(to_get) > 0:
            out = set_accumulator()
            pre_arg_override = {
                "function_name": "get_metadata",
                "desc": "Preprocessing",
                "unit": "file",
                "compression": None,
            }
            if isinstance(self.pre_executor, (FuturesExecutor, ParslExecutor)):
                pre_arg_override.update({"tailtimeout": None})
            if isinstance(self.pre_executor, (DaskExecutor)):
                self.pre_executor.heavy_input = None
                pre_arg_override.update({"worker_affinity": False})
            pre_executor = self.pre_executor.copy(**pre_arg_override)
            closure = partial(
                self.automatic_retries,
                self.retries,
                self.skipbadfiles,
                partial(
                    self.metadata_fetcher_root, self.xrootdtimeout, self.align_clusters
                ),
            )
            out, _ = pre_executor(to_get, closure, out)
            while out:
                item = out.pop()
                self.metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

    def _preprocess_fileset_parquet(self, fileset: dict) -> None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = {
            filemeta
            for filemeta in fileset
            if not filemeta.populated(clusters=self.align_clusters)
        }
        if len(to_get) > 0:
            out = set_accumulator()
            pre_arg_override = {
                "function_name": "get_metadata",
                "desc": "Preprocessing",
                "unit": "file",
                "compression": None,
            }
            if isinstance(self.pre_executor, (FuturesExecutor, ParslExecutor)):
                pre_arg_override.update({"tailtimeout": None})
            if isinstance(self.pre_executor, (DaskExecutor)):
                self.pre_executor.heavy_input = None
                pre_arg_override.update({"worker_affinity": False})
            pre_executor = self.pre_executor.copy(**pre_arg_override)
            closure = partial(
                self.automatic_retries,
                self.retries,
                self.skipbadfiles,
                self.metadata_fetcher_parquet,
            )
            out, _ = pre_executor(to_get, closure, out)
            while out:
                item = out.pop()
                self.metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

    def _filter_badfiles(self, fileset: dict) -> list:
        final_fileset = []
        for filemeta in fileset:
            if filemeta.populated(clusters=self.align_clusters):
                final_fileset.append(filemeta)
            elif not self.skipbadfiles:
                raise RuntimeError(
                    f"Metadata for file {filemeta.filename} could not be accessed."
                )
        return final_fileset

    def _chunk_generator(self, fileset: dict, treename: str) -> Generator:
        config = None
        if self.use_skyhook:
            config = Runner.read_coffea_config()
        if not self.use_skyhook and (self.format == "root" or self.format == "parquet"):
            if self.maxchunks is None:
                last_chunksize = self.chunksize
                for filemeta in fileset:
                    last_chunksize = yield from filemeta.chunks(
                        last_chunksize,
                        self.align_clusters,
                    )
            else:
                # get just enough file info to compute chunking
                nchunks = defaultdict(int)
                chunks = []
                for filemeta in fileset:
                    if nchunks[filemeta.dataset] >= self.maxchunks:
                        continue
                    for chunk in filemeta.chunks(self.chunksize, self.align_clusters):
                        chunks.append(chunk)
                        nchunks[filemeta.dataset] += 1
                        if nchunks[filemeta.dataset] >= self.maxchunks:
                            break
                yield from (c for c in chunks)
        else:
            if self.use_skyhook and not config.get("skyhook", None):
                print("No skyhook config found, using defaults")
                config["skyhook"] = dict()

            dataset_filelist_map = {}
            if self.use_skyhook:
                import pyarrow.dataset as ds

                for dataset, basedir in fileset.items():
                    ds_ = ds.dataset(basedir, format="parquet")
                    dataset_filelist_map[dataset] = ds_.files
            else:
                for dataset, maybe_filelist in fileset.items():
                    if isinstance(maybe_filelist, list):
                        dataset_filelist_map[dataset] = maybe_filelist
                    elif isinstance(maybe_filelist, dict):
                        if "files" not in maybe_filelist:
                            raise ValueError(
                                "Dataset definition must have key 'files' defined!"
                            )
                        dataset_filelist_map[dataset] = maybe_filelist["files"]
                    else:
                        raise ValueError(
                            "Dataset definition in fileset must be dict[str: list[str]] or dict[str: dict[str: Any]]"
                        )
            chunks = []
            for dataset, filelist in dataset_filelist_map.items():
                for filename in filelist:
                    # If skyhook config is provided and is not empty,
                    if self.use_skyhook:
                        ceph_config_path = config["skyhook"].get(
                            "ceph_config_path", "/etc/ceph/ceph.conf"
                        )
                        ceph_data_pool = config["skyhook"].get(
                            "ceph_data_pool", "cephfs_data"
                        )
                        filename = f"{ceph_config_path}:{ceph_data_pool}:{filename}"
                    chunks.append(
                        WorkItem(
                            dataset,
                            filename,
                            treename,
                            0,
                            0,
                            "",
                            (
                                fileset[dataset]["metadata"]
                                if "metadata" in fileset[dataset]
                                else None
                            ),
                        )
                    )
            yield from iter(chunks)

    @staticmethod
    def _work_function(
        format: str,
        xrootdtimeout: int,
        schema: schemas.BaseSchema,
        use_dataframes: bool,
        savemetrics: bool,
        item: WorkItem,
        processor_instance: ProcessorABC,
        uproot_options: dict,
    ) -> dict:
        if "timeout" in uproot_options:
            xrootdtimeout = uproot_options["timeout"]
        if processor_instance == "heavy":
            item, processor_instance = item
        if not isinstance(processor_instance, ProcessorABC):
            processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

        if format == "root":
            filecontext = uproot.open(
                {item.filename: None},
                timeout=xrootdtimeout,
                **uproot_options,
            )
        elif format == "parquet":
            raise NotImplementedError("Parquet format is not supported yet.")

        metadata = {
            "dataset": item.dataset,
            "filename": item.filename,
            "treename": item.treename,
            "entrystart": item.entrystart,
            "entrystop": item.entrystop,
            "fileuuid": (
                str(uuid.UUID(bytes=item.fileuuid)) if len(item.fileuuid) > 0 else ""
            ),
        }
        if item.usermeta is not None:
            metadata.update(item.usermeta)

        with filecontext as file:
            if schema is None:
                raise ValueError("Schema must be set")
            elif issubclass(schema, schemas.BaseSchema):
                # change here
                if format == "root":
                    materialized = []
                    factory = NanoEventsFactory.from_root(
                        file=file,
                        treepath=item.treename,
                        schemaclass=schema,
                        metadata=metadata,
                        access_log=materialized,
                        mode="virtual",
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                    )
                    events = factory.events()
                elif format == "parquet":
                    raise NotImplementedError("Parquet format is not supported yet.")
            else:
                raise ValueError(
                    "Expected schema to derive from nanoevents.BaseSchema, instead got %r"
                    % schema
                )
            tic = time.time()
            try:
                out = processor_instance.process(events)
            except Exception as e:
                raise Exception(
                    f"Failed processing file: {item!r}. The error was: {e!r}."
                ) from e
            if out is None:
                raise ValueError(
                    "Output of process() should not be None. Make sure your processor's process() function returns an accumulator."
                )
            toc = time.time()
            if use_dataframes:
                return out
            else:
                if savemetrics:
                    metrics = {}
                    if isinstance(file, uproot.ReadOnlyDirectory):
                        metrics["bytesread"] = file.file.source.num_requested_bytes
                    if schema is not None and issubclass(schema, schemas.BaseSchema):
                        metrics["columns"] = set(materialized)
                        metrics["entries"] = len(events)
                    metrics["processtime"] = toc - tic
                    return {"out": out, "metrics": metrics, "processed": {item}}
                return {"out": out, "processed": {item}}

    def __call__(
        self,
        fileset: dict,
        processor_instance: ProcessorABC,
        treename: Optional[str] = None,
        uproot_options: Optional[dict] = {},
    ) -> Accumulatable:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict
                A dictionary ``{dataset: [file, file], }``
                Optionally, if some files' tree name differ, the dictionary can be specified:
                ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
            processor_instance : ProcessorABC
                An instance of a class deriving from ProcessorABC
            treename : str
                name of tree inside each root file, can be ``None``;
                treename can also be defined in fileset, which will override the passed treename
            uproot_options : dict, optional
                Any options to pass to ``uproot.open``
        """
        wrapped_out = self.run(fileset, processor_instance, treename, uproot_options)
        if self.use_dataframes:
            return wrapped_out  # not wrapped anymore
        if self.savemetrics:
            return wrapped_out["out"], wrapped_out["metrics"]
        return wrapped_out["out"]

    def preprocess(
        self,
        fileset: dict,
        treename: Optional[str] = None,
    ) -> Generator:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict
                A dictionary ``{dataset: [file, file], }``
                Optionally, if some files' tree name differ, the dictionary can be specified:
                ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
                You can also define a different tree name per file in the dictionary:
                ``{dataset: {'files': {file: 'name'}}, }``
            treename : str
                name of tree inside each root file, can be ``None``;
                treename can also be defined in fileset, which will override the passed treename
        """

        if not isinstance(fileset, (Mapping, str)):
            raise ValueError(
                "Expected fileset to be a mapping dataset: list(files) or filename"
            )
        if self.format == "root":
            fileset = list(self._normalize_fileset(fileset, treename))
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

            self._preprocess_fileset_root(fileset)
            fileset = self._filter_badfiles(fileset)

            # reverse fileset list to match the order of files as presented in version
            # v0.7.4. This fixes tests using maxchunks.
            fileset.reverse()
        elif self.format == "parquet":
            raise NotImplementedError("Parquet format is not supported yet.")

        return self._chunk_generator(fileset, treename)

    def run(
        self,
        fileset: Union[dict, str, list[WorkItem], Generator],
        processor_instance: ProcessorABC,
        treename: Optional[str] = None,
        uproot_options: Optional[dict] = {},
    ) -> Accumulatable:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict | str | List[WorkItem] | Generator
                - A dictionary ``{dataset: [file, file], }``
                  Optionally, if some files' tree name differ, the dictionary can be specified:
                  ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
                  You can also define a different tree name per file in the dictionary:
                ``{dataset: {'files': {file: 'name'}}, }``
                - A single file name
                - File chunks for self.preprocess()
                - Chunk generator
            processor_instance : ProcessorABC
                An instance of a class deriving from ProcessorABC
            treename : str, optional
                name of tree inside each root file, can be ``None``;
                treename can also be defined in fileset, which will override the passed treename
                Not needed if processing premade chunks
            uproot_options : dict, optional
                Any options to pass to ``uproot.open``
        """

        meta = False
        if not isinstance(fileset, (Mapping, str)):
            if isinstance(fileset, Generator) or isinstance(fileset[0], WorkItem):
                meta = True
            else:
                raise ValueError(
                    "Expected fileset to be a mapping dataset: list(files) or filename"
                )
        if not isinstance(processor_instance, ProcessorABC):
            raise ValueError("Expected processor_instance to derive from ProcessorABC")

        if meta:
            chunks = fileset
        else:
            chunks = self.preprocess(fileset, treename)

        if self.processor_compression is None:
            pi_to_send = processor_instance
        else:
            pi_to_send = lz4f.compress(
                cloudpickle.dumps(processor_instance),
                compression_level=self.processor_compression,
            )
        # hack around dask/dask#5503 which is really a silly request but here we are
        if isinstance(self.executor, DaskExecutor):
            self.executor.heavy_input = pi_to_send
            closure = partial(
                self._work_function,
                self.format,
                self.xrootdtimeout,
                self.schema,
                self.use_dataframes,
                self.savemetrics,
                processor_instance="heavy",
                uproot_options=uproot_options,
            )
        else:
            closure = partial(
                self._work_function,
                self.format,
                self.xrootdtimeout,
                self.schema,
                self.use_dataframes,
                self.savemetrics,
                processor_instance=pi_to_send,
                uproot_options=uproot_options,
            )

        chunks = list(chunks)

        exe_args = {
            "unit": "chunk",
            "function_name": type(processor_instance).__name__,
        }

        closure = partial(
            self.automatic_retries, self.retries, self.skipbadfiles, closure
        )

        executor = self.executor.copy(**exe_args)
        wrapped_out, e = executor(chunks, closure, None)
        if wrapped_out is None:
            raise ValueError(
                "No chunks returned results, verify ``processor`` instance structure.\n\
                if you used skipbadfiles=True, it is possible all your files are bad."
            )
        wrapped_out["exception"] = e

        if not self.use_dataframes:
            processor_instance.postprocess(wrapped_out["out"])

        if "metrics" in wrapped_out.keys():
            wrapped_out["metrics"]["chunks"] = len(chunks)
            for k, v in wrapped_out["metrics"].items():
                if isinstance(v, set):
                    wrapped_out["metrics"][k] = list(v)
        if self.use_dataframes:
            return wrapped_out["out"]
        else:
            return wrapped_out
