from __future__ import print_function, division
import concurrent.futures
from functools import partial
from itertools import repeat
import os
import time
import pickle
import sys
import math
import json
import cloudpickle
import toml
import uproot
import uuid
import warnings
import traceback
import shutil
from collections import defaultdict
from cachetools import LRUCache
from io import BytesIO
import lz4.frame as lz4f
from contextlib import ExitStack
from .processor import ProcessorABC
from .accumulator import accumulate, set_accumulator, Accumulatable
from .dataframe import LazyDataFrame
from ..nanoevents import NanoEventsFactory, schemas
from ..util import _hash, _exception_chain, rich_bar

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field, asdict
from typing import (
    Iterable,
    Callable,
    Optional,
    List,
    Set,
    Generator,
    Dict,
    Union,
    Tuple,
    Awaitable,
)


try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


try:
    from functools import cached_property
except ImportError:
    cached_property = property


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


class FileMeta(object):
    __slots__ = ["dataset", "filename", "treename", "metadata"]

    def __init__(self, dataset, filename, treename, metadata=None):
        self.dataset = dataset
        self.filename = filename
        self.treename = treename
        self.metadata = metadata

    def __str__(self):
        return "FileMeta(%s:%s)" % (self.filename, self.treename)

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
    usermeta: Optional[Dict] = field(default=None, compare=False)

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


class _compression_wrapper(object):
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
    def __init__(self, futures: Set[Awaitable], refresh=2):
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

    def fetch(self, N: int) -> List[Accumulatable]:
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
class WorkQueueExecutor(ExecutorBase):
    """Execute using Work Queue

    For more information, see :ref:`intro-coffea-wq`

    Parameters
    ----------
        items : sequence or generator
            Sequence of input arguments
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
            Compress accumulator outputs in flight with LZ4, at level specified (default 9)
            `None`` sets level to 1 (minimal compression)
        # work queue specific options:
        cores : int
            Maximum number of cores for work queue task. If unset, use a whole worker.
        memory : int
            Maximum amount of memory (in MB) for work queue task. If unset, use a whole worker.
        disk : int
            Maximum amount of disk space (in MB) for work queue task. If unset, use a whole worker.
        gpus : int
            Number of GPUs to allocate to each task.  If unset, use zero.
        resource_monitor : str
            If given, one of 'off', 'measure', or 'watchdog'. Default is 'off'.
            - 'off': turns off resource monitoring. Overriden to 'watchdog' if resources_mode
                     is not set to 'fixed'.
            - 'measure': turns on resource monitoring for Work Queue. The
                        resources used per task are measured.
            - 'watchdog': in addition to measuring resources, tasks are terminated if they
                        go above the cores, memory, or disk specified.
        resources_mode : str
            one of 'fixed', 'max-seen', or 'max-throughput'. Default is 'max-seen'.
            Sets the strategy to automatically allocate resources to tasks.
            - 'fixed': allocate cores, memory, and disk specified for each task.
            - 'max-seen' or 'auto': use the cores, memory, and disk given as maximum values to allocate,
                          but first try each task by allocating the maximum values seen. Leads
                          to a good compromise between parallelism and number of retries.
            - 'max-throughput': Like max-seen, but first tries the task with an
                          allocation that maximizes overall throughput.
            If resources_mode is other than 'fixed', preprocessing and
            accumulation tasks always use the 'max-seen' strategy, as the
            former tasks always use the same resources, the latter has a
            distribution of resources that increases over time.
        split_on_exhaustion: bool
            Whether to split a processing task in half according to its chunksize when it exhausts its
            the cores, memory, or disk allocated to it. If False, a task that exhausts resources
            permanently fails. Default is True.
        fast_terminate_workers: int
            Terminate workers on which tasks have been running longer than average.
            The time limit is computed by multiplying the average runtime of tasks
            by the value of 'fast_terminate_workers'. Since there are
            legitimately slow tasks, no task may trigger fast termination in
            two distinct workers. Less than 1 disables it.

        manager_name : str
            Name to refer to this work queue manager.
            Sets port to 0 (any available port) if port not given.
        port : int or tuple(int, int)
            Port number or range (inclusive of ports )for work queue manager program.
            Defaults to 9123 if manager_name not given.
        password_file: str
            Location of a file containing a password used to authenticate workers.
        ssl: bool or tuple(str, str)
            Enable ssl encryption between manager and workers. If a tuple, then it
            should be of the form (key, cert), where key and cert are paths to the files
            containing the key and certificate in pem format. If True, auto-signed temporary
            key and cert are generated for the session.

        extra_input_files: list
            A list of files in the current working directory to send along with each task.
            Useful for small custom libraries and configuration files needed by the processor.
        x509_proxy : str
            Path to the X509 user proxy. If None (the default), use the value of the
            environment variable X509_USER_PROXY, or fallback to the file /tmp/x509up_u${UID} if
            exists.  If False, disables the default behavior and no proxy is sent.

        environment_file : optional, str
            Conda python environment tarball to use. If not given, assume that
            the python environment is already setup at the execution site.
        wrapper : str
            Wrapper script to run/open python environment tarball. Defaults to python_package_run found in PATH.

        treereduction : int
            Number of processed chunks per accumulation task. Defaults is 20.

        verbose : bool
            If true, emit a message on each task submission and completion.
            Default is false.
        print_stdout : bool
            If true (default), print the standard output of work queue task on completion.

        debug_log : str
            Filename for debug output
        stats_log : str
            Filename for tasks statistics output
        transactions_log : str
            Filename for tasks lifetime reports output
        tasks_accum_log : str
            Filename for the log of tasks that have been processed and accumulated.

        filepath: str
            Path to the parent directory where to create the staging directory.
            Default is "." (current working directory).

        custom_init : function, optional
            A function that takes as an argument the queue's WorkQueue object.
            The function is called just before the first work unit is submitted
            to the queue.
    """

    # Standard executor options:
    compression: Optional[int] = 9  # as recommended by lz4
    retries: int = 2  # task executes at most 3 times
    # wq executor options:
    manager_name: Optional[str] = None
    port: Optional[Union[int, Tuple[int, int]]] = None
    filepath: str = "."
    events_total: Optional[int] = None
    x509_proxy: Optional[str] = None
    verbose: bool = False
    print_stdout: bool = False
    status_display_interval: Optional[int] = 10
    debug_log: Optional[str] = None
    stats_log: Optional[str] = None
    transactions_log: Optional[str] = None
    tasks_accum_log: Optional[str] = None
    password_file: Optional[str] = None
    ssl: Union[bool, Tuple[str, str]] = False
    environment_file: Optional[str] = None
    extra_input_files: List = field(default_factory=list)
    wrapper: Optional[str] = shutil.which("poncho_package_run")
    resource_monitor: Optional[str] = "off"
    resources_mode: Optional[str] = "max-seen"
    split_on_exhaustion: Optional[bool] = True
    fast_terminate_workers: Optional[int] = None
    cores: Optional[int] = None
    memory: Optional[int] = None
    disk: Optional[int] = None
    gpus: Optional[int] = None
    treereduction: int = 20
    chunksize: int = 100000
    dynamic_chunksize: Optional[Dict] = None
    custom_init: Optional[Callable] = None

    # deprecated
    bar_format: Optional[str] = None
    chunks_accum_in_mem: Optional[int] = None
    master_name: Optional[str] = None
    chunks_per_accum: Optional[int] = None

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        from .work_queue_tools import run

        return (
            run(
                self,
                items,
                function,
                accumulator,
            ),
            0,
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
            You can pass an instance instead of a class to re-use an executor
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
            Supply an additional executor to process merge jobs indepedently.
            An ``int`` will be interpretted as ``ProcessPoolExecutor(max_workers=int)``.
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
    merging: Union[bool, Tuple[int, int, int]] = False
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
                set(pool.submit(function, item) for item in items), refresh=2
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
                    print("Exception occured, recovering progress...")
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
                            work.result()
                            if self.compression is None
                            else _decompress(work.result())
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
    merging: Optional[Union[bool, Tuple[int, int, int]]] = False
    jobs_executors: Union[str, List] = "all"
    merges_executors: Union[str, List] = "all"

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
                print("Exception occured, recovering progress...")
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
            self.branchnames = set(
                item.path.split(".")[0] for item in self.filehandle.schema
            )

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
    reading, via NanoEvents or LazyDataFrame.  For more customized processing,
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
        dynamic_chunksize : dict, optional
            Whether to adapt the chunksize for units of work to run in the targets given.
            Currently supported are 'wall_time' (in seconds), and 'memory' (in MB).
            E.g., with {"wall_time": 120, "memory": 2048}, the chunksize will
            be dynamically adapted so that processing jobs each run in about
            two minutes, using two GB of memory. (Currently only for the WorkQueueExecutor.)
    """

    executor: ExecutorBase
    pre_executor: Optional[ExecutorBase] = None
    chunksize: int = 100000
    maxchunks: Optional[int] = None
    metadata_cache: Optional[MutableMapping] = None
    dynamic_chunksize: Optional[Dict] = None
    skipbadfiles: bool = False
    xrootdtimeout: Optional[int] = 60
    align_clusters: bool = False
    savemetrics: bool = False
    mmap: bool = False
    schema: Optional[schemas.BaseSchema] = schemas.BaseSchema
    cachestrategy: Optional[
        Union[Literal["dask-worker"], Callable[..., MutableMapping]]
    ] = None  # fmt: skip
    processor_compression: int = 1
    use_skyhook: Optional[bool] = False
    skyhook_options: Optional[Dict] = field(default_factory=dict)
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

        if self.align_clusters and self.dynamic_chunksize:
            raise RuntimeError(
                "align_clusters and dynamic_chunksize cannot be used simultaneously"
            )
        if self.maxchunks and self.dynamic_chunksize:
            raise RuntimeError(
                "maxchunks and dynamic_chunksize cannot be used simultaneously"
            )
        if self.dynamic_chunksize and not isinstance(self.executor, WorkQueueExecutor):
            raise RuntimeError(
                "dynamic_chunksize currently only supported by the WorkQueueExecutor"
            )

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
    def get_cache(cachestrategy):
        cache = None
        if cachestrategy == "dask-worker":
            from distributed import get_worker
            from coffea.processor.dask import ColumnCache

            worker = get_worker()
            try:
                cache = worker.plugins[ColumnCache.name]
            except KeyError:
                # emit warning if not found?
                pass
        elif callable(cachestrategy):
            cache = cachestrategy()
        return cache

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
        fileset: Dict,
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
            for filename in filelist:
                yield FileMeta(dataset, filename, local_treename, user_meta)

    @staticmethod
    def metadata_fetcher(
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

    def _preprocess_fileset(self, fileset: Dict) -> None:
        # this is a bit of an abuse of map-reduce but ok
        to_get = set(
            filemeta
            for filemeta in fileset
            if not filemeta.populated(clusters=self.align_clusters)
        )
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
                partial(self.metadata_fetcher, self.xrootdtimeout, self.align_clusters),
            )
            out, _ = pre_executor(to_get, closure, out)
            while out:
                item = out.pop()
                self.metadata_cache[item] = item.metadata
            for filemeta in fileset:
                filemeta.maybe_populate(self.metadata_cache)

    def _filter_badfiles(self, fileset: Dict) -> List:
        final_fileset = []
        for filemeta in fileset:
            if filemeta.populated(clusters=self.align_clusters):
                final_fileset.append(filemeta)
            elif not self.skipbadfiles:
                raise RuntimeError(
                    f"Metadata for file {filemeta.filename} could not be accessed."
                )
        return final_fileset

    def _chunk_generator(self, fileset: Dict, treename: str) -> Generator:
        config = None
        if self.use_skyhook:
            config = Runner.read_coffea_config()
        if self.format == "root":
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
                            fileset[dataset]["metadata"]
                            if "metadata" in fileset[dataset]
                            else None,
                        )
                    )
            yield from iter(chunks)

    @staticmethod
    def _work_function(
        format: str,
        xrootdtimeout: int,
        mmap: bool,
        schema: schemas.BaseSchema,
        cache_function: Callable[[], MutableMapping],
        use_dataframes: bool,
        savemetrics: bool,
        item: WorkItem,
        processor_instance: ProcessorABC,
    ) -> Dict:
        if processor_instance == "heavy":
            item, processor_instance = item
        if not isinstance(processor_instance, ProcessorABC):
            processor_instance = cloudpickle.loads(lz4f.decompress(processor_instance))

        if format == "root":
            filecontext = uproot.open(
                {item.filename: None},
                timeout=xrootdtimeout,
                file_handler=uproot.MemmapSource
                if mmap
                else uproot.MultithreadedFileSource,
            )
        elif format == "parquet":
            filecontext = ParquetFileContext(item.filename)

        metadata = {
            "dataset": item.dataset,
            "filename": item.filename,
            "treename": item.treename,
            "entrystart": item.entrystart,
            "entrystop": item.entrystop,
            "fileuuid": str(uuid.UUID(bytes=item.fileuuid))
            if len(item.fileuuid) > 0
            else "",
        }
        if item.usermeta is not None:
            metadata.update(item.usermeta)

        with filecontext as file:
            if schema is None:
                # To deprecate
                tree = None
                if format == "root":
                    tree = file[item.treename]
                elif format == "parquet":
                    tree = file
                else:
                    raise ValueError("Format can only be root or parquet!")
                events = LazyDataFrame(
                    tree, item.entrystart, item.entrystop, metadata=metadata
                )
            elif issubclass(schema, schemas.BaseSchema):
                # change here
                if format == "root":
                    materialized = []
                    factory = NanoEventsFactory.from_root(
                        file=file,
                        treepath=item.treename,
                        entry_start=item.entrystart,
                        entry_stop=item.entrystop,
                        persistent_cache=cache_function(),
                        schemaclass=schema,
                        metadata=metadata,
                        access_log=materialized,
                    )
                    events = factory.events()
                elif format == "parquet":
                    skyhook_options = {}
                    if ":" in item.filename:
                        (
                            ceph_config_path,
                            ceph_data_pool,
                            filename,
                        ) = item.filename.split(":")
                        # patch back filename into item
                        item = WorkItem(**dict(asdict(item), filename=filename))
                        skyhook_options["ceph_config_path"] = ceph_config_path
                        skyhook_options["ceph_data_pool"] = ceph_data_pool

                    factory = NanoEventsFactory.from_parquet(
                        file=item.filename,
                        treepath=item.treename,
                        schemaclass=schema,
                        metadata=metadata,
                        skyhook_options=skyhook_options,
                    )
                    events = factory.events()
            else:
                raise ValueError(
                    "Expected schema to derive from nanoevents.BaseSchema, instead got %r"
                    % schema
                )
            tic = time.time()
            try:
                out = processor_instance.process(events)
            except Exception as e:
                raise Exception(f"Failed processing file: {item!r}") from e
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
                    else:
                        metrics["columns"] = set(events.materialized)
                        metrics["entries"] = events.size
                    metrics["processtime"] = toc - tic
                    return {"out": out, "metrics": metrics, "processed": set([item])}
                return {"out": out, "processed": set([item])}

    def __call__(
        self,
        fileset: Dict,
        treename: str,
        processor_instance: ProcessorABC,
    ) -> Accumulatable:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict
                A dictionary ``{dataset: [file, file], }``
                Optionally, if some files' tree name differ, the dictionary can be specified:
                ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
            treename : str
                name of tree inside each root file, can be ``None``;
                treename can also be defined in fileset, which will override the passed treename
            processor_instance : ProcessorABC
                An instance of a class deriving from ProcessorABC
        """

        wrapped_out = self.run(fileset, processor_instance, treename)
        if self.use_dataframes:
            return wrapped_out  # not wrapped anymore
        if self.savemetrics:
            return wrapped_out["out"], wrapped_out["metrics"]
        return wrapped_out["out"]

    def preprocess(
        self,
        fileset: Dict,
        treename: str,
    ) -> Generator:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict
                A dictionary ``{dataset: [file, file], }``
                Optionally, if some files' tree name differ, the dictionary can be specified:
                ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
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

            self._preprocess_fileset(fileset)
            fileset = self._filter_badfiles(fileset)

            # reverse fileset list to match the order of files as presented in version
            # v0.7.4. This fixes tests using maxchunks.
            fileset.reverse()

        return self._chunk_generator(fileset, treename)

    def run(
        self,
        fileset: Union[Dict, str, List[WorkItem], Generator],
        processor_instance: ProcessorABC,
        treename: str = None,
    ) -> Accumulatable:
        """Run the processor_instance on a given fileset

        Parameters
        ----------
            fileset : dict | str | List[WorkItem] | Generator
                - A dictionary ``{dataset: [file, file], }``
                  Optionally, if some files' tree name differ, the dictionary can be specified:
                  ``{dataset: {'treename': 'name', 'files': [file, file]}, }``
                - A single file name
                - File chunks for self.preprocess()
                - Chunk generator
            treename : str, optional
                name of tree inside each root file, can be ``None``;
                treename can also be defined in fileset, which will override the passed treename
                Not needed if processing premade chunks
            processor_instance : ProcessorABC
                An instance of a class deriving from ProcessorABC
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
                self.mmap,
                self.schema,
                partial(self.get_cache, self.cachestrategy),
                self.use_dataframes,
                self.savemetrics,
                processor_instance="heavy",
            )
        else:
            closure = partial(
                self._work_function,
                self.format,
                self.xrootdtimeout,
                self.mmap,
                self.schema,
                partial(self.get_cache, self.cachestrategy),
                self.use_dataframes,
                self.savemetrics,
                processor_instance=pi_to_send,
            )

        if self.format == "root" and isinstance(self.executor, WorkQueueExecutor):
            # keep chunks in generator, use a copy to count number of events
            # this is cheap, as we are reading from the cache
            chunks_to_count = self.preprocess(fileset, treename)
        else:
            # materialize chunks to list, then count that list
            chunks = list(chunks)
            chunks_to_count = chunks

        events_total = sum(len(c) for c in chunks_to_count)

        exe_args = {
            "unit": "chunk",
            "function_name": type(processor_instance).__name__,
        }
        if isinstance(self.executor, WorkQueueExecutor):
            exe_args.update(
                {
                    "unit": "event",
                    "events_total": events_total,
                    "dynamic_chunksize": self.dynamic_chunksize,
                    "chunksize": self.chunksize,
                }
            )

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


def run_spark_job(
    fileset,
    processor_instance,
    executor,
    executor_args={},
    spark=None,
    partitionsize=200000,
    thread_workers=16,
):
    """A wrapper to submit spark jobs

    A convenience wrapper to submit jobs for spark datasets, which is a
    dictionary of dataset: [file list] entries.  Presently supports reading of
    parquet files converted from root.  For more customized processing,
    e.g. to read other objects from the files and pass them into data frames,
    one can write a similar function in their user code.

    Parameters
    ----------
        fileset : dict
            dictionary {dataset: [file, file], }
        processor_instance : ProcessorABC
            An instance of a class deriving from ProcessorABC

            .. note:: The processor instance must define all the columns in data and MC that it reads as ``.columns``
        executor:
            anything that inherits from `SparkExecutor` like `spark_executor`

            In general, a function that takes 3 arguments: items, function accumulator
            and performs some action equivalent to:
            for item in items: accumulator += function(item)
        executor_args:
            arguments to send to the creation of a spark session
        spark:
            an optional already created spark instance

            if ``None`` then we create an ephemeral spark instance using a config
        partitionsize:
            partition size to try to aim for (coalescese only, repartition too expensive)
        thread_workers:
            how many spark jobs to let fly in parallel during processing steps
    """

    try:
        import pyspark
    except ImportError as e:
        print(
            "you must have pyspark installed to call run_spark_job()!", file=sys.stderr
        )
        raise e

    from packaging import version
    import pyarrow as pa
    import warnings

    arrow_env = ("ARROW_PRE_0_15_IPC_FORMAT", "1")
    if version.parse(pa.__version__) >= version.parse("0.15.0") and version.parse(
        pyspark.__version__
    ) < version.parse("3.0.0"):
        import os

        if arrow_env[0] not in os.environ or os.environ[arrow_env[0]] != arrow_env[1]:
            warnings.warn(
                "If you are using pyarrow >= 0.15.0, make sure to set %s=%s in your environment!"
                % arrow_env
            )

    import pyspark.sql
    from .spark.spark_executor import SparkExecutor
    from .spark.detail import _spark_initialize, _spark_stop, _spark_make_dfs

    if not isinstance(fileset, Mapping):
        raise ValueError("Expected fileset to be a mapping dataset: list(files)")
    if not isinstance(processor_instance, ProcessorABC):
        raise ValueError("Expected processor_instance to derive from ProcessorABC")
    if not isinstance(executor, SparkExecutor):
        raise ValueError("Expected executor to derive from SparkExecutor")

    executor_args.setdefault("config", None)
    executor_args.setdefault("file_type", "parquet")
    executor_args.setdefault("laurelin_version", "1.1.1")
    executor_args.setdefault("treeName", "Events")
    executor_args.setdefault("schema", None)
    executor_args.setdefault("cache", True)
    executor_args.setdefault("skipbadfiles", False)
    executor_args.setdefault("retries", 0)
    executor_args.setdefault("xrootdtimeout", None)
    file_type = executor_args["file_type"]
    treeName = executor_args["treeName"]
    schema = executor_args["schema"]
    if "flatten" in executor_args:
        raise ValueError(
            "Executor argument 'flatten' is deprecated, please refactor your processor to accept awkward arrays"
        )
    if "nano" in executor_args:
        raise ValueError(
            "Awkward0 NanoEvents no longer supported.\n"
            "Please use 'schema': processor.NanoAODSchema to enable awkward NanoEvents processing."
        )
    use_cache = executor_args["cache"]

    if executor_args["config"] is None:
        executor_args.pop("config")

    # initialize spark if we need to
    # if we initialize, then we deconstruct
    # when we're done
    killSpark = False
    if spark is None:
        spark = _spark_initialize(**executor_args)
        killSpark = True
        use_cache = False  # if we always kill spark then we cannot use the cache
    else:
        if not isinstance(spark, pyspark.sql.session.SparkSession):
            raise ValueError(
                "Expected 'spark' to be a pyspark.sql.session.SparkSession"
            )

    dfslist = {}
    if executor._cacheddfs is None:
        dfslist = _spark_make_dfs(
            spark,
            fileset,
            partitionsize,
            processor_instance.columns,
            thread_workers,
            file_type,
            treeName,
        )

    output = executor(
        spark, dfslist, processor_instance, None, thread_workers, use_cache, schema
    )
    processor_instance.postprocess(output)

    if killSpark:
        _spark_stop(spark)
        del spark
        spark = None

    return output
