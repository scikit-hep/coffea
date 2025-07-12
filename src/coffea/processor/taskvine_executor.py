import collections
import math
import os
import re
import signal
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from os.path import join
from typing import Callable, Optional

from coffea.util import rich_bar

from .accumulator import (
    Accumulatable,
    accumulate,
)
from .executor import (
    ExecutorBase,
    WorkItem,
    _compression_wrapper,
    _decompress,
)

# The TaskVine object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
manager = None

# If set to True, workflow stops processing and outputs only the results that
# have been already processed.
early_terminate = False


def accumulate_result_files(files_to_accumulate, accumulator=None, concurrent_reads=2):
    def read_file(f):
        with open(f, "rb") as rf:
            return _decompress(rf.read())

    from coffea.processor import accumulate

    with ThreadPool(max(concurrent_reads, 1)) as pool:
        for result in pool.imap_unordered(read_file, list(files_to_accumulate), 1):
            if not accumulator:
                accumulator = result
            else:
                accumulator = accumulate([result], accumulator)

    return accumulator


try:
    import ndcctools.taskvine as vine
    from ndcctools.taskvine import Manager, PythonTask, PythonTaskNoResult
except ImportError:
    vine = None
    print("ndcctools.taskvine module not available")

    class PythonTask:
        def __init__(self, *args, **kwargs):
            raise ImportError("ndcctools.taskvine not available")

    class Manager:
        def __init__(self, *args, **kwargs):
            raise ImportError("ndcctools.taskvine not available")


@dataclass
class TaskVineExecutor(ExecutorBase):
    """Execute using TaskVine distributed computing framework

    Parameters
    ----------
        port : int, optional
            Port to listen on for workers (default: 9123)
        manager_name : str, optional
            Name for the manager (default: None)
        status_display_interval : int, optional
            Interval for status display updates (default: 5)
        ssl : bool, optional
            Enable SSL for connections (default: False)
        filepath : str, optional
            Path for temporary files (default: "/tmp")
        extra_input_files : list, optional
            List of extra input files to stage (default: [])
        x509_proxy : str, optional
            Path to X509 proxy file (default: None)
        environment_file : str, optional
            Path to environment file for workers (default: None)
        status : bool, optional
            Enable status display (default: True)
        verbose : bool, optional
            Enable verbose output (default: False)
        print_stdout : bool, optional
            Print stdout from tasks (default: False)
        password_file : str, optional
            Path to password file (default: None)
        treereduction : int, optional
            Tree reduction factor for output accumulators (default: 20)
        cores : int, optional
            Number of cores per task (default: 1)
        memory : int, optional
            Memory requirement per task in MB (default: None)
        disk : int, optional
            Disk requirement per task in MB (default: None)
        gpus : int, optional
            Number of GPUs per task (default: None)
        replicas : int, optional
            Number of replicas for tasks (default: 1)
        disable_worker_transfers : bool, optional
            Disable worker-to-worker transfers (default: False)
        resource_monitor : str, optional
            Resource monitoring mode: 'off', 'measure', or 'watchdog' (default: 'off')
        resources_mode : str, optional
            Resource allocation mode: 'fixed', 'max', or 'max-throughput' (default: 'fixed')
        fast_terminate_workers : int, optional
            Fast termination threshold (default: None)
        retries : int, optional
            Number of retries for failed tasks (default: 3)
        split_on_exhaustion : bool, optional
            Split tasks on resource exhaustion (default: True)
        checkpoint_proportion : float, optional
            Proportion of tasks to checkpoint (default: 0.1)
        concurrent_reads : int, optional
            Number of concurrent file reads (default: 2)
        custom_init : callable, optional
            Custom initialization function (default: None)
        compression : int, optional
            Compression level for data transfer (default: 1)
        unit : str, optional
            Unit for progress display (default: "events")
        desc : str, optional
            Description for progress display (default: "Processing")
        function_name : str, optional
            Name of the function being executed (default: None)
    """

    port: int = 9123
    manager_name: Optional[str] = None
    status_display_interval: int = 5
    ssl: bool = False
    filepath: str = "/tmp"
    extra_input_files: Optional[list] = None
    x509_proxy: Optional[str] = None
    environment_file: Optional[str] = None
    verbose: bool = False
    print_stdout: bool = False
    password_file: Optional[str] = None
    treereduction: int = 20
    cores: int = 1
    memory: Optional[int] = None
    disk: Optional[int] = None
    gpus: Optional[int] = None
    replicas: int = 1
    disable_worker_transfers: bool = False
    resource_monitor: str = "off"
    resources_mode: str = "fixed"
    fast_terminate_workers: Optional[int] = None
    retries: int = 3
    split_on_exhaustion: bool = True
    checkpoint_proportion: float = 0.1
    concurrent_reads: int = 2
    custom_init: Optional[Callable] = None

    def __post_init__(self):
        if self.extra_input_files is None:
            self.extra_input_files = []
        if self.treereduction < 2:
            raise ValueError("TaskVineExecutor: treereduction should be at least 2.")
        if not self.port:
            self.port = 0 if self.manager_name else 9123
        # taskvine always needs serialization to files, thus compression is always on
        if self.compression is None:
            self.compression = 1
        # activate monitoring if it has not been explicitly activated and we are
        # using an automatic resource allocation.
        if self.resources_mode != "fixed" and self.resource_monitor == "off":
            self.resource_monitor = "watchdog"
        self.verbose = self.verbose or self.print_stdout
        self.x509_proxy = _get_x509_proxy(self.x509_proxy)

    def __call__(
        self,
        items: Iterable,
        function: Callable,
        accumulator: Accumulatable,
    ):
        """Execute the function on items using TaskVine

        Parameters
        ----------
            items : Iterable
                Items to process
            function : Callable
                Function to apply to each item
            accumulator : Accumulatable
                Initial accumulator for results
        """
        if not vine:
            print("You must have TaskVine installed to use TaskVineExecutor!")
            import ndcctools.taskvine  # noqa

        global manager
        if manager is None:
            manager = CoffeaVine(self)
        else:
            # if manager already listening on port, update the parameters given by
            # the executor
            manager.executor = self

        try:
            if self.custom_init:
                self.custom_init(manager)

            if self.desc == "Preprocessing":
                result = manager._preprocessing(items, function, accumulator)
                # we do not shutdown manager after preprocessing, as we want to
                # keep the connected workers for processing/accumulation
            else:
                result = manager._processing(items, function, accumulator)
                manager = None
        except Exception as e:
            manager = None
            raise e

        return result, 0


class CoffeaVine(Manager):
    def __init__(
        self,
        executor,
    ):
        self.executor = executor

        self.stats_coffea = Stats()

        # set to keep track of the final work items the workflow consists of.
        # When a work item needs to be split, it is replaced from this set by
        # its constituents.
        self.known_workitems = set()

        # list that keeps results as they finish to construct accumulation
        # tasks.
        self.tasks_to_accumulate = []

        super().__init__(
            port=self.executor.port,
            name=self.executor.manager_name,
            staging_path=executor.filepath,
            status_display_interval=self.executor.status_display_interval,
            ssl=self.executor.ssl,
        )

        self.extra_input_files = {
            name: self.declare_file(name, cache=True)
            for name in executor.extra_input_files
        }

        if self.executor.x509_proxy:
            self.extra_input_files["x509.pem"] = self.declare_file(
                executor.x509_proxy, cache=True
            )

        self.poncho_file = False
        if self.executor.environment_file:
            self.poncho_file = self.declare_poncho(
                executor.environment_file, cache=True
            )

        self.bar = StatusBar(enabled=executor.status)
        self.console = VerbosePrint(self.bar.console, executor.status, executor.verbose)

        self._declare_resources()

        # Make use of the stored password file, if enabled.
        if self.executor.password_file:
            self.set_password_file(self.executor.password_file)

        self.console.printf(f"Listening for TaskVine workers on port {self.port}.")
        # perform a wait to print any warnings before progress bars
        self.wait(0)

    def submit(self, task):
        taskid = super().submit(task)
        self.console(
            "submitted {category} task id {id} item {item}, with {size} {unit}(s)",
            category=task.category,
            id=taskid,
            item=task.itemid,
            size=len(task),
            unit=self.executor.unit,
        )
        return taskid

    def wait(self, timeout=None):
        task = super().wait(timeout)
        if task:
            task.report(self)
            # Evaluate and display details of the completed task
            if task.successful():
                self.stats_coffea.max(
                    "bytes_received", task.get_metric("bytes_received") / 1e6
                )
                self.stats_coffea.max("bytes_sent", task.get_metric("bytes_sent") / 1e6)
                # Remove input files as we go to avoid unbounded disk. We do not
                # remove outputs, as they are used by further accumulate tasks.
                task.cleanup_inputs(self)
            return task
        return None

    def application_info(self):
        return {
            "application_info": {
                "values": dict(self.stats_coffea),
                "units": {
                    "bytes_received": "MB",
                    "bytes_sent": "MB",
                },
            }
        }

    @property
    def executor(self):
        return self._executor

    @executor.setter
    def executor(self, new_value):
        self._executor = new_value

    def soft_terminate(self, task=None):
        if task:
            self.console.warn(f"item {task.itemid} failed permanently.")

        if not early_terminate:
            # trigger soft termination
            _handle_early_terminate(0, None, raise_on_repeat=False)

    def _preprocessing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        for item in items:
            task = PreProcTask(self, function, item)
            self.submit(task)

        self.bar.add_task("Preprocessing", total=len(items), unit=self.executor.unit)
        while not self.empty():
            task = self.wait(5)
            if task:
                if task.successful():
                    accumulator = accumulate([task.output], accumulator)
                    self.bar.advance("Preprocessing", 1)
                    task.cleanup_outputs(self)
                else:
                    task.resubmit(self)
                self.bar.refresh()

        self.bar.stop_task("Preprocessing")
        return accumulator

    def _submit_processing_tasks(self, proc_fn, items):
        while True:
            if early_terminate or self._items_empty:
                return
            sc = self.stats_coffea
            if sc["chunks_submitted"] >= sc["chunks_total"]:
                return
            if not self.hungry():
                return
            try:
                item = next(items)
                self._submit_processing_task(proc_fn, item)
            except StopIteration:
                self.console.warn("Ran out of items to process.")
                self._items_empty = True
                return

    def _submit_processing_task(self, proc_fn, item):
        self.known_workitems.add(item)
        t = ProcTask(self, proc_fn, item)
        self.submit(t)
        self.stats_coffea["chunks_submitted"] += 1

    def all_to_accumulate_local(self):
        return all([t.is_checkpoint() for t in self.tasks_to_accumulate])

    def _final_accumulation(self, accumulator):
        if len(self.tasks_to_accumulate) < 1:
            self.console.warn("No results available.")
            return accumulator

        self.console("Merging with local final accumulator...")
        accumulator = accumulate_result_files(
            [t.output_file.source() for t in self.tasks_to_accumulate], accumulator
        )

        for t in self.tasks_to_accumulate:
            t.cleanup_outputs(self)

        sc = self.stats_coffea
        if sc["chunks_processed"] != sc["chunks_total"]:
            self.console.warn(
                f"Number of chunks processed ({sc['chunks_processed']}) is different from total ({sc['chunks_total']})!"
            )

        return accumulator

    def _processing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        accumulate_fn = _compression_wrapper(
            self.executor.compression,
            accumulate_result_files,
            self.executor.concurrent_reads,
        )

        sc = self.stats_coffea

        # Keep track of total tasks in each state.
        sc["chunks_processed"] = 0
        sc["chunks_submitted"] = 0
        sc["chunks_total"] = len(items)
        sc["accumulations_submitted"] = 0

        # make sure items looks like a generator for a list.
        # easier than keeping track of current chunk.
        items = (item for item in items)

        self._make_process_bars()

        signal.signal(signal.SIGINT, _handle_early_terminate)

        self._process_events(function, accumulate_fn, items)

        # merge results with original accumulator given by the executor
        accumulator = self._final_accumulation(accumulator)

        self._update_bars(final_update=True)
        return accumulator

    def _process_events(self, proc_fn, accum_fn, items):
        self.known_workitems = set()
        sc = self.stats_coffea
        self._items_empty = False

        while True:
            if self.empty():
                if self._items_empty:
                    break
                if (
                    sc["chunks_total"] <= sc["chunks_processed"]
                    and len(self.tasks_to_accumulate) <= self.executor.treereduction
                    and self.all_to_accumulate_local()
                ):
                    break

            self._submit_processing_tasks(proc_fn, items)

            # When done submitting, look for completed tasks.
            task = self.wait(5)
            if task:
                if not task.successful():
                    task.resubmit(self)
                    continue
                self.tasks_to_accumulate.append(task)
                if re.match("processing", task.category):
                    sc["chunks_processed"] += 1
                elif task.category == "accumulating":
                    sc["accumulations_done"] += 1
                else:
                    raise RuntimeError(f"Unrecognized task category {task.category}")

            self._submit_accum_tasks(accum_fn)
            self._update_bars()

    def _submit_accum_tasks(self, accum_fn):
        def _group(lst, n):
            total = len(lst)
            jump = math.ceil(total / n)
            for start in range(jump):
                yield lst[start::jump]

        treereduction = self.executor.treereduction

        sc = self.stats_coffea
        bring_back = False
        force = False

        factor = 3

        if sc["chunks_processed"] >= sc["chunks_total"] or early_terminate:
            s = self.stats
            factor = 2
            if s.tasks_waiting == 0:
                factor = 1
                bring_back = True

            if s.tasks_waiting + s.tasks_on_workers == 0:
                force = True
                if (
                    len(self.tasks_to_accumulate) <= treereduction
                    and self.all_to_accumulate_local()
                ):
                    return

        if (len(self.tasks_to_accumulate) < (factor * treereduction) - 1) and not force:
            return

        self.tasks_to_accumulate.sort(
            key=lambda t: t.get_metric("time_workers_execute_last")
        )
        self.tasks_to_accumulate.sort(key=lambda t: len(t))

        if force:
            work_list = self.tasks_to_accumulate
            self.tasks_to_accumulate = []
        else:
            split = max(1, (factor - 1)) * treereduction
            work_list = self.tasks_to_accumulate[0:split]
            self.tasks_to_accumulate = self.tasks_to_accumulate[split:]

        for next_to_accum in _group(work_list, treereduction):
            if len(next_to_accum) < 2 and not force:
                self.tasks_to_accumulate.extend(next_to_accum)
                continue

            nall = len(next_to_accum)
            ncps = sum(t.is_checkpoint() for t in next_to_accum)
            bring_back = bring_back or (
                (ncps / nall) <= self.executor.checkpoint_proportion
            )

            accum_task = AccumTask(
                self, accum_fn, next_to_accum, bring_back_output=bring_back
            )
            self.submit(accum_task)
            sc["accumulations_submitted"] += 1

    def _declare_resources(self):
        executor = self.executor

        # If explicit resources are given, collect them into default_resources
        default_resources = {"cores": 1}
        if executor.cores:
            default_resources["cores"] = executor.cores
        if executor.memory:
            default_resources["memory"] = executor.memory
        if executor.disk:
            default_resources["disk"] = executor.disk
        if executor.gpus:
            default_resources["gpus"] = executor.gpus

        # Enable monitoring and auto resource consumption, if desired:
        self.tune("temp-replica-count", max(executor.replicas, 1))
        self.tune("category-steady-n-tasks", 1)
        self.tune("prefer-dispatch", 1)
        self.tune("immediate-recovery", 1)
        self.tune("wait-for-workers", 0)
        self.tune("attempt-schedule-depth", 200)
        self.tune("hungry-minimum", 100)

        if executor.disable_worker_transfers:
            self.disable_peer_transfers()

        # if resource_monitor is given, and not 'off', then monitoring is activated.
        # anything other than 'measure' is assumed to be 'watchdog' mode, where in
        # addition to measuring resources, tasks are killed if they go over their
        # resources.
        monitor_enabled = True
        watchdog_enabled = True
        if not executor.resource_monitor or executor.resource_monitor == "off":
            monitor_enabled = False
        elif executor.resource_monitor == "measure":
            watchdog_enabled = False

        if monitor_enabled:
            self.enable_monitoring(watchdog=watchdog_enabled)

        # set the auto resource modes
        mode = "max"
        if executor.resources_mode == "fixed":
            mode = "fixed"
        for category in "default preprocessing processing accumulating".split():
            self.set_category_mode(category, mode)
            self.set_category_resources_max(category, default_resources)

        # use auto mode max-throughput only for processing tasks
        if executor.resources_mode == "max-throughput":
            self.set_category_mode("processing", "max throughput")

        # enable fast termination of workers
        fast_terminate = executor.fast_terminate_workers
        for category in "default preprocessing processing accumulating".split():
            if fast_terminate and fast_terminate > 1:
                self.activate_fast_abort_category(category, fast_terminate)

    def _make_process_bars(self):
        sc = self.stats_coffea
        accums = self._estimate_accum_tasks()

        self.bar.add_task(
            "Submitted", total=sc["chunks_total"], unit=self.executor.unit
        )
        self.bar.add_task(
            "Processed", total=sc["chunks_total"], unit=self.executor.unit
        )
        self.bar.add_task("Accumulated", total=math.ceil(accums), unit="tasks")

        self.stats_coffea["chunks_processed"] = 0
        self.stats_coffea["accumulations_done"] = 0
        self.stats_coffea["accumulations_submitted"] = 0
        self.stats_coffea["estimated_total_accumulations"] = accums

        self._update_bars()

    def _estimate_accum_tasks(self):
        sc = self.stats_coffea

        try:
            # return immediately if there is no more work to do
            if sc["chunks_total"] <= sc["chunks_processed"]:
                if sc["accumulations_submitted"] <= sc["accumulations_done"]:
                    return sc["accumulations_done"]

            items_to_accum = sc["chunks_processed"]
            items_to_accum += sc["accumulations_submitted"]

            chunks_left = sc["chunks_total"] - sc["chunks_processed"]
            items_to_accum += chunks_left

            accums = 1
            while True:
                if items_to_accum <= self.executor.treereduction:
                    accums += 1
                    break
                step = math.floor(items_to_accum / self.executor.treereduction)
                accums += step
                items_to_accum -= step * self.executor.treereduction
            return accums
        except Exception:
            return 0

    def _update_bars(self, final_update=False):
        sc = self.stats_coffea
        total = sc["chunks_total"]

        accums = self._estimate_accum_tasks()

        self.bar.update("Submitted", completed=sc["chunks_submitted"], total=total)
        self.bar.update("Processed", completed=sc["chunks_processed"], total=total)
        self.bar.update("Accumulated", completed=sc["accumulations_done"], total=accums)

        sc["estimated_total_accumulations"] = accums

        self.bar.refresh()
        if final_update:
            self.bar.stop()


class CoffeaVineTask(PythonTask):
    tasks_counter = 0

    def __init__(self, m, fn, item_args, itemid, bring_back_output=False):
        CoffeaVineTask.tasks_counter += 1
        self.itemid = itemid
        self.retries_to_go = m.executor.retries
        self.function = fn
        self._checkpoint = False

        super().__init__(self.function, *item_args)

        # disable vine serialization as coffea does its own.
        self.disable_output_serialization()

        if bring_back_output or m.executor.disable_worker_transfers:
            self.set_output_cache(True)
        else:
            self.enable_temp_output()

        for name, f in m.extra_input_files.items():
            self.add_input(f, name)

        if m.executor.x509_proxy:
            self.set_env_var("X509_USER_PROXY", "x509.pem")

        if m.poncho_file:
            self.add_environment(m.poncho_file)

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.itemid)

    def _has_result(self):
        return not isinstance(self.output, PythonTaskNoResult)

    def is_checkpoint(self):
        return self._checkpoint

    # use output to return python result, rather than stdout as regular vine
    @property
    def output(self):
        return _decompress(super().output)

    def cleanup_inputs(self, m, force=False):
        pass

    def cleanup_outputs(self, m):
        try:
            name = self.output_file.source()
            if name:
                os.remove(name)
        except FileNotFoundError:
            pass
        m.undeclare_file(self.output_file)

    def clone(self, m):
        raise NotImplementedError

    def resubmit(self, m):
        if self.retries_to_go < 1:
            return m.soft_terminate(self)

        t = self.clone(m)
        t.retries_to_go = self.retries_to_go - 1

        m.console(
            "resubmitting {} as {} with {} events. {} attempt(s) left.",
            self.itemid,
            t.itemid,
            len(t),
            t.retries_to_go,
        )

        m.submit(t)

    def split(self, m):
        # if tasks do not overwrite this method, then is is assumed they cannot
        # be split.
        m.soft_terminate(self)

    def debug_info(self):
        msg = f"{self.itemid} with '{self.result}' result."
        return msg

    def exhausted(self):
        return self.result == "resource exhaustion"

    def report(self, m):
        if (not m.console.verbose_mode) and self.successful():
            return self.successful()

        m.console.printf(
            "{} task id {} item {} with {} events on {}. return code {} ({})",
            self.category,
            self.id,
            self.itemid,
            len(self),
            self.hostname,
            self.exit_code,
            self.result,
        )

        m.console.printf(
            "    allocated cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}",
            self.resources_allocated.cores,
            self.resources_allocated.memory,
            self.resources_allocated.disk,
            self.resources_allocated.gpus,
        )

        if m.executor.resource_monitor and m.executor.resource_monitor != "off":
            m.console.printf(
                "    measured cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}, runtime {:.1f} s",
                self.resources_measured.cores + 0.0,  # +0.0 trick to clear any -0.0
                self.resources_measured.memory,
                self.resources_measured.disk,
                self.resources_measured.gpus,
                (self.get_metric("time_workers_execute_last")) / 1e6,
            )

        if (
            m.executor.print_stdout
            or not (self.successful() or self.exhausted())
            or self.category == "accumulating"
        ):
            if self.std_output:
                m.console.print("    output:")
                m.console.print(self.std_output)

        if not (self.successful() or self.exhausted()):
            info = self.debug_info()
            m.console.warn(
                "task id {} item {} failed: {}\n    {}",
                self.id,
                self.itemid,
                self.result,
                info,
            )
        return self.successful()


class PreProcTask(CoffeaVineTask):
    def __init__(self, m, fn, item, itemid=None):
        if not itemid:
            itemid = f"pre_{CoffeaVineTask.tasks_counter}"

        self.item = item

        self.size = 1
        super().__init__(m, fn, [self.item], itemid, bring_back_output=True)

        self.set_category("preprocessing")
        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transferring file.
            pass
        else:
            f = m.declare_file(item.filename, cache=False)
            self.add_input(f, item.filename)

    def clone(self, m):
        return PreProcTask(
            m,
            self.function,
            self.item,
            self.itemid,
        )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return f"{(i.dataset, i.filename, i.treename)} {msg}"


class ProcTask(CoffeaVineTask):
    def __init__(self, m, fn, item, itemid=None, bring_back_output=False):
        self.size = len(item)

        if not itemid:
            itemid = f"p_{CoffeaVineTask.tasks_counter}"

        self.item = item
        super().__init__(m, fn, [item], itemid, bring_back_output=bring_back_output)

        self.set_category("processing")
        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transferring file.
            pass
        else:
            f = m.declare_file(item.filename, cache=False)
            self.add_input(f, item.filename)

    def clone(self, m):
        return ProcTask(
            m,
            self.function,
            self.item,
            self.itemid,
        )

    def resubmit(self, m):
        if self.retries_to_go < 1:
            return m.soft_terminate(self)

        if self.exhausted():
            if m.executor.split_on_exhaustion:
                return self.split(m)
            else:
                return m.soft_terminate(self)
        else:
            return super().resubmit(m)

    def split(self, m):
        m.console.warn(f"splitting task id {self.id} after resource exhaustion.")

        total = len(self.item)
        if total < 2:
            return m.soft_terminate()

        m.stats_coffea[
            "chunks_total"
        ] += 1  # 1 chunk is split into 2, so we need to add 1 to the total
        m.stats_coffea["chunks_split"] += 1

        # remove the original item from the known work items, as it is being
        # split into two or more work items.
        m.known_workitems.remove(self.item)

        i = self.item
        chunksize = math.ceil((i.entrystop - i.entrystart) / 2)
        start = i.entrystart
        while start < i.entrystop:
            stop = min(i.entrystop, start + chunksize)
            w = WorkItem(
                i.dataset, i.filename, i.treename, start, stop, i.fileuuid, i.usermeta
            )
            t = self.__class__(m, self.function, w)
            start = stop

            m.submit(t)
            m.known_workitems.add(w)

            m.console(
                "resubmitting {} partly as {} with {} events. {} attempt(s) left.",
                self.itemid,
                t.itemid,
                len(t),
                t.retries_to_go,
            )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return "{} {}".format(
            (i.dataset, i.filename, i.treename, i.entrystart, i.entrystop), msg
        )


class AccumTask(CoffeaVineTask):
    def __init__(
        self, m, fn, tasks_to_accumulate, itemid=None, bring_back_output=False
    ):
        if not itemid:
            itemid = f"accum_{CoffeaVineTask.tasks_counter}"

        self.tasks_to_accumulate = tasks_to_accumulate
        self.size = sum(len(t) for t in self.tasks_to_accumulate)

        names = [f"file.{i}" for (i, t) in enumerate(self.tasks_to_accumulate)]

        super().__init__(m, fn, [names], itemid, bring_back_output=bring_back_output)

        self._checkpoint = bring_back_output

        self.set_category("accumulating")
        for name, t in zip(names, self.tasks_to_accumulate):
            self.add_input(t.output_file, name)

    def is_checkpoint(self):
        return self._checkpoint

    def cleanup_inputs(self, m, force=False):
        super().cleanup_inputs(m)
        # cleanup files associated with results already accumulated
        if self.is_checkpoint() or force:
            if self.tasks_to_accumulate:
                for t in self.tasks_to_accumulate:
                    # cleanup up the tree as the cleanup was triggered from a checkpoint.
                    t.cleanup_inputs(m, force=True)
                    t.cleanup_outputs(m)
            self.tasks_to_accumulate = None

    def clone(self, m):
        return AccumTask(
            m,
            self.function,
            self.tasks_to_accumulate,
            self.itemid,
        )

    def debug_info(self):
        tasks = self.tasks_to_accumulate

        msg = super().debug_info()
        if tasks:
            return "{} accumulating: [{}] ".format(
                msg, "\n".join([t.result for t in tasks])
            )
        else:
            return f"{msg} accumulating: []"


def _handle_early_terminate(signum, frame, raise_on_repeat=True):
    global early_terminate
    raise KeyboardInterrupt

    if early_terminate and raise_on_repeat:
        raise KeyboardInterrupt
    else:
        manager.console.printf(
            "********************************************************************************"
        )
        manager.console.printf("Canceling processing tasks for final accumulation.")
        manager.console.printf("C-c now to immediately terminate.")
        manager.console.printf(
            "********************************************************************************"
        )
        early_terminate = True
        manager.cancel_by_category("processing")
        manager.cancel_by_category("accumulating")


def _get_x509_proxy(x509_proxy=None):
    if x509_proxy:
        return x509_proxy

    x509_proxy = os.environ.get("X509_USER_PROXY", None)
    if x509_proxy:
        return x509_proxy

    x509_proxy = join(os.environ.get("TMPDIR", "/tmp"), f"x509up_u{os.getuid()}")
    if os.path.exists(x509_proxy):
        return x509_proxy

    return None


class ResultUnavailable(Exception):
    pass


class Stats(collections.defaultdict):
    def __init__(self, *args, **kwargs):
        super().__init__(int, *args, **kwargs)

    def min(self, stat, value):
        try:
            self[stat] = min(self[stat], value)
        except KeyError:
            self[stat] = value

    def max(self, stat, value):
        try:
            self[stat] = max(self[stat], value)
        except KeyError:
            self[stat] = value


class VerbosePrint:
    def __init__(self, console, status_mode=True, verbose_mode=True):
        self.console = console
        self.status_mode = status_mode
        self.verbose_mode = verbose_mode

    def __call__(self, format_str, *args, **kwargs):
        if self.verbose_mode:
            self.printf(format_str, *args, **kwargs)

    def print(self, msg):
        if self.status_mode:
            self.console.print(msg)
        else:
            print(msg)

    def printf(self, format_str, *args, **kwargs):
        msg = format_str.format(*args, **kwargs)
        self.print(msg)

    def warn(self, format_str, *args, **kwargs):
        if self.status_mode:
            format_str = "[red]WARNING:[/red] " + format_str
        else:
            format_str = "WARNING: " + format_str
        self.printf(format_str, *args, **kwargs)


# Support for rich_bar so that we can keep track of bars by their names, rather
# than the changing bar ids.
class StatusBar:
    def __init__(self, enabled=True):
        self._prog = rich_bar()
        self._ids = {}
        if enabled:
            self._prog.start()

    def add_task(self, desc, *args, **kwargs):
        b = self._prog.add_task(desc, *args, **kwargs)
        self._ids[desc] = b
        self._prog.start_task(self._ids[desc])
        return b

    def stop_task(self, desc, *args, **kwargs):
        return self._prog.stop_task(self._ids[desc], *args, **kwargs)

    def update(self, desc, *args, **kwargs):
        return self._prog.update(self._ids[desc], *args, **kwargs)

    def advance(self, desc, *args, **kwargs):
        return self._prog.advance(self._ids[desc], *args, **kwargs)

    # redirect anything else to rich_bar
    def __getattr__(self, name):
        return getattr(self._prog, name)
