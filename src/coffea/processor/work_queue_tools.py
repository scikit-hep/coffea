import collections
import math
import os
import random
import re
import signal
import textwrap
from os.path import basename, getsize, join
from tempfile import NamedTemporaryFile, TemporaryDirectory

import cloudpickle
import numpy
import scipy

from coffea.util import deprecate, rich_bar

from .accumulator import accumulate
from .executor import WorkItem, _compression_wrapper, _decompress

# The Work Queue object is global b/c we want to
# retain state between runs of the executor, such
# as connections to workers, cached data, etc.
_wq_queue = None

# If set to True, workflow stops processing and outputs only the results that
# have been already processed.
early_terminate = False


# This function that accumulates results from files does not require wq.
# We declare it before checking for wq so that we do not need to install wq at
# the remote site.
def accumulate_result_files(files_to_accumulate, accumulator=None):
    from coffea.processor import accumulate

    # work on local copy of list
    files_to_accumulate = list(files_to_accumulate)
    while files_to_accumulate:
        f = files_to_accumulate.pop()

        with open(f, "rb") as rf:
            result = _decompress(rf.read())
        if not accumulator:
            accumulator = result
            continue

        accumulator = accumulate([result], accumulator)
        del result
    return accumulator


try:
    import work_queue as wq
    from work_queue import Task, WorkQueue
except ImportError:
    wq = None
    print("work_queue module not available")

    class Task:
        def __init__(self, *args, **kwargs):
            raise ImportError("work_queue not available")

    class WorkQueue:
        def __init__(self, *args, **kwargs):
            raise ImportError("work_queue not available")


TaskReport = collections.namedtuple(
    "TaskReport", ["events_count", "wall_time", "memory"]
)


class CoffeaWQ(WorkQueue):
    def __init__(
        self,
        executor,
    ):
        self._staging_dir_obj = TemporaryDirectory("wq-tmp-", dir=executor.filepath)

        self.executor = executor
        self.stats_coffea = Stats()

        # set to keep track of the final work items the workflow consists of.
        # When a work item needs to be split, it is replaced from this set by
        # its constituents.
        self.known_workitems = set()

        # list that keeps results as they finish to construct accumulation
        # tasks.
        self.tasks_to_accumulate = []

        # list of TaskReport tuples with the statistics to compute the dynamic
        # chunksize
        self.task_reports = []

        super().__init__(
            port=self.executor.port,
            name=self.executor.manager_name,
            debug_log=self.executor.debug_log,
            stats_log=self.executor.stats_log,
            transactions_log=self.executor.transactions_log,
            status_display_interval=self.executor.status_display_interval,
            ssl=self.executor.ssl,
        )

        self.bar = StatusBar(enabled=executor.status)
        self.console = VerbosePrint(self.bar.console, executor.status, executor.verbose)

        self._declare_resources()

        # Make use of the stored password file, if enabled.
        if self.executor.password_file:
            self.specify_password_file(self.executor.password_file)

        self.function_wrapper = self._write_fn_wrapper()

        if self.executor.tasks_accum_log:
            with open(self.executor.tasks_accum_log, "w") as f:
                f.write(
                    "id,category,status,dataset,file,range_start,range_stop,accum_parent,time_start,time_end,cpu_time,memory,fin,fout\n"
                )

        self.console.printf(f"Listening for work queue workers on port {self.port}.")
        # perform a wait to print any warnings before progress bars
        self.wait(0)

    def __del__(self):
        try:
            self._staging_dir_obj.cleanup()
        finally:
            super().__del__()

    def _check_executor_parameters(self, executor):
        if executor.environment_file and not executor.wrapper:
            raise ValueError(
                "WorkQueueExecutor: Could not find poncho_package_run. Use 'wrapper' argument."
            )

        if not executor.treereduction and executor.chunks_per_accum:
            deprecate(
                RuntimeError("chunks_per_accum is deprecated. Use treereduction."),
                "v0.8.0",
                "31 Dec 2022",
            )
            executor.treereduction = executor.chunks_per_accum

        if executor.treereduction < 2:
            raise ValueError("WorkQueueExecutor: treereduction should be at least 2.")

        if not executor.manager_name and executor.master_name:
            deprecate(
                RuntimeError("master_name is deprecated. Use manager_name."),
                "v0.8.0",
                "31 Dec 2022",
            )
            executor.manager_name = executor.master_name

        if not executor.port:
            executor.port = 0 if executor.manager_name else 9123

        # wq always needs serializaiton to files, thus compression is always on
        if executor.compression is None:
            executor.compression = 1

        # activate monitoring if it has not been explicitly activated and we are
        # using an automatic resource allocation.
        if executor.resources_mode != "fixed" and executor.resource_monitor == "off":
            executor.resource_monitor = "watchdog"

        deprecated = ["master_name", "chunks_accum_in_mem", "bar_format"]
        for field in deprecated:
            if getattr(executor, field):
                deprecate(
                    RuntimeError(f"{field} is deprecated"), "v0.8.0", "31 Dec 2022"
                )

        executor.verbose = executor.verbose or executor.print_stdout
        executor.x509_proxy = _get_x509_proxy(executor.x509_proxy)

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
                task.fout_size = getsize(task.outfile_output) / 1e6
                if task.fin_size > 0:
                    # record only if task used any intermediate inputs
                    self.stats_coffea.max("size_max_input", task.fin_size)
                self.stats_coffea.max("size_max_output", task.fout_size)
                # Remove input files as we go to avoid unbounded disk we do not
                # remove outputs, as they are used by further accumulate tasks
                task.cleanup_inputs()
            return task
        return None

    def application_info(self):
        return {
            "application_info": {
                "values": dict(self.stats_coffea),
                "units": {
                    "size_max_output": "MB",
                    "size_max_input": "MB",
                },
            }
        }

    @property
    def staging_dir(self):
        return self._staging_dir_obj.name

    @property
    def chunksize_current(self):
        return self._chunksize_current

    @chunksize_current.setter
    def chunksize_current(self, new_value):
        self._chunksize_current = new_value
        self.stats_coffea["chunksize_current"] = self._chunksize_current

    @property
    def executor(self):
        return self._executor

    @executor.setter
    def executor(self, new_value):
        self._executor = new_value
        self._check_executor_parameters(self._executor)

    def function_to_file(self, function, name=None):
        with NamedTemporaryFile(
            prefix=name, suffix=".p", dir=self.staging_dir, delete=False
        ) as f:
            cloudpickle.dump(function, f)
            return f.name

    def soft_terminate(self, task=None):
        if task:
            self.console.warn(f"item {task.itemid} failed permanently.")

        if not early_terminate:
            # trigger soft termination
            _handle_early_terminate(0, None, raise_on_repeat=False)

    def _add_task_report(self, task):
        r = TaskReport(
            len(task), task.cmd_execution_time / 1e6, task.resources_measured.memory
        )
        self.task_reports.append(r)

    def _preprocessing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        infile_pre_fn = self.function_to_file(function, "preproc")
        for item in items:
            task = PreProcCoffeaWQTask(self, infile_pre_fn, item)
            self.submit(task)

        self.bar.add_task("Preprocessing", total=len(items), unit=self.executor.unit)
        while not self.empty():
            task = self.wait(5)
            if task:
                if task.successful():
                    accumulator = accumulate([task.output], accumulator)
                    self.bar.advance("Preprocessing", 1)
                    task.cleanup_outputs()
                    task.task_accum_log(self.executor.tasks_accum_log, "", "done")
                else:
                    task.resubmit(self)
                self.bar.refresh()

        self.bar.stop_task("Preprocessing")
        return accumulator

    def _submit_processing_tasks(self, infile_procc_fn, items):
        while True:
            if early_terminate or self._items_empty:
                return
            if not self.hungry():
                return
            sc = self.stats_coffea
            if sc["events_submitted"] >= sc["events_total"]:
                return

            try:
                if sc["events_submitted"] > 0:
                    # can't send if generator not initialized first with a next
                    chunksize = _sample_chunksize(self.chunksize_current)
                    item = items.send(chunksize)
                else:
                    item = next(items)
                self._submit_processing_task(infile_procc_fn, item)
            except StopIteration:
                self.console.warn("Ran out of items to process.")
                self._items_empty = True
                return

    def _submit_processing_task(self, infile_procc_fn, item):
        self.known_workitems.add(item)
        t = ProcCoffeaWQTask(self, infile_procc_fn, item)
        self.submit(t)
        self.stats_coffea["events_submitted"] += len(t)

    def _final_accumulation(self, accumulator):
        if len(self.tasks_to_accumulate) < 1:
            self.console.warn("No results available.")
            return accumulator

        self.console("Merging with local final accumulator...")
        accumulator = accumulate_result_files(
            [t.outfile_output for t in self.tasks_to_accumulate], accumulator
        )

        total_accumulated_events = 0
        for t in self.tasks_to_accumulate:
            total_accumulated_events += len(t)
            t.cleanup_outputs()
            t.task_accum_log(self.executor.tasks_accum_log, "accumulated", 0)

        sc = self.stats_coffea
        if sc["events_processed"] != sc["events_total"]:
            self.console.warn(
                f"Number of events processed ({sc['events_processed']}) is different from total ({sc['events_total']})!"
            )

        if total_accumulated_events != sc["events_processed"]:
            self.console.warn(
                f"Number of events accumulated ({total_accumulated_events}) is different from processed ({sc['events_processed']})!"
            )

        return accumulator

    def _fill_unprocessed_items(self, accumulator, items):
        chunksize = max(self.chunksize_current, self.executor.chunksize)
        try:
            while True:
                if chunksize != self.executor.chunksize:
                    item = items.send(chunksize)
                else:
                    item = next(items)
                self.known_workitems.add(item)
        except StopIteration:
            pass

        unproc = self.known_workitems - accumulator["processed"]
        accumulator["unprocessed"] = unproc
        if unproc:
            count = sum(len(item) for item in unproc)
            self.console.warn(f"{len(unproc)} unprocessed item(s) ({count} event(s)).")

    def _processing(self, items, function, accumulator):
        function = _compression_wrapper(self.executor.compression, function)
        accumulate_fn = _compression_wrapper(
            self.executor.compression, accumulate_result_files
        )

        infile_procc_fn = self.function_to_file(function, "proc")
        infile_accum_fn = self.function_to_file(accumulate_fn, "accum")

        executor = self.executor
        sc = self.stats_coffea

        # Ensure that the items looks like a generator
        if not isinstance(items, collections.abc.Generator):
            items = (item for item in items)

        # Keep track of total tasks in each state.
        sc["events_processed"] = 0
        sc["events_submitted"] = 0
        sc["events_total"] = executor.events_total
        sc["accumulations_submitted"] = 0
        sc["chunksize_original"] = executor.chunksize

        self.chunksize_current = executor.chunksize

        self._make_process_bars()

        signal.signal(signal.SIGINT, _handle_early_terminate)

        self._process_events(infile_procc_fn, infile_accum_fn, items)

        # merge results with original accumulator given by the executor
        accumulator = self._final_accumulation(accumulator)

        # compute the set of unprocessed work items, if any
        self._fill_unprocessed_items(accumulator, items)

        if self.chunksize_current != sc["chunksize_original"]:
            self.console.printf(f"final chunksize {self.chunksize_current}")

        self._update_bars(final_update=True)
        return accumulator

    def _process_events(self, infile_procc_fn, infile_accum_fn, items):
        self.known_workitems = set()
        sc = self.stats_coffea
        self._items_empty = False

        while True:
            if self.empty():
                if early_terminate or self._items_empty:
                    break
                if sc["events_total"] <= sc["events_processed"]:
                    break

            self._submit_processing_tasks(infile_procc_fn, items)

            # When done submitting, look for completed tasks.
            task = self.wait(5)

            if not task:
                continue

            if not task.successful():
                task.resubmit(self)
                continue

            self.tasks_to_accumulate.append(task)

            if re.match("processing", task.category):
                self._add_task_report(task)
                sc["events_processed"] += len(task)
                sc["chunks_processed"] += 1
                self._update_chunksize()
            elif task.category == "accumulating":
                sc["accumulations_done"] += 1
            else:
                raise RuntimeError(f"Unrecognized task category {task.category}")

            self._submit_accum_tasks(infile_accum_fn)
            self._update_bars()

    def _submit_accum_tasks(self, infile_accum_fn):
        treereduction = self.executor.treereduction

        sc = self.stats_coffea
        force = early_terminate
        force |= sc["events_processed"] >= sc["events_total"]

        if len(self.tasks_to_accumulate) < (2 * treereduction) - 1 and (not force):
            return

        if force:
            min_accum = 2
        else:
            min_accum = treereduction

        self.tasks_to_accumulate.sort(key=lambda t: t.fout_size)

        for start in range(0, len(self.tasks_to_accumulate), treereduction):
            if len(self.tasks_to_accumulate) < min_accum:
                break

            end = min(len(self.tasks_to_accumulate), treereduction)
            next_to_accum = self.tasks_to_accumulate[0:end]
            self.tasks_to_accumulate = self.tasks_to_accumulate[end:]

            accum_task = AccumCoffeaWQTask(self, infile_accum_fn, next_to_accum)
            self.submit(accum_task)
            sc["accumulations_submitted"] += 1

            # log the input tasks to this accumulation task
            for t in next_to_accum:
                t.task_accum_log(self.executor.tasks_accum_log, "done", t.id)

    def _update_chunksize(self):
        ex = self.executor
        if ex.dynamic_chunksize:
            chunksize = _compute_chunksize(
                ex.chunksize, ex.dynamic_chunksize, self.task_reports
            )
            self.chunksize_current = chunksize
            self.console("current chunksize {}", self.chunksize_current)
        return self.chunksize_current

    def _declare_resources(self):
        executor = self.executor

        # If explicit resources are given, collect them into default_resources
        default_resources = {}
        if executor.cores:
            default_resources["cores"] = executor.cores
        if executor.memory:
            default_resources["memory"] = executor.memory
        if executor.disk:
            default_resources["disk"] = executor.disk
        if executor.gpus:
            default_resources["gpus"] = executor.gpus

        # Enable monitoring and auto resource consumption, if desired:
        self.tune("category-steady-n-tasks", 3)

        # Evenly divide resources in workers per category
        self.tune("force-proportional-resources", 1)

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
        mode = wq.WORK_QUEUE_ALLOCATION_MODE_MAX
        if executor.resources_mode == "fixed":
            mode = wq.WORK_QUEUE_ALLOCATION_MODE_FIXED
        for category in "default preprocessing processing accumulating".split():
            self.specify_category_max_resources(category, default_resources)
            self.specify_category_mode(category, mode)

        # use auto mode max-throughput only for processing tasks
        if executor.resources_mode == "max-throughput":
            self.specify_category_mode(
                "processing", wq.WORK_QUEUE_ALLOCATION_MODE_MAX_THROUGHPUT
            )

        # enable fast termination of workers
        fast_terminate = executor.fast_terminate_workers
        for category in "default preprocessing processing accumulating".split():
            if fast_terminate and fast_terminate > 1:
                self.activate_fast_abort_category(category, fast_terminate)

    def _write_fn_wrapper(self):
        """Writes a wrapper script to run serialized python functions and arguments.
        The wrapper takes as arguments the name of three files: function, argument, and output.
        The files function and argument have the serialized function and argument, respectively.
        The file output is created (or overwritten), with the serialized result of the function call.
        The wrapper created is created/deleted according to the lifetime of the WorkQueueExecutor.
        """

        proxy_basename = ""
        if self.executor.x509_proxy:
            proxy_basename = basename(self.executor.x509_proxy)

        contents = textwrap.dedent(
            """\
                        #!/usr/bin/env python3
                        import os
                        import sys
                        import cloudpickle
                        import coffea

                        if "{proxy}":
                            os.environ['X509_USER_PROXY'] = "{proxy}"

                        (fn, args, out) = sys.argv[1], sys.argv[2], sys.argv[3]

                        with open(fn, 'rb') as f:
                            exec_function = cloudpickle.load(f)
                        with open(args, 'rb') as f:
                            exec_args = cloudpickle.load(f)

                        pickled_out = exec_function(*exec_args)
                        with open(out, 'wb') as f:
                            f.write(pickled_out)

                        # Force an OS exit here to avoid a bug in xrootd finalization
                        os._exit(0)
                        """
        )
        with NamedTemporaryFile(
            prefix="fn_wrapper", dir=self.staging_dir, delete=False
        ) as f:
            f.write(contents.format(proxy=proxy_basename).encode())
            return f.name

    def _make_process_bars(self):
        accums = self._estimate_accum_tasks()

        self.bar.add_task(
            "Submitted", total=self.executor.events_total, unit=self.executor.unit
        )
        self.bar.add_task(
            "Processed", total=self.executor.events_total, unit=self.executor.unit
        )
        self.bar.add_task("Accumulated", total=math.ceil(accums), unit="tasks")

        self.stats_coffea["chunks_processed"] = 0
        self.stats_coffea["accumulations_done"] = 0
        self.stats_coffea["accumulations_submitted"] = 0
        self.stats_coffea["estimated_total_accumulations"] = accums

        self._update_bars()

    def _estimate_accum_tasks(self):
        sc = self.stats_coffea

        # return immediately if there is no more work to do
        if sc["events_total"] <= sc["events_processed"]:
            if sc["accumulations_submitted"] <= sc["accumulations_done"]:
                return sc["accumulations_done"]

        items_to_accum = sc["chunks_processed"]
        items_to_accum += sc["accumulations_submitted"]

        events_left = sc["events_total"] - sc["events_processed"]
        chunks_left = math.ceil(events_left / sc["chunksize_current"])
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

    def _update_bars(self, final_update=False):
        sc = self.stats_coffea
        total = sc["events_total"]

        accums = self._estimate_accum_tasks()

        self.bar.update("Submitted", completed=sc["events_submitted"], total=total)
        self.bar.update("Processed", completed=sc["events_processed"], total=total)
        self.bar.update("Accumulated", completed=sc["accumulations_done"], total=accums)

        sc["estimated_total_accumulations"] = accums

        self.bar.refresh()
        if final_update:
            self.bar.stop()


class CoffeaWQTask(Task):
    tasks_counter = 0

    def __init__(self, queue, infile_fn, item_args, itemid):
        CoffeaWQTask.tasks_counter += 1

        self.itemid = itemid

        self.py_result = ResultUnavailable()
        self._stdout = None

        self.infile_fn = infile_fn

        self.infile_args = join(queue.staging_dir, f"args_{self.itemid}.p")
        self.outfile_output = join(queue.staging_dir, f"out_{self.itemid}.p")
        self.outfile_stdout = join(queue.staging_dir, f"stdout_{self.itemid}.p")

        with open(self.infile_args, "wb") as wf:
            cloudpickle.dump(item_args, wf)

        executor = queue.executor
        self.retries_to_go = executor.retries

        super().__init__(self.remote_command(env_file=executor.environment_file))

        self.specify_input_file(queue.function_wrapper, "fn_wrapper", cache=True)
        self.specify_input_file(infile_fn, "function.p", cache=True)
        self.specify_input_file(self.infile_args, "args.p", cache=False)
        self.specify_output_file(self.outfile_output, "output.p", cache=False)
        self.specify_output_file(self.outfile_stdout, "stdout.log", cache=False)

        for f in executor.extra_input_files:
            self.specify_input_file(f, cache=True)

        if executor.x509_proxy:
            self.specify_input_file(executor.x509_proxy, cache=True)

        if executor.wrapper and executor.environment_file:
            self.specify_input_file(executor.wrapper, "py_wrapper", cache=True)
            self.specify_input_file(executor.environment_file, "env_file", cache=True)

    def __len__(self):
        return self.size

    def __str__(self):
        return str(self.itemid)

    def remote_command(self, env_file=None):
        fn_command = "python fn_wrapper function.p args.p output.p >stdout.log 2>&1"
        command = fn_command

        if env_file:
            wrap = (
                './py_wrapper -d -e env_file -u "$WORK_QUEUE_SANDBOX"/{}-env-{} -- {}'
            )
            command = wrap.format(basename(env_file), os.getpid(), fn_command)

        return command

    @property
    def std_output(self):
        if not self._stdout:
            try:
                with open(self.outfile_stdout) as rf:
                    self._stdout = rf.read()
            except Exception:
                self._stdout = None
        return self._stdout

    def _has_result(self):
        return not (
            self.py_result is None or isinstance(self.py_result, ResultUnavailable)
        )

    # use output to return python result, rather than stdout as regular wq
    @property
    def output(self):
        if not self._has_result():
            try:
                with open(self.outfile_output, "rb") as rf:
                    result = _decompress(rf.read())
                    self.py_result = result
            except Exception as e:
                self.py_result = ResultUnavailable(e)
        return self.py_result

    def cleanup_inputs(self):
        os.remove(self.infile_args)

    def cleanup_outputs(self):
        os.remove(self.outfile_output)

    def clone(self, queue):
        raise NotImplementedError

    def resubmit(self, queue):
        if self.retries_to_go < 1:
            return queue.soft_terminate(self)

        t = self.clone(queue)
        t.retries_to_go = self.retries_to_go - 1

        queue.console(
            "resubmitting {} as {} with {} events. {} attempt(s) left.",
            self.itemid,
            t.itemid,
            len(t),
            t.retries_to_go,
        )

        queue.submit(t)

    def split(self, queue):
        # if tasks do not overwrite this method, then is is assumed they cannot
        # be split.
        queue.soft_terminate(self)

    def debug_info(self):
        self.output  # load results, if needed

        has_output = "" if self._has_result() else "out"
        msg = f"{self.itemid} with{has_output} result."
        return msg

    def successful(self):
        return (self.result == 0) and (self.return_status == 0)

    def exhausted(self):
        return self.result == wq.WORK_QUEUE_RESULT_RESOURCE_EXHAUSTION

    def report(self, queue):
        if (not queue.console.verbose_mode) and self.successful():
            return self.successful()

        result_str = self.result_str.lower().replace("_", " ")
        if not self.successful() and self.result == 0:
            result_str = "task error"

        queue.console.printf(
            "{} task id {} item {} with {} events on {}. return code {} ({})",
            self.category,
            self.id,
            self.itemid,
            len(self),
            self.hostname,
            self.return_status,
            result_str,
        )

        queue.console.printf(
            "    allocated cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}",
            self.resources_allocated.cores,
            self.resources_allocated.memory,
            self.resources_allocated.disk,
            self.resources_allocated.gpus,
        )

        if queue.executor.resource_monitor and queue.executor.resource_monitor != "off":
            queue.console.printf(
                "    measured cores: {:.1f}, memory: {:.0f} MB, disk {:.0f} MB, gpus: {:.1f}, runtime {:.1f} s",
                self.resources_measured.cores + 0.0,  # +0.0 trick to clear any -0.0
                self.resources_measured.memory,
                self.resources_measured.disk,
                self.resources_measured.gpus,
                (self.cmd_execution_time) / 1e6,
            )

        if queue.executor.print_stdout or not (self.successful() or self.exhausted()):
            if self.std_output:
                queue.console.print("    output:")
                queue.console.print(self.std_output)

        if not (self.successful() or self.exhausted()):
            info = self.debug_info()
            queue.console.warn(
                "task id {} item {} failed: {}\n    {}",
                self.id,
                self.itemid,
                result_str,
                info,
            )
        return self.successful()

    def task_accum_log(self, log_filename, accum_parent, status):
        # Should call write_task_accum_log with the appropriate arguments
        return NotImplementedError

    def write_task_accum_log(
        self, log_filename, accum_parent, dataset, filename, start, stop, status
    ):
        if not log_filename:
            return

        with open(log_filename, "a") as f:
            f.write(
                "{id},{cat},{status},{set},{file},{start},{stop},{accum},{time_start},{time_end},{cpu},{mem},{fin},{fout}\n".format(
                    id=self.id,
                    cat=self.category,
                    status=status,
                    set=dataset,
                    file=filename,
                    start=start,
                    stop=stop,
                    accum=accum_parent,
                    time_start=self.resources_measured.start,
                    time_end=self.resources_measured.end,
                    cpu=self.resources_measured.cpu_time,
                    mem=self.resources_measured.memory,
                    fin=self.fin_size,
                    fout=self.fout_size,
                )
            )


class PreProcCoffeaWQTask(CoffeaWQTask):
    def __init__(self, queue, infile_fn, item, itemid=None):
        if not itemid:
            itemid = f"pre_{CoffeaWQTask.tasks_counter}"

        self.item = item

        self.size = 1
        super().__init__(queue, infile_fn, [item], itemid)

        self.specify_category("preprocessing")

        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transferring file.
            pass
        else:
            self.specify_input_file(
                item.filename, remote_name=item.filename, cache=True
            )

        self.fin_size = 0

    def clone(self, queue):
        return PreProcCoffeaWQTask(
            queue,
            self.infile_fn,
            self.item,
            self.itemid,
        )

    def debug_info(self):
        i = self.item
        msg = super().debug_info()
        return f"{(i.dataset, i.filename, i.treename)} {msg}"

    def task_accum_log(self, log_filename, accum_parent, status):
        meta = list(self.output)[0].metadata
        i = self.item
        self.write_task_accum_log(
            log_filename, "", i.dataset, i.filename, 0, meta["numentries"], "done"
        )


class ProcCoffeaWQTask(CoffeaWQTask):
    def __init__(self, queue, infile_fn, item, itemid=None):
        self.size = len(item)

        if not itemid:
            itemid = f"p_{CoffeaWQTask.tasks_counter}"

        self.item = item

        super().__init__(queue, infile_fn, [item], itemid)

        self.specify_category("processing")

        if re.search("://", item.filename) or os.path.isabs(item.filename):
            # This looks like an URL or an absolute path (assuming shared
            # filesystem). Not transferring file.
            pass
        else:
            self.specify_input_file(
                item.filename, remote_name=item.filename, cache=True
            )

        self.fin_size = 0

    def clone(self, queue):
        return ProcCoffeaWQTask(
            queue,
            self.infile_fn,
            self.item,
            self.itemid,
        )

    def resubmit(self, queue):
        if self.retries_to_go < 1:
            return queue.soft_terminate(self)

        if self.exhausted():
            if queue.executor.split_on_exhaustion:
                return self.split(queue)
            else:
                return queue.soft_terminate(self)
        else:
            return super().resubmit(queue)

    def split(self, queue):
        queue.console.warn(f"splitting task id {self.id} after resource exhaustion.")

        total = len(self.item)
        if total < 2:
            return queue.soft_terminate()

        # if the chunksize was updated to be less than total, then use that.
        # Otherwise, partition the task in two and update the current chunksize.
        chunksize_target = queue.chunksize_current
        if total <= chunksize_target:
            chunksize_target = math.ceil(total / 2)
            queue.chunksize_current = chunksize_target

        n = max(math.ceil(total / chunksize_target), 1)
        chunksize_actual = int(math.ceil(total / n))

        queue.stats_coffea["chunks_split"] += 1

        # remove the original item from the known work items, as it is being
        # split into two or more work items.
        queue.known_workitems.remove(self.item)

        i = self.item
        start = i.entrystart
        while start < self.item.entrystop:
            stop = min(i.entrystop, start + chunksize_actual)
            w = WorkItem(
                i.dataset, i.filename, i.treename, start, stop, i.fileuuid, i.usermeta
            )
            t = self.__class__(queue, self.infile_fn, w)
            start = stop

            queue.submit(t)
            queue.known_workitems.add(w)

            queue.console(
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

    def task_accum_log(self, log_filename, accum_parent, status):
        i = self.item
        self.write_task_accum_log(
            log_filename,
            accum_parent,
            i.dataset,
            i.filename,
            i.entrystart,
            i.entrystop,
            status,
        )


class AccumCoffeaWQTask(CoffeaWQTask):
    def __init__(
        self,
        queue,
        infile_fn,
        tasks_to_accumulate,
        itemid=None,
    ):
        if not itemid:
            itemid = f"accum_{CoffeaWQTask.tasks_counter}"

        self.tasks_to_accumulate = tasks_to_accumulate
        self.size = sum(len(t) for t in self.tasks_to_accumulate)

        args = [[basename(t.outfile_output) for t in self.tasks_to_accumulate]]

        super().__init__(queue, infile_fn, args, itemid)

        self.specify_category("accumulating")

        for t in self.tasks_to_accumulate:
            self.specify_input_file(t.outfile_output, cache=False)

        self.fin_size = sum(t.fout_size for t in tasks_to_accumulate)

    def cleanup_inputs(self):
        super().cleanup_inputs()
        # cleanup files associated with results already accumulated
        for t in self.tasks_to_accumulate:
            t.cleanup_outputs()

    def clone(self, queue):
        return AccumCoffeaWQTask(
            queue,
            self.infile_fn,
            self.tasks_to_accumulate,
            self.itemid,
        )

    def debug_info(self):
        tasks = self.tasks_to_accumulate

        msg = super().debug_info()

        results = [
            CoffeaWQTask.debug_info(t)
            for t in tasks
            if isinstance(t, AccumCoffeaWQTask)
        ]
        results += [
            t.debug_info() for t in tasks if not isinstance(t, AccumCoffeaWQTask)
        ]

        return "{} accumulating: [{}] ".format(msg, "\n".join(results))

    def task_accum_log(self, log_filename, status, accum_parent=None):
        self.write_task_accum_log(
            log_filename, accum_parent, "", "", 0, len(self), status
        )


def run(executor, items, function, accumulator):
    """Execute using Work Queue
    For more information, see :ref:`intro-coffea-wq`
    """
    if not wq:
        print("You must have Work Queue installed to use WorkQueueExecutor!")
        # raise an import error for work queue
        import work_queue  # noqa

    global _wq_queue
    if _wq_queue is None:
        _wq_queue = CoffeaWQ(executor)
    else:
        # if queue already listening on port, update the parameters given by
        # the executor
        _wq_queue.executor = executor

    try:
        if executor.custom_init:
            executor.custom_init(_wq_queue)

        if executor.desc == "Preprocessing":
            result = _wq_queue._preprocessing(items, function, accumulator)
            # we do not shutdown queue after preprocessing, as we want to
            # keep the connected workers for processing/accumulation
        else:
            result = _wq_queue._processing(items, function, accumulator)
            _wq_queue = None
    except Exception as e:
        _wq_queue = None
        raise e

    return result


def _handle_early_terminate(signum, frame, raise_on_repeat=True):
    global early_terminate

    if early_terminate and raise_on_repeat:
        raise KeyboardInterrupt
    else:
        _wq_queue.console.printf(
            "********************************************************************************"
        )
        _wq_queue.console.printf("Canceling processing tasks for final accumulation.")
        _wq_queue.console.printf("C-c now to immediately terminate.")
        _wq_queue.console.printf(
            "********************************************************************************"
        )
        early_terminate = True
        _wq_queue.cancel_by_category("processing")


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


# Functions related to dynamic chunksize, independent of Work Queue
def _floor_to_pow2(value):
    if value < 1:
        return 1
    return pow(2, math.floor(math.log2(value)))


def _sample_chunksize(chunksize):
    # sample between value found and half of it, to better explore the
    # space.  we take advantage of the fact that the function that
    # generates chunks tries to have equally sized work units per file.
    # Most files have a different number of events, which is unlikely
    # to be a multiple of the chunsize computed. Just in case all files
    # have the same number of events, we return chunksize/2 10% of the
    # time.
    return int(random.choices([chunksize, max(chunksize / 2, 1)], weights=[90, 10])[0])


def _compute_chunksize(base_chunksize, resource_targets, task_reports):
    chunksize_time = None
    chunksize_memory = None

    if resource_targets is not None and len(task_reports) > 1:
        target_time = resource_targets.get("wall_time", None)
        if target_time:
            chunksize_time = _compute_chunksize_target(
                target_time, [(time, evs) for (evs, time, mem) in task_reports]
            )

        target_memory = resource_targets["memory"]
        if target_memory:
            chunksize_memory = _compute_chunksize_target(
                target_memory, [(mem, evs) for (evs, time, mem) in task_reports]
            )

    candidate_sizes = [c for c in [chunksize_time, chunksize_memory] if c]
    if candidate_sizes:
        chunksize = min(candidate_sizes)
    else:
        chunksize = base_chunksize

    try:
        chunksize = int(_floor_to_pow2(chunksize))
    except ValueError:
        chunksize = base_chunksize

    return chunksize


def _compute_chunksize_target(target, pairs):
    # if no info to compute dynamic chunksize (e.g. they info is -1), return nothing
    if len(pairs) < 1 or pairs[0][0] < 0:
        return None

    avgs = [e / max(1, target) for (target, e) in pairs]
    quantiles = numpy.quantile(avgs, [0.25, 0.5, 0.75], interpolation="nearest")

    # remove outliers below the 25%
    pairs_filtered = []
    for i, avg in enumerate(avgs):
        if avg >= quantiles[0]:
            pairs_filtered.append(pairs[i])

    try:
        # separate into time, numevents arrays
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
            [rep[0] for rep in pairs_filtered],
            [rep[1] for rep in pairs_filtered],
        )
    except Exception:
        slope = None

    if (
        slope is None
        or numpy.isnan(slope)
        or numpy.isnan(intercept)
        or slope < 0
        or intercept > 0
    ):
        # we assume that chunksize and target have a positive
        # correlation, with a non-negative overhead (-intercept/slope). If
        # this is not true because noisy data, use the avg chunksize/time.
        # slope and intercept may be nan when data falls in a vertical line
        # (specially at the start)
        slope = quantiles[1]
        intercept = 0

    org = (slope * target) + intercept

    return org
