from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, islice, repeat, starmap
from queue import Queue, ShutDown
from threading import Condition, Thread
import time
from typing import Callable, Iterator, Literal, Protocol, Sized, TypeVar

from coffea.compute.protocol import Computable, ResultType


T = TypeVar("T")


class EventsArray(Sized, Protocol):
    "Awkward array of events or similar"

    # metadata: dict[str, Any]


EventsFunc = Callable[[EventsArray], ResultType]
"Function that processes an EventsArray and returns a ResultType"


class ProcessorABC(ABC):
    @abstractmethod
    def process(self, events: EventsArray) -> ResultType:
        """Process a chunk of events and return a result."""
        raise NotImplementedError


@dataclass(slots=True)
class Step:
    path: str
    entry_range: tuple[int, int]

    def __len__(self) -> int:
        return self.entry_range[1] - self.entry_range[0]


@dataclass(slots=True)
class StepWorkElement:
    func: EventsFunc
    step: Step

    def __call__(self) -> ResultType:
        # Dummy implementation of event loading
        info = str(self.step.entry_range)
        events = info + "A" * (len(self.step) - len(info))  # Dummy events
        return self.func(events)


class StepIterable(ABC):
    @abstractmethod
    def iter_steps(self) -> Iterator[Step]:
        """Return an iterator over steps in the computation."""
        raise NotImplementedError

    def apply(self, func: EventsFunc | ProcessorABC) -> "StepwiseComputable":
        if isinstance(func, ProcessorABC):
            func = func.process
        return StepwiseComputable(func=func, iterable=self)


@dataclass
class StepwiseComputable:
    func: EventsFunc
    iterable: StepIterable

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        return map(StepWorkElement, repeat(self.func), self.iterable.iter_steps())


@dataclass
class File(StepIterable):
    path: str
    steps: list[tuple[int, int]]

    def iter_steps(self) -> Iterator[Step]:
        return map(Step, repeat(self.path), self.steps)


@dataclass
class Dataset(StepIterable):
    files: list[File]
    traversal: Literal["depth", "breadth"] = "depth"
    """The traversal strategy for iterating over files and steps in the dataset.

    "depth" means to process all steps in one file before moving to the next file.
    "breadth" means to process the first step in all files, then the second step in all files, etc.
    """

    def iter_steps(self) -> Iterator[Step]:
        if self.traversal == "breadth":
            raise NotImplementedError
        return chain.from_iterable(map(File.iter_steps, self.files))


@dataclass
class ResumableComputation:
    original: Computable
    index: int

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        return islice(self.original, self.index, None)


@dataclass
class RetryableComputation:
    original: Computable
    failed_indices: frozenset[int]

    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        for i, task_element in enumerate(self.original):
            if i in self.failed_indices:
                yield task_element


class EmptyResult:
    def __add__(self, other: T, /) -> T:
        return other

    def __radd__(self, other: T, /) -> T:
        return other


@dataclass(slots=True)
class TaskElement:
    index: int
    func: Callable[[], ResultType]

    def __call__(self) -> ResultType:
        return self.func()


@dataclass(slots=True)
class FailedTaskElement(TaskElement):
    exception: Exception
    retries: int
    # TODO: last attempt time


ErrorAction = Literal["retry", "continue", "abort"]


@dataclass
class ErrorPolicy:
    fail_fast: bool = False
    """If True, abort the computation on the first error encountered."""
    # TODO: exception-type dependent policy, max retries, backoff

    def first_action(
        self, element: TaskElement, exception: Exception
    ) -> tuple[FailedTaskElement, ErrorAction]:
        """Determine action to take on first failure of a task element."""
        new_element = FailedTaskElement(
            index=element.index,
            func=element.func,
            exception=exception,
            retries=0,
        )
        if self.fail_fast:
            return new_element, "abort"
        return new_element, "continue"

    def retry_action(self, element: FailedTaskElement) -> ErrorAction:
        """Determine action to take on retry failure of a task element."""
        if self.fail_fast:
            return "abort"
        return "continue"


class SimpleTask:
    item: Computable
    error_policy: ErrorPolicy
    _iter: Iterator[TaskElement]
    _index: int
    "Index of the next task element to process, consistent with _output and _failures"
    _output: ResultType
    _failures: list[FailedTaskElement]
    _done: bool
    _cv: Condition

    def __init__(self, item: Computable) -> None:
        self.item = item
        self.error_policy = ErrorPolicy()
        self._iter = starmap(TaskElement, enumerate(item))
        self._index = 0
        self._output = EmptyResult()
        self._failures = []
        self._done = False
        self._cv = Condition()

    def result(self) -> ResultType:
        self.wait()
        if self._failures:
            # Reraise the first error encountered
            # TODO: friendly error explaining how to access partial results,
            # adjust error policy to continue on errors, etc.
            raise self._failures[0].exception from None
        return self._output

    def partial_result(self) -> tuple[ResultType, Computable]:
        with self._cv:
            if not self.done():
                if self._failures:
                    raise NotImplementedError(
                        "Partial result with errors not implemented"
                    )
                resumable = ResumableComputation(
                    original=self.item,
                    index=self._index,
                )
                return self._output, resumable
        return self._output, RetryableComputation(
            original=self.item,
            failed_indices=frozenset(element.index for element in self._failures),
        )

    def wait(self) -> None:
        with self._cv:
            self._cv.wait_for(self.done)

    def _set_done(self) -> None:
        with self._cv:
            self._done = True
            # only notification needed, since all waiters are in wait()
            self._cv.notify_all()

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        pass

    def _run(self) -> None:
        for task_element in self._iter:
            try:
                result = task_element()
            except Exception as ex:
                new_element, action = self.error_policy.first_action(task_element, ex)
                with self._cv:
                    if action == "abort":
                        self._failures.append(new_element)
                        self._index += 1
                        return self._set_done()
                    elif action == "retry":
                        raise NotImplementedError("Retry action not implemented")
                    elif action == "continue":
                        self._failures.append(new_element)
                        self._index += 1
                        continue
            else:
                with self._cv:
                    # This could use a more sophisticated merging strategy
                    self._output = self._output + result
                    self._index += 1
        return self._set_done()


def _work(task_queue: Queue[SimpleTask]) -> None:
    while True:
        try:
            task = task_queue.get()
        except ShutDown:
            break
        task._run()
        task_queue.task_done()


class SimpleBackend:
    task_queue: Queue[SimpleTask]
    _thread: Thread

    def __init__(self) -> None:
        self.task_queue = Queue()
        self._thread = Thread(
            target=_work,
            name="SingleThreadedBackend",
            args=(self.task_queue,),
            daemon=True,
        )
        self._thread.start()

    def __del__(self) -> None:
        self.task_queue.shutdown()
        self._thread.join()

    def compute(self, item: Computable) -> SimpleTask:
        task = SimpleTask(item)
        self.task_queue.put(task)
        return task


if __name__ == "__main__":
    # Stress test performance of compute with a dummy dataset and processor
    stepsize = 10_000
    steps_per_file = 100
    num_files = 10_000
    dataset = Dataset(
        files=[
            File(
                path=f"file{j:04d}.root",
                steps=[
                    (i, i + stepsize)
                    for i in range(0, stepsize * steps_per_file, stepsize)
                ],
            )
            for j in range(num_files)
        ]
    )

    class SimpleProcessor(ProcessorABC):
        def process(self, events: EventsArray) -> ResultType:
            return len(events)  # Dummy processing: return number of events

    processor = SimpleProcessor()

    # Binding the processor to the dataset makes the computable object not easy to serialize
    # but on the other hand, the ProcessableDataset has them separated out easily enough
    computable = dataset.apply(processor)

    backend = SimpleBackend()

    tic = time.monotonic()
    task = backend.compute(computable)
    task.wait()
    toc = time.monotonic()
    print(f"Computation took {toc - tic:.2f} seconds")
    print(f"Processed {num_files * steps_per_file} steps")
    assert task.result() == stepsize * steps_per_file * num_files

    # Try a partial result
    task = backend.compute(computable)
    dt = 0.4
    time.sleep(dt)
    part, resumable = task.partial_result()
    print(f"Partial result after {dt}s: {part}")
    print(f"Resumable computation has {len(list(resumable))} remaining steps")

    resumed_task = backend.compute(resumable)
    final_result = part + resumed_task.result()
    print(final_result)
    assert final_result == stepsize * steps_per_file * num_files

    class BuggyProcessor(ProcessorABC):
        def process(self, events: EventsArray) -> ResultType:
            if hash(events) % 100 == 0:
                raise RuntimeError("Simulated processing error")
            return len(events)

    buggy_computable = dataset.apply(BuggyProcessor())
    buggy_task = backend.compute(buggy_computable)
    buggy_task.result()
