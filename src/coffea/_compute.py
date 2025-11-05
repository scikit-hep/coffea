from dataclasses import dataclass
from itertools import chain, islice, repeat
from queue import Queue, ShutDown
from threading import Condition, Thread
import time
from typing import Callable, Iterator, Literal, Protocol, Any, Sized, TypeVar

import numpy

T = TypeVar("T")


class ResultType(Protocol):
    def __add__(self: T, other: T, /) -> T:
        """Merge two results together."""
        ...


EventsArray = Sized
"Awkward array of events or similar"


class Processor(Protocol):
    def process(self, events: EventsArray) -> ResultType:
        """Process a chunk of events and return a result."""
        ...


class Computable(Protocol):
    def __iter__(self) -> Iterator[Callable[[], ResultType]]:
        """Return an iterator over callables that each compute a part of the result.

        This establishes a global index over the computation. We can use filters
        and slicing on this basis to create resumable and retryable computations.

        A potential optimization is to yield a tuple of (index, callable) so that
        the index can be smarter than just the integer position in the iterator.
        Then Computable could be a Container type.
        """
        ...


@dataclass(slots=True)
class ProcessStep:
    path: str
    step: tuple[int, int]
    processor: Processor

    def __call__(self) -> ResultType:
        """Load the events for this step and process them."""
        events = "A" * (self.step[1] - self.step[0])  # Dummy implementation
        return self.processor.process(events)


@dataclass
class File:
    path: str
    steps: list[tuple[int, int]]

    def apply(self, processor: Processor) -> "ProcessableFile":
        return ProcessableFile(file=self, processor=processor)


@dataclass(slots=True)
class ProcessableFile:
    file: File
    processor: Processor

    def __iter__(self):
        return map(
            ProcessStep, repeat(self.file.path), self.file.steps, repeat(self.processor)
        )


@dataclass
class Dataset:
    files: list[File]

    def apply(self, processor: Processor) -> "ProcessableDataset":
        return ProcessableDataset(dataset=self, processor=processor)


@dataclass(slots=True)
class ProcessableDataset:
    dataset: Dataset
    processor: Processor
    traversal: Literal["chain", "roundrobin"] = "chain"
    """The traversal strategy for iterating over files and steps in the dataset.

    "chain" means to process all steps in one file before moving to the next file.
    "roundrobin" means to process the first step in all files, then the second step in all files, etc.
    """

    def __iter__(self):
        if self.traversal == "roundrobin":
            raise NotImplementedError
        return chain.from_iterable(
            map(ProcessableFile, self.dataset.files, repeat(self.processor))
        )


class Task(Protocol):
    def result(self) -> ResultType:
        """Get the full final result of the computation."""
        ...

    def partial_result(self) -> tuple[ResultType, Computable]:
        """Get a partial result and the corresponding continuation computation.

        The partial result may either be because the task is not yet complete,
        or because the computation failed on a subset of the data.
        """
        ...

    def wait(self) -> None: ...

    def done(self) -> bool: ...

    def cancel(self) -> None: ...


class Backend(Protocol):
    def compute(self, item: Computable) -> Task:
        """Launch a computation and return a Task representing it.

        The backend holds any resources needed to perform the computation,
        such as a thread pool, process pool, or cluster connection. It should
        manage a queue of tasks and execute them in FIFO order.
        """
        ...


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


class SimpleTask:
    item: Computable
    _iter: Iterator[Callable[[], ResultType]]
    _index: int
    _output: ResultType
    _errors: list[tuple[int, Exception]]
    _done: bool
    _cv: Condition

    def __init__(self, item: Computable) -> None:
        self.item = item
        self._iter = iter(item)
        self._index = 0
        self._output = EmptyResult()
        self._errors = []
        self._done = False
        self._cv = Condition()

    def result(self) -> ResultType:
        self.wait()
        if self._errors:
            # Reraise the first error encountered
            raise self._errors[0][1] from None
        return self._output

    def partial_result(self) -> tuple[ResultType, Computable]:
        with self._cv:
            if not self.done():
                if self._errors:
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
            failed_indices=frozenset(idx for idx, _ in self._errors),
        )

    def wait(self) -> None:
        with self._cv:
            self._cv.wait_for(self.done)

    def done(self) -> bool:
        return self._done

    def cancel(self) -> None:
        pass

    def _run(self) -> None:
        for task_element in self._iter:
            result = task_element()
            with self._cv:
                self._index += 1
                # This could use a more sophisticated merging strategy
                self._output = self._output + result
                self._cv.notify_all()
        with self._cv:
            self._done = True
            self._cv.notify_all()


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


def default_backend() -> Backend:
    """Get the default computation backend."""
    return SimpleBackend()


def compute(item: Computable, backend: Backend = default_backend()) -> Task:
    """Compute the given item using the specified backend."""
    return backend.compute(item)


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

    class SimpleProcessor:
        def process(self, events: EventsArray) -> ResultType:
            return len(events)  # Dummy processing: return number of events

    processor = SimpleProcessor()

    # Binding the processor to the dataset makes the computable object not easy to serialize
    # but on the other hand, the ProcessableDataset has them separated out easily enough
    computable = dataset.apply(processor)

    tic = time.monotonic()
    task = compute(computable)
    task.wait()
    toc = time.monotonic()
    print(f"Computation took {toc - tic:.2f} seconds")
    print(f"Processed {num_files * steps_per_file} steps")
    assert task.result() == stepsize * steps_per_file * num_files

    # Try a partial result
    computable = dataset.apply(processor)
    task = compute(computable)
    time.sleep(0.4)
    part, resumable = task.partial_result()
    print(f"Partial result after 1s: {part}")
    print(f"Resumable computation has {len(list(resumable))} remaining steps")

    resumed_task = compute(resumable)
    final_result = part + resumed_task.result()
    assert final_result == stepsize * steps_per_file * num_files
