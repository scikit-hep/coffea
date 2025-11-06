import time

from coffea.compute import (
    Dataset,
    EventsArray,
    File,
    ProcessorABC,
    ResultType,
)
from coffea.compute.backends.threaded import SingleThreadedBackend
from coffea.compute.errors import ErrorPolicy


class DummyProcessor(ProcessorABC):
    def process(self, events: EventsArray) -> ResultType:
        return len(events)


class BuggyProcessor(ProcessorABC):
    def process(self, events: EventsArray) -> ResultType:
        if hash(events) % 100 == 0:
            raise ValueError("Simulated processing error")
        return len(events)


def test_threaded_backend_compute():
    "Stress test performance of compute with a dummy dataset and processor"
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

    with SingleThreadedBackend() as backend:
        # Binding the processor to the dataset makes the computable object not easy to serialize
        # but on the other hand, the ProcessableDataset has them separated out easily enough
        computable = dataset.apply(DummyProcessor())
        tic = time.monotonic()
        task = backend.compute(computable)
        task.wait()
        toc = time.monotonic()
        print(f"Computation took {toc - tic:.2f} seconds")
        print(f"Processed {num_files * steps_per_file} steps with dummy processor")
        assert task.result() == stepsize * steps_per_file * num_files

        computable = dataset.apply(BuggyProcessor())
        tic = time.monotonic()
        task = backend.compute(
            computable, error_policy=ErrorPolicy(continue_on=(ValueError,))
        )
        task.wait()
        toc = time.monotonic()
        print(f"Computation took {toc - tic:.2f} seconds")
        print(f"Processed {num_files * steps_per_file} steps with buggy processor")
        part, resumable = task.partial_result()
        print(f"Partial result: {part}, resumable length: {len(list(resumable))}")


if __name__ == "__main__":
    # For use with profiling, e.g.
    # py-spy record -f speedscope tests/test_compute_integration.py
    test_threaded_backend_compute()
