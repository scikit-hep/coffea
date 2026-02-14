import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from coffea.compute.backends.threaded import ThreadedBackend
from coffea.compute.data import ContextDataset, Dataset, File
from coffea.compute.errors import ErrorPolicy
from coffea.compute.func import EventsArray


class DummyProcessor:
    def process(self, events: EventsArray) -> int:
        return len(events)


class BuggyProcessor:
    def process(self, events: EventsArray) -> int:
        if hash(events) % 1000 == 0:
            raise ValueError("Simulated processing error")
        return len(events)


def test_threaded_backend_compute():
    "Stress test performance of compute with a dummy dataset and processor"
    stepsize = 10_000
    steps_per_file = 100
    num_files = 1_000
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
        ],
        metadata=ContextDataset(dataset_name="test_dataset", cross_section=1.0),
    )

    with ThreadedBackend(
        pool_factory=partial(ThreadPoolExecutor, max_workers=2)
    ) as backend:
        # Binding the processor to the dataset makes the computable object not easy to serialize
        computable = dataset.map_steps(DummyProcessor())
        tic = time.monotonic()
        task = backend.compute(computable)
        task.wait()
        toc = time.monotonic()
        print(f"Computation took {toc - tic:.2f} seconds")
        print(f"Processed {num_files * steps_per_file} steps with dummy processor")
        assert task.result() == stepsize * steps_per_file * num_files

        computable = dataset.map_steps(BuggyProcessor())
        tic = time.monotonic()
        task = backend.compute(
            computable, error_policy=ErrorPolicy(continue_on=(ValueError,))
        )
        task.wait()
        toc = time.monotonic()
        print(f"Computation took {toc - tic:.2f} seconds")
        print(f"Processed {num_files * steps_per_file} steps with buggy processor")
        part = task.partial_result()
        print(f"Partial result: {part}")


if __name__ == "__main__":
    # For use with profiling, e.g.
    # py-spy record -f speedscope python tests/test_compute_integration.py
    test_threaded_backend_compute()
    # On a M3 mac with py3.13, this is about 55k steps per second with
    # DummyProcessor and about 45k steps per second with BuggyProcessor
    # TODO: use pytest-benchmark to track this performance over time and catch regressions
