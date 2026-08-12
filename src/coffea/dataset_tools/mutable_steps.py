"""Prototype for mutable (resizable) steps: generator-driven step iteration where the
consumer may renegotiate the step size mid-stream.

Stored ``steps`` on a file spec are a static tiling chosen at preprocess time. Execution
backends with runtime resource information (e.g. a resource-monitoring scheduler that sees a
worker near memory exhaustion) can do better by shrinking or growing subsequent chunks while
a file is being processed. This module provides that as a ``send``-channel generator
protocol: iterating yields ``[start, stop]`` steps, and ``generator.send(new_size)`` requests
that subsequent steps be *at most* ``new_size`` entries, with the remainder of the current
region re-tiled as evenly as possible::

    n = ceil(remaining / new_size)
    actual = ceil(remaining / n)

so a resize triggered by resource exhaustion can never produce a larger-than-requested step,
and the remaining steps stay near-uniform. The same channel shape is used by
``Computable.gen_steps`` in the ``coffea.compute`` protocol, so consumers written against
this prototype translate directly.

Everything here is a prototype: APIs may change, and nothing is re-exported from
``coffea.dataset_tools``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

__all__ = [
    "resizable_steps",
    "coverage_regions",
    "iter_file_steps",
    "iter_dataset_steps",
    "remaining_regions",
    "completed_spec",
    "WallTimeStepPolicy",
    "run_adaptive_steps",
]


def _validate_size(size: int) -> None:
    if size is None or size < 1:
        raise ValueError(f"step size must be a positive integer (>= 1), got {size!r}.")


def resizable_steps(
    start: int, stop: int, step_size: int
) -> Generator[list[int], int | None, None]:
    """Yield ``[begin, end]`` steps tiling ``[start, stop)``, honoring resize requests.

    Steps are at most the current target size and as even as possible: each iteration
    re-tiles the remaining entries as ``ceil(remaining / ceil(remaining / size))``, which is
    stable while the size is unchanged. Sending a positive integer sets the target size for
    all subsequent steps; sending ``None`` (plain iteration) keeps the current size.
    """
    _validate_size(step_size)
    current = step_size
    pos = start
    while pos < stop:
        remaining = stop - pos
        n_steps = math.ceil(remaining / current)
        actual = math.ceil(remaining / n_steps)
        end = min(pos + actual, stop)
        sent = yield [pos, end]
        pos = end
        if sent is not None:
            _validate_size(sent)
            current = sent


def coverage_regions(filespec) -> list[list[int]]:
    """The contiguous entry regions a file spec covers, as ``[begin, end]`` pairs.

    Adjacent steps (``stop == next start``) merge into one region. A spec without steps
    covers ``[0, num_entries)``; a spec with neither steps nor num_entries cannot be tiled
    and raises a ValueError.
    """
    if filespec.steps is not None:
        regions: list[list[int]] = []
        for begin, end in filespec.steps:
            if regions and regions[-1][1] == begin:
                regions[-1][1] = end
            else:
                regions.append([begin, end])
        return regions
    if filespec.num_entries is not None:
        return [[0, filespec.num_entries]] if filespec.num_entries > 0 else []
    raise ValueError(
        "Cannot iterate steps for a file spec with neither steps nor num_entries; "
        "preprocess the file first."
    )


def iter_file_steps(
    filespec, step_size: int | None = None
) -> Generator[list[int], int | None, None]:
    """Yield resizable ``[start, stop]`` steps over a file spec's covered regions.

    ``step_size`` is the initial target size (default: one step per contiguous region).
    ``send(new_size)`` re-tiles from the next step onward, carrying across regions.
    """
    current = step_size
    for begin, end in coverage_regions(filespec):
        size = current if current is not None else end - begin
        gen = resizable_steps(begin, end, size)
        try:
            step = next(gen)
            while True:
                sent = yield step
                if sent is not None:
                    current = sent
                step = gen.send(sent)
        except StopIteration:
            continue


def iter_dataset_steps(
    dataset, step_size: int | None = None
) -> Generator[tuple[str, list[int]], int | None, None]:
    """Yield resizable ``(filename, [start, stop])`` steps over every file of a DatasetSpec.

    A resize request applies from the next step onward and carries across file boundaries,
    so a consumer that shrinks the step size mid-file keeps the smaller size for later files
    until it requests otherwise.
    """
    current = step_size
    for fname, filespec in dataset.files.items():
        gen = iter_file_steps(filespec, current)
        try:
            step = next(gen)
            while True:
                sent = yield (fname, step)
                if sent is not None:
                    current = sent
                step = gen.send(sent)
        except StopIteration:
            continue


def remaining_regions(filespec, completed: list[list[int]]) -> list[list[int]]:
    """The parts of a file spec's coverage not contained in ``completed`` ranges.

    ``completed`` are arbitrary ``[start, stop]`` pairs (they need not align with the spec's
    stored steps -- resized steps generally do not). Returns ``[begin, end]`` regions suitable
    for resuming iteration via :func:`resizable_steps`.
    """
    merged: list[list[int]] = []
    for begin, end in sorted([list(pair) for pair in completed]):
        if merged and begin <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([begin, end])
    remaining: list[list[int]] = []
    for begin, end in coverage_regions(filespec):
        pos = begin
        for done_begin, done_end in merged:
            if done_end <= pos or done_begin >= end:
                continue
            if done_begin > pos:
                remaining.append([pos, done_begin])
            pos = max(pos, done_end)
            if pos >= end:
                break
        if pos < end:
            remaining.append([pos, end])
    return remaining


def completed_spec(filespec, completed: list[list[int]]):
    """A copy of ``filespec`` whose steps are the ``completed`` ranges.

    Useful for persisting progress: the result round-trips like any spec, and adding it to a
    spec built from later completions accumulates coverage via the ordinary step arithmetic.
    Returns ``None`` when nothing has completed.
    """
    if not completed:
        return None
    spec = filespec.model_dump()
    spec["steps"] = sorted([list(pair) for pair in completed])
    return type(filespec)(**spec)


@dataclass
class WallTimeStepPolicy:
    """Toy resize policy targeting a fixed wall time per step.

    After each step, the next target size is ``current * target_seconds / elapsed``, clamped
    to ``[min_step_size, max_step_size]`` and damped by ``max_growth`` per adjustment so a
    single fast outlier does not balloon the step size.
    """

    target_seconds: float
    min_step_size: int = 1
    max_step_size: int | None = None
    max_growth: float = 2.0

    def propose(self, current_size: int, elapsed_seconds: float) -> int | None:
        """The new target size, or None to keep the current one."""
        if elapsed_seconds <= 0.0:
            scaled = current_size * self.max_growth
        else:
            scaled = current_size * self.target_seconds / elapsed_seconds
        scaled = min(scaled, current_size * self.max_growth)
        new_size = max(self.min_step_size, int(scaled))
        if self.max_step_size is not None:
            new_size = min(new_size, self.max_step_size)
        return None if new_size == current_size else new_size


@dataclass
class AdaptiveRun:
    """Result of :func:`run_adaptive_steps`."""

    results: list
    completed: dict[str, list[list[int]]]
    step_sizes: list[int] = field(default_factory=list)


def run_adaptive_steps(
    dataset,
    work: Callable[[str, list[int]], object],
    step_size: int,
    policy: WallTimeStepPolicy,
    clock: Callable[[], float] = time.monotonic,
) -> AdaptiveRun:
    """Toy driver: process a DatasetSpec step by step, resizing steps from measured wall time.

    ``work(filename, [start, stop])`` is called for each step; its wall time (measured with
    ``clock``) feeds ``policy.propose``, and any proposed size is sent into the step
    generator, re-tiling the remainder. This is the single-consumer analogue of what a
    resource-monitoring scheduler does with worker feedback; ``clock`` is injectable so the
    control loop is testable without real waiting.
    """
    _validate_size(step_size)
    gen = iter_dataset_steps(dataset, step_size)
    run = AdaptiveRun(results=[], completed={})
    current = step_size
    try:
        item = next(gen)
        while True:
            fname, step = item
            begin = clock()
            run.results.append(work(fname, step))
            elapsed = clock() - begin
            run.completed.setdefault(fname, []).append(step)
            run.step_sizes.append(step[1] - step[0])
            proposed = policy.propose(current, elapsed)
            if proposed is not None:
                current = proposed
                item = gen.send(proposed)
            else:
                item = next(gen)
    except StopIteration:
        pass
    return run
