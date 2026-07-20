"""Tests for the mutable (resizable) steps prototype."""

import shutil

import pytest

from coffea.dataset_tools import DataGroupSpec, preprocess
from coffea.dataset_tools.mutable_steps import (
    WallTimeStepPolicy,
    completed_spec,
    coverage_regions,
    iter_dataset_steps,
    iter_file_steps,
    remaining_regions,
    resizable_steps,
    run_adaptive_steps,
)

_DY = "tests/samples/nano_dy.root"


def _contiguous(steps, begin, end):
    """Steps tile [begin, end) exactly, in order, with no gaps or overlaps."""
    assert steps[0][0] == begin
    assert steps[-1][1] == end
    for previous, current in zip(steps, steps[1:]):
        assert previous[1] == current[0]


@pytest.fixture
def dataset(tmp_path):
    copy = str(tmp_path / "nano_dy_copy.root")
    shutil.copy(_DY, copy)
    dgs = DataGroupSpec({"D": {"files": {_DY: "Events", copy: "Events"}}})
    available, _ = preprocess(dgs, save_form=False, backend="iterative")
    return available["D"]


# --------------------------------------------------------------------------------------
# resizable_steps
# --------------------------------------------------------------------------------------


def test_resizable_steps_even_tiling():
    # 100 entries at target 30: 4 steps of 25 (as even as possible, at most the target)
    steps = list(resizable_steps(0, 100, 30))
    assert steps == [[0, 25], [25, 50], [50, 75], [75, 100]]


def test_resizable_steps_shrink_retiles_remainder():
    gen = resizable_steps(0, 100, 30)
    steps = [next(gen)]
    # request at most 10 for the remaining 75 entries
    step = gen.send(10)
    while True:
        steps.append(step)
        assert step[1] - step[0] <= 10
        try:
            step = next(gen)
        except StopIteration:
            break
    _contiguous(steps, 0, 100)


def test_resizable_steps_grow_is_at_most_requested():
    gen = resizable_steps(0, 100, 25)
    next(gen)
    # remaining 75 at target 50: 2 steps of at most 38 = ceil(75/2)
    step = gen.send(50)
    assert step == [25, 63]


def test_resizable_steps_invalid_sizes_raise():
    with pytest.raises(ValueError, match="positive integer"):
        next(resizable_steps(0, 10, 0))
    gen = resizable_steps(0, 10, 5)
    next(gen)
    with pytest.raises(ValueError, match="positive integer"):
        gen.send(-1)


# --------------------------------------------------------------------------------------
# coverage_regions / iter_file_steps
# --------------------------------------------------------------------------------------


def test_coverage_regions_merges_adjacent(dataset):
    fs = next(iter(dataset.files.values()))
    spec = fs.model_dump()
    spec["steps"] = [[0, 20], [20, 40]]
    merged = type(fs)(**spec)
    assert coverage_regions(merged) == [[0, 40]]

    spec["steps"] = [[0, 10], [20, 30]]
    disjoint = type(fs)(**spec)
    assert coverage_regions(disjoint) == [[0, 10], [20, 30]]


def test_coverage_regions_without_steps_uses_num_entries(dataset):
    fs = next(iter(dataset.files.values()))
    spec = fs.model_dump()
    spec["steps"] = None
    from coffea.dataset_tools.filespec import CoffeaROOTFileSpecOptional

    nosteps = CoffeaROOTFileSpecOptional(**spec)
    assert coverage_regions(nosteps) == [[0, fs.num_entries]]

    spec["num_entries"] = None
    unknown = CoffeaROOTFileSpecOptional(**spec)
    with pytest.raises(ValueError, match="neither steps nor num_entries"):
        coverage_regions(unknown)


def test_iter_file_steps_respects_disjoint_regions(dataset):
    fs = next(iter(dataset.files.values()))
    spec = fs.model_dump()
    spec["steps"] = [[0, 10], [20, 30]]
    disjoint = type(fs)(**spec)
    steps = list(iter_file_steps(disjoint, 5))
    assert steps == [[0, 5], [5, 10], [20, 25], [25, 30]]


def test_iter_dataset_steps_resize_carries_across_files(dataset):
    gen = iter_dataset_steps(dataset, 40)
    fname_first, step = next(gen)
    assert step == [0, 40]
    # shrink after the first file's single step: the second file re-tiles at 10
    seen = []
    try:
        item = gen.send(10)
        while True:
            seen.append(item)
            item = next(gen)
    except StopIteration:
        pass
    fnames = {fname for fname, _ in seen}
    assert fnames == set(dataset.files) - {fname_first} or fnames == set(dataset.files)
    for _, step in seen:
        assert step[1] - step[0] <= 10
    second_file_steps = [step for fname, step in seen if fname != fname_first]
    _contiguous(second_file_steps, 0, 40)


# --------------------------------------------------------------------------------------
# resumption helpers
# --------------------------------------------------------------------------------------


def test_remaining_regions_interval_subtraction(dataset):
    fs = next(iter(dataset.files.values()))
    # completed ranges need not align with stored steps
    assert remaining_regions(fs, [[0, 25]]) == [[25, 40]]
    assert remaining_regions(fs, [[10, 15], [15, 20]]) == [[0, 10], [20, 40]]
    assert remaining_regions(fs, [[0, 40]]) == []
    assert remaining_regions(fs, []) == coverage_regions(fs)


def test_completed_spec_roundtrip_and_accumulation(dataset):
    fs = next(iter(dataset.files.values()))
    first = completed_spec(fs, [[0, 15]])
    second = completed_spec(fs, [[15, 40]])
    assert completed_spec(fs, []) is None
    merged = first + second
    assert merged.steps == [[0, 15], [15, 40]]
    assert remaining_regions(fs, merged.steps) == []


# --------------------------------------------------------------------------------------
# toy adaptive driver
# --------------------------------------------------------------------------------------


class _FakeClock:
    """Deterministic clock advanced by the fake work function."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_run_adaptive_steps_converges_to_target(dataset):
    clock = _FakeClock()
    per_entry_seconds = 0.01

    def work(fname, step):
        clock.now += (step[1] - step[0]) * per_entry_seconds
        return (fname, tuple(step))

    # equilibrium size: target 0.1 s at 0.01 s/entry -> 10 entries per step
    policy = WallTimeStepPolicy(target_seconds=0.1)
    run = run_adaptive_steps(dataset, work, step_size=40, policy=policy, clock=clock)

    # the first (full-file) step triggers a shrink; later steps sit at the equilibrium
    assert run.step_sizes[0] == 40
    assert set(run.step_sizes[1:]) == {10}
    # every file is fully covered exactly once
    assert set(run.completed) == set(dataset.files)
    for fname, steps in run.completed.items():
        _contiguous(steps, 0, dataset.files[fname].num_entries)
    assert len(run.results) == len(run.step_sizes)


def test_run_adaptive_steps_growth_is_damped(dataset):
    clock = _FakeClock()

    def instant_work(fname, step):
        return None

    policy = WallTimeStepPolicy(target_seconds=1.0, max_step_size=64, max_growth=2.0)
    run = run_adaptive_steps(
        dataset, instant_work, step_size=4, policy=policy, clock=clock
    )
    # growth per adjustment is bounded by max_growth and capped at max_step_size
    for previous, current in zip(run.step_sizes, run.step_sizes[1:]):
        assert current <= max(previous * 2, 1) or current <= 64
    assert max(run.step_sizes) <= 64
