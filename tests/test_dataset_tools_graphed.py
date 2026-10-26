import contextlib
import dataclasses
import pickle
import subprocess
import sys
import textwrap

import awkward as ak
import numpy as np
import pytest
import uproot

from coffea.dataset_tools import apply_to_dataset, apply_to_fileset, preprocess
from coffea.dataset_tools.filespec import DataGroupSpec
from coffea.nanoevents import BaseSchema
from coffea.util import decompress_form

pytest.importorskip("graphed")
from graphed.core.execution import SequentialRunner  # noqa: E402

N = 20


def _write_flag_files(tmp_path):
    """ROOT and parquet pairs where only the first file of each pair has a bool ``flag``."""
    has_root, missing_root = str(tmp_path / "has.root"), str(tmp_path / "missing.root")
    with uproot.recreate(has_root) as f:
        f.mktree("Events", {"x": np.float32, "flag": np.bool_})
        f["Events"].extend(
            {"x": np.arange(N, dtype=np.float32), "flag": np.ones(N, dtype=bool)}
        )
    with uproot.recreate(missing_root) as f:
        f.mktree("Events", {"x": np.float32})
        f["Events"].extend({"x": np.arange(N, dtype=np.float32) + 100})
    has_pq, missing_pq = str(tmp_path / "has.parquet"), str(
        tmp_path / "missing.parquet"
    )
    ak.to_parquet(
        ak.zip({"x": np.arange(N, dtype=np.float32), "flag": np.ones(N, dtype=bool)}),
        has_pq,
    )
    ak.to_parquet(ak.zip({"x": np.arange(N, dtype=np.float32) + 100}), missing_pq)
    return has_root, missing_root, has_pq, missing_pq


class _PicklingRunner:
    """A graphed executor that ships the plan through pickle, as a process pool does."""

    def __init__(self):
        self.n_tasks = 0

    def run(self, plan):
        self.n_tasks += len(plan.tasks)
        return SequentialRunner().run(pickle.loads(pickle.dumps(plan)))


def _root_fileset(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    bad = str(tmp_path / "does_not_exist.root")
    return {
        "union": {"files": {has_root: "Events", missing_root: "Events", bad: "Events"}},
        "single": {"files": {has_root: "Events"}},
    }


@pytest.mark.parametrize("files_per_batch", [1, 2])
@pytest.mark.parametrize("save_form", [False, True])
@pytest.mark.parametrize("fileset_kind", ["legacy", "dict", "mixed"])
def test_preprocess_graphed_matches_dask(
    tmp_path, fileset_kind, save_form, files_per_batch
):
    pytest.importorskip("dask_awkward")
    fileset = _root_fileset(tmp_path)
    kwargs = dict(
        step_size=7,
        files_per_batch=files_per_batch,
        skip_bad_files=True,
        save_form=save_form,
        preprocess_legacy_root=fileset_kind == "legacy",
    )
    if fileset_kind == "mixed":
        _, _, has_pq, missing_pq = _write_flag_files(tmp_path)
        fileset["parquet"] = {"files": {has_pq: None, missing_pq: None}}
        fileset = DataGroupSpec(fileset)

    with (
        pytest.warns(DeprecationWarning)
        if fileset_kind == "dict"
        else contextlib.nullcontext()
    ):
        dask_out = preprocess(fileset, scheduler="sync", **kwargs)
    with (
        pytest.warns(DeprecationWarning)
        if fileset_kind == "dict"
        else contextlib.nullcontext()
    ):
        runner = _PicklingRunner()
        graphed_out = preprocess(fileset, scheduler=runner, backend="graphed", **kwargs)

    assert graphed_out == dask_out
    # one task per batch: union has 3 files, single 1, parquet 2
    n_files = [3, 1] + ([2] if fileset_kind == "mixed" else [])
    assert runner.n_tasks == sum(-(-n // files_per_batch) for n in n_files)

    available = graphed_out[0]
    union = available["union"]
    assert len(union["files"] if isinstance(union, dict) else union.files) == 2
    if save_form:
        datasets = ["union"] + (["parquet"] if fileset_kind == "mixed" else [])
        for name in datasets:
            ds = available[name]
            if isinstance(ds, dict):
                form = decompress_form(ds.get("form") or ds["compressed_form"])
            else:
                form = ds.form
            assert "IndexedOption" in str(form)


def _preprocessed_graphed(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    available, _ = preprocess(
        DataGroupSpec({"ds": {"files": {has_root: "Events", missing_root: "Events"}}}),
        step_size=7,
        save_form=True,
        backend="graphed",
    )
    return available


def test_apply_to_dataset_graphed_union_form_maybe_missing_branch(tmp_path):
    dataset = _preprocessed_graphed(tmp_path)["ds"]
    assert "IndexedOption" in str(dataset.form)

    (flag,) = apply_to_dataset(
        lambda events: events["flag"], dataset, schemaclass=BaseSchema, mode="graphed"
    )
    values = flag.session.materialize(flag)
    # file order is has, missing: real values first, then the absent branch as None
    assert ak.to_list(values) == [True] * N + [None] * N


def test_apply_to_dataset_graphed_refuses_read_error_report(tmp_path):
    dataset = _preprocessed_graphed(tmp_path)["ds"]
    with pytest.raises(NotImplementedError, match="graphed mode has no report tuple"):
        apply_to_dataset(
            lambda events: events["x"],
            dataset,
            schemaclass=BaseSchema,
            uproot_options={"allow_read_errors_with_report": True},
            mode="graphed",
        )


def test_apply_to_fileset_graphed(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    available, _ = preprocess(
        DataGroupSpec(
            {
                "has": {"files": {has_root: "Events"}, "metadata": {"tag": "A"}},
                "missing": {"files": {missing_root: "Events"}},
            }
        ),
        step_size=7,
        backend="graphed",
    )

    def analysis(events):
        return {"x": events["x"], "dataset": events.metadata["dataset"]}

    out = apply_to_fileset(analysis, available, schemaclass=BaseSchema, mode="graphed")
    assert list(out) == ["has", "missing"]
    for name, offset in [("has", 0), ("missing", 100)]:
        x = out[name]["x"]
        assert ak.to_list(x.session.materialize(x)) == list(range(offset, offset + N))
        assert out[name]["dataset"] == name


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda fs: preprocess(fs, backend="eager"), "backend must be"),
        (lambda fs: apply_to_fileset(lambda e: e, fs, mode="eager"), "mode must be"),
    ],
)
def test_dataset_tools_refuse_unknown_mode(tmp_path, call, match):
    has_root, *_ = _write_flag_files(tmp_path)
    with pytest.raises(ValueError, match=match):
        call(DataGroupSpec({"ds": {"files": {has_root: "Events"}}}))


def test_preprocess_graphed_refuses_dask_scheduler(tmp_path):
    has_root, *_ = _write_flag_files(tmp_path)
    with pytest.raises(TypeError, match="graphed executor"):
        preprocess(
            DataGroupSpec({"ds": {"files": {has_root: "Events"}}}),
            scheduler="sync",
            backend="graphed",
        )


def test_dataset_tools_graphed_without_dask(tmp_path):
    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    script = textwrap.dedent(f"""
        import sys

        class _BlockDask:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("dask", "dask_awkward", "distributed"):
                    raise ImportError(f"blocked {{name}}")

        sys.meta_path.insert(0, _BlockDask())
        import awkward as ak
        from coffea.dataset_tools import apply_to_dataset, preprocess
        from coffea.dataset_tools.filespec import DataGroupSpec
        from coffea.nanoevents import BaseSchema

        files = {{{has_root!r}: "Events", {missing_root!r}: "Events"}}
        available, _ = preprocess(
            DataGroupSpec({{"ds": {{"files": files}}}}), save_form=True, backend="graphed"
        )
        (flag,) = apply_to_dataset(
            lambda e: e["flag"], available["ds"], schemaclass=BaseSchema, mode="graphed"
        )
        values = ak.to_list(flag.session.materialize(flag))
        assert not any(m.split(".")[0] in ("dask", "dask_awkward") for m in sys.modules)
        print(values.count(True), values.count(None))
        """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [str(N), str(N)]


class _ReverseOrderRunner:
    """Runs a plan's tasks, and so combines its batches, in reverse key order."""

    def run(self, plan):
        from graphed.core.execution import Task

        n = len(plan.tasks)
        tasks = [Task(n - 1 - t.key, t.partition) for t in plan.tasks]
        return SequentialRunner().run(dataclasses.replace(plan, tasks=tasks))


def test_preprocess_graphed_does_not_depend_on_combine_order(tmp_path):
    """With files_per_batch=1 and a skipped file first, results match dask whatever order
    the executor combines the batches in, for DataGroupSpec and dict filesets."""
    from coffea.dataset_tools import preprocess_legacy

    has_root, missing_root, _, _ = _write_flag_files(tmp_path)
    absent = str(tmp_path / "does_not_exist.root")
    files = {absent: "Events", has_root: "Events", missing_root: "Events"}
    kwargs = dict(step_size=7, save_form=True, skip_bad_files=True)

    spec = DataGroupSpec({"ds": {"files": files}})
    dask_out = preprocess(spec, scheduler="sync", **kwargs)
    graphed_out = preprocess(
        spec, backend="graphed", scheduler=_ReverseOrderRunner(), **kwargs
    )
    assert graphed_out == dask_out
    assert list(graphed_out[1]["ds"].files) == list(files)

    legacy = {"ds": {"files": files}}
    assert preprocess_legacy(
        legacy, backend="graphed", scheduler=_ReverseOrderRunner(), **kwargs
    ) == preprocess_legacy(legacy, scheduler="sync", **kwargs)
