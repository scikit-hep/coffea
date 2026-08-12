"""Tests for the switchable preprocessing backends in coffea.dataset_tools.preprocess_backends."""

import shutil

import awkward
import pytest

from coffea.dataset_tools import (
    DaskBackend,
    DataGroupSpec,
    FuturesBackend,
    IterativeBackend,
    PreprocessBackend,
    PreprocessJob,
    preprocess,
)
from coffea.dataset_tools.preprocess_backends import (
    _iter_batches,
    ordered_concat,
    resolve_backend,
)


def _multi_file_fileset(tmp_path):
    """A single dataset with two identical-form ROOT files (so form union is trivial)."""
    src = "tests/samples/nano_dy.root"
    copy = tmp_path / "nano_dy_copy.root"
    shutil.copy(src, copy)
    return DataGroupSpec(
        {
            "ZJets": {"files": {src: "Events", str(copy): "Events"}},
            "ZJets2": {"files": {src: "Events"}},
        }
    )


# --------------------------------------------------------------------------------------
# ordered_concat / _iter_batches unit tests
# --------------------------------------------------------------------------------------


def test_ordered_concat_preserves_order():
    a = awkward.Array([1, 2])
    b = awkward.Array([3, 4])
    c = awkward.Array([5])
    assert ordered_concat([a, b, c]).to_list() == [1, 2, 3, 4, 5]


def test_ordered_concat_drops_none_and_handles_empty():
    a = awkward.Array([1, 2])
    assert ordered_concat([None, a, None]).to_list() == [1, 2]
    # single element is returned as-is
    assert ordered_concat([a]) is a
    # nothing to concatenate
    assert ordered_concat([]) is None
    assert ordered_concat([None, None]) is None


def test_iter_batches_slices_in_order():
    arr = awkward.Array([{"x": i} for i in range(5)])
    batches = list(_iter_batches(arr, files_per_batch=2))
    assert [b.x.to_list() for b in batches] == [[0, 1], [2, 3], [4]]


def test_iter_batches_empty_yields_once():
    arr = awkward.Array([{"x": 1}])[0:0]
    batches = list(_iter_batches(arr, files_per_batch=1))
    assert len(batches) == 1
    assert len(batches[0]) == 0


# --------------------------------------------------------------------------------------
# resolve_backend
# --------------------------------------------------------------------------------------


def test_resolve_backend_strings():
    assert isinstance(resolve_backend("dask"), DaskBackend)
    assert isinstance(resolve_backend(None), DaskBackend)
    assert isinstance(resolve_backend("iterative"), IterativeBackend)
    assert isinstance(resolve_backend("futures"), FuturesBackend)


def test_resolve_backend_instance_passthrough():
    inst = IterativeBackend()
    assert resolve_backend(inst) is inst


def test_resolve_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown preprocessing backend"):
        resolve_backend("nonsense")


def test_resolve_backend_scheduler_warns_for_non_dask_string():
    with pytest.warns(UserWarning, match="only affects the dask backend"):
        resolve_backend("iterative", scheduler="synchronous")


def test_resolve_backend_scheduler_injected_into_dask_instance(recwarn):
    """A DaskBackend instance with no scheduler of its own receives the scheduler argument
    (as a copy; the original instance is not mutated), without warning."""
    inst = DaskBackend(split_every=4)
    resolved = resolve_backend(inst, scheduler="synchronous")
    assert resolved.scheduler == "synchronous"
    assert resolved.split_every == 4
    assert inst.scheduler is None
    assert not any("ignored" in str(w.message) for w in recwarn)


def test_resolve_backend_scheduler_warns_for_instance():
    # scheduler cannot be injected into a non-dask instance or a DaskBackend that already
    # carries its own scheduler, so it warns instead of silently dropping the argument
    with pytest.warns(UserWarning, match="ignored when a PreprocessBackend instance"):
        resolve_backend(FuturesBackend(), scheduler="synchronous")
    with pytest.warns(UserWarning, match="ignored when a PreprocessBackend instance"):
        inst = DaskBackend(scheduler="threads")
        assert resolve_backend(inst, scheduler="synchronous") is inst


def test_resolve_backend_scheduler_no_warn_for_dask_string(recwarn):
    resolve_backend("dask", scheduler="synchronous")
    assert not any("ignored" in str(w.message) for w in recwarn)


# --------------------------------------------------------------------------------------
# Backend equivalence (dask-free backends do not import dask)
# --------------------------------------------------------------------------------------


def test_iterative_and_threads_agree(tmp_path):
    dgs = _multi_file_fileset(tmp_path)
    a_iter, u_iter = preprocess(dgs, step_size=7, save_form=True, backend="iterative")
    a_thr, u_thr = preprocess(
        dgs, step_size=7, save_form=True, backend=FuturesBackend(workers=2)
    )
    # DatasetSpec.__eq__ compares decoded forms (ignoring non-deterministic compressed bytes)
    assert a_iter == a_thr
    assert u_iter == u_thr
    # order of files within a dataset is preserved by the ordered reduce
    src = "tests/samples/nano_dy.root"
    assert list(a_iter["ZJets"].files)[0] == src


def test_iterative_preserves_steps_and_form(tmp_path):
    dgs = _multi_file_fileset(tmp_path)
    available, _ = preprocess(dgs, step_size=7, save_form=True, backend="iterative")
    ds = available["ZJets"]
    assert len(ds.files) == 2
    for fs in ds.files.values():
        assert fs.num_entries == 40
        assert fs.uuid is not None
        assert fs.steps[0][0] == 0
    assert ds.compressed_form is not None


def test_dask_matches_dask_free(tmp_path):
    pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    dgs = _multi_file_fileset(tmp_path)
    a_dask, u_dask = preprocess(
        dgs, step_size=7, save_form=True, backend="dask", scheduler="synchronous"
    )
    a_iter, u_iter = preprocess(dgs, step_size=7, save_form=True, backend="iterative")
    assert a_dask == a_iter
    assert u_dask == u_iter


# --------------------------------------------------------------------------------------
# dask-unavailable fallback hint
# --------------------------------------------------------------------------------------


def test_dask_import_failure_prints_hint(tmp_path, monkeypatch):
    """When the default dask backend can't import dask, a hint about the dask-free
    backends is emitted before the ModuleNotFoundError propagates."""
    import importlib

    backends_mod = importlib.import_module("coffea.dataset_tools.preprocess_backends")
    # the package re-exports the `preprocess` function, shadowing the submodule attribute,
    # so fetch the module object explicitly rather than via attribute access
    preprocess_mod = importlib.import_module("coffea.dataset_tools.preprocess")

    def _boom():
        raise ModuleNotFoundError("no dask here")

    monkeypatch.setattr(backends_mod, "_import_dask", _boom)

    called = {"hint": False}
    orig_hint = preprocess_mod.print_dask_backend_fallback_hint

    def _hint():
        called["hint"] = True
        orig_hint()

    monkeypatch.setattr(preprocess_mod, "print_dask_backend_fallback_hint", _hint)

    dgs = _multi_file_fileset(tmp_path)
    with pytest.raises(ModuleNotFoundError):
        preprocess(dgs, step_size=7, save_form=True, backend="dask")
    assert called["hint"] is True


def test_worker_modulenotfound_does_not_trigger_dask_hint(tmp_path, monkeypatch):
    """A ModuleNotFoundError raised by a worker during compute (surfaced in Task.result())
    must NOT print the 'dask could not be imported' hint -- only a failure to import dask in
    submit() should."""
    import importlib

    preprocess_mod = importlib.import_module("coffea.dataset_tools.preprocess")

    class _RaisingTask:
        def result(self):
            raise ModuleNotFoundError("missing xrootd inside a worker")

    class _FakeDaskBackend(DaskBackend):
        def submit(self, jobs):  # submit succeeds; the error is deferred to result()
            return _RaisingTask()

    called = {"hint": False}
    monkeypatch.setattr(
        preprocess_mod,
        "print_dask_backend_fallback_hint",
        lambda: called.__setitem__("hint", True),
    )
    monkeypatch.setattr(
        preprocess_mod, "resolve_backend", lambda b, s=None: _FakeDaskBackend()
    )

    dgs = _multi_file_fileset(tmp_path)
    with pytest.raises(ModuleNotFoundError, match="missing xrootd"):
        preprocess(dgs, step_size=7, save_form=True, backend="dask")
    assert called["hint"] is False


def test_empty_parquet_file_does_not_crash(tmp_path):
    """A 0-row parquet file must yield steps=[[0, 0]] instead of a ZeroDivisionError,
    mirroring the ROOT get_steps num_entries==0 guard."""
    from coffea.dataset_tools import preprocess_parquet

    path = tmp_path / "empty.parquet"
    awkward.to_parquet(awkward.Array([{"x": 1.0, "y": 2}])[0:0], str(path))
    dgs = DataGroupSpec({"E": {"files": {str(path): None}}})

    # recalculate_steps=True forces the step-computation branch
    _available, updated = preprocess_parquet(
        dgs, recalculate_steps=True, save_form=True, backend="iterative"
    )
    fs = next(iter(updated["E"].files.values()))
    assert fs.num_entries == 0
    assert fs.steps == [[0, 0]]


def test_fallback_hint_mentions_backends(capsys):
    from coffea.dataset_tools.preprocess_backends import (
        print_dask_backend_fallback_hint,
    )

    print_dask_backend_fallback_hint()
    out = capsys.readouterr().out
    assert "iterative" in out
    assert "futures" in out


@pytest.mark.parametrize(
    "sample, is_rntuple",
    [
        ("tests/samples/nano_dy.root", False),
        ("tests/samples/nano_dy_rntuple.root", True),
    ],
)
def test_awkward_form_json_matches_uproot_dask(sample, is_rntuple):
    """The dask-free form builder must stay byte-identical to uproot.dask's form for both TTree
    and RNTuple, otherwise the iterative/futures backends would store forms incompatible with the
    dask read path used at analysis time."""
    pytest.importorskip("dask")
    pytest.importorskip("dask_awkward")
    from functools import partial

    import uproot
    from uproot._util import no_filter

    from coffea.dataset_tools.preprocess import _FORM_AK_ADD_DOC, _awkward_form_json
    from coffea.util import _is_interpretable

    tree = uproot.open({sample: None})["Events"]
    filt = partial(_is_interpretable, emit_warning=False)
    if is_rntuple:
        # RNTuples cannot build a form from an already-open object via uproot.dask; pass the spec
        dask_form = uproot.dask(
            {sample: "Events"},
            open_files=False,
            full_paths=True,
            ak_add_doc=_FORM_AK_ADD_DOC,
            filter_name=no_filter,
            filter_typename=no_filter,
            filter_branch=filt,
        ).layout.form.to_json()
    else:
        dask_form = uproot.dask(
            tree,
            ak_add_doc=_FORM_AK_ADD_DOC,
            filter_name=no_filter,
            filter_typename=no_filter,
            filter_branch=filt,
        ).layout.form.to_json()
    assert _awkward_form_json(tree, is_rntuple) == dask_form


def test_step_size_zero_or_negative_raises(tmp_path):
    """step_size < 1 is rejected with a clear ValueError at the entrypoint rather than a bare
    ZeroDivisionError from deep inside a worker."""
    dgs = _multi_file_fileset(tmp_path)
    for bad in (0, -5):
        with pytest.raises(ValueError, match="step_size must be a positive integer"):
            preprocess(dgs, step_size=bad, save_form=False, backend="iterative")
    # None and >= 1 are accepted (smoke: no raise)
    preprocess(dgs, step_size=1, save_form=False, backend="iterative")


def test_step_size_zero_raises_in_legacy_path():
    """The legacy path validates step_size too, before importing dask or opening files."""
    from coffea.dataset_tools import preprocess_legacy

    with pytest.raises(ValueError, match="step_size must be a positive integer"):
        preprocess_legacy(
            {"ds": {"files": {"nonexistent.root": "Events"}}}, step_size=0
        )


def test_backends_are_preprocessbackend_instances():
    assert isinstance(DaskBackend(), PreprocessBackend)
    assert isinstance(IterativeBackend(), PreprocessBackend)
    assert isinstance(FuturesBackend(), PreprocessBackend)


def test_skipped_bad_file_assembled_by_filename(tmp_path):
    """A skipped bad file is absent from `available` but retained in `updated` with its original
    input info, assembled by filename in the original input order (no positional zip).
    """
    good = "tests/samples/nano_dy.root"
    bad = str(tmp_path / "does_not_exist.root")
    dgs = DataGroupSpec({"ZJets": {"files": {bad: "Events", good: "Events"}}})
    available, updated = preprocess(
        dgs, step_size=20, save_form=True, skip_bad_files=True, backend="iterative"
    )
    # updated keeps both files, in the original input order; available drops the bad one
    assert list(updated["ZJets"].files) == [bad, good]
    assert bad not in available["ZJets"].files
    assert good in available["ZJets"].files


def test_partial_result_equals_result_when_complete():
    """partial_result() agrees with result() once all work has finished, for the eager and
    futures backends."""
    arr = awkward.Array([{"x": i} for i in range(4)])
    jobs = {
        "d": PreprocessJob(array=arr, map_fn=lambda batch: batch, files_per_batch=1)
    }
    for backend in (IterativeBackend(), FuturesBackend(workers=2)):
        task = backend.submit(jobs)
        task.wait()
        partial = task.partial_result()
        result = task.result()
        assert result["d"].to_list() == arr.to_list()
        assert partial["d"].to_list() == result["d"].to_list()


def test_futures_default_workers_uses_executor_default():
    """FuturesBackend defaults workers to None, deferring pool sizing to the executor
    (parallel by default), including via the string selector."""
    assert FuturesBackend().workers is None
    assert resolve_backend("futures").workers is None


def test_futures_string_backend_matches_iterative(tmp_path):
    dgs = _multi_file_fileset(tmp_path)
    a_fut, u_fut = preprocess(dgs, step_size=7, save_form=True, backend="futures")
    a_iter, u_iter = preprocess(dgs, step_size=7, save_form=True, backend="iterative")
    assert a_fut == a_iter
    assert u_fut == u_iter


def test_futures_result_failure_cancels_pending():
    """When a batch fails, result() re-raises after cancelling batches that have not started,
    so a fatal error does not wait for the rest of the fileset to be processed."""
    import threading
    from concurrent.futures import Future, ThreadPoolExecutor

    from coffea.dataset_tools.preprocess_backends import _FuturesTask

    release = threading.Event()
    ran = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(release.wait)  # occupies the only worker
        pending = pool.submit(ran.set)  # queued behind the blocker

        failed = Future()
        failed.set_exception(ValueError("boom"))

        task = _FuturesTask({"d": [failed, pending]}, pool, owns_pool=True)
        with pytest.raises(ValueError, match="boom"):
            task.result()
        assert pending.cancelled()
        assert not ran.is_set()
    finally:
        release.set()


def test_preprocess_rntuple_skips_ttree_file_when_requested():
    """With require_rntuple, a TTree object raises a ValueError that participates in
    skip_bad_files/file_exceptions like any other per-file error."""
    from coffea.dataset_tools import preprocess_rntuple

    rnt = "tests/samples/nano_dy_rntuple.root"
    ttree = "tests/samples/nano_dy.root"
    dgs = DataGroupSpec({"D": {"files": {rnt: "Events", ttree: "Events"}}})

    with pytest.raises(ValueError, match="not an RNTuple"):
        preprocess_rntuple(dgs, save_form=False, backend="iterative")

    available, updated = preprocess_rntuple(
        dgs,
        save_form=False,
        backend="iterative",
        skip_bad_files=True,
        file_exceptions=(OSError, ValueError),
    )
    assert list(available["D"].files) == [rnt]
    assert list(updated["D"].files) == [rnt, ttree]


def test_preprocess_rntuple_rejects_parquet(tmp_path):
    """require_rntuple rejects parquet-format datasets with a clear error."""
    from coffea.dataset_tools import preprocess_rntuple

    path = tmp_path / "d.parquet"
    awkward.to_parquet(awkward.Array([{"x": 1.0}]), str(path))
    dgs = DataGroupSpec({"P": {"files": {str(path): None}}})
    with pytest.raises(ValueError, match="parquet-format"):
        preprocess_rntuple(dgs, save_form=False, backend="iterative")


def test_workers_validate_step_size():
    """get_steps and get_parquet_form_uuid_steps reject step_size < 1 directly, including
    negative values (which would otherwise silently produce a single step per file)."""
    from coffea.dataset_tools.preprocess import get_parquet_form_uuid_steps, get_steps

    normed = awkward.Array(
        [
            {
                "file": "tests/samples/nano_dy.root",
                "object_path": "Events",
                "steps": None,
                "num_entries": None,
                "uuid": None,
            }
        ]
    )
    for bad in (0, -5):
        with pytest.raises(ValueError, match="step_size must be a positive integer"):
            get_steps(normed, step_size=bad)
        with pytest.raises(ValueError, match="step_size must be a positive integer"):
            get_parquet_form_uuid_steps(normed, step_size=bad)
