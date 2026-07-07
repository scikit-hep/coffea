import os.path as osp

import pyarrow
import pytest

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor import Err, Ok
from coffea.processor.executor import UprootMissTreeError
from coffea.processor.test_items import NanoEventsProcessor

_exceptions = (FileNotFoundError, UprootMissTreeError, pyarrow.ArrowInvalid)


@pytest.mark.parametrize("filetype", ["ttree", "rntuple", "parquet"])
@pytest.mark.parametrize("skipbadfiles", [False, True, _exceptions])
@pytest.mark.parametrize("maxchunks", [None, 1000])
@pytest.mark.parametrize("compression", [None, 0])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
@pytest.mark.parametrize("mode", ["eager", "virtual"])
@pytest.mark.parametrize("processor_type", ["ProcessorABC", "Callable"])
def test_nanoevents_analysis(
    executor, compression, maxchunks, skipbadfiles, filetype, mode, processor_type
):
    if processor_type == "Callable":
        processor_instance = NanoEventsProcessor(mode=mode, check_filehandle=True)
    else:
        processor_instance = NanoEventsProcessor(
            mode=mode, check_filehandle=True
        ).process

    suffix = {"ttree": ".root", "rntuple": "_rntuple.root", "parquet": ".parquet"}[
        filetype
    ]
    runner_format = "parquet" if filetype == "parquet" else "root"
    # for parquet, treename-mismatch isn't a failure mode; substitute a malformed
    # parquet file so the dataset still has a non-OSError raise like root gets
    # from UprootMissTreeError
    bad_second_file = (
        "tests/samples/nano_dy_malformed.parquet"
        if filetype == "parquet"
        else f"tests/samples/nano_dy_SpecialTree{suffix}"
    )
    bad_only_file = (
        "tests/samples/nano_dy_malformed.parquet"
        if filetype == "parquet"
        else f"tests/samples/nano_dy{suffix}"
    )

    filelist = {
        "DummyBadMissingFile": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/non_existent{suffix}")],
        },
        "ZJetsBadMissingTree": {
            "treename": "NotEvents",
            "files": [
                osp.abspath(f"tests/samples/nano_dy{suffix}"),
                osp.abspath(bad_second_file),
            ],
        },
        "ZJetsBadMissingTreeAllFiles": {
            "treename": "NotEvents",
            "files": [osp.abspath(bad_only_file)],
        },
        "ZJets": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dy{suffix}")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dimuon{suffix}")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    executor = executor(compression=compression)
    run = processor.Runner(
        executor=executor,
        skipbadfiles=skipbadfiles,
        schema=schemas.NanoAODSchema,
        maxchunks=maxchunks,
        format=runner_format,
    )

    if skipbadfiles == _exceptions:
        hists = run(
            filelist,
            processor_instance=processor_instance,
            treename="Events",
        )
        assert hists["cutflow"]["ZJets_pt"] == 18
        assert hists["cutflow"]["ZJets_mass"] == 6
        assert hists["cutflow"]["ZJetsBadMissingTree_pt"] == 18
        assert hists["cutflow"]["ZJetsBadMissingTree_mass"] == 6
        assert hists["cutflow"]["Data_pt"] == 84
        assert hists["cutflow"]["Data_mass"] == 66

    else:
        with pytest.raises(_exceptions):
            hists = run(
                filelist,
                processor_instance=processor_instance,
                treename="Events",
            )
        with pytest.raises(_exceptions):
            hists = run(
                filelist,
                processor_instance=processor_instance,
                treename="NotEvents",
            )


@pytest.mark.parametrize("filetype", ["ttree", "rntuple", "parquet"])
@pytest.mark.parametrize("align_clusters", [False, True])
def test_preprocessing(align_clusters, filetype):
    suffix = {"ttree": ".root", "rntuple": "_rntuple.root", "parquet": ".parquet"}[
        filetype
    ]
    runner_format = "parquet" if filetype == "parquet" else "root"
    nano_dy = f"tests/samples/nano_dy{suffix}"
    nano_dy_empty = f"tests/samples/nano_dy_empty{suffix}"

    fileset = {
        "only_empty": {
            "files": {
                nano_dy_empty: "Events",
            },
        },
        "nonempty_and_empty": {
            "files": {
                nano_dy: "Events",
                nano_dy_empty: "Events",
            },
        },
        "empty_and_nonempty": {
            "files": {
                nano_dy_empty: "Events",
                nano_dy: "Events",
            },
        },
        "only_nonempty": {
            "files": {
                nano_dy: "Events",
            },
        },
    }

    executor = processor.IterativeExecutor()
    if filetype == "parquet" and align_clusters:
        with pytest.raises(
            ValueError, match="align_clusters is only supported for ROOT"
        ):
            processor.Runner(
                executor=executor,
                schema=schemas.NanoAODSchema,
                chunksize=7,
                align_clusters=align_clusters,
                format=runner_format,
            )
        return

    run = processor.Runner(
        executor=executor,
        schema=schemas.NanoAODSchema,
        chunksize=7,
        align_clusters=align_clusters,
        format=runner_format,
    )
    chunks = list(run.preprocess(fileset))
    if align_clusters:
        assert len(chunks) == 6
        for chunk in chunks:
            if chunk.dataset == "only_empty":
                assert chunk.filename == nano_dy_empty
                assert chunk.entrystart == 0
                assert chunk.entrystop == 0
            elif (
                chunk.dataset == "nonempty_and_empty"
                or chunk.dataset == "empty_and_nonempty"
            ):
                assert chunk.filename in [nano_dy, nano_dy_empty]
                if chunk.filename == nano_dy:
                    assert chunk.entrystart == 0
                    assert chunk.entrystop == 40
                else:
                    assert chunk.entrystart == 0
                    assert chunk.entrystop == 0
            elif chunk.dataset == "only_nonempty":
                assert chunk.filename == nano_dy
                assert chunk.entrystart == 0
                assert chunk.entrystop == 40
    else:
        assert len(chunks) == 21
        for chunk in chunks:
            if chunk.dataset == "only_empty":
                assert chunk.filename == nano_dy_empty
                assert chunk.entrystart == 0
                assert chunk.entrystop == 0
            elif (
                chunk.dataset == "nonempty_and_empty"
                or chunk.dataset == "empty_and_nonempty"
            ):
                assert chunk.filename in [nano_dy, nano_dy_empty]
                if chunk.filename == nano_dy:
                    assert chunk.entrystart in [0, 7, 14, 21, 28, 35]
                    assert (
                        chunk.entrystop == chunk.entrystart + 7
                        if chunk.entrystart != 35
                        else 40
                    )
                else:
                    assert chunk.entrystart == 0
                    assert chunk.entrystop == 0
            elif chunk.dataset == "only_nonempty":
                assert chunk.filename == nano_dy
                assert chunk.entrystart in [0, 7, 14, 21, 28, 35]
                assert (
                    chunk.entrystop == chunk.entrystart + 7
                    if chunk.entrystart != 35
                    else 40
                )

        def data_manipulation(events):
            dataset = events.metadata["dataset"]
            n_events = len(events)
            assert events.attrs["@events_factory"].access_log == []
            return {dataset: n_events}

        out = run(chunks, data_manipulation)
        assert out == {
            "only_empty": 0,
            "nonempty_and_empty": 40,
            "empty_and_nonempty": 40,
            "only_nonempty": 40,
        }


_good_fileset = {
    "ZJets": {
        "treename": "Events",
        "files": [osp.abspath("tests/samples/nano_dy.root")],
    }
}

_bad_fileset = {
    "Missing": {
        "treename": "Events",
        "files": [osp.abspath("tests/samples/non_existent.root")],
    }
}


@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_use_result_type_ok(executor):
    """use_result_type=True returns Ok(output) on a successful run."""
    run = processor.Runner(
        executor=executor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=True,
    )
    processor_instance = NanoEventsProcessor(mode="eager")
    result = run(
        _good_fileset, processor_instance=processor_instance, treename="Events"
    )
    assert isinstance(result, Ok), f"Expected Ok, got {result!r}"
    assert result.is_ok()
    out = result.unwrap()
    assert "cutflow" in out


@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_use_result_type_err(executor):
    """use_result_type=True returns Err(exception) for an exception type
    matching skipbadfiles instead of raising."""
    run = processor.Runner(
        executor=executor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=True,  # default tuple matches OSError → FileNotFoundError
    )
    processor_instance = NanoEventsProcessor(mode="eager")
    result = run(_bad_fileset, processor_instance=processor_instance, treename="Events")
    assert isinstance(result, Err), f"Expected Err, got {result!r}"
    assert result.is_err()
    assert isinstance(result.exception, BaseException)


@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_use_result_type_run_method_ok(executor):
    """use_result_type=True also works when calling run() directly."""
    run = processor.Runner(
        executor=executor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=True,
    )
    processor_instance = NanoEventsProcessor(mode="eager")
    result = run.run(
        _good_fileset, processor_instance=processor_instance, treename="Events"
    )
    assert isinstance(result, Ok), f"Expected Ok, got {result!r}"
    assert result.unwrap() is not None


@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
def test_use_result_type_run_method_err(executor):
    """use_result_type=True via run() returns Err for matching exception types."""
    run = processor.Runner(
        executor=executor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=True,
    )
    processor_instance = NanoEventsProcessor(mode="eager")
    result = run.run(
        _bad_fileset, processor_instance=processor_instance, treename="Events"
    )
    assert isinstance(result, Err), f"Expected Err, got {result!r}"
    assert isinstance(result.exception, BaseException)


def test_err_carries_partial_value():
    """Err.value preserves a partial accumulator passed alongside the exception."""
    boom = ValueError("boom")
    err = Err(boom, value={"out": "partial", "metrics": {"chunks": 3}})
    assert err.is_err()
    assert err.exception is boom
    assert err.value == {"out": "partial", "metrics": {"chunks": 3}}


def test_err_value_defaults_to_none():
    err = Err(RuntimeError("no partial"))
    assert err.value is None


def test_use_result_type_err_preserves_partial_recoverable_output():
    """When a recoverable executor surfaces a matching exception via
    wrapped_out['exception'], __call__ returns Err(value=partial_output)
    instead of dropping the partial."""
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
        savemetrics=True,
        use_result_type=True,
        skipbadfiles=(RuntimeError,),
    )

    boom = RuntimeError("simulated partial failure")
    partial_out = {"cutflow": {"events": 42}}
    metrics = {"chunks": 1}

    def fake_run(**kwargs):
        return Ok(
            {
                "out": partial_out,
                "metrics": metrics,
                "exception": boom,
            }
        )

    run.run = fake_run
    result = run(
        {"x": {"files": {"f.root": "Events"}}},
        processor_instance=NanoEventsProcessor(mode="eager"),
    )
    assert isinstance(result, Err), f"Expected Err, got {result!r}"
    assert result.exception is boom
    assert result.value == (partial_out, metrics)


def test_use_result_type_recoverable_non_matching_propagates():
    """A recoverable exception that doesn't match skipbadfiles' filter must
    still propagate (it's a real bug, not an expected failure)."""
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=(OSError,),  # filter doesn't include AssertionError
    )

    boom = AssertionError("real bug")

    def fake_run(**kwargs):
        return Ok({"out": {"x": 1}, "exception": boom})

    run.run = fake_run
    with pytest.raises(AssertionError, match="real bug"):
        run(
            {"x": {"files": {"f.root": "Events"}}},
            processor_instance=NanoEventsProcessor(mode="eager"),
        )


def test_use_result_type_run_non_matching_propagates():
    """A non-matching exception raised inside _run must propagate, not become Err."""
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=(OSError,),
    )
    # An empty/invalid fileset triggers ValueError in _run; ValueError is not
    # OSError, so it must propagate rather than be captured as Err.
    with pytest.raises(ValueError):
        run.run({}, processor_instance=NanoEventsProcessor(mode="eager"))


def test_use_result_type_matches_through_exception_chain():
    """A wrapped exception whose __cause__ matches skipbadfiles must become Err,
    consistent with automatic_retries' chain-based matching."""
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
        use_result_type=True,
        skipbadfiles=(OSError,),
    )

    # head=RuntimeError, cause=OSError; head doesn't match the filter but cause does.
    cause = OSError("disk gone")
    head = RuntimeError("processor blew up")
    head.__cause__ = cause

    def fake_run(**kwargs):
        return Ok({"out": {"x": 1}, "exception": head})

    run.run = fake_run
    result = run(
        {"x": {"files": {"f.root": "Events"}}},
        processor_instance=NanoEventsProcessor(mode="eager"),
    )
    assert isinstance(result, Err), f"Expected Err, got {result!r}"
    assert result.exception is head


def test_use_result_type_requires_skipbadfiles():
    """use_result_type=True must be paired with skipbadfiles."""
    with pytest.raises(ValueError, match="requires skipbadfiles"):
        processor.Runner(
            executor=processor.IterativeExecutor(),
            use_result_type=True,
        )

    # accepted shapes
    processor.Runner(
        executor=processor.IterativeExecutor(),
        use_result_type=True,
        skipbadfiles=True,
    )
    processor.Runner(
        executor=processor.IterativeExecutor(),
        use_result_type=True,
        skipbadfiles=(FileNotFoundError,),
    )


def _one(x):
    return {"n": 1}


def test_processor_compression_none_runs():
    # Bug 11: processor_compression=None must not feed raw processor to lz4f.decompress
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
        processor_compression=None,
    )
    out = run(
        _good_fileset,
        processor_instance=NanoEventsProcessor(mode="eager"),
        treename="Events",
    )
    assert out["cutflow"]["ZJets_pt"] == 18
    assert out["cutflow"]["ZJets_mass"] == 6


@pytest.mark.parametrize("seam", ["metadata", "run"])
def test_uproot_options_timeout(seam):
    # Bug 21: a timeout inside uproot_options must not be passed twice to uproot.open
    from coffea.processor.executor import FileMeta, Runner

    if seam == "metadata":
        item = FileMeta("ZJets", osp.abspath("tests/samples/nano_dy.root"), "Events")
        out = Runner.metadata_fetcher_root(60, False, {"timeout": 30}, item)
        (fetched,) = out
        assert fetched.metadata["numentries"] == 40
    else:
        run = processor.Runner(
            executor=processor.IterativeExecutor(),
            schema=schemas.NanoAODSchema,
        )
        out = run(
            _good_fileset,
            processor_instance=NanoEventsProcessor(mode="eager"),
            treename="Events",
            uproot_options={"timeout": 30},
        )
        assert out["cutflow"]["ZJets_pt"] == 18


@pytest.mark.parametrize("skipbadfiles", [False, (ValueError,)])
def test_automatic_retries_retry_without_skipbadfiles(skipbadfiles):
    # Bug 22: retries must happen regardless of skipbadfiles
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient")
        return "ok"

    out = processor.Runner.automatic_retries(3, skipbadfiles, flaky)
    assert out == "ok"
    assert calls["n"] == 3


@pytest.mark.parametrize("form", ["int", "class", "instance", "bool"])
def test_futures_mergepool_forms(form):
    # Bug 25: every documented mergepool form must work and not mutate self.mergepool
    import concurrent.futures as cf

    mp = {
        "int": 2,
        "class": cf.ProcessPoolExecutor,
        "instance": cf.ProcessPoolExecutor(max_workers=2),
        "bool": True,
    }[form]
    ex = processor.FuturesExecutor(
        workers=2, merging=True, mergepool=mp, compression=None, status=False
    )
    out, code = ex(list(range(6)), _one, None)
    assert out == {"n": 6}
    assert code == 0
    assert ex.mergepool is mp
    if isinstance(mp, cf.Executor):
        mp.shutdown()


def test_recoverable_exception_raised_in_default_path():
    # Bug 47: use_result_type=False must not silently drop a captured exception
    run = processor.Runner(
        executor=processor.IterativeExecutor(),
        schema=schemas.NanoAODSchema,
    )
    boom = RuntimeError("recovered failure")

    def fake_run(**kwargs):
        return {"out": {"cutflow": {"events": 1}}, "exception": boom}

    run.run = fake_run
    with pytest.raises(RuntimeError, match="recovered failure"):
        run(
            {"x": {"files": {"f.root": "Events"}}},
            processor_instance=NanoEventsProcessor(mode="eager"),
        )


def test_maxchunks_limits_preprocess_file_opening():
    # Bug 56: maxchunks=1 must not open every file during preprocessing
    fileset = {
        "ZJets": {
            "treename": "Events",
            "files": [
                osp.abspath("tests/samples/nano_dy.root"),
                osp.abspath("tests/samples/non_existent.root"),
            ],
        }
    }
    run = processor.Runner(
        executor=processor.IterativeExecutor(retries=0),
        schema=schemas.NanoAODSchema,
        maxchunks=1,
    )
    chunks = list(run.preprocess(fileset))
    assert len(chunks) == 1
    assert chunks[0].filename.endswith("nano_dy.root")


def test_watcher_recompresses_only_on_change(monkeypatch):
    # Bug 55: the running accumulator must not be recompressed on idle poll cycles
    from coffea.processor import executor as ex_mod

    item = ex_mod._compress({"n": 1}, 1)
    calls = {"n": 0}
    real_compress = ex_mod._compress

    def counting(value, compression):
        calls["n"] += 1
        return real_compress(value, compression)

    monkeypatch.setattr(ex_mod, "_compress", counting)

    class StubExec:
        desc = "Processing"
        unit = "items"
        merging = False
        compression = 1

    class StubFH:
        def __init__(self):
            self.running = 1
            self.done = {"futures": 0, "merges": 0}
            self.completed = []
            self.futures = []
            self.merges = []
            self._polls = 0

        def update(self):
            self._polls += 1
            if self._polls >= 3:
                self.completed = [item]
                self.running = 0

        def fetch(self, n):
            batch = self.completed[:n]
            self.completed = self.completed[n:]
            return batch

    out = ex_mod._watcher(StubFH(), StubExec(), lambda b: b, None)
    assert calls["n"] == 1
    assert ex_mod._decompress(out) == {"n": 1}


def test_auto_dask_client_single_and_cleanup(monkeypatch):
    # Bug 24: at most one auto-created dask Client, assigned to both executors and
    # cleaned up afterwards
    from coffea.processor import executor as ex_mod

    created = []

    class FakeClient:
        def __init__(self, **kwargs):
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    class FakeDistributed:
        class client:
            Client = FakeClient

    monkeypatch.setattr(ex_mod, "_import_distributed", lambda: FakeDistributed)

    executor = processor.DaskExecutor()
    runner = processor.Runner(executor=executor)
    assert executor.client is None
    with runner._auto_dask_client() as client:
        assert client is created[0]
        assert executor.client is client
        assert runner.pre_executor.client is client
        assert len(created) == 1
    assert len(created) == 1
    assert executor.client is None
    assert created[0].closed is True
