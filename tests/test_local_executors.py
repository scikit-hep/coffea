import os.path as osp

import pyarrow
import pytest
from test_processors import NanoEventsProcessor

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor.executor import UprootMissTreeError

_exceptions = (FileNotFoundError, UprootMissTreeError, pyarrow.ArrowInvalid)


@pytest.mark.parametrize("filetype", ["ttree", "rntuple", "parquet"])
@pytest.mark.parametrize("skipbadfiles", [False, True, _exceptions])
@pytest.mark.parametrize("maxchunks", [None, 1000])
@pytest.mark.parametrize("compression", [None, 0, 2])
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
