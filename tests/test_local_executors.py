import os.path as osp

import pytest

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor.executor import UprootMissTreeError
from coffea.processor.test_items import NanoEventsProcessor

_exceptions = (FileNotFoundError, UprootMissTreeError)


@pytest.mark.parametrize("filetype", ["root", "parquet"])
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
        processor_instance = NanoEventsProcessor(mode=mode)
    else:
        processor_instance = NanoEventsProcessor(mode=mode).process

    if filetype == "parquet":
        pytest.xfail("parquet nanoevents not supported yet")

    filelist = {
        "DummyBadMissingFile": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/non_existent.{filetype}")],
        },
        "ZJetsBadMissingTree": {
            "treename": "NotEvents",
            "files": [
                osp.abspath(f"tests/samples/nano_dy.{filetype}"),
                osp.abspath(f"tests/samples/nano_dy_SpecialTree.{filetype}"),
            ],
        },
        "ZJetsBadMissingTreeAllFiles": {
            "treename": "NotEvents",
            "files": [osp.abspath(f"tests/samples/nano_dy.{filetype}")],
        },
        "ZJets": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dy.{filetype}")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.abspath(f"tests/samples/nano_dimuon.{filetype}")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    executor = executor(compression=compression)
    run = processor.Runner(
        executor=executor,
        skipbadfiles=skipbadfiles,
        schema=schemas.NanoAODSchema,
        maxchunks=maxchunks,
        format=filetype,
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


@pytest.mark.parametrize("align_clusters", [False, True])
def test_preprocessing(align_clusters):
    fileset = {
        "only_empty": {
            "files": {
                "tests/samples/nano_dy_empty.root": "Events",
            },
        },
        "nonempty_and_empty": {
            "files": {
                "tests/samples/nano_dy.root": "Events",
                "tests/samples/nano_dy_empty.root": "Events",
            },
        },
        "empty_and_nonempty": {
            "files": {
                "tests/samples/nano_dy_empty.root": "Events",
                "tests/samples/nano_dy.root": "Events",
            },
        },
        "only_nonempty": {
            "files": {
                "tests/samples/nano_dy.root": "Events",
            },
        },
    }

    executor = processor.IterativeExecutor()
    run = processor.Runner(
        executor=executor,
        schema=schemas.NanoAODSchema,
        chunksize=7,
        align_clusters=align_clusters,
    )
    chunks = list(run.preprocess(fileset))
    if align_clusters:
        assert len(chunks) == 6
        for chunk in chunks:
            if chunk.dataset == "only_empty":
                assert chunk.filename == "tests/samples/nano_dy_empty.root"
                assert chunk.entrystart == 0
                assert chunk.entrystop == 0
            elif (
                chunk.dataset == "nonempty_and_empty"
                or chunk.dataset == "empty_and_nonempty"
            ):
                assert chunk.filename in [
                    "tests/samples/nano_dy.root",
                    "tests/samples/nano_dy_empty.root",
                ]
                if chunk.filename == "tests/samples/nano_dy.root":
                    assert chunk.entrystart == 0
                    assert chunk.entrystop == 40
                else:
                    assert chunk.entrystart == 0
                    assert chunk.entrystop == 0
            elif chunk.dataset == "only_nonempty":
                assert chunk.filename == "tests/samples/nano_dy.root"
                assert chunk.entrystart == 0
                assert chunk.entrystop == 40
    else:
        assert len(chunks) == 21
        for chunk in chunks:
            if chunk.dataset == "only_empty":
                assert chunk.filename == "tests/samples/nano_dy_empty.root"
                assert chunk.entrystart == 0
                assert chunk.entrystop == 0
            elif (
                chunk.dataset == "nonempty_and_empty"
                or chunk.dataset == "empty_and_nonempty"
            ):
                assert chunk.filename in [
                    "tests/samples/nano_dy.root",
                    "tests/samples/nano_dy_empty.root",
                ]
                if chunk.filename == "tests/samples/nano_dy.root":
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
                assert chunk.filename == "tests/samples/nano_dy.root"
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
