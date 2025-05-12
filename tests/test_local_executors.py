import os.path as osp

import pytest

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor.executor import UprootMissTreeError


@pytest.mark.parametrize("filetype", ["root", "parquet"])
@pytest.mark.parametrize("skipbadfiles", [True, False])
@pytest.mark.parametrize("maxchunks", [None, 1000])
@pytest.mark.parametrize("compression", [None, 0, 2])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor, processor.FuturesExecutor]
)
@pytest.mark.parametrize("mode", ["eager", "virtual"])
def test_nanoevents_analysis(
    executor, compression, maxchunks, skipbadfiles, filetype, mode
):
    from coffea.processor.test_items import NanoEventsProcessor

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

    if skipbadfiles:
        hists = run(
            filelist,
            processor_instance=NanoEventsProcessor(mode=mode),
            treename="Events",
        )
        assert hists["cutflow"]["ZJets_pt"] == 18
        assert hists["cutflow"]["ZJets_mass"] == 6
        assert hists["cutflow"]["ZJetsBadMissingTree_pt"] == 18
        assert hists["cutflow"]["ZJetsBadMissingTree_mass"] == 6
        assert hists["cutflow"]["Data_pt"] == 84
        assert hists["cutflow"]["Data_mass"] == 66

    else:
        LookForError = (FileNotFoundError, UprootMissTreeError)
        with pytest.raises(LookForError):
            hists = run(
                filelist,
                processor_instance=NanoEventsProcessor(mode=mode),
                treename="Events",
            )
        with pytest.raises(LookForError):
            hists = run(
                filelist,
                processor_instance=NanoEventsProcessor(mode=mode),
                treename="NotEvents",
            )
