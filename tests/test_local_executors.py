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
        processor_instance = NanoEventsProcessor(mode=mode, check_filehandle=True)
    else:
        processor_instance = NanoEventsProcessor(
            mode=mode, check_filehandle=True
        ).process

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
