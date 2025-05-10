import os.path as osp
import sys

import pytest

from coffea import processor
from coffea.nanoevents import schemas
from coffea.processor.executor import UprootMissTreeError

if sys.platform.startswith("win"):
    pytest.skip("skipping tests that only function in linux", allow_module_level=True)


@pytest.mark.parametrize("filetype", ["root", "parquet"])
@pytest.mark.parametrize("skipbadfiles", [True, False])
@pytest.mark.parametrize("maxchunks", [1, None])
@pytest.mark.parametrize("chunksize", [100000, 5])
@pytest.mark.parametrize("schema", [None, schemas.BaseSchema])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor]  # , processor.FuturesExecutor
)
def test_dataframe_analysis(
    executor, schema, chunksize, maxchunks, skipbadfiles, filetype
):
    from coffea.processor.test_items import NanoTestProcessor

    if schema is not None and filetype == "parquet":
        pytest.xfail("parquet nanoevents not supported yet")

    filelist = {
        "ZJets": {"files": [osp.abspath(f"tests/samples/nano_dy.{filetype}")]},
        "Data": {"files": [osp.abspath(f"tests/samples/nano_dimuon.{filetype}")]},
    }

    executor = executor()
    run = processor.Runner(
        executor=executor,
        schema=schema,
        chunksize=chunksize,
        maxchunks=maxchunks,
        skipbadfiles=skipbadfiles,
        format=filetype,
    )

    hists = run(filelist, "Events", processor_instance=NanoTestProcessor())

    if maxchunks is None:
        assert hists["cutflow"]["ZJets_pt"] == 18
        assert hists["cutflow"]["ZJets_mass"] == 6
        assert hists["cutflow"]["Data_pt"] == 84
        assert hists["cutflow"]["Data_mass"] == 66
    else:
        assert maxchunks == 1
        print(hists["cutflow"]["ZJets_pt"])
        assert hists["cutflow"]["ZJets_pt"] == (18 if chunksize == 100_000 else 2)
        assert hists["cutflow"]["ZJets_mass"] == (6 if chunksize == 100_000 else 1)
        assert hists["cutflow"]["Data_pt"] == (84 if chunksize == 100_000 else 13)
        assert hists["cutflow"]["Data_mass"] == (66 if chunksize == 100_000 else 12)


@pytest.mark.parametrize("filetype", ["root", "parquet"])
@pytest.mark.parametrize("skipbadfiles", [True, False])
@pytest.mark.parametrize("maxchunks", [None, 1000])
@pytest.mark.parametrize("compression", [None, 0, 2])
@pytest.mark.parametrize(
    "executor", [processor.IterativeExecutor]  # , processor.FuturesExecutor
)
def test_nanoevents_analysis(executor, compression, maxchunks, skipbadfiles, filetype):
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
        schema=processor.NanoAODSchema,
        maxchunks=maxchunks,
        format=filetype,
    )

    if skipbadfiles:
        hists = run(filelist, "Events", processor_instance=NanoEventsProcessor())
        assert hists["cutflow"]["ZJets_pt"] == 18
        assert hists["cutflow"]["ZJets_mass"] == 6
        assert hists["cutflow"]["ZJetsBadMissingTree_pt"] == 18
        assert hists["cutflow"]["ZJetsBadMissingTree_mass"] == 6
        assert hists["cutflow"]["Data_pt"] == 84
        assert hists["cutflow"]["Data_mass"] == 66

    else:
        LookForError = (FileNotFoundError, UprootMissTreeError)
        with pytest.raises(LookForError):
            hists = run(filelist, "Events", processor_instance=NanoEventsProcessor())
        with pytest.raises(LookForError):
            hists = run(filelist, "NotEvents", processor_instance=NanoEventsProcessor())
