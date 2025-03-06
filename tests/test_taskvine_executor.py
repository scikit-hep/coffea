import sys
import os.path as osp
import pytest
from coffea import processor


try:
    import ndcctools.taskvine as vine
except ModuleNotFoundError:
    pytest.skip("could not import taskvine!")

if sys.platform.startswith("win"):
    pytest.skip("skipping tests that only function in linux", allow_module_level=True)


def test_taskvine_executor_nanoevents_analysis():
    from coffea.processor.test_items import NanoEventsProcessor

    filelist = {
        "ZJets": {
            "treename": "Events",
            "files": [osp.abspath("tests/samples/nano_dy.root")],
            "metadata": {"checkusermeta": True, "someusermeta": "hello"},
        },
        "Data": {
            "treename": "Events",
            "files": [osp.abspath("tests/samples/nano_dimuon.root")],
            "metadata": {"checkusermeta": True, "someusermeta2": "world"},
        },
    }

    port = 9123
    executor = processor.TaskVineExecutor(port=port)

    workers = vine.Factory(manager_host_port="localhost:9123")
    workers.max_workers = 1
    workers.min_workers = 1
    workers.cores = 1
    workers.disk = 2000

    with workers:
        run = processor.Runner(
            executor=executor,
            skipbadfiles=True,
            schema=processor.NanoAODSchema,
            maxchunks=10000,
            chunksize=1000,
        )

    hists = run(filelist, "Events", processor_instance=NanoEventsProcessor())
    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66
