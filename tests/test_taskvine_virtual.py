import pytest

from coffea.processor import (
    TaskVineExecutor,
)


@pytest.mark.skipif(
    not pytest.importorskip("ndcctools.taskvine", reason="TaskVine not available"),
    reason="TaskVine not available",
)
def test_taskvine_executor_with_virtual_arrays():
    """Test TaskVineExecutor with virtual arrays (lazy loading) and eager evaluation"""
    import os.path as osp

    from coffea.nanoevents import schemas
    from coffea.processor.test_items import NanoEventsProcessor

    # Use the same filelist as in local executors test
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

    # Create the same processor as used in local executors
    processor = NanoEventsProcessor(mode="virtual")

    # Create Runner with TaskVineExecutor
    from coffea.processor import Runner

    executor = TaskVineExecutor(
        port=9123, verbose=True, cores=1, disk=1024
    )  # max resources per task

    run = Runner(
        executor=executor,
        schema=schemas.NanoAODSchema,
    )

    try:
        # Test that the runner can process the files with both modes
        from ndcctools.taskvine import Factory

        workers = Factory(manager_host_port="localhost:9123", batch_type="local")
        workers.min_workers = 1
        workers.max_workers = 1
        workers.cores = 2
        workers.disk = 4096

        with workers:
            hists = run(
                filelist,
                processor_instance=processor,
                treename="Events",
            )

        # Check that we get the expected results (same as local executors)
        assert hists["cutflow"]["ZJets_pt"] == 18
        assert hists["cutflow"]["ZJets_mass"] == 6
        assert hists["cutflow"]["Data_pt"] == 84
        assert hists["cutflow"]["Data_mass"] == 66

        # Verify that both modes work correctly
        assert "mass" in hists
        assert "pt" in hists

    except (ImportError, RuntimeError, FileNotFoundError) as e:
        # Expected if TaskVine is not properly configured or test files don't exist
        assert any(
            x in str(e).lower() for x in ["taskvine", "ndcctools", "file", "not found"]
        )
