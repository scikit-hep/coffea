import multiprocessing
import sys

import pytest

from coffea import processor
from coffea.nanoevents import BaseSchema


def test_parsl_start_stop():
    pytest.importorskip("parsl", minversion="0.7.2")

    from coffea.processor.parsl.detail import (
        _default_cfg,
        _parsl_initialize,
        _parsl_stop,
    )

    _parsl_initialize(config=_default_cfg)

    _parsl_stop()


def do_parsl_job(filelist, flatten=False, compression=0, config=None):
    from coffea.processor.test_items import NanoTestProcessor

    executor = processor.ParslExecutor(compression=compression, config=config)
    run = processor.Runner(executor=executor, schema=BaseSchema)

    hists = run(
        filelist,
        processor_instance=NanoTestProcessor(mode="virtual"),
        treename="Events",
    )

    assert hists["cutflow"]["ZJets_pt"] == 18
    assert hists["cutflow"]["ZJets_mass"] == 6
    assert hists["cutflow"]["Data_pt"] == 84
    assert hists["cutflow"]["Data_mass"] == 66


# @pytest.mark.skipif(sys.platform.startswith('darwin'), reason='parsl htex not working on osx again')
def test_parsl_htex_executor():
    pytest.importorskip("parsl", minversion="0.7.2")
    import os
    import os.path as osp

    import parsl
    from parsl.config import Config
    from parsl.executors import HighThroughputExecutor
    from parsl.providers import LocalProvider

    parsl_version = tuple(map(int, parsl.__version__.split(".")))
    if parsl_version >= (2024, 3, 4):
        max_workers_arg = {"max_workers_per_node": 1}
    else:
        max_workers_arg = {"max_workers": 1}

    parsl_config = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_default",
                address="127.0.0.1",
                cores_per_worker=max(multiprocessing.cpu_count() // 2, 1),
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                    nodes_per_block=1,
                ),
                **max_workers_arg,
            )
        ],
        strategy=None,
    )
    parsl.load(parsl_config)

    filelist = {
        "ZJets": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        "Data": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
    }

    do_parsl_job(filelist)
    do_parsl_job(filelist, compression=1)

    filelist = {
        "ZJets": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dy.root")],
        },
        "Data": {
            "treename": "Events",
            "files": [osp.join(os.getcwd(), "tests/samples/nano_dimuon.root")],
        },
    }

    do_parsl_job(filelist)


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="signals are different on windows"
)
def test_timeout():
    import signal

    from coffea.processor.parsl.timeout import timeout

    @timeout
    def too_long(timeout=None):
        import time

        time.sleep(20)

    @timeout
    def make_except(timeout=None):
        import time

        time.sleep(1)
        raise Exception("oops!")

    try:
        too_long(timeout=5)
    except Exception as e:
        assert e.args[0] == "Timeout hit"

    try:
        make_except(timeout=20)
    except Exception as e:
        assert e.args[0] == "oops!"

    # reset alarms for other tests, this is suspicious
    signal.alarm(0)


def test_parsl_condor_cfg():
    pytest.importorskip("parsl", minversion="0.7.2")

    from coffea.processor.parsl.condor_config import condor_config

    print(condor_config())


def test_parsl_slurm_cfg():
    pytest.importorskip("parsl", minversion="0.7.2")
    import os

    x509_proxy = "x509up_u%s" % (os.getuid())
    fname = "/tmp/%s" % x509_proxy
    with open(fname, "w+"):
        os.utime(fname, None)

    from coffea.processor.parsl.slurm_config import slurm_config

    print(slurm_config())
