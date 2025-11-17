import os
import os.path as osp

import parsl
from parsl.providers import CondorProvider
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from parsl.addresses import address_by_hostname

x509_proxy = "x509up_u%s" % (os.getuid(),)


def condor_config(
    cores_per_job=4,
    mem_per_core=2048,
    total_workers=24,
    max_workers=200,
    pyenv_dir="%s/.local" % (os.environ["HOME"],),
    grid_proxy_dir="/tmp",
    htex_label="coffea_parsl_condor_htex",
    wrk_init=None,
    condor_cfg=None,
):
    pyenv_relpath = pyenv_dir.split("/")[-1]

    if wrk_init is None:
        wrk_init = """
        source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc7-opt/setup.sh
        export PATH=`pwd`/%s:$PATH
        export PYTHONPATH=`pwd`/%s:$PYTHONPATH

        export X509_USER_PROXY=`pwd`/%s
        mkdir -p ./%s
        """ % (
            "%s/bin" % pyenv_relpath,
            "%s/lib/python3.6/site-packages" % pyenv_relpath,
            x509_proxy,
            htex_label,
        )

    if condor_cfg is None:
        condor_cfg = """
        transfer_output_files = %s
        RequestMemory = %d
        RequestCpus = %d
        """ % (
            htex_label,
            mem_per_core * cores_per_job,
            cores_per_job,
        )

    xfer_files = [pyenv_dir, osp.join(grid_proxy_dir, x509_proxy)]

    parsl_version = tuple(map(int, parsl.__version__.split(".")))
    if parsl_version >= (2024, 3, 4):
        max_workers_arg = {"max_workers_per_node": cores_per_job}
    else:
        max_workers_arg = {"max_workers": cores_per_job}

    condor_htex = Config(
        executors=[
            HighThroughputExecutor(
                label=htex_label,
                address=address_by_hostname(),
                prefetch_capacity=0,
                cores_per_worker=1,
                worker_logdir_root="./",
                provider=CondorProvider(
                    init_blocks=total_workers,
                    max_blocks=max_workers,
                    nodes_per_block=1,
                    worker_init=wrk_init,
                    transfer_input_files=xfer_files,
                    scheduler_options=condor_cfg,
                ),
                **max_workers_arg,
            )
        ],
        strategy=None,
    )

    return condor_htex
