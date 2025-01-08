import os
import shutil
import os.path as osp

import parsl
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

from parsl.addresses import address_by_hostname

x509_proxy = "x509up_u%s" % (os.getuid())


def slurm_config(
    cores_per_job=16,
    mem_per_core=2048,
    jobs_per_worker=1,
    initial_workers=4,
    max_workers=8,
    work_dir="./",
    grid_proxy_dir="/tmp",
    partition="",
    walltime="02:00:00",
    htex_label="coffea_parsl_slurm_htex",
):
    shutil.copy2(osp.join(grid_proxy_dir, x509_proxy), osp.join(work_dir, x509_proxy))

    wrk_init = """
    export XRD_RUNFORKHANDLER=1
    export X509_USER_PROXY=%s
    """ % (
        osp.join(work_dir, x509_proxy)
    )

    sched_opts = """
    #SBATCH --cpus-per-task=%d
    #SBATCH --mem-per-cpu=%d
    """ % (
        cores_per_job,
        mem_per_core,
    )

    parsl_version = tuple(map(int, parsl.__version__.split(".")))
    if parsl_version >= (2024, 3, 4):
        max_workers_arg = {"max_workers_per_node": cores_per_job}
    else:
        max_workers_arg = {"max_workers": cores_per_job}

    slurm_htex = Config(
        executors=[
            HighThroughputExecutor(
                label=htex_label,
                address=address_by_hostname(),
                prefetch_capacity=0,
                provider=SlurmProvider(
                    launcher=SrunLauncher(),
                    init_blocks=initial_workers,
                    max_blocks=max_workers,
                    nodes_per_block=jobs_per_worker,
                    partition=partition,
                    scheduler_options=sched_opts,  # Enter scheduler_options if needed
                    worker_init=wrk_init,  # Enter worker_init if needed
                    walltime=walltime,
                ),
                **max_workers_arg,
            )
        ],
        strategy=None,
    )

    return slurm_htex
