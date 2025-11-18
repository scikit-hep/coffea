"""A framework for analysis scale-out


"""
from .processor import ProcessorABC
from .dataframe import (
    LazyDataFrame,
)
from .helpers import Weights, PackedSelection
from .executor import (
    IterativeExecutor,
    FuturesExecutor,
    DaskExecutor,
    ParslExecutor,
    WorkQueueExecutor,
    TaskVineExecutor,
    Runner,
    run_spark_job,
)
from .accumulator import (
    accumulate,
    Accumulatable,
    AccumulatorABC,
    value_accumulator,
    list_accumulator,
    set_accumulator,
    dict_accumulator,
    defaultdict_accumulator,
    column_accumulator,
)
from coffea.nanoevents.schemas import (
    NanoAODSchema,
    TreeMakerSchema,
)


# deprecated run_uproot_job & executor usage:
from functools import partial


def _run_x_job(
    fileset,
    treename,
    processor_instance,
    executor,
    executor_args={},
    pre_executor=None,
    pre_args=None,
    chunksize=100000,
    maxchunks=None,
    metadata_cache=None,
    dynamic_chunksize=None,
    format="root",
):
    """
    Please use instead, e.g.:

        executor = IterativeExecutor()
        run = processor.Runner(
            executor=executor,
            schema=processor.NanoAODSchema,
        )
        hists = run(filelist, "Events", processor_instance=processor_instance)
    """

    # turn this deprecation warning on from coffea.__version__ >= 0.8 on
    # from coffea.util import deprecate
    # deprecate(
    #     RuntimeError(f"This method is deprecated, please use directly the new: {executor} and {Runner} classes.\n {_run_x_job.__doc__}"),  # noqa: E501
    #     0.9,
    # )

    # extract executor kwargs
    exe_args = {}
    exe_fields = executor.__dataclass_fields__.keys()
    exe_keys = list(executor_args.keys())
    for k in exe_keys:
        if k in exe_fields:
            exe_args[k] = executor_args.pop(k)

    executor = executor(**exe_args)

    # extract preexecutor kwargs
    if pre_executor is not None and pre_args is not None:
        pre_exe_args = {}
        pre_exe_fields = pre_executor.__dataclass_fields__.keys()
        pre_exe_keys = list(pre_args.keys())
        for k in pre_exe_keys:
            if k in pre_exe_fields:
                pre_exe_args[k] = pre_args.pop(k)

        pre_executor = pre_executor(**pre_exe_args)

    # make Runner instance, assume other args are for _work_function & co.
    run = Runner(
        executor=executor,
        chunksize=chunksize,
        maxchunks=maxchunks,
        metadata_cache=metadata_cache,
        dynamic_chunksize=dynamic_chunksize,
        format=format,
        **executor_args,
    )

    return run(
        fileset,
        treename,
        processor_instance=processor_instance,
    )


run_uproot_job = partial(_run_x_job, format="root")
run_parquet_job = partial(_run_x_job, format="parquet")

iterative_executor = IterativeExecutor
futures_executor = FuturesExecutor
dask_executor = DaskExecutor
parsl_executor = ParslExecutor
work_queue_executor = WorkQueueExecutor


__all__ = [
    "ProcessorABC",
    "LazyDataFrame",
    "Weights",
    "PackedSelection",
    "IterativeExecutor",
    "FuturesExecutor",
    "DaskExecutor",
    "ParslExecutor",
    "WorkQueueExecutor",
    "TaskVineExecutor",
    "Runner",
    "run_spark_job",
    "accumulate",
    "Accumulatable",
    "AccumulatorABC",
    "value_accumulator",
    "list_accumulator",
    "set_accumulator",
    "dict_accumulator",
    "defaultdict_accumulator",
    "column_accumulator",
    "NanoAODSchema",
    "TreeMakerSchema",
    # following methods are deprecated
    "run_uproot_job",
    "run_parquet_job",
    "iterative_executor",
    "futures_executor",
    "dask_executor",
    "parsl_executor",
    "work_queue_executor",
]
