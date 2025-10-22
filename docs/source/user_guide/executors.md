---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to scale with executors

Coffea separates analysis code from execution strategy. Executors manage concurrency and chunk scheduling while the processor logic stays unchanged.
Each executor implements the same interface and is consumed by {class}`coffea.processor.Runner`.

## Choosing an executor

- `IterativeExecutor`: the simplest option for debugging and unit tests; executes chunks in one thread.
- `FuturesExecutor`: uses `concurrent.futures` pools to fan out work across local CPU cores.
- `DaskExecutor`: connects to a [Dask Distributed](https://distributed.dask.org/en/latest/) cluster for interactive or batch workloads.
- `ParslExecutor`: targets HPC facilities via [Parsl](http://parsl-project.org/).
- `TaskVineExecutor`: dispatches work to opportunistic and heterogeneous resources managed by [TaskVine](https://cctools.readthedocs.io/en/latest/taskvine/).
- Work Queue: see {doc}`wq` for details on the Work Queue executor.

All executors accept the same arguments when invoked by {class}`~coffea.processor.Runner`, making it easy to prototype locally and scale out later. The snippets below assume that `fileset` and `my_processor`
are defined (see the example at the end of this page for a full context).

## Local executors

### IterativeExecutor

{class}`~coffea.processor.IterativeExecutor` processes one chunk at a time in the current Python process. It has no extra dependencies and is ideal for debugging, tutorials, and deterministic testing.

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

runner = processor.Runner(
    executor=processor.IterativeExecutor(status=True),
    schema=NanoAODSchema,
)

result = runner(fileset, processor_instance=my_processor, treename="Events")
```

### FuturesExecutor

{class}`~coffea.processor.FuturesExecutor` builds on `concurrent.futures` to parallelize across CPU cores on the same machine. By default it creates a `ProcessPoolExecutor`; you can pass `pool=concurrent.futures.ThreadPoolExecutor` or a constructed executor to reuse an existing pool. The `workers` argument controls the number of tasks scheduled in parallel, and options such as `merging` and `compression` improve throughput for large reductions.

```python
from concurrent.futures import ThreadPoolExecutor
from coffea import processor
from coffea.nanoevents import NanoAODSchema

executor = processor.FuturesExecutor(
    workers=8,
    pool=ThreadPoolExecutor,
    status=True,
)
runner = processor.Runner(executor=executor, schema=NanoAODSchema)

result = runner(fileset, processor_instance=my_processor, treename="Events")
```

## Distributed executors

### DaskExecutor

{class}`~coffea.processor.DaskExecutor` integrates with an existing Dask cluster. Provide a `distributed.Client` (or allow the executor to create one) and coffea will submit chunked tasks to the scheduler. Set `use_dataframes=True` when individual tasks return pandas objects and you want a Dask DataFrame back.

```python
from dask.distributed import Client
from coffea import processor
from coffea.nanoevents import NanoAODSchema

client = Client("tcp://scheduler:8786")

runner = processor.Runner(
    executor=processor.DaskExecutor(client=client, status=True),
    pre_executor=processor.IterativeExecutor(status=True),
    schema=NanoAODSchema,
    savemetrics=True,
)

result, metrics = runner(
    fileset,
    processor_instance=my_processor,
    treename="Events",
)
```

The optional `pre_executor` controls how metadata is gathered before submitting work; by default it reuses the main executor.

### ParslExecutor

{class}`~coffea.processor.ParslExecutor` uses Parsl's `DataFlowKernel` to launch tasks onto HPC systems. Load a Parsl configuration up front (for example with `parsl.load(config)`) and pass the same configuration to the executor.

```python
import parsl
from coffea import processor
from coffea.nanoevents import NanoAODSchema
from parsl.config import Config
from parsl.executors import HighThroughputExecutor

config = Config(executors=[HighThroughputExecutor(label="jobs")])
parsl.load(config)

executor = processor.ParslExecutor(config=config, status=True)
runner = processor.Runner(executor=executor, schema=NanoAODSchema)

result = runner(fileset, processor_instance=my_processor, treename="Events")
```

### TaskVineExecutor

{class}`~coffea.processor.TaskVineExecutor` brings the TaskVine workflow engine to coffea. It stages processor code, data, and optional environment archives to workers that connect back to the manager.

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

executor = processor.TaskVineExecutor(
    port=9123,
    cores=2,
    disk=2048,
    status=True,
)
runner = processor.Runner(executor=executor, schema=NanoAODSchema)

result = runner(fileset, processor_instance=my_processor, treename="Events")
```

Coordinate worker factories using the TaskVine CLI or Python APIs; see the TaskVine documentation for examples.

### Work Queue

The Work Queue executor shares the same concepts but is covered in depth in {doc}`wq`.

## Switching executors

The processor code stays identical no matter which executor you choose. The example below counts events per dataset and runs locally, then on a Dask cluster, without modifying the processor.

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

class CountEvents(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator(
            {"events": processor.defaultdict_accumulator(int)}
        )

    @property
    def accumulator(self):
        return self._accumulator.identity()

    def process(self, events):
        out = self.accumulator
        out["events"][events.metadata["dataset"]] += len(events)
        return out

    def postprocess(self, accumulator):
        return accumulator

fileset = {
    "ZJets": {"treename": "Events", "files": ["/data/nano_dy.root"]},
    "Data": {"treename": "Events", "files": ["/data/nano_dimuon.root"]},
}

my_processor = CountEvents()

# Local development
local_runner = processor.Runner(
    executor=processor.IterativeExecutor(status=True),
    schema=NanoAODSchema,
)
local_result = local_runner(fileset, processor_instance=my_processor, treename="Events")

# Scale to a cluster
from dask.distributed import Client

client = Client("tcp://scheduler:8786")
cluster_runner = processor.Runner(
    executor=processor.DaskExecutor(client=client, status=True),
    schema=NanoAODSchema,
)
cluster_result = cluster_runner(
    fileset,
    processor_instance=my_processor,
    treename="Events",
)
```

## Tips & tricks

- Set `savemetrics=True` on {class}`~coffea.processor.Runner` to collect bytes read, columns touched, and runtime statistics for every executor.
- Use `processor.SimpleCheckpointer` with the `checkpointer` argument when running long jobs so partially completed chunks persist across restarts.
- For `FuturesExecutor`, pass a pre-created `ProcessPoolExecutor` if you need to control affinity or reuse workers across multiple runs.
- When using `DaskExecutor`, call `client.upload_file` or package your environment so workers have the same code version as the driver.
- Disable compression (`compression=None`) only if the accumulator payloads are small; otherwise LZ4 saves network transfer time on distributed backends.
