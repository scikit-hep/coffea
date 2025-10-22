# Getting Started

Coffea couples the columnar data model of Awkward Array with a thin execution layer so that an analysis can move from a laptop to a cluster without rewrites.
The workflow always follows the same pattern:

1. Implement a {class}`coffea.processor.ProcessorABC` that turns NanoEvents into accumulators.
2. Execute it with {class}`coffea.processor.Runner` using a local executor while you iterate.
3. Swap the executor when you are ready to scale out.

Below is a minimal processor that applies muon scale factors from `correctionlib` and produces a histogram.

```python
import awkward as ak
import correctionlib
import hist
from coffea import processor


class MuonProcessor(processor.ProcessorABC):
    def __init__(self, sf_path: str):
        self.corrections = correctionlib.CorrectionSet.from_file(sf_path)
        self.muon_sf = self.corrections["muon_sf"]
        self._accumulator = processor.dict_accumulator(
            {
                "mass": hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="dataset"),
                    hist.axis.Regular(60, 60, 120, name="mass", label="mμμ [GeV]"),
                ),
                "events": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator.identity()

    def process(self, events):
        out = self.accumulator
        dataset = events.metadata["dataset"]

        # select OS dimuons
        muons = events.Muon[events.Muon.tightId]
        dimuons = ak.combinations(muons, 2, fields=["lead", "trail"])
        dimuons = dimuons[dimuons.lead.charge != dimuons.trail.charge]

        # correctionlib returns per-muon weights; take product per event
        sf_lead = self.muon_sf.evaluate(dimuons.lead.eta, dimuons.lead.pt)
        sf_trail = self.muon_sf.evaluate(dimuons.trail.eta, dimuons.trail.pt)
        event_weight = sf_lead * sf_trail

        mass = (dimuons.lead + dimuons.trail).mass
        out["mass"].fill(
            dataset=dataset,
            mass=ak.to_numpy(ak.flatten(mass, axis=None)),
            weight=ak.to_numpy(ak.flatten(event_weight, axis=None)),
        )
        out["events"][dataset] += len(events)
        return out

    def postprocess(self, accumulator):
        return accumulator
```

### Run locally

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

fileset = {
    "DYJets": {"treename": "Events", "files": ["nano_dy.root"]},
    "Data": {"treename": "Events", "files": ["nano_data.root"]},
}

runner = processor.Runner(
    executor=processor.IterativeExecutor(status=True),
    schema=NanoAODSchema,
    savemetrics=True,
)

result, metrics = runner(
    fileset,
    processor_instance=MuonProcessor("muon_sf.json.gz"),
    treename="Events",
)
```

`result` is a nested accumulator that includes histograms and cutflow counters. The `metrics` dictionary captures runtime information such as bytes read and columns touched.

### Scale out

Scaling does not require modifying the processor. Replace the executor and, if needed, provide configuration for the backing service.

```python
from dask.distributed import Client

client = Client("tcp://scheduler:8786")

cluster_runner = processor.Runner(
    executor=processor.DaskExecutor(client=client, status=True),
    schema=NanoAODSchema,
    savemetrics=True,
)

result_cluster, metrics_cluster = cluster_runner(
    fileset,
    processor_instance=MuonProcessor("muon_sf.json.gz"),
    treename="Events",
)
```

You can follow the same pattern with {class}`~coffea.processor.FuturesExecutor`, {class}`~coffea.processor.ParslExecutor`, or {class}`~coffea.processor.TaskVineExecutor`. See {doc}`concepts` for background on processors and executors.

## Table of Contents

```{toctree}
:maxdepth: 1
installation.md
concepts.md
```
