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

    def process(self, events):
        dataset = events.metadata["dataset"]

        # Create histogram with category axis
        h_mass = hist.Hist.new.StrCat([], growth=True, name="dataset").Reg(
            60, 60, 120, name="mass", label="mμμ [GeV]"
        ).Weight()

        # select OS dimuons
        muons = events.Muon[events.Muon.tightId]
        dimuons = ak.combinations(muons, 2, fields=["lead", "trail"])
        dimuons = dimuons[dimuons.lead.charge != dimuons.trail.charge]

        # correctionlib returns per-muon weights; take product per event
        sf_lead = self.muon_sf.evaluate(dimuons.lead.eta, dimuons.lead.pt)
        sf_trail = self.muon_sf.evaluate(dimuons.trail.eta, dimuons.trail.pt)
        event_weight = sf_lead * sf_trail

        mass = (dimuons.lead + dimuons.trail).mass
        h_mass.fill(
            dataset=dataset,
            mass=mass,
            weight=event_weight,
        )

        return {
            dataset: {
                "mass": h_mass,
                "events": len(events),
            }
        }

    def postprocess(self, accumulator):
        return accumulator
```

## Run locally

```python
from coffea import processor
from coffea.nanoevents import NanoAODSchema

fileset = {
    "DYJets": {
        "files": {"nano_dy.root": "Events"},
        "metadata": {"is_mc": True},
    },
    "Data": {
        "files": {"nano_data.root": "Events"},
        "metadata": {"is_mc": False},
    },
}

runner = processor.Runner(
    executor=processor.IterativeExecutor(),
    schema=NanoAODSchema,
    savemetrics=True,
)

result, metrics = runner(fileset, processor_instance=MuonProcessor("muon_sf.json.gz"))
```

`result` is a nested accumulator that includes histograms and cutflow counters. The `metrics` dictionary captures runtime information such as bytes read and columns touched.

## Scale out

Scaling does not require modifying the processor. Replace the executor and, if needed, provide configuration for the backing service.

```python
from dask.distributed import Client

client = Client("tcp://scheduler:8786")

cluster_runner = processor.Runner(
    executor=processor.DaskExecutor(client=client),
    schema=NanoAODSchema,
    savemetrics=True,
)

result_cluster, metrics_cluster = cluster_runner(
    fileset, processor_instance=MuonProcessor("muon_sf.json.gz")
)
```

You can follow the same pattern with {class}`~coffea.processor.FuturesExecutor`, {class}`~coffea.processor.ParslExecutor`, or {class}`~coffea.processor.TaskVineExecutor`. See {doc}`concepts` for background on processors and executors.

## Table of Contents

```{toctree}
:maxdepth: 1
installation.md
concepts.md
```
