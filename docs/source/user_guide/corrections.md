---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to apply corrections

Coffea integrates naturally with [correctionlib](https://cms-nanoaod.github.io/correctionlib/) and other lookup tools so that scale factors and systematic variations flow through your processor. This guide demonstrates typical patterns.

## Load a correction set once

```python
import correctionlib
from coffea import processor


class MuonSFProcessor(processor.ProcessorABC):
    def __init__(self, payload: str):
        self.cset = correctionlib.CorrectionSet.from_file(payload)
        self.sf = self.cset["muon_sf"]
        ...
```

- Load the JSON once in `__init__`, coffea knows how to distribute it with the Processor onto a cluster.
- Keep references to individual `Correction` objects you will call frequently.

## Evaluate per-object scale factors

```python
import awkward as ak

def process(self, events):
    dataset = events.metadata["dataset"]
    muons = events.Muon[(events.Muon.tightId) & (events.Muon.pt > 20)]

    sf = self.sf.evaluate(
        muons.eta,
        muons.pt,
        "nominal",
    )

    event_weight = ak.prod(sf, axis=1)
    ...
```

Correctionlib broadcasts over Awkward arrays automatically; the output matches the shape of the inputs.

## Handle systematic variations

```python
sf_up = self.sf.evaluate(muons.eta, muons.pt, "syst_up")
sf_down = self.sf.evaluate(muons.eta, muons.pt, "syst_down")

event_weight = ak.prod(sf, axis=1)
event_weight_up = ak.prod(sf_up, axis=1)
event_weight_down = ak.prod(sf_down, axis=1)
```

Store alternative weights in your output dictionary so downstream steps can build envelopes.

## Combine multiple corrections

```python
trig = self.cset["trigger_sf"].evaluate(muons.pt, muons.eta, "nominal")
iso = self.cset["iso_sf"].evaluate(muons.pt, muons.eta, "nominal")

per_muon = sf * trig * iso
event_weight = ak.prod(per_muon, axis=1)
```

Multiply per-object corrections before reducing across the event dimension.

## Apply event-level weights

Not all weights depend on per-object kinematics. Use metadata for global factors.

```python
xsec = events.metadata["cross_section"]
luminosity = 35.9
event_weight *= xsec * luminosity / events.metadata["n_events"]
```

Keep bookkeeping inputs (sum of generator weights, number of events) in the fileset metadata.

## Report weights in the output

```python
return {
    dataset: {
        "cutflow": {
            "weighted": float(ak.sum(event_weight)),
        },
        "systematics": {
            "muon_sf_up": float(ak.sum(event_weight_up)),
            "muon_sf_down": float(ak.sum(event_weight_down)),
        },
    }
}
```

Return systematic variations in your output dictionary so they can be merged across chunks.

## Tips & tricks

- Persist helper arrays (like absolute eta) on the events object if multiple corrections need them; this avoids recomputing inside every evaluation.
- Record the correction versions you used in the output dictionary or metadata to streamline reproducibility and cross-checks.
- Use the `Weights` class from `coffea.analysis_tools` to manage multiple corrections and their systematics together (see the applying_corrections notebook for examples).
