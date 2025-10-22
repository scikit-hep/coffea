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

- Load the JSON once in `__init__`.
- Keep references to individual `Correction` objects you will call frequently.

## Evaluate per-object scale factors

```python
import awkward as ak

def process(self, events):
    out = self.accumulator
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

Store alternative weights in the accumulator so downstream steps can build envelopes.

## Combine multiple corrections

```python
trig = self.cset["trigger_sf"].evaluate(muons.pt, muons.eta, "nominal")
iso = self.cset["iso_sf"].evaluate(muons.pt, muons.eta, "nominal")

per_muon = sf * trig * iso
event_weight = ak.prod(per_muon, axis=1)
```

Multiply per-object corrections before reducing across the event dimension.

## Patch missing phase space

When a correction is undefined for a value, correctionlib raises an exception. Guard with masks or clamps.

```python
eta = ak.clip(muons.eta, -2.39, 2.39)
pt = ak.where(muons.pt < 20, 20, muons.pt)

sf = self.sf.evaluate(eta, pt, "nominal")
```

Apply the same pre-processing to all systematic branches to stay consistent.

## Apply event-level weights

Not all weights depend on per-object kinematics. Use metadata for global factors.

```python
xsec = events.metadata["cross_section"]
luminosity = 35.9
event_weight *= xsec * luminosity / events.metadata["n_events"]
```

Keep bookkeeping inputs (sum of generator weights, number of events) in the fileset metadata.

## Report weights in the accumulator

```python
out["cutflow"]["weighted"] += float(ak.sum(event_weight))
out["systematics"]["muon_sf_up"] += float(ak.sum(event_weight_up))
```

Use `defaultdict_accumulator` or `dict_accumulator` to accumulate per-systematic yields.

## Share corrections across processors

For large teams, store correction payloads on CVMFS or in object stores and load them by URL.

```python
import fsspec
import json

with fsspec.open("https://.../muon_sf.json.gz") as fin:
    payload = json.load(fin)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
```

Caching the parsed `CorrectionSet` in your processor avoids re-reading the file per chunk.

## Tips & tricks

- Print `self.sf.inputs` to confirm the order and names of arguments—especially when migrating to new payload versions.
- Use `self.sf.to_evaluator()` for a fast, vectorized callable if you need to evaluate the same correction millions of times outside coffea.
- Persist helper arrays (like absolute eta) on the events object if multiple corrections need them; this avoids recomputing inside every evaluation.
- Record the correction versions you used in the accumulator or metadata to streamline reproducibility and cross-checks.
