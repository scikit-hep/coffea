---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to work with NanoEvents

NanoEvents turns columnar ROOT or Parquet files into Pythonic objects with Awkward Array behaviors.
This guide walks through exploring branches, creating selections, and reducing data inside a coffea processor.

## Inspect collections interactively

```python
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    {"nano_dy.root": "Events"},
    schemaclass=NanoAODSchema,
    entry_stop=10_000,
).events()

print(events.fields)            # top-level collections
print(events.Muon.fields)       # attributes on the Muon collection
print(events.Muon.pt.type)      # awkward type
```

Use this pattern in notebooks to discover the structure of a sample before writing a processor.

## Columnar selections

Selections stay lazy until you materialize them. Compose masks with vectorized operations.

```python
import awkward as ak

tight_muons = events.Muon[
    (events.Muon.tightId)
    & (events.Muon.pt > 25)
    & (abs(events.Muon.eta) < 2.4)
]

os_pairs = (
    (tight_muons[:, :, None].charge + tight_muons[:, None, :].charge) == 0
)
```

`tight_muons` retains the Awkward structure, so per-event lengths remain variable.

## Use vector behaviors

NanoAODSchema associates Lorentz-vector behaviors.

```python
lead, trail = ak.unzip(ak.combinations(tight_muons, 2))
dimuon = lead + trail

mass = dimuon.mass        # automatically computed invariant mass
pt = dimuon.pt
```

Behaviors follow you into the processor environment, enabling the same concise syntax.

## Access metadata inside processors

`events.metadata` carries dataset-level information from the fileset.

```python
from coffea import processor


class ExampleProcessor(processor.ProcessorABC):
    ...
    def process(self, events):
        year = events.metadata["year"]
        is_mc = events.metadata.get("is_mc", False)
```

Enroll cross sections, era flags, and other attributes when preparing the fileset.

## Convert to pandas or numpy

Use Awkward utilities when you require flat arrays.

```python
import awkward as ak

flat_mass = ak.to_numpy(ak.flatten(mass, axis=None))
df = ak.to_dataframe({"mass": mass, "pt": pt})
```

`ak.to_dataframe` preserves jagged offsets by creating a multi-index; flatten the data before conversion if you prefer a simple index.

## Keep processing columnar

Avoid explicit Python loops over events or particles. Coffea’s executors thrive on vectorized operations because they minimize interpreter overhead and play well with batching. If you must fall back to a loop, wrap the hot section in a `numba.njit`-decorated function—see the [Awkward Array numba guide](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba.html)—so it compiles to machine code while preserving chunk-level parallelism.

## Tips & tricks

- Call `ak.num(collection, axis=1)` to see how many objects each event contains.
- If a branch is missing, confirm that it is interpretable by NanoEvents; warnings of the schema often identify incompatible forms.
- Apply selections with boolean masks before combinations to avoid forming unnecessary pairings.
