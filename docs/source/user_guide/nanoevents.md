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

## Record the analysis instead of running it: `mode="graphed"`

`mode="graphed"` returns a deferred [graphed](https://github.com/graphed-org/graphed) array rather
than an Awkward one. Building the analysis reads no event data: field access, behaviors and
cross-references are recorded as a graph, and the branches each partition needs are derived from
that graph when it runs. Install it with `pip install coffea[graphed]` (Python 3.11 or newer).

```python
import awkward as ak
import graphed
from graphed.awkward import gak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    {"nano_dy.root": "Events"}, schemaclass=NanoAODSchema, mode="graphed"
).events()

matched_pt = events.Muon.matched_jet.pt
n_jet = gak.num(events.Jet, axis=1)
```

Nothing has been read yet. `graphed.aggregate_plan` turns the recorded outputs into a plan an
executor runs partition by partition. The plan names coffea's recording backend by its class, so it
carries no schema behaviors:

```python
from graphed.core.execution import SequentialRunner


def reduce(values):
    return tuple(ak.to_list(value) for value in values)


def combine(left, right):
    if left is None or right is None:
        return left if right is None else right
    return tuple(a + b for a, b in zip(left, right))


def empty():
    return None


plan = graphed.aggregate_plan(
    matched_pt,
    n_jet,
    reduce=reduce,
    combine=combine,
    empty=empty,
    steps_per_file=2,
)
print(sorted(plan.process.columns))
# ['Jet_pt', 'Muon_jetIdx', 'nJet', 'nMuon']

matched, n_jet_per_event = SequentialRunner().run(plan).value
print(n_jet_per_event[:5])
# [5, 8, 5, 3, 5]
```

`n_jet` costs only the `nJet` counter: a plan reads what the graph touches, not what the schema
describes.

A plan run re-raises a failing operation's error as awkward raised it. To have it point at the
analysis line that recorded the operation, run the output with
`graphed.debug.run(events.session, output)` after `import graphed.debug`, which raises a
`StageError` naming that line.

### What graphed mode refuses

Graphed mode raises `NotImplementedError` instead of falling back silently, and each message names
where the capability lives instead:

| Refused | Instead |
| --- | --- |
| `from_root(..., steps_per_file=...)` | `graphed.aggregate_plan(..., steps_per_file=...)` |
| `uproot_options={"allow_read_errors_with_report": True}` | `graphed.checkpoint.run_resumable` |
| `add_systematic`, `_ensure_systematics` | `graphed.vary(...)` |
| a callable argument, e.g. `metric_table(..., metric=fn)` | `graphed.vary(...)` for a variation; otherwise record the metric with array operations |
| `from_parquet(..., mode="graphed")` | `from_root(..., mode="graphed")` |
| `coffea.util.maybe_map_partitions` | record the operation on the graph |

A schema is admitted only if it declares `__graphed_capable__ = True` in its own class body, which
`NanoAODSchema` and `PFNanoAODSchema` do. Subclasses do not inherit the flag: declare it once the
schema's cross-references have been checked in graphed mode. A plan reaches workers through plain
`pickle`, so its `reduce`/`combine`/`empty` callables and any custom schema class must live in a
module the workers can import, not in a notebook or `__main__`.

## Keep processing columnar

Avoid explicit Python loops over events or particles. Coffea’s executors thrive on vectorized operations because they minimize interpreter overhead and play well with batching. If you must fall back to a loop, wrap the hot section in a `numba.njit`-decorated function—see the [Awkward Array numba guide](https://awkward-array.org/doc/main/user-guide/how-to-use-in-numba.html)—so it compiles to machine code while preserving chunk-level parallelism.

## Tips & tricks

- Call `ak.num(collection, axis=1)` to see how many objects each event contains.
- If a branch is missing, confirm that it is interpretable by NanoEvents; warnings of the schema often identify incompatible forms.
- Apply selections with boolean masks before combinations to avoid forming unnecessary pairings.
