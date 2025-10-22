---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to collect results

Accumulators are coffea’s mechanism for aggregating per-chunk outputs. This guide covers building histograms, cutflow counters, and structured outputs that merge cleanly after parallel execution.

## Create a reusable accumulator

```python
import hist
from coffea import processor


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator(
            {
                "mass": hist.Hist(
                    hist.axis.StrCategory([], growth=True, name="dataset"),
                    hist.axis.Regular(60, 60, 120, name="mass", label="mμμ [GeV]"),
                ),
                "cutflow": processor.defaultdict_accumulator(int),
            }
        )

    @property
    def accumulator(self):
        return self._accumulator.identity()
```

- `dict_accumulator` nests other accumulators.
- The `identity` method returns an empty copy so every worker modifies its own instance.

## Fill histograms

Fill the accumulator histograms in each Processor. Keep in mind to flatten the awkward array first, e.g.:

```python
import awkward as ak

def process(self, events):
    out = self.accumulator
    dataset = events.metadata["dataset"]

    muons = events.Muon[(events.Muon.tightId) & (events.Muon.pt > 25)]
    lead, trail = ak.unzip(ak.combinations(muons, 2))
    mass = (lead + trail).mass

    out["mass"].fill(
        dataset=dataset,
        mass=ak.to_numpy(ak.flatten(mass, axis=None)),
    )
    out["cutflow"]["pairs"] += ak.sum(ak.num(mass, axis=1))
    out["cutflow"]["events"] += len(events)
    return out
```

- Histogram axes grow dynamically when you use `growth=True`.
- Use `ak.num` to count objects per event.
- Track metadata such as the number of processed and selected events inside the accumulator for later efficiency calculations.

## Add non-histogram outputs

You can mix arbitrary accumulator types, such as `set_accumulator` or `list_accumulator`.

```python
from coffea.processor import list_accumulator

self._accumulator = processor.dict_accumulator(
    {
        "mass": hist.Hist(
            hist.axis.StrCategory([], growth=True, name="dataset"),
            hist.axis.Regular(60, 60, 120, name="mass"),
        ),
        "cutflow": processor.defaultdict_accumulator(int),
        "snapshots": list_accumulator([]),
    }
)
```

Append small summaries or debug payloads that you want to merge across workers.

## Use postprocess to finalize results

`postprocess` runs after all chunks are merged. Treat it as a hook to manipulate the aggregated outputs before you hand them to the next stage.

```python
def postprocess(self, accumulator):
    cutflow = accumulator["cutflow"]

    pairs = cutflow.get("pairs", 0.0)
    events_total = cutflow.get("events", 0.0)
    cutflow["efficiency"] = pairs / events_total if events_total else 0.0

    return {"mass": hist_mass, "cutflow": cutflow}
```

## Serialize results

Histograms are picklable; save them to disk using `coffea.util.save`.

```python
import coffea

result = runner(...)

coffea.util.save(result, "out.coffea")
```

## Debug accumulator merging

- Make sure that every processor returns the accumulator.
- Avoid early returns, i.e. `if len(events) == 0: return`
- When introducing new accumulator components or modifying it, run with `processor.IterativeExecutor` to validate the output structure before scaling up.

## Tips & tricks

- Store intermediate values (e.g. per-cut yields) in `defaultdict_accumulator(int)` so that keys appear automatically when you increment them.
- For large numpy arrays avoid returning them through the accumulator; write them to disk inside `process` instead to avoid memory spikes during merging.
