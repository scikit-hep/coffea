---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# How to collect results

Processors aggregate per-chunk outputs by returning dictionaries from the `process()` method. This guide covers building histograms, cutflow counters, and structured outputs that merge cleanly after parallel execution.

## Return results from process()

In modern coffea, simply return a dictionary from your `process()` method. For virtual/eager modes, nest results under the dataset name; for dask mode, return a flat dictionary.

```python
import hist
import awkward as ak
from coffea import processor


class DimuonProcessor(processor.ProcessorABC):
    def __init__(self, mode="virtual"):
        assert mode in ["virtual", "eager", "dask"]
        self._mode = mode

    def process(self, events):
        dataset = events.metadata["dataset"]

        # Create histogram (use hist.dask.Hist for dask mode)
        if self._mode == "dask":
            hist_class = hist.dask.Hist
        else:
            hist_class = hist.Hist

        h_mass = hist_class.new.Reg(60, 60, 120, name="mass", label="mμμ [GeV]").Int64()

        muons = events.Muon[(events.Muon.tightId) & (events.Muon.pt > 25)]
        lead, trail = ak.unzip(ak.combinations(muons, 2))
        mass = (lead + trail).mass
        h_mass.fill(mass=mass)

        # For dask mode, return flat dict
        if self._mode == "dask":
            return {
                "entries": ak.num(events, axis=0),
                "mass": h_mass,
                "npairs": ak.sum(ak.num(mass, axis=1)),
            }
        # For virtual/eager mode, nest under dataset name
        else:
            return {
                dataset: {
                    "entries": len(events),
                    "mass": h_mass,
                    "npairs": int(ak.sum(ak.num(mass, axis=1))),
                }
            }

    def postprocess(self, accumulator):
        pass
```

- No need for `_accumulator`, `@property accumulator`, or `.identity()` methods
- Just return a plain dictionary
- The framework automatically merges dictionaries and hist objects across chunks

## Fill histograms

Use hist directly without wrapper classes:

```python
import awkward as ak
import hist

def process(self, events):
    dataset = events.metadata["dataset"]

    # Create histogram with category axis
    h_mass = hist.Hist.new.StrCat([], name="region", growth=True).Reg(
        60, 60, 120, name="mass", label="mμμ [GeV]"
    ).Weight()

    muons = events.Muon[(events.Muon.tightId) & (events.Muon.pt > 25)]
    lead, trail = ak.unzip(ak.combinations(muons, 2))
    mass = (lead + trail).mass

    # Fill with event weights
    h_mass.fill(region="signal", mass=mass, weight=events.genWeight)

    return {
        dataset: {
            "entries": len(events),
            "mass": h_mass,
        }
    }
```

- Histogram axes grow dynamically when you use `growth=True`
- Use `ak.num` to count objects per event
- No need to call `ak.to_numpy()` or `ak.flatten()` - hist handles awkward arrays directly

## Track multiple results

Return multiple items in your dictionary:

```python
from coffea.analysis_tools import Weights

def process(self, events):
    dataset = events.metadata["dataset"]

    # Track weights (use Weights(None) for dask mode)
    weights = Weights(len(events))
    weights.add("genWeight", events.genWeight)

    # Create histograms
    h_mass = hist.Hist.new.Reg(60, 60, 120, name="mass").Weight()
    h_pt = hist.Hist.new.Reg(50, 0, 200, name="pt").Weight()

    muons = events.Muon[(events.Muon.tightId) & (events.Muon.pt > 25)]
    lead, trail = ak.unzip(ak.combinations(muons, 2))
    dimuon = lead + trail

    h_mass.fill(mass=dimuon.mass, weight=weights.weight())
    h_pt.fill(pt=dimuon.pt, weight=weights.weight())

    return {
        dataset: {
            "entries": len(events),
            "sumw": float(ak.sum(events.genWeight)),
            "mass": h_mass,
            "pt": h_pt,
            "weightStats": weights.weightStatistics,
        }
    }
```

- Include scalar values like event counts and sum of weights
- Include hist objects for histograms
- Include `weightStatistics` for debugging
- All types will be automatically merged across chunks

## Use postprocess to finalize results

`postprocess` runs after all chunks are merged. Use it to compute derived quantities:

```python
def postprocess(self, accumulator):
    # accumulator is the merged dictionary from all chunks
    for dataset in accumulator:
        entries = accumulator[dataset]["entries"]
        sumw = accumulator[dataset]["sumw"]
        # Add derived quantities
        accumulator[dataset]["avgWeight"] = sumw / entries if entries > 0 else 0

    return accumulator
```

## Serialize results

Histograms and dictionaries are picklable; save them to disk using `coffea.util.save`:

```python
import coffea.util

result = runner(fileset, processor_instance=my_processor)
coffea.util.save(result, "out.coffea")

# Reload later
loaded = coffea.util.load("out.coffea")
```

## When to use accumulator classes

You don't need accumulator classes for most use cases:
- **Histograms**: Use `hist.Hist` directly - they already support merging
- **Scalar sums**: Return plain Python `int` or `float` - they're automatically summed
- **Dictionaries**: Return plain Python `dict` - nested values are automatically merged

Use accumulator classes when you need specialized merging behavior:
- **Concatenation**: Use `column_accumulator` or `list_accumulator` to append arrays or lists
- **Union**: Use `set_accumulator` to collect unique values
- **Auto-initialization**: Use `defaultdict_accumulator` for counters that self-initialize

## Accumulator classes reference

### column_accumulator - Collect arrays across chunks

Use `column_accumulator` to append numpy or awkward arrays from each chunk:

```python
from coffea.processor import column_accumulator
import awkward as ak

def process(self, events):
    dataset = events.metadata["dataset"]

    # Select events and collect their kinematics
    selected_events = events[ak.max(events.Muon.pt, axis=1) > 50]
    event_ids = ak.to_numpy(selected_events.event)

    return {
        dataset: {
            "entries": len(events),
            "selected_event_ids": column_accumulator(event_ids),
        }
    }
```

The framework automatically concatenates arrays across chunks.

### list_accumulator - Collect lists across chunks

Use `list_accumulator` to collect small lists of values:

```python
from coffea.processor import list_accumulator
import awkward as ak

def process(self, events):
    dataset = events.metadata["dataset"]

    # Collect metadata about interesting events
    mask = events.MET.pt > 200
    runs = ak.to_numpy(events.run[mask])
    lumis = ak.to_numpy(events.luminosityBlock[mask])
    event_ids = ak.to_numpy(events.event[mask])

    interesting = [(int(r), int(l), int(e))
                   for r, l, e in zip(runs, lumis, event_ids)]

    return {
        dataset: {
            "entries": len(events),
            "interesting_events": list_accumulator(interesting),
        }
    }
```

### set_accumulator - Collect unique values

Use `set_accumulator` to collect unique values across chunks:

```python
from coffea.processor import set_accumulator
import awkward as ak

def process(self, events):
    dataset = events.metadata["dataset"]

    # Collect unique run numbers
    unique_runs = set_accumulator(set(ak.to_numpy(events.run)))

    return {
        dataset: {
            "entries": len(events),
            "runs": unique_runs,
        }
    }
```

### defaultdict_accumulator - Simple counters

Use `defaultdict_accumulator` for counters that automatically initialize:

```python
from coffea.processor import defaultdict_accumulator

def process(self, events):
    dataset = events.metadata["dataset"]

    cutflow = defaultdict_accumulator(int)
    cutflow["total"] += len(events)
    cutflow["has_muon"] += int(ak.sum(ak.num(events.Muon, axis=1) > 0))
    cutflow["has_2muon"] += int(ak.sum(ak.num(events.Muon, axis=1) >= 2))

    return {
        dataset: {
            "cutflow": cutflow,
        }
    }
```

### value_accumulator - Accumulate arbitrary types

Use `value_accumulator` to accumulate values with custom types:

```python
from coffea.processor import value_accumulator

def process(self, events):
    dataset = events.metadata["dataset"]

    # Track sum of weights
    sumw = value_accumulator(float, float(ak.sum(events.genWeight)))

    return {
        dataset: {
            "sumw": sumw,
        }
    }
```

## Tips & tricks

- Return plain dictionaries - the framework handles merging automatically
- Use `hist.Hist` for virtual/eager modes and `hist.dask.Hist` for dask mode
- For simple cutflow counters, use regular Python dicts with integer values or `defaultdict_accumulator(int)`
- Use accumulator classes when you need specialized merging behavior (concatenation, union, etc.)
- Avoid early returns like `if len(events) == 0: return` - always return the full dictionary structure
- Test with `processor.IterativeExecutor` first to validate the output structure
- For large numpy arrays (>100 MB), consider writing them to disk inside `process` instead of using `column_accumulator`
