---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
---

# Migrating to latest coffea

This guide is for analyses written against coffea 0.7.x that need to run on current coffea (CalVer releases, e.g. `2026.9.0`) in its default virtual-array mode.
The processor idiom is the same, so most analyses port with a handful of mechanical changes.
A few changes, however, either fail with an unhelpful error or change physics results without failing at all; those are marked **⚠️**.

Each section shows the 0.7 code first and the current equivalent second.

## Quick checklist

| 0.7.x | Current coffea | Section |
|---|---|---|
| `awkward` 1, `uproot` 4, `numpy<2` | `awkward` 2, `uproot` 5, `numpy` 2 supported | [Dependencies](#dependencies) |
| `NanoEventsFactory.from_root(file, "Events")` | `NanoEventsFactory.from_root({file: "Events"}, mode="virtual")` | [Opening files](#opening-files) |
| `processor.run_uproot_job(...)` | `processor.Runner(...)(fileset, processor_instance, treename=...)` | [Running a processor](#running-a-processor) |
| `run(fileset, "Events", proc)` | `run(fileset, proc, treename="Events")` | [Running a processor](#running-a-processor) |
| `processor.futures_executor` | `processor.FuturesExecutor(workers=...)` | [Running a processor](#running-a-processor) |
| `coffea.hist` | `hist` | [Histograms](#histograms) |
| `processor.Weights`, `processor.PackedSelection` | `coffea.analysis_tools.Weights`, `.PackedSelection` | [Moved imports](#moved-imports) |
| `events.Electron.x = ...` | `events["Electron", "x"] = ...` | [Adding fields](#adding-fields) |
| ⚠️ `jets["rho"] = ...` for JEC | `jets["Rho"] = ...` | [Vector behaviors](#vector-behaviors) |
| ⚠️ float32 kinematics | upcast to float64 where precision matters | [Vector behaviors](#vector-behaviors) |

## Dependencies

coffea 0.7 pins `awkward<2`, `uproot==4.*`, and `numpy<2`.
Current coffea requires `awkward>=2`, `uproot>=5`, and scikit-hep `vector`, and supports `numpy` 2, so install it in a fresh environment rather than upgrading in place:

```bash
python -m venv coffea-env
source coffea-env/bin/activate
pip install coffea
```

Awkward Array 2 changed some behavior that is independent of coffea (for example, arrays no longer accept attribute assignment; see [Adding fields](#adding-fields)).
If your analysis calls `awkward` functions directly, check them against the Awkward Array 2 documentation as well.

## Opening files

In 0.7, the tree name was the second positional argument:

```python
# coffea 0.7
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    "nano_dy.root",
    "Events",
    schemaclass=NanoAODSchema,
).events()
```

All arguments after the file are now keyword-only, so this raises `TypeError: NanoEventsFactory.from_root() takes 2 positional arguments but 3 positional arguments ... were given`.
Pass the file and tree together as a dictionary, the same form `uproot.open()` accepts, and choose a `mode`:

```python
# current coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    {"nano_dy.root": "Events"},
    mode="virtual",
    schemaclass=NanoAODSchema,
).events()
```

`mode="virtual"` (the default) reads columns lazily when they are first used, which is the closest match to 0.7 behavior.
`mode="eager"` reads everything up front, and `mode="dask"` returns a `dask-awkward` array for building a task graph.
`treepath="Events"` as a keyword also still works.

## Running a processor

### `run_uproot_job` is gone

0.7 analyses commonly used the `run_uproot_job` helper with lower-case executor names and an `executor_args` dictionary:

```python
# coffea 0.7
from coffea import processor
from coffea.nanoevents import NanoAODSchema

output = processor.run_uproot_job(
    fileset,
    "Events",
    MyProcessor(),
    processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 4},
)
```

`run_uproot_job`, `run_parquet_job`, `run_spark_job`, the lower-case executor aliases (`iterative_executor`, `futures_executor`, `dask_executor`, ...), `WorkQueueExecutor`, and `LazyDataFrame` have been removed.
Construct the executor and a {class}`~coffea.processor.Runner` directly:

```python
# current coffea
from coffea import processor
from coffea.nanoevents import NanoAODSchema

run = processor.Runner(
    executor=processor.FuturesExecutor(workers=4),
    schema=NanoAODSchema,
)
output = run(fileset, MyProcessor(), treename="Events")
```

Options that were passed in `executor_args` are now split by where they belong: executor options (`workers`, `compression`, `status`, ...) go to the executor, and run options (`schema`, `chunksize`, `maxchunks`, `skipbadfiles`, ...) go to `Runner`.
`dynamic_chunksize`, `mmap`, and the Skyhook options are no longer `Runner` fields.

### ⚠️ The argument order of `Runner.__call__` changed

If you already used `Runner` in 0.7, note that the tree name moved from the second positional argument to a keyword-only argument at the end:

```python
# coffea 0.7
output = run(fileset, "Events", processor_instance=MyProcessor())
output = run(fileset, "Events", MyProcessor())

# current coffea
output = run(fileset, MyProcessor(), treename="Events")
```

The old positional form raises `TypeError: Runner.__call__() takes 3 positional arguments but 4 were given`.
The same applies to `Runner.run()` and `Runner.preprocess()`.

### Other `Runner` defaults

- `schema` now defaults to `NanoAODSchema` (it was `BaseSchema` in 0.7). Pass `schema=BaseSchema` explicitly if you relied on the old default.
- `postprocess()` is no longer an abstract method of {class}`~coffea.processor.ProcessorABC`, so you can drop empty `postprocess()` implementations.
- Accumulators no longer need the `self._accumulator` / `@property accumulator` / `.identity()` pattern. Return a plain dictionary from `process()`; see {doc}`accumulators`.

## Histograms

`coffea.hist` was deprecated in favour of [`hist`](https://hist.readthedocs.io) and has been removed; `import coffea.hist` raises `ImportError`.

```python
# coffea 0.7
from coffea import hist

h = hist.Hist(
    "Events",
    hist.Cat("dataset", "Dataset"),
    hist.Bin("pt", "Muon $p_T$ [GeV]", 50, 0, 200),
)
h.fill(dataset=dataset, pt=ak.flatten(events.Muon.pt))
```

```python
# current coffea
import hist

h = (
    hist.Hist.new
    .StrCat([], name="dataset", growth=True)
    .Reg(50, 0, 200, name="pt", label="Muon $p_T$ [GeV]")
    .Weight()
)
h.fill(dataset=dataset, pt=ak.flatten(events.Muon.pt))
```

`hist.Hist` objects returned from `process()` are merged across chunks automatically.
Use `.Weight()` storage if you fill with weights, or `.Double()` / `.Int64()` for unweighted counts.
`hist` also provides plotting (via `mplhep`) and is compatible with `uproot` for writing ROOT files.

## Moved imports

`coffea.processor` no longer re-exports helpers that live in other modules:

```python
# coffea 0.7
from coffea.processor import Weights, PackedSelection, NanoAODSchema

# current coffea
from coffea.analysis_tools import Weights, PackedSelection
from coffea.nanoevents import NanoAODSchema
```

## Adding fields

Awkward Array 2 does not allow attribute assignment on arrays:

```python
# coffea 0.7
events.Electron.myvar = events.Electron.pt * 2
```

This raises `AttributeError: only private attributes (started with an underscore) can be set on arrays`.
Use item assignment with a tuple path instead:

```python
# current coffea
events["Electron", "myvar"] = events.Electron.pt * 2
```

The same applies to any collection or top-level field, e.g. `events["myflag"] = ...`.

## Vector behaviors

The 2D, 3D, and Lorentz-vector mixins in `coffea.nanoevents.methods.vector` are now built on [scikit-hep/vector](https://vector.readthedocs.io) instead of coffea's own implementation.
Most analysis code is unaffected (`.pt`, `.eta`, `.delta_r()`, `a + b`, `.mass` all work as before), with two exceptions.

### ⚠️ Field names that are vector coordinate aliases

`vector` treats several names as aliases for coordinates.
In particular, `rho` is the azimuthal magnitude, i.e. a synonym for `pt` on momentum vectors.
Attaching an unrelated quantity under that name to a jet collection therefore collides with the jet momentum.

This matters for jet energy corrections, because the 0.7 `jetmet_tools` example attached the event-level pileup density to jets as `rho`:

```python
# coffea 0.7
jets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
name_map["Rho"] = "rho"
```

On early coffea 202X releases this could silently overwrite the jets' transverse momentum (see [discussion #1529](https://github.com/scikit-hep/coffea/discussions/1529)).
Current releases detect the collision when the collection is next accessed and raise:

```text
ValueError: JetArray: multiple azimuthal radial aliases present: ['pt', 'rho']
```

Use a field name that is not a coordinate alias, as the current {doc}`corrections` example does:

```python
# current coffea
jets["Rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, jets.pt)[0]
name_map["Rho"] = "Rho"
```

The same check applies to the other coordinate aliases. On a `PtEtaPhiMLorentzVector` collection such as `Jet`, adding any of these fields raises a similar `ValueError`:

- `rho` (alias of `pt`)
- `x`, `y` (Cartesian azimuthal coordinates, conflicting with `pt`/`phi`)
- `z`, `theta` (longitudinal coordinates, conflicting with `eta`)
- `t`, `E`, `energy`, `M`, `tau` (temporal coordinates, conflicting with `mass`; `tau` is the proper time, i.e. the mass)

Names that merely contain an alias, such as NanoAOD's N-subjettiness branches `tau1`/`tau2` or a capitalised `Rho`, are fine.

### ⚠️ float32 precision

NanoAOD stores kinematics as float32, and computations on float32 inputs stay in float32.
Quantities computed by cancellation, such as the invariant mass of two nearly collinear massless objects, can lose all precision:

```python
from coffea.nanoevents.methods import vector

def p4(phi, dtype):
    return ak.zip(
        {
            "pt": np.array([1000.0], dtype),
            "eta": np.array([5.0], dtype),
            "phi": np.array([phi], dtype),
            "mass": np.array([0.0], dtype),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )

(p4(0.0, np.float32) + p4(1e-4, np.float32)).mass  # [0.0]
(p4(0.0, np.float64) + p4(1e-4, np.float64)).mass  # [0.0999...]
```

In the conversion described in discussion #1529, float32 inputs also produced `inf` outputs after the move to `vector`.
Where precision matters, upcast the inputs before building vectors, for example with `ak.values_astype(events.Jet.pt, np.float64)`.

## Complete example

A minimal 0.7-style processor and its current equivalent, side by side.

```python
# coffea 0.7
import awkward as ak
from coffea import hist, processor
from coffea.nanoevents import NanoAODSchema


class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator({
            "pt": hist.Hist("Events", hist.Cat("dataset", "Dataset"),
                            hist.Bin("pt", "Muon $p_T$ [GeV]", 50, 0, 200)),
            "nevents": processor.defaultdict_accumulator(int),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        out = self.accumulator.identity()
        dataset = events.metadata["dataset"]
        events.Muon.ptx2 = events.Muon.pt * 2
        out["pt"].fill(dataset=dataset, pt=ak.flatten(events.Muon.pt))
        out["nevents"][dataset] += len(events)
        return out

    def postprocess(self, accumulator):
        return accumulator


output = processor.run_uproot_job(
    {"DY": ["nano_dy.root"]},
    "Events",
    MyProcessor(),
    processor.futures_executor,
    executor_args={"schema": NanoAODSchema, "workers": 4},
)
```

```python
# current coffea
import awkward as ak
import hist
from coffea import processor
from coffea.nanoevents import NanoAODSchema


class MyProcessor(processor.ProcessorABC):
    def process(self, events):
        dataset = events.metadata["dataset"]
        events["Muon", "ptx2"] = events.Muon.pt * 2
        h = (
            hist.Hist.new
            .StrCat([], name="dataset", growth=True)
            .Reg(50, 0, 200, name="pt", label="Muon $p_T$ [GeV]")
            .Weight()
        )
        h.fill(dataset=dataset, pt=ak.flatten(events.Muon.pt))
        return {"pt": h, "nevents": {dataset: len(events)}}


run = processor.Runner(
    executor=processor.FuturesExecutor(workers=4),
    schema=NanoAODSchema,
)
output = run({"DY": ["nano_dy.root"]}, MyProcessor(), treename="Events")
```
