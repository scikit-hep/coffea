# AGENTS.md — working in the coffea repository

Guidance for AI coding agents (and humans) working in **coffea**, the Columnar
Object Framework For Effective Analysis. Coffea is a scikit-hep library for
columnar High-Energy-Physics analysis built on `awkward`, `uproot`, `hist`,
`vector`, and (optionally) `dask` / `dask-awkward`. It wraps flat ROOT/Parquet
nTuples into physics-aware awkward arrays, provides schemas, corrections,
histogramming and selection helpers, and scales analyses from a laptop to a
batch/distributed cluster.

This file describes how the code is laid out, the idioms to follow, and — in a
dedicated section at the end — **how to migrate analyses across coffea's major
eras**. `CLAUDE.md` imports this file; keep shared guidance here.

---

## Environment & commands

Target runtime is **Python ≥ 3.10**. Releases follow **CalVer** (`vYYYY.M.P`).

```bash
# Editable install with the dev toolchain (tests, linters, docs)
pip install -e '.[dev]'

# Add the optional stacks you need:
pip install -e '.[dask,dask-awkward]'   # distributed / delayed execution
pip install -e '.[parsl]'               # Parsl executor
pip install -e '.[rucio]'               # CMS dataset discovery
pip install -e '.[xrootd]'              # fsspec-xrootd remote reads
pip install -e '.[triton]'              # Triton inference (ml_tools)

# Run the checks before proposing changes (both must pass — see CONTRIBUTING):
pre-commit run --all-files              # black, flake8, and the rest
pytest                                  # test suite (testpaths = tests/)
pytest -n auto                          # parallelize (pytest-xdist)
pytest tests/test_nanoevents.py -k dask # a single file / selection
```

As of **v2026.7.0**, `dask`, `distributed`, `dask-awkward`, and `dask-histogram`
are **optional** dependencies. Core eager/virtual workflows import none of them;
only `mode="dask"`, `apply_to_fileset`, and the dask executors pull them in.

Key dependency floors (see `pyproject.toml`): `awkward>=2.8.11`, `uproot>=5.7.0`,
`vector>=1.4.1,!=1.6.0`, `hist>=2`, `numpy>=1.22`, and (when installed)
`dask-awkward>=2025.9.0`.

---

## Source layout (`src/coffea/`)

- `nanoevents/` — `NanoEventsFactory` + schemas (NanoAOD, PHYSLITE, PFNano,
  Scouting, TreeMaker, FCC/EDM4HEP, Delphes, PDUNE) and physics-method mixins in
  `methods/`. Lazy/virtual/dask materialization and form handling live here
  (`factory.py`, `transforms.py`, `schemas/`).
- `dataset_tools/` — dataset specification, `preprocess`, `apply_to_fileset`, and
  fileset manipulation. The fileset is modelled with **pydantic** (`filespec.py`).
- `processor/` — `ProcessorABC`, `Runner`, and executors (`IterativeExecutor`,
  `FuturesExecutor`, `DaskExecutor`, `parsl`, `taskvine_executor.py`);
  `accumulator.py` for reducible outputs.
- `analysis_tools.py` — `PackedSelection`, `Weights`, N-1 and systematics helpers.
- `jetmet_tools/`, `btag_tools/`, `lookup_tools/`, `lumi_tools/`, `ml_tools/` —
  corrections, scale factors, lookup tables, lumi masks, ML inference adapters
  (correctionlib, Triton, ONNX, torch).
- `util.py` — shared helpers (`compress_form`/`decompress_form`, `coffea_console`,
  lazy `_import_dask` / `_import_dask_awkward`).

Tests mirror modules under `tests/`; sample ROOT/Parquet files live in
`tests/samples/`. Docs are MyST/notebook sources under `docs/source/`.

---

## Core idioms (write new code this way)

### Reading data: `NanoEventsFactory` and execution modes

`from_root` / `from_parquet` take a **`mode`** argument — `"eager"`, `"virtual"`,
or `"dask"` — that selects the backend:

- **`"eager"`** — arrays are materialized immediately into memory (awkward 2).
  Simplest mental model; closest to old in-memory workflows.
- **`"virtual"`** — *the default* for `from_root`/`from_parquet`. Arrays are
  virtual: columns are read from disk on first access and cached, with no task
  graph. Memory-efficient and dask-free.
- **`"dask"`** — builds a `dask-awkward` task graph; nothing runs until
  `.compute()` (or `dask.compute(...)`). Use for distributed scale-out.

```python
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema

events = NanoEventsFactory.from_root(
    {"file.root": "Events"},
    mode="virtual",                 # or "eager" / "dask"
    schemaclass=NanoAODSchema,
    metadata={"dataset": "DYJets"},
).events()
```

`from_preloaded` defaults to `mode="eager"`. `steps_per_file` replaces the old
"chunks" terminology. `mode="dask"` is incompatible with `entry_start`/
`entry_stop`/`steps_per_file`, and silently falls back to `"virtual"` for a
schema that is not dask-capable.

Available schemas: `NanoAODSchema`, `PFNanoAODSchema`, `ScoutingNanoAODSchema`,
`PHYSLITESchema`, `TreeMakerSchema`, `FCCSchema`, `EDM4HEPSchema`,
`DelphesSchema`, `PDUNESchema`, and the `BaseSchema` they derive from.

### Manipulating awkward arrays

- **Assign fields with setitem, not attribute assignment:**
  `events["Muon", "pt2"] = events.Muon.pt ** 2` — **not** `events.Muon.pt2 = ...`
  (attribute assignment does not persist on awkward-2 records).
- Prefer vectorized awkward operations (`ak.num`, `ak.mask`, `ak.combinations`,
  `ak.firsts`, broadcasting) over Python loops.
- Vectors use **scikit-hep `vector`** behaviors. Watch reserved coordinate names
  (`rho`, `phi`, `eta`, `theta`, `tau`, …): attaching a field named `rho` to a
  Momentum record silently reinterprets geometry. Upcast `float32`→`float64`
  before kinematic math to avoid overflow to `inf`.

### Histogramming

Use scikit-hep **`hist`** (+ `mplhep` for CMS/ATLAS styling); coffea's old
`coffea.hist` is gone. Categorical axes are `hist.axis.StrCategory([], growth=True)`;
binned axes are `hist.axis.Regular(...)` / `Variable(...)`; opt into variances with
`storage=hist.storage.Weight()`. `weight` and `sample` are reserved axis names.

### Preprocessing and scaling

`dataset_tools.preprocess` computes per-file `steps`, `num_entries`, `uuid`, and
(with `save_form=True`, the default) the awkward form, returning
`(available, all)` filesets. Filesets are pydantic `DataGroupSpec` models but the
APIs also accept — and return — the legacy plain-`dict` format.

```python
from coffea.dataset_tools import preprocess, apply_to_fileset, max_chunks

available, allfiles = preprocess(fileset, step_size=100_000, save_form=True)

# Scale a processor / analysis over the fileset (builds a dask graph):
out, report = apply_to_fileset(
    MyProcessor(), max_chunks(available, 5),
    schemaclass=NanoAODSchema,
    uproot_options={"allow_read_errors_with_report": True},
)
```

### Processors and executors

`ProcessorABC` requires `process(events)`; `postprocess` is optional. Run it
either through `apply_to_fileset` (dask) or the `Runner` + executor API:

```python
from coffea.processor import Runner, FuturesExecutor
run = Runner(executor=FuturesExecutor(workers=4), schema=NanoAODSchema)
out = run(fileset, treename="Events", processor_instance=MyProcessor())
```

`analysis_tools.PackedSelection` (cut bookkeeping), `Weights` (event weights +
systematic variations), and the N-1 helpers are the standard building blocks.

---

## Conventions

- Keep PRs focused; add/adjust tests for every behavior change; keep or improve
  coverage. `pre-commit run --all-files` and `pytest` must pass.
- Match the existing formatting (black, flake8); do not hand-fight the formatters.
- **Comments describe current behavior, not history.** No issue/PR numbers, no
  "used to…/before the fix…" narration; state what the code does now and why a
  non-obvious shape exists.
- Prefer editing existing modules over adding new ones; don't add abstractions
  for hypothetical needs.

---

## Migration guide

Coffea has moved through three broad eras. Identify where an analysis starts and
apply the relevant subsection. **Keep this guide current**: the migration target
is the newest release (**v2026.7.0** as of this writing) and the moving parts
below should be updated as new versions ship.

Primary references (read these when migrating):
[coffea 0.7→virtual-array notes (#1529)](https://github.com/scikit-hep/coffea/discussions/1529),
[`coffea.hist`→`hist` migration (#705)](https://github.com/scikit-hep/coffea/discussions/705),
[virtual-arrays / `mode=` (#1368)](https://github.com/scikit-hep/coffea/discussions/1368),
and the [release notes](https://github.com/scikit-hep/coffea/releases).

### A. From coffea 0.7 (awkward 1, lazy arrays) → latest

Coffea 0.7 is the last `awkward`-1 line (Python 3.8/3.9, `coffea.hist`, coffea's
built-in vector behaviors, lazy `NanoEvents`). It is unmaintained; migrate to the
CalVer line. The large-scale changes:

1. **awkward 1 → awkward 2.** Field assignment moves to setitem:
   `events.Electron.myvar = x` → `events["Electron", "myvar"] = x`. Many
   `ak.*` signatures changed; re-test every array manipulation.
2. **Lazy arrays → explicit modes.** 0.7's implicit lazy `events` becomes a
   `mode=` choice. `mode="eager"` is the closest in-memory analogue;
   `mode="virtual"` (the default) reads columns on demand without a dask graph.
3. **`coffea.hist` → `hist` + `mplhep`** (#705): `Cat(...)` →
   `axis.StrCategory([], growth=True)`; `Bin(...)` → `axis.Regular/Variable`;
   sparse categorical axes become dense (watch memory for many categories);
   `storage="weight"` is opt-in; `weight`/`sample` are reserved axis names;
   `h.sum("x")` → `h[{"x": sum}]`.
4. **Vector → scikit-hep `vector`** (#1529): upcast `float32`→`float64` before
   kinematics (float32 can overflow to `inf`); audit custom field names against
   `vector`'s reserved coordinate names (`rho`, `tau2`, …) that silently shadow
   momentum components.
5. **Processors/executors** still exist (`ProcessorABC`, `Runner`, executors) but
   several executor arguments are now keyword-only, and `postprocess` is optional.
6. **Histograms/selection helpers**: replace `coffea.processor.PackedSelection`
   /`Weights` imports with `coffea.analysis_tools`.

### B. From coffea 2023/2024/early-2025 (dask-awkward-only, `delayed=`) → latest

These CalVer versions were **dask-first**: `NanoEventsFactory.from_root` took
`delayed=True/False`, analyses built `dask-awkward` graphs and called
`.compute()`. Virtual arrays (introduced in the **2025.7** line) changed the
default execution model.

1. **`delayed=` is removed → use `mode=`.** `delayed=True` → `mode="dask"`;
   `delayed=False` → `mode="eager"`; the **new default is `mode="virtual"`**.
   There is no compatibility shim — update every `from_root`/`from_parquet` call.
2. **Default is now dask-free.** Code that only called `.compute()` to
   materialize a local result usually no longer needs `dask` at all: switch to
   `mode="virtual"` (or `"eager"`) and drop the compute. Keep `mode="dask"` +
   `apply_to_fileset` only for genuine distributed scale-out.
3. **dask is optional (2026.7).** If you keep the dask path, install it
   explicitly: `pip install 'coffea[dask,dask-awkward]'`.
4. **Preprocessing.** `preprocess()` returns `(available, all)` and saves forms
   by default; `steps_per_file`/`step_size` replace "chunks"; pass
   `{"allow_read_errors_with_report": True}` in `uproot_options` for access
   reports. RNTuple inputs are supported (2025.11); buffer caches (2026.4) and
   `split_fileset`/`Result` (2026.6) are available.
5. **Schemas & tools** added along the way: EDM4HEP/updated FCC, Scouting,
   weighted N-1, correctionlib adapter classes for `CorrectedJetsFactory`.

### C. Pydantic dataset specification — *placeholder, keep updated*

Recent releases model the fileset with **pydantic** (`DataGroupSpec`,
`DatasetSpec`, `ROOTFileSpec`/`ParquetFileSpec`, `ModelFactory`) instead of raw
dicts, adding construction-time validation, typed concrete-vs-optional specs, and
round-trippable JSON — while still accepting and returning the legacy dict format
(dict-in → dict-out).

> **Do not yet instruct users to fully migrate off plain-dict filesets.** Several
> extensions to the pydantic dataset tools — union forms via `DatasetSpec`
> addition with per-file field bitsets, user metadata extraction during
> preprocessing, mutable/resizable steps, switchable execution backends
> (`iterative`/`futures`), RNTuple/Parquet form extraction, and ServiceX /
> RDataFrame / `universal_pathlib` interop — are staged on **feature branches not
> yet merged upstream**. Fill in concrete, versioned migration steps here as those
> branches land in a tagged release, and only then recommend the typed models as
> the default fileset representation.

---

## Version targets

- **Latest / recommended:** `v2026.7.0` (July 2026). Track new releases and
  update the migration guide and version floors above as they ship.
- **Floors:** Python ≥ 3.10, `awkward>=2.8.11`, `uproot>=5.7.0`,
  `vector>=1.4.1,!=1.6.0`, `hist>=2`.
- **Legacy:** the `0.7.x` line (latest `0.7.31`) is awkward-1 and maintenance-only.

## Common gotchas

- Attribute assignment on awkward records is silently dropped — use setitem.
- `float32` kinematics can overflow to `inf` under scikit-hep `vector`; upcast.
- Reserved `vector` coordinate names shadow momentum fields — rename custom fields.
- `mode="dask"` requires the optional dask stack; without it, install `[dask,dask-awkward]`.
- The access `report` from `apply_to_fileset` must be computed together with the
  analysis output to be accurate.
