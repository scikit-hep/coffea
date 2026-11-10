# Migration guide (coffea eras)

Read this when migrating an analysis, not on every task. The target is always the
newest [release](https://github.com/scikit-hep/coffea/releases); keep the moving
parts below current as versions ship.

References:
[0.7 to virtual arrays (#1529)](https://github.com/scikit-hep/coffea/discussions/1529),
[`coffea.hist` to `hist` (#705)](https://github.com/scikit-hep/coffea/discussions/705),
[virtual arrays / `mode=` (#1368)](https://github.com/scikit-hep/coffea/discussions/1368).

---

## A. From coffea 0.7 (awkward 1, lazy arrays)

0.7 is the last awkward-1 line (Python 3.8/3.9, `coffea.hist`, coffea's own vector
behaviors, lazy `NanoEvents`) and is unmaintained.

1. **awkward 1 to awkward 2.** Field assignment becomes setitem:
   `events.Electron.myvar = x` becomes `events["Electron", "myvar"] = x`; the
   attribute form now raises. Many `ak.*` signatures changed, so re-test every
   array manipulation.
2. **Lazy arrays to explicit modes.** The implicit lazy `events` becomes a `mode=`
   choice. `mode="virtual"` (the default) is the closest analogue: columns
   materialize on first access and are cached, with no task graph. `mode="eager"`
   reads everything up front; `mode="dask"` builds a `dask-awkward` graph.
3. **`coffea.hist` to `hist` + `mplhep`.** `Cat(...)` becomes
   `axis.StrCategory([], growth=True)`, `Bin(...)` becomes
   `axis.Regular`/`Variable`, `h.sum("x")` becomes `h[{"x": sum}]`, weight storage
   is opt-in, and `weight`/`sample` are reserved axis names. Sparse categorical
   axes are now dense — watch memory when there are many categories.
4. **Vectors go through scikit-hep `vector`.** It keeps to the input coordinate
   system, so `float32` fields can lose precision or overflow to `inf` where
   coffea's older behaviors did not; upcasting to `float64` is the usual fix, to be
   verified against your own kinematics. A custom field that collides with a
   coordinate its vector already carries is now rejected at construction rather
   than silently shadowing it (see *Common gotchas* in `ARCHITECTURE.md`).
5. **Processors and executors** still exist, but several executor arguments are
   keyword-only and `postprocess` is optional.
6. **`PackedSelection` and `Weights`** import from `coffea.analysis_tools`; the
   `coffea.processor` re-export is gone.

---

## B. From the dask-first CalVer releases (2023 to early 2025, `delayed=`)

Those versions took `delayed=True/False`, built `dask-awkward` graphs and called
`.compute()`. Virtual arrays (the 2025.7 line) changed the default execution model.
This is the hardest migration: the code that builds and computes the graph goes,
not only the flags.

> Dask and distributed remain a **first-tier job executor** (`DaskExecutor` in
> `Runner`, a distributed `Client`, `apply_to_fileset`). What is de-emphasized is
> the lazy `dask-awkward` **DAG** (`mode="dask"`) as the default columnar-compute
> model. Migrating off dask below means off that data model, not off the scheduler.

1. **`delayed=` is gone — use `mode=`.** `delayed=True` becomes `mode="dask"`,
   `delayed=False` becomes `mode="eager"`, and the new default is
   `mode="virtual"`. There is no shim: update every `from_root`/`from_parquet`
   call. Keep `mode="dask"` only where the graph is the point; `preprocess` and
   `apply_to_fileset` remain that pipeline (`ARCHITECTURE.md`, *"dask" mode*).
2. **The processor body loses the graph.** `dask_awkward` calls become the
   `awkward` calls they mirror (`dak.num` to `ak.num`, `dak.to_parquet` to
   `ak.to_parquet`), `hist.dask.Hist` becomes `hist.Hist` and fills at once, and
   every `.compute()` / `dask.compute(...)` goes. `process()` returns materialized
   accumulatables (`hist.Hist`, dicts, `column_accumulator`) that `accumulate`
   merges. `PackedSelection`, `Weights` and the correction factories keep their
   API. Columns are still read one at a time on first access; `ak.materialize`
   forces a read when you need one.
3. **Scaling out moves from `apply_to_fileset` to `Runner`.** `preprocess`, then
   `apply_to_fileset`, then `dask.compute(out, report)` becomes one call on a
   plain fileset that `Runner` chunks itself:

   ```python
   from coffea.processor import Runner, DaskExecutor  # or Futures/Iterative/TaskVine/Parsl

   run = Runner(DaskExecutor(client=client), chunksize=250_000, maxchunks=None,
                skipbadfiles=True, savemetrics=True, schema=NanoAODSchema)
   out, metrics = run(fileset, processor_instance=MyProcessor(), treename="Events")
   ```

   `fileset` is `{dataset: {"files": {path: "Events"}, "metadata": {...}}}`, or
   `{dataset: [paths]}` with `treename=`. `max_chunks(...)` becomes `maxchunks=`;
   the read-error report becomes `skipbadfiles=` plus the `(out, metrics)` pair
   from `savemetrics=True`. `events.metadata` carries `dataset`, `filename`,
   `treename`, `entrystart`, `entrystop` and `fileuuid` beside your own keys,
   which must not reuse those names.
4. **The dask stack is optional (2026.7).** Eager and virtual workflows import no
   dask; install `'coffea[dask,dask-awkward]'` if you need it.
5. **Preprocessing, in the dask pipeline.** `preprocess()` returns
   `(available, all)` and saves forms by default; `steps_per_file`/`step_size`
   replace "chunks"; pass `{"allow_read_errors_with_report": True}` in
   `uproot_options` for access reports. RNTuple inputs landed in 2025.11, buffer
   caches in 2026.4, and `split_fileset`/`Result` in 2026.6.
6. **Schemas and tools** added along the way: EDM4HEP and updated FCC, Scouting,
   weighted N-1, correctionlib adapters for `CorrectedJetsFactory`.

---

## C. Pydantic dataset specification — placeholder

Filesets are modelled with pydantic (`DataGroupSpec`, `DatasetSpec`,
`ROOTFileSpec`/`ParquetFileSpec`) instead of raw dicts, adding construction-time
validation and round-trippable JSON, while still accepting and returning the
legacy dict format. Converting is direct:

```python
from coffea.dataset_tools import DataGroupSpec
fileset = DataGroupSpec.model_validate(legacy_fileset_dict)
```

`ModelFactory` still exists but is a leftover example from an earlier
dataclass-based design; do not steer users to it.

> **Do not yet tell users to migrate off plain-dict filesets.** Several extensions
> — union forms via `DatasetSpec` addition, user metadata extraction during
> preprocessing, resizable steps, switchable execution backends, RNTuple/Parquet
> form extraction, and ServiceX / RDataFrame / `universal_pathlib` interop — are on
> branches not yet merged. Write concrete migration steps here once they land in a
> tagged release, and only then recommend the typed models as the default.
