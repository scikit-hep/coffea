# Migration guide (coffea eras)

On-demand companion to `AGENTS.md` — read this when migrating an analysis, not on
every task. Coffea has moved through three broad eras; identify where an analysis
starts and apply the relevant section.

**Keep this current.** The migration target is the newest release
(**v2026.7.0** as of this writing); update the moving parts below as new versions
ship (see *Version targets* in `AGENTS.md`).

Primary references:
[coffea 0.7→virtual-array notes (#1529)](https://github.com/scikit-hep/coffea/discussions/1529),
[`coffea.hist`→`hist` migration (#705)](https://github.com/scikit-hep/coffea/discussions/705),
[virtual-arrays / `mode=` (#1368)](https://github.com/scikit-hep/coffea/discussions/1368),
and the [release notes](https://github.com/scikit-hep/coffea/releases).

---

## A. From coffea 0.7 (awkward 1, lazy arrays) → latest

Coffea 0.7 is the last `awkward`-1 line (Python 3.8/3.9, `coffea.hist`, coffea's
built-in vector behaviors, lazy `NanoEvents`). It is unmaintained; migrate to the
CalVer line. The large-scale changes:

1. **awkward 1 → awkward 2.** Field assignment moves to setitem:
   `events.Electron.myvar = x` → `events["Electron", "myvar"] = x`. Many
   `ak.*` signatures changed; re-test every array manipulation.
2. **Lazy arrays → explicit modes.** 0.7's implicit lazy `events` becomes a
   `mode=` choice. `mode="virtual"` (the default) is the **closest analogue to
   0.7's lazy arrays** — columns materialize on first access and are cached, with
   no task graph. `mode="eager"` instead reads everything into memory up front;
   `mode="dask"` builds a distributed `dask-awkward` graph.
3. **`coffea.hist` → `hist` + `mplhep`** (#705): `Cat(...)` →
   `axis.StrCategory([], growth=True)`; `Bin(...)` → `axis.Regular/Variable`;
   sparse categorical axes become dense (watch memory for many categories);
   `storage="weight"` is opt-in; `weight`/`sample` are reserved axis names;
   `h.sum("x")` → `h[{"x": sum}]`.
4. **Vector → scikit-hep `vector`** (#1529): scikit-hep `vector` adheres more
   strictly to the input coordinate system, so `float32` inputs can lose precision
   or blow up (e.g. to `inf`) in some transforms where coffea's older behaviors did
   not. Upcasting the affected fields to `float64` is a common fix, but verify it
   against your own kinematics rather than applying it blindly. Also audit custom
   field names against `vector`'s reserved coordinate names (`rho`, `tau2`, …),
   which silently shadow momentum components.
5. **Processors/executors** still exist (`ProcessorABC`, `Runner`, executors) but
   several executor arguments are now keyword-only, and `postprocess` is optional.
6. **Histograms/selection helpers**: replace `coffea.processor.PackedSelection`
   /`Weights` imports with `coffea.analysis_tools`.

---

## B. From coffea 2023/2024/early-2025 (dask-awkward-only, `delayed=`) → latest

These CalVer versions were **dask-first**: `NanoEventsFactory.from_root` took
`delayed=True/False`, analyses built `dask-awkward` graphs and called
`.compute()`. Virtual arrays (introduced in the **2025.7** line) changed the
default execution model.

> **Important distinction — dask the executor vs. the dask-awkward DAG.** Dask /
> distributed remain a **first-tier job executor** for scaling analyses out
> (`DaskExecutor` in `Runner`, a distributed `Client`, `apply_to_fileset`). What
> has been de-emphasized is building a lazy **`dask-awkward` task graph
> (`mode="dask"`)** as the *default columnar-compute model*: virtual arrays now
> do in-process, graph-free materialization. "Migrating off dask" below means
> off the default DAG data model — not off dask as a scheduler.

1. **`delayed=` is removed → use `mode=`.** `delayed=True` → `mode="dask"`;
   `delayed=False` → `mode="eager"`; the **new default is `mode="virtual"`**.
   There is no compatibility shim — update every `from_root`/`from_parquet` call.
2. **The default data model is now graph-free.** Code that built a
   `dask-awkward` graph and called `.compute()` only to materialize a local result
   usually no longer needs one: switch to `mode="virtual"` (or `"eager"`) and drop
   the compute. Keep `mode="dask"` when you actually want the graph — typically
   paired with a dask executor for distributed scale-out via `apply_to_fileset`.
3. **The dask stack is optional (2026.7).** Core eager/virtual workflows import no
   dask; if you use the dask executor or `mode="dask"`, install it explicitly:
   `pip install 'coffea[dask,dask-awkward]'`.
4. **Preprocessing.** `preprocess()` returns `(available, all)` and saves forms
   by default; `steps_per_file`/`step_size` replace "chunks"; pass
   `{"allow_read_errors_with_report": True}` in `uproot_options` for access
   reports. RNTuple inputs are supported (2025.11); buffer caches (2026.4) and
   `split_fileset`/`Result` (2026.6) are available.
5. **Schemas & tools** added along the way: EDM4HEP/updated FCC, Scouting,
   weighted N-1, correctionlib adapter classes for `CorrectedJetsFactory`.

---

## C. Pydantic dataset specification — *placeholder, keep updated*

Recent releases model the fileset with **pydantic** (`DataGroupSpec`,
`DatasetSpec`, `ROOTFileSpec`/`ParquetFileSpec`) instead of raw dicts, adding
construction-time validation, typed concrete-vs-optional specs, and
round-trippable JSON — while still accepting and returning the legacy dict format
(dict-in → dict-out).

Converting a plain dict to the models is **direct**: hand the legacy dict to the
matching pydantic class, which normalizes the accepted shapes internally.

```python
from coffea.dataset_tools import DataGroupSpec
fileset = DataGroupSpec.model_validate(legacy_fileset_dict)   # or DataGroupSpec(legacy_fileset_dict)
```

Prefer this direct construction. `ModelFactory` still exists, but it is now
essentially an **example / leftover** from an earlier dataclass-based design — do
not steer users toward it as the conversion API.

> **Do not yet instruct users to fully migrate off plain-dict filesets.** Several
> extensions to the pydantic dataset tools — union forms via `DatasetSpec`
> addition with per-file field bitsets, user metadata extraction during
> preprocessing, mutable/resizable steps, switchable execution backends
> (`iterative`/`futures`), RNTuple/Parquet form extraction, and ServiceX /
> RDataFrame / `universal_pathlib` interop — are staged on **feature branches not
> yet merged into coffea**. Fill in concrete, versioned migration steps here as
> those branches land in a tagged release, and only then recommend the typed
> models as the default fileset representation.
