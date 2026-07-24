# CLAUDE.md

Guidance for Claude Code (claude.ai/code) working in this repository.

The full, tool-agnostic guide — repository layout, coffea idioms (NanoEvents
modes, awkward-2 patterns, histogramming, preprocessing, processors), coding
conventions, and the **migration guide** across coffea's 0.7 / dask-awkward /
virtual-array / pydantic eras — lives in `AGENTS.md` and is imported here:

@AGENTS.md

## Claude Code specifics

- **Read `AGENTS.md` first.** It is the source of truth; the notes below only add
  Claude-workflow reminders. Do not duplicate its content here — extend
  `AGENTS.md` instead so both stay in sync.
- **Before changing behavior**, locate the code with a search, read the relevant
  module and its mirrored test in `tests/`, then make the smallest change that
  satisfies the request. Prefer editing existing modules over adding new ones.
- **Validate locally** the way CI does: `pre-commit run --all-files` and
  `pytest` (use `pytest -n auto` for speed, or a `-k` selection while iterating).
  Both must pass before you propose changes.
- **Match the house comment style**: comments state current behavior and non-obvious
  rationale — no issue/PR numbers, no "used to…/before the fix…" narration.
- **The dask stack is optional** (v2026.7.0+). Don't assume `dask`/`dask-awkward`
  are importable; guard dask-only paths and mirror the lazy-import pattern in
  `coffea/util.py` (`_import_dask`, `_import_dask_awkward`).
- **When advising users on migration**, follow the eras in `AGENTS.md`: the
  pydantic-dataset-tools migration is a *placeholder* — do not tell users to
  abandon plain-dict filesets until the staged feature branches merge into a
  tagged release.
- **Keep the guide current.** When a new coffea version ships or a staged feature
  lands, update the version targets and the migration section in `AGENTS.md`.
