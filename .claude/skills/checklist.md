# Reviewer checklist

## Previous findings

When a previous review file of the same kind exists, open the review with one
line per finding in it, heading copied verbatim, prefixed:

- `RESOLVED` — the defect is gone
- `REPEAT` — the defect is still present (your judgment, not a string match)
- `SUPERSEDED` — the design changed so the finding no longer applies

Omit the block for the first review of a kind.

## Hunt for

- **Silent wrongness** — wrong numbers with no error: a dropped field, a
  variation from the wrong baseline, a mask on the wrong axis. CRITICAL.
- **Tests that cannot fail** — pass against the unchanged behavior. HIGH.
- **Undeclared deviation** — the diff or plan does something not sanctioned and
  not recorded. HIGH.
- **Hard rules** — the `AGENTS.md` rules: setitem field assignment, no Python
  loop over `axis=0` outside numba, guarded dask imports.
- **Scope creep** — steps or hunks beyond the goal, abstractions with one caller.
- **Comment bloat** — restated code, narrated history, issue numbers.
- **Weakened checks** — a loosened tolerance, softened assertion, added skip.
- **Fidelity set too low** — `plan.md` names a level below what its paths give
  (`coffea-change`, *Fidelity buckets*). HIGH.

## Finding format and verdict

Most severe first, then exactly one verdict line:

```
### [SEVERITY] short claim
where:   plan section or path:line
why:     the concrete failure — inputs or state, and the wrong result
fix:     what would resolve it

VERDICT: BLOCKING (n CRITICAL, n HIGH, n MEDIUM)
VERDICT: CLEAN (only LOW/NIT, or none)
```

Do not fix what you find. A reviewer that edits the artifact has stopped
reviewing it.
