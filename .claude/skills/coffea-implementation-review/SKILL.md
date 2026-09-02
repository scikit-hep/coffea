---
name: coffea-implementation-review
description: Adversarially review the implemented diff against .agent-work/plan.md and write findings to .agent-work/impl-review-NN.md. Use after each implementation round, until no CRITICAL, HIGH or MEDIUM findings remain. Default capability tier C1.
---

# Implementation review

Review the diff that was actually produced against the plan that was agreed.
Protocol, severities and tiers: `.claude/skills/README.md`.

**Default tier C1** — the last automated check before a human reads the change.
`economy` fidelity may drop to C2 for docs-only diffs.

## Inputs

- the working diff (`git diff` against the merge base)
- `.agent-work/plan.md` and `.agent-work/impl-notes.md`

Fresh session: judge the diff, not the explanation of it. Where `impl-notes.md`
and the diff disagree, the diff is the truth.

## Read the diff in place

A hunk can be correct in isolation and wrong in place. Open the files around each
change: what else calls this, what invariant the old code held, what the mirrored
test actually asserts.

## What to hunt for

- **Silent wrongness** — wrong results with no error raised. The dominant risk in
  coffea: a dropped field, a variation computed from the wrong baseline, a mask on
  the wrong axis, a schema that truncates a name. CRITICAL.
- **Tests that cannot fail** — a test exists but would pass against the unchanged
  behavior, so the change is untested in substance. HIGH.
- **Undeclared deviation from the plan** — the diff does something the plan did
  not sanction and `impl-notes.md` does not mention. HIGH: it never went through
  planning review.
- **Hard-rule violations** — attribute field assignment, a Python loop over
  `axis=0` outside numba, an unguarded dask import.
- **Scope creep** — changes unrelated to the goal, including drive-by cleanup.
- **Comment bloat** — paragraphs restating the code, narration of the bug that was
  fixed, issue or PR numbers.
- **Weakened checks** — a loosened tolerance, a softened assertion, a skip added to
  make a test pass.

## Output

Write `.agent-work/impl-review-NN.md`, numbering from `01`, most severe first:

```
### [SEVERITY] short claim
where:   path:line in the diff
why:     the concrete failure — inputs or state, and the wrong result produced
fix:     what would resolve it
```

End with one verdict line, either:

```
VERDICT: BLOCKING (n CRITICAL, n HIGH, n MEDIUM)
VERDICT: CLEAN (only LOW/NIT, or none)
```

`CLEAN` ends the loop. Remaining LOW and NIT findings are either applied or
recorded in `impl-notes.md` with a one-line reason.

Do not fix what you find. A reviewer that edits the code has stopped reviewing it.

## Before returning CLEAN

Confirm `pre-commit run --all-files` and `pytest` pass on the current tree. A
verdict asserted without running them is not a verdict.
