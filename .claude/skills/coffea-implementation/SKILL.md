---
name: coffea-implementation
description: Implement the agreed .agent-work/plan.md in the coffea repository, recording what changed in .agent-work/impl-notes.md. Use after planning is clean, and again after each implementation review. Default capability tier C2.
---

# Implementation

Build what `.agent-work/plan.md` specifies. Protocol, severities and tiers:
`.claude/skills/README.md`.

**Default tier C2** — the hard reasoning was done in planning. Mechanical steps
may drop to C3; raise to C1 for correctness-sensitive code under `critical`
fidelity.

## Inputs

- `.agent-work/plan.md`, the final plan
- `.agent-work/impl-review-NN.md`, the highest-numbered one, if any

Fresh session: read the code before changing it, even if the plan describes it.

## Rules

- **Implement the plan, not your own better idea.** If the plan is wrong, stop —
  see below. Quietly improving on it destroys the guarantee that what was reviewed
  is what got built.
- **Smallest change that works.** No adjacent cleanup, no refactor in passing, no
  error handling for cases that cannot occur.
- **Tests are part of the step.** Every behavior change gets its mirrored test
  under `tests/`, confirmed to fail against the unchanged behavior.
- **Comments are sparse** — a line of *why* where the code is non-obvious.
  `ARCHITECTURE.md`, *Conventions*, gives the density.
- **The `AGENTS.md` hard rules bind absolutely**: setitem field assignment, no
  Python loop over `axis=0`, guarded dask imports.

## Before you finish

`pre-commit run --all-files` and `pytest` must pass. If an unrelated test fails,
find out why; do not route around it, and never weaken an assertion or a tolerance
to get green.

## Output

Write `.agent-work/impl-notes.md`:

- **What changed** — per file, one line: what and why.
- **Deviations** — every place the implementation differs from the plan, with the
  reason. An undeclared deviation is the most expensive thing to find in review.
- **Test evidence** — which tests cover the change, and that they fail without it.
- **Left undone** — anything from the plan not implemented, and why.

## When the plan does not survive

If the plan cannot be implemented as written — it rests on a misreading of the
code, the approach does not work, or it would need changes well outside its scope
— stop rather than improvise.

Write `.agent-work/blocked.md`: what you attempted, what in the code contradicts
the plan (with `path:line`), and which plan assumptions are false. Leave the tree
clean, then hand back to planning. This is a normal outcome. A wrong plan caught
in contact with the code is worth more than a plausible implementation of it.
