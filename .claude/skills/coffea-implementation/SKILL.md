---
name: coffea-implementation
description: Implement the agreed .agent-work/plan.md in the coffea repository, recording what changed in .agent-work/impl-notes.md. Use after planning is clean and after each implementation review.
---

# Implementation

Build what `.agent-work/plan.md` specifies. Protocol, verification scope and
recipe: `.claude/skills/README.md`.

## Inputs

- `.agent-work/plan.md`, the final plan
- `.agent-work/impl-review-NN.md`, the highest-numbered one, if any
- the code itself, before changing it, even if the plan describes it

## Rules

- **Implement the plan, not your own better idea.** If the plan is wrong, stop
  (below); quiet improvement breaks the guarantee that what was reviewed is built.
- **Smallest change that works.** No adjacent cleanup, no refactor in passing.
- **Tests are part of the step.** Every behavior change gets its mirrored test
  under `tests/`, shown to fail without the change by the README recipe.
- **Comments are sparse** (a line of *why* where the code is non-obvious) and
  **the `AGENTS.md` hard rules bind absolutely.**
- **Per round, run the README verification scope.** Never weaken an assertion,
  tolerance or skip to get green; diagnose an unrelated failure.

## Output

`.agent-work/impl-notes.md`: **What changed** (per file, one line); **Deviations**
(every difference from the plan, with the reason); **Test evidence** (per new or
changed test, `fails without the fix: <test id>` or `no baseline: <test id>`);
**Left undone** (anything not implemented, and why).

## When the plan does not survive

If the plan cannot be implemented as written, stop rather than improvise: write
`.agent-work/blocked.md` (what you attempted, what in the code contradicts the
plan with `path:line`, which assumptions are false), leave the tree clean. Normal.
