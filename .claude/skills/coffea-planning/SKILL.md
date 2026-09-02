---
name: coffea-planning
description: Write or revise the implementation plan for a coffea change, in .agent-work/plan.md. Use at the start of a multi-step change, and again after each planning review. Default capability tier C1.
---

# Planning

Produce `.agent-work/plan.md`: the plan a separate agent will implement without
asking you anything. Protocol, severities and tiers: `.claude/skills/README.md`.

**Default tier C1** — planning errors propagate into every later step. `economy`
fidelity may drop to C2.

## Inputs

- the task
- `.agent-work/plan-review-NN.md`, the highest-numbered one, if any
- `.agent-work/blocked.md`, if implementation fell back to planning

You are in a fresh session. Read the repository yourself.

## Before writing

Read `AGENTS.md` and the one `ARCHITECTURE.md` section covering the subsystem you
are changing (`grep -n '^#' ARCHITECTURE.md` for the table of contents). Then read
the code you intend to change and the tests that already cover it. A plan written
from file names alone will not survive review.

State what you verified and what you assumed. A flagged assumption is a finding
the reviewer can check; a hidden one is a bug.

## The plan

Write `.agent-work/plan.md` containing:

1. **Goal** — one paragraph. What changes for a user of coffea, and what does not.
2. **Current behavior** — what the code does today, with `path:line` references
   you have actually read.
3. **Steps** — ordered, each independently implementable and each naming the files
   it touches. A step a reviewer cannot check is too vague.
4. **Tests** — for every behavior change, the mirrored test under `tests/` that
   will prove it and what it asserts. Name the discriminating case: a test that
   passes against the unchanged code is worthless.
5. **Risks and assumptions** — what could be wrong, what you could not verify,
   what would make you abandon this approach.
6. **Out of scope** — what you are deliberately not doing.

## Revising

Address every finding in the review: apply it, or record one line in the plan
saying why not. Silently dropping a finding restarts the loop for no reason.

On the folding round — invoked when the review found only LOW and NIT — absorb
those and stop. Do not reopen settled decisions.

If `blocked.md` exists, the previous plan failed in contact with the code. Treat
its account as evidence and change the approach rather than restating it.

## Constraints

- Smallest change that works. No abstraction for hypothetical needs, no adjacent
  cleanup, no feature flags unless asked.
- The `AGENTS.md` hard rules bind the plan as much as the code.
- Prefer editing existing modules to adding new ones.
