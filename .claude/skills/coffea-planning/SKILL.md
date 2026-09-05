---
name: coffea-planning
description: Write or revise the implementation plan for a coffea change in .agent-work/plan.md. Use at the start of a multi-step change and after each planning review.
---

# Planning

Produce `.agent-work/plan.md`: the plan a separate agent will implement without
asking you anything. Protocol and severities: `.claude/skills/README.md`.

## Inputs

- the task, and the fidelity level the driver passed
- `.agent-work/plan-review-NN.md`, the highest-numbered one, if any
- `.agent-work/blocked.md`, if implementation fell back to planning
- *Fidelity buckets* in `.claude/skills/coffea-change/SKILL.md`

Fresh session: read `AGENTS.md`, the `ARCHITECTURE.md` section for the subsystem
(`grep -n '^#'`), the code you will change and its tests. State what you assumed.

## The plan

Line 1: `fidelity: <level>`, the highest bucket of any path the change will
touch. This line, not the driver's guess, decides who runs every later step. Then:

1. **Goal** — one paragraph: what changes for a user of coffea, and what does not.
2. **Current behavior** — what the code does today, with `path:line` you have read.
3. **Steps** — ordered, independently implementable, each naming its files.
4. **Tests** — per behavior change, the mirrored test under `tests/` and what it
   asserts; name the case that fails against the unchanged code.
5. **Risks and assumptions** — what could be wrong, what you could not verify.
6. **Out of scope** — what you are deliberately not doing.

## Revising

Address every finding: apply it, or record one line in the plan saying why not.
On the folding round (only LOW and NIT found) apply exactly those findings and
stop; any other edit re-enters review. If `blocked.md` exists, the previous plan
failed in contact with the code: change the approach. Smallest change that works,
no adjacent cleanup; the `AGENTS.md` hard rules bind the plan as much as the code.
