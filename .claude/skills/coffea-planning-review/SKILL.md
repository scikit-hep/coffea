---
name: coffea-planning-review
description: Adversarially review .agent-work/plan.md for a coffea change and write findings to .agent-work/plan-review-NN.md. Use after each planning round until only LOW and NIT remain.
---

# Planning review

Try to break `.agent-work/plan.md` before anyone writes code. Protocol:
`.claude/skills/README.md`; hunt list, format and verdict: `.claude/skills/checklist.md`.

## Inputs

- `.agent-work/plan.md`
- the previous `.agent-work/plan-review-NN.md`, if any
- *Fidelity buckets* in `.claude/skills/coffea-change/SKILL.md`

You did not write this plan and you are not obliged to like it. No findings on a
non-trivial plan is more likely a lazy review than a perfect plan.

## Verify against the code

Do not review the plan as prose. Open the files it names and check:

- Does *Current behavior* match what the code does?
- Do the `path:line` references exist and say what the plan claims? An
  unverifiable claim is itself a finding.
- Do the named tests exist and cover what the plan assumes?
- Is each step implementable as written, in the order given, and checkable?
  Too vague, or a pre-written bad diff, is a finding.

## Output

`.agent-work/plan-review-NN.md`, numbered from `01`: the *Previous findings*
block, findings in the checklist format, one verdict line. `CLEAN` sends the
plan to a single folding round, then to implementation.
