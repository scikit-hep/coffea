---
name: coffea-planning-review
description: Adversarially review .agent-work/plan.md for a coffea change and write findings to .agent-work/plan-review-NN.md. Use after each planning round, until only LOW and NIT findings remain. Default capability tier C1.
---

# Planning review

Try to break `.agent-work/plan.md` before anyone writes code. Protocol, severities
and tiers: `.claude/skills/README.md`.

**Default tier C1** — the cheapest place in the loop to catch a mistake. Do not
economize here.

## Stance

You did not write this plan and you are not obliged to like it. Find what is wrong
with it rather than confirming it. No findings on a non-trivial plan is more
likely a lazy review than a perfect plan.

## Verify against the code

Do not review the plan as prose. Open the files it names and check its claims:

- Does *Current behavior* match what the code actually does?
- Do the `path:line` references exist and say what the plan claims?
- Do the named tests exist, and cover what the plan assumes they cover?
- Is each step implementable as written, in the order given?

An unverifiable claim is itself a finding.

## What to hunt for

- **Silent wrongness** — a change that produces wrong numbers rather than an
  error: dropped fields, a variation that corrupts its baseline, a mask on the
  wrong axis. CRITICAL in a physics library.
- **Missing discriminating test** — a behavior change whose test would pass
  against the unchanged code proves nothing. HIGH.
- **Violated hard rules** — the `AGENTS.md` rules, especially setitem field
  assignment and no Python loops over `axis=0`.
- **Scope creep** — steps beyond the stated goal, or abstractions with one caller.
- **Unstated assumptions** — anything the plan needs to be true and has not
  checked.
- **Wrong altitude** — too vague to implement, or so detailed it has pre-written
  the diff badly.

## Output

Write `.agent-work/plan-review-NN.md`, numbering from `01`, most severe first:

```
### [SEVERITY] short claim
where:   plan section, and path:line in the repo if it concerns real code
why:     the concrete failure this causes — inputs or state, and the wrong result
fix:     what would resolve it
```

End with one verdict line, either:

```
VERDICT: BLOCKING (n CRITICAL, n HIGH, n MEDIUM)
VERDICT: CLEAN-ENOUGH (only LOW/NIT)
```

`CLEAN-ENOUGH` sends the plan to a single folding round, then to implementation.
Anything else sends it back for revision.

Do not rewrite the plan yourself. Finding and fixing in one pass is how a reviewer
talks itself into accepting its own reasoning.
