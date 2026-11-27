---
name: coffea-implementation-review
description: Adversarially review the implemented diff against .agent-work/plan.md and write findings to .agent-work/impl-review-NN.md. Use after each implementation round until no CRITICAL, HIGH or MEDIUM remain.
---

# Implementation review

Review the diff that was produced against the plan that was agreed. Protocol,
verification scope and recipe: `.claude/skills/README.md`; hunt list, format and
verdict: `.claude/skills/checklist.md`.

## Inputs

- the diff: `git diff "$(git merge-base origin/master HEAD)"`
- `.agent-work/plan.md` and `.agent-work/impl-notes.md`
- the previous `.agent-work/impl-review-NN.md`, if any

Fresh session: judge the diff, not the explanation of it. Where `impl-notes.md`
and the diff disagree, the diff is the truth. A hunk can be correct in isolation
and wrong in place: open the files around each change, what else calls this,
what invariant the old code held, what the mirrored test actually asserts.

## Test evidence

Re-run every `fails without the fix` line in `impl-notes.md` with the README
recipe. A missing line for a new or changed test, or one that does not
reproduce, is HIGH.

## Before returning CLEAN

Run the full `pytest` and `pre-commit run --all-files` yourself on the current
tree; a failure is BLOCKING with the failing test named, and a verdict asserted
without running them is not a verdict.

## Output

`.agent-work/impl-review-NN.md`, numbered from `01`: the *Previous findings*
block, findings in the checklist format, one verdict line. `CLEAN` ends the
loop; remaining LOW and NIT are applied or recorded in `impl-notes.md` with a
reason.
