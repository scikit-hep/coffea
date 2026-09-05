# Review-loop skills

Five skills take a change from idea to merge-ready through two review loops:
planning to planning-review until the plan holds up, then implementation to
implementation-review until the code does. `coffea-change` drives the sequence
and picks who runs each step; the other four are the steps, plain markdown
naming no model or vendor (`coffea-change` names one harness, in *Sequence*
only). Reviewers also read `.claude/skills/checklist.md`.

| skill | reads | writes |
| --- | --- | --- |
| `coffea-change` | the task; `plan.md` | nothing; it sequences the others and clears `.agent-work/` after a block |
| `coffea-planning` | task; fidelity; latest `plan-review-NN.md`; `blocked.md` if present | `plan.md` |
| `coffea-planning-review` | `plan.md`; previous `plan-review-NN.md` | `plan-review-NN.md` |
| `coffea-implementation` | final `plan.md`; latest `impl-review-NN.md` | code, tests, `impl-notes.md` |
| `coffea-implementation-review` | `plan.md`, `impl-notes.md`, the diff; previous `impl-review-NN.md` | `impl-review-NN.md` |

## Fresh context per step

Every step runs in a new agent session, inheriting no transcript. Its only inputs
are the repository and the files named above, so a reviewer cannot inherit the
author's justification for a decision it is meant to judge.

## Artifacts

All loop state lives in `.agent-work/`, gitignored scratch that never appears in a commit.

```
.agent-work/
  plan.md               # current plan; planning overwrites it each round
  plan-review-01.md     # one file per planning-review round, numbered
  impl-notes.md         # what was built, and any deviation from the plan
  impl-review-01.md     # one file per implementation-review round, numbered
  blocked.md            # only when implementation hits an unrecoverable problem
  escalated.md          # only when coffea-change has redone a step one tier up
```

## Severities

| severity | meaning |
| --- | --- |
| **CRITICAL** | silently wrong results, data corruption, or a security hole |
| **HIGH** | incorrect behavior in a plausible case; a behavior change with no test; a public API broken without cause |
| **MEDIUM** | a design flaw that will cause bugs or rework; scope that cannot be implemented as written |
| **LOW** | a real improvement that blocks nothing |
| **NIT** | naming, wording, or formatting preference |

## The loop

```
        ┌──────────────────────────────────────┐
        v                                      │
  coffea-planning ──> coffea-planning-review ──┘  while any CRITICAL/HIGH/MEDIUM
        │
        │  one folding round: absorb the remaining LOW/NIT into the plan
        v
  coffea-implementation ──> coffea-implementation-review ──┐
        │       ^                                          │
        │       └──────────────────────────────────────────┘  while any CRITICAL/HIGH/MEDIUM
        │
        │  unrecoverable problem: write blocked.md
        v
  restart at coffea-planning
```

No iteration cap: a loop exits on its exit condition. `coffea-change` holds the
stop rule for a finding that returns unchanged.

## Verification scope

Per round: `pre-commit run --files <touched paths>` (prek is a drop-in) and the
test modules mirroring the touched code. The full `pytest` and
`pre-commit run --all-files` run once per candidate `CLEAN`, by the reviewer; a
failure there is BLOCKING with the failing test named. Reviewers take the diff as
`git diff "$(git merge-base origin/master HEAD)"`, which ignores staging.

## Showing a test discriminates

A test proves a change only if it fails without it. Evidence, per test:
```
base=$(git merge-base origin/master HEAD)
git add -A
git show "$base:<path>" > <path>   # each source file the test exercises;
                                    # rm <path> if it does not exist at base
pytest <test id>                    # expect failure or a collection error
git checkout -- <path>              # restore the index copy
```

A test whose only subject is a new module records `no baseline` instead.
