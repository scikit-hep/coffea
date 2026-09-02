# Review-loop skills

Four skills that take a change from idea to merged-ready through two review
loops: planning to planning-review until the plan holds up, then implementation
to implementation-review until the code does.

They are plain markdown and name no model or vendor. Any agent that can read
files, edit files and run tests can follow them.

| skill | reads | writes |
| --- | --- | --- |
| `coffea-planning` | task; latest `plan-review-NN.md`; `blocked.md` if present | `plan.md` |
| `coffea-planning-review` | `plan.md` | `plan-review-NN.md` |
| `coffea-implementation` | final `plan.md`; latest `impl-review-NN.md` | code, tests, `impl-notes.md` |
| `coffea-implementation-review` | `plan.md`, `impl-notes.md`, the diff | `impl-review-NN.md` |

## Fresh context per step

Every step runs in a new agent session, inheriting no transcript. Its only inputs
are the repository and the files named above, so a reviewer cannot inherit the
author's justification for a decision it is meant to judge. The file hand-off is
the mechanism: an agent either read `plan.md` or it did not.

## Artifacts

All loop state lives in `.agent-work/`, which is gitignored. It is scratch and
must never appear in a commit.

```
.agent-work/
  plan.md               # current plan; planning overwrites it each round
  plan-review-01.md     # one file per planning-review round, numbered
  impl-notes.md         # what was built, and any deviation from the plan
  impl-review-01.md     # one file per implementation-review round, numbered
  blocked.md            # only when implementation hits an unrecoverable problem
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
        ^                            │                     │
        └────────────────────────────┘  while any CRITICAL/HIGH/MEDIUM
                                     │
                          unrecoverable problem
                                     │
                                     v
                            write blocked.md,
                          restart at coffea-planning
```

There is no iteration cap; a loop exits on its exit condition, not on a count. If
the same finding returns unchanged twice, say so and hand back to the human
rather than looping again.

A clean implementation review is not a substitute for the author reading their
own diff before opening a PR.

## Capability tiers

Skills name a tier, never a model:

- **C1** — strongest available reasoning
- **C2** — strong general purpose
- **C3** — fast and cheap

Fidelity scales the tier to the blast radius: `critical` (core correctness paths)
pins every step to C1, `standard` uses the per-step defaults, `economy` (docs,
isolated tests) drops one tier where the skill allows it.

If a review rejects the same artifact twice at a tier, redo it one tier stronger
rather than trying a third time at the same one.

The mapping from tiers to concrete models belongs to whichever harness is driving,
not here: it is the fastest-rotting part of this document.
