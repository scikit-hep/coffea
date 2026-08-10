# Review-loop skills

Four skills that take a change from idea to implementation through two review
loops: **planning → planning-review** until the plan holds up, then
**implementation → implementation-review** until the code does.

They are plain markdown and name no tool, model, or vendor. Any agent that can
read files, edit files, and run tests can follow them.

| skill | reads | writes |
| --- | --- | --- |
| `coffea-planning` | task; latest `plan-review-NN.md`; `blocked.md` if present | `plan.md` |
| `coffea-planning-review` | `plan.md` | `plan-review-NN.md` |
| `coffea-implementation` | final `plan.md`; latest `impl-review-NN.md` | code, tests, `impl-notes.md` |
| `coffea-implementation-review` | `plan.md`, `impl-notes.md`, the diff | `impl-review-NN.md` |

## Fresh context per step

**Every step runs in a new agent session.** No step inherits another's
transcript; its only inputs are the repository and the files named above. This
keeps each context small and stops one step's reasoning from anchoring the next —
in particular, it stops a reviewer from inheriting the author's justification for
a decision it is supposed to judge independently.

The file hand-off *is* the mechanism. Passing state on disk rather than in
conversation is what makes the freshness rule checkable and portable: an agent
either read `plan.md` or it did not.

## Artifacts

All loop state lives in `.agent-work/`, which is gitignored — it is scratch, not
deliverable, and must never appear in a commit.

```
.agent-work/
  plan.md               # current plan; planning overwrites it each round
  plan-review-01.md     # one file per planning-review round, numbered
  impl-notes.md         # what was built, and any deviation from the plan
  impl-review-01.md     # one file per implementation-review round, numbered
  blocked.md            # only when implementation hits an unrecoverable problem
```

## Severities

Reviews classify every finding. The severity decides whether the loop continues.

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

**There is no iteration cap.** A loop exits on its exit condition, not on a
count. If rounds stop making progress — the same finding returns unchanged twice
— say so plainly and hand back to the human rather than looping again.

## Capability tiers

Steps differ in how much reasoning they need, and changes differ in how much care
they are worth. Skills refer to capability tiers, never to a named model:

- **C1** — strongest available reasoning
- **C2** — strong general purpose
- **C3** — fast and cheap

Fidelity scales the tier to the blast radius of the change: `critical` (core
correctness paths) pins every step to C1; `standard` uses the per-step defaults
each skill states; `economy` (docs, isolated tests) drops one tier where the skill
allows it.

If a review rejects the same artifact twice at a tier, **redo it one tier
stronger** rather than attempting a third time at the same tier. Looping a model
that cannot do the task is the most expensive way to fail.

> The mapping from C1/C2/C3 to concrete models is intentionally not here: it is
> the fastest-rotting part and is specific to whichever tool is driving. Each
> harness supplies its own mapping (for Claude Code, `CLAUDE.md`). **This mapping
> is not yet agreed — see the skills pull request discussion.**
