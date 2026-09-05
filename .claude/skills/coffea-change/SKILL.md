---
name: coffea-change
description: Drive a coffea change through the planning and implementation review loops, choosing the fidelity level and the agent tier for each step. Use for any multi-step change; a human can follow it by hand, one skill per session.
---

# Driving a change

You sequence the four loop skills; you do not plan, implement or review.
Protocol: `.claude/skills/README.md`.

## Fidelity buckets

The highest bucket of any path the change touches wins.

| level | paths |
| --- | --- |
| `critical` | `AGENTS.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `docs/agents/**`, `.claude/**`, `.github/CODEOWNERS`, `.github/zizmor.yml`, `.github/workflows/agent-file-guard.yml`; `src/coffea/**` except the executor transports `src/coffea/processor/dask/**`, `src/coffea/processor/parsl/**`, `src/coffea/processor/taskvine_executor.py` |
| `economy` | `docs/source/**`, `binder/**` |
| `standard` | everything else: the transports, `tests/**`, `.github/**`, `pyproject.toml` |

Before a plan exists, guess from the task text and pass the guess to planning.
From `plan.md` on, its first line (`fidelity: <level>`) governs, in either
direction.

## Tiers

| level | planning | planning-review | implementation | implementation-review |
| --- | --- | --- | --- | --- |
| `critical` | C1 | C1 | C1 | C1 |
| `standard` | C1 | C1 | C2 | C2 |
| `economy` | C2 | C2 | C3 | C2 |

C1 is the strongest reasoning available, C2 strong general purpose, C3 fast
and cheap. The concrete models live only in `.claude/agents/coffea-c1.md`,
`.claude/agents/coffea-c2.md` and `.claude/agents/coffea-c3.md`. A harness
offering a model above C1's may pass it for `critical` steps.

## Sequence

Each step is a fresh session. By hand: open a new session, invoke the skill,
paste the task or the artifact names. In Claude Code: the Agent tool with
`subagent_type` `coffea-c1`, `coffea-c2` or `coffea-c3` and the prompt
`Read .claude/skills/<skill>/SKILL.md and follow it. Inputs: ...`; one
sub-agent per step, never reused.

1. `coffea-planning` with the task and the fidelity level: the guess on the
   first pass, `plan.md`'s `fidelity:` line thereafter.
2. If planning ran at C2 and `grep -m1 '^fidelity:' .agent-work/plan.md` names a
   level whose planning tier is C1, re-run step 1 at C1 with that level.
3. `coffea-planning-review`. `BLOCKING`: back to step 1 with the review.
   `CLEAN`: one folding round of `coffea-planning`; if `plan.md` now ends with
   `folding: exceeded`, delete that line and go back to step 3, else step 4.
4. `coffea-implementation`.
5. If `.agent-work/blocked.md` exists: remove every other file in `.agent-work/`;
   if `git status --porcelain` is not empty, stop and hand to the human; else go
   to step 1. Planning deletes `blocked.md` once it has read it.
6. `coffea-implementation-review`. `BLOCKING`: back to step 4. `CLEAN`: done.

## Stop rule

A `BLOCKING` review whose *Previous findings* block has a `REPEAT` line carrying
CRITICAL, HIGH or MEDIUM stops the loop. If `.agent-work/escalated.md` does not
name the rejected step and that step ran below C1: append `<step> C<n>` to
`escalated.md`, redo that one step at the next tier up, later steps keeping
their table tiers. Otherwise stop and hand to the human with the review file.

## Done

Hand back to the human with the diff and `.agent-work/impl-notes.md`. The human
reads both and opens the PR; the driver never opens it.
