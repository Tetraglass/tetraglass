---
name: next-task
description: Runs the standard development loop for this project. It picks the next open task from docs/roadmap.md, plans it, implements it with tests, verifies it and logs progress. Use whenever the user says "continue", "next", "keep going", "work on the roadmap", "start milestone Mx", names a roadmap task id such as M0.4, or asks what to do next.
---

# Next task

One pass of this loop should finish one roadmap task completely. Finished small tasks are worth more than several half-done ones, because the developer works in short evening sessions.

## 1. Orient

- If the repository root has no `Cargo.toml`, run `/bootstrap-workspace` instead and stop.
- Read the last three entries of `docs/progress.md`.
- Open `docs/roadmap.md` and find the current milestone: the first one with unticked tasks.
- Skim `docs/decisions.md` for decisions that touch the task area.

## 2. Pick

- If the user named a task, use it. Otherwise take the first unticked task in the current milestone whose listed dependencies are ticked.
- If the task looks bigger than one session (roughly more than 500 changed lines or more than two crates), split it in `docs/roadmap.md` into numbered sub-tasks first (M2.3a, M2.3b) and take the first.
- Never pick a task from a later milestone unless the user asks.

## 3. Plan

Write an implementation plan before editing code. It contains:

- the goal in one sentence and the roadmap id,
- the files and crates touched,
- public API or scene schema changes (new types, functions, fields),
- the tests that prove it works,
- risks, and anything in `AGENTS.md` invariants or `.agents/rules/` that constrains it.

Wait for the user's approval if the plan changes a public API after 0.1, changes the scene schema, adds a dependency or contradicts a decision. Otherwise continue.

## 4. Implement

Work in small steps that each compile. Use the specialized skills where they apply:

| Task kind | Skill |
| --- | --- |
| A new or changed filter | `/add-filter` |
| A new or changed file reader | `/add-reader` |
| Scene files or gallery entries | `/author-scene` |
| Anything that changes rendered pixels | `/render-check` |
| Test meshes | `/make-test-data` |
| Blender export or Cycles rendering | `/blender-bridge` |

## 5. Verify

Run `/verify` and fix every failure. Do not weaken, skip or delete tests to get green; if a test seems wrong, explain why and ask.

## 6. Document

- Rustdoc for every new public item.
- `docs/scene-format.md` when the scene changed.
- `docs/design-plan.md` only if the design itself changed, together with a new entry in `docs/decisions.md`.

## 7. Record and commit

Tick the task in `docs/roadmap.md`. Append to `docs/progress.md`:

```
## YYYY-MM-DD: M0.4 Core grid types
- Done: what now works, in one or two lines.
- Tests: what was added.
- Decisions: none, or D-0xx with a one-line summary.
- Dependencies added: none, or name and reason.
- Follow-ups: anything discovered but not done.
```

Stage all task changes and create a local git commit using a Conventional Commits message, for example `feat(core): add unstructured grid and typed arrays`. Do not push unless the user asks.

## 8. Next

If every task of the milestone is ticked, suggest `/milestone-gate`. Otherwise name the next task in one line so the user can say "continue".
