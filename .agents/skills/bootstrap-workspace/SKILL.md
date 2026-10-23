---
name: bootstrap-workspace
description: Creates the Cargo workspace, crate skeletons, xtask, CI pipeline and license policy for this project, following docs/design-plan.md, then commits the result to git and offers to push it to origin. Use once at the very start, whenever the repository root has no Cargo.toml, or when the user asks to set up, scaffold, initialize or name the project.
---

# Bootstrap the workspace

This turns the planning files into a compiling, verified Cargo workspace, committed to git. It covers roadmap task M0.1.

## 0. Preconditions

- If a `Cargo.toml` already exists at the repository root, stop. Report what exists and ask whether the user wants a repair instead of a fresh bootstrap.
- Read `AGENTS.md`, `docs/decisions.md` and the Architecture section of `docs/design-plan.md`.

## 1. Check git

Run these and keep the results for later steps:

```
git rev-parse --is-inside-work-tree
git branch --show-current
git status --porcelain
git remote get-url origin
```

- **Not a git repository:** tell the user, and offer to run `git init -b main`. Explain that the remote repository (for example on GitHub) is theirs to create; this skill never creates one. Continue only after they answer.
- **Uncommitted changes:** list them. If they are only the planning files from the starter kit, offer to commit them first as a separate commit, `docs: add design plan and agent setup`. Otherwise ask the user to commit or stash them, so the bootstrap commit contains only bootstrap changes.
- **No `origin` remote:** warn that the repository URL in `Cargo.toml` will stay a placeholder and nothing can be pushed. Continue, and add a follow-up to `docs/progress.md` in step 8.
- **Branch is not `main`:** remember the branch name for step 4, so CI runs on pushes to it.

### Repository URL

Derive a public HTTPS URL from `origin`, without the `.git` suffix:

| `origin` | Repository URL |
| --- | --- |
| `git@github.com:Tetraglass/tetraglass.git` | `https://github.com/Tetraglass/tetraglass` |
| `ssh://git@github.com/Tetraglass/tetraglass.git` | `https://github.com/Tetraglass/tetraglass` |
| `https://github.com/Tetraglass/tetraglass.git` | `https://github.com/Tetraglass/tetraglass` |
| `https://alice:<token>@github.com/Tetraglass/tetraglass.git` | `https://github.com/Tetraglass/tetraglass` |

Always remove user names, passwords and tokens: the URL is published in every crate's metadata. The expected result is `https://github.com/Tetraglass/tetraglass` (D-017). If `origin` points anywhere else, show the URL and ask before using it. Without an `origin`, use the expected URL and warn that nothing can be pushed yet.

## 2. Confirm the name

The name is decided: **Tetraglass**, crate prefix `tetraglass` (D-017). Anyone can claim a crates.io name at any time, so check again that these are still free: `tetraglass`, `tetraglass-core`, `tetraglass-io`, `tetraglass-filters`, `tetraglass-scene`, `tetraglass-render`, `tetraglass-egui`, `tetraglass-export`, `tetraglass-cli` and `tetraglass-viewer`. The exact check is the crates.io index: `https://index.crates.io/te/tr/<name>` answers 404 when a name is free. If any name is taken, stop and ask the user how to proceed.

If `AGENTS.md` says `Project name: TBD` (the kit reused for another project), ask the user for a name, offer three short, pronounceable suggestions if they have none, run the same check, and record the result: the name and prefix lines in `AGENTS.md`, a decision in `docs/decisions.md`, and the copyright line in `LICENSE-MIT`.

## 3. Check the toolchain

Run and report versions; install what is missing (ask first if installing requires admin rights):

```
rustc --version
rustup target add wasm32-unknown-unknown
cargo install cargo-deny --locked
```

## 4. Create the workspace

Copy and fill the templates in `resources/`, replacing `{{prefix}}` with the prefix (`tetraglass`) and `{{repository}}` with the URL from step 1:

| Template | Destination |
| --- | --- |
| `resources/workspace-Cargo.toml` | `Cargo.toml` |
| `resources/rust-toolchain.toml` | `rust-toolchain.toml` |
| `resources/cargo-config.toml` | `.cargo/config.toml` |
| `resources/ci.yml` | `.github/workflows/ci.yml` (if the branch from step 1 is not `main`, put that branch in `on.push.branches`) |
| `resources/xtask/Cargo.toml`, `resources/xtask/main.rs` | `xtask/Cargo.toml`, `xtask/src/main.rs` |

Create these crates under `crates/`, each from `resources/crate-Cargo.toml` and `resources/lib.rs` (fill `{{crate}}` and `{{description}}`):

| Crate | Description |
| --- | --- |
| `<prefix>-core` | Data model: unstructured grids, cell types, typed arrays, time series. No GPU, no I/O. |
| `<prefix>-io` | Readers and writers for VTK and Gmsh files, working on byte buffers. |
| `<prefix>-filters` | Linearization, surface extraction and visualization filters on tetrahedral views. |
| `<prefix>-scene` | Serializable scene model, JSON Schema and cached pipeline evaluation. |
| `<prefix>-render` | wgpu renderer drawing into host-provided textures, plus a headless helper. |
| `<prefix>-egui` | egui viewport widget built on the renderer. |
| `<prefix>-cli` | Command-line tool: inspect, render, validate, schema, view, export. |

The CLI crate is a binary: use `resources/cli-main.rs` as `src/main.rs` and add a `[[bin]]` section named `<prefix>`. The `export` and `viewer` crates are created later, in the milestones that need them.

Wire the internal dependencies exactly as in the design plan, through `[workspace.dependencies]` path entries: io and filters depend on core; scene on core, io and filters; render on core and scene; egui on render; cli on everything it needs. Do not add external dependencies beyond the templates yet, except as step 5 says.

## 5. Pin the GPU stack

The wgpu version must match egui-wgpu. Determine it instead of guessing:

```
cargo add -p <prefix>-egui egui egui-wgpu
cargo tree -p <prefix>-egui -i wgpu --depth 0
```

Move `egui`, `egui-wgpu` and the reported `wgpu` version into `[workspace.dependencies]`, then add `wgpu` and `bytemuck` (with its `derive` feature) to the render crate. `cargo tree -i wgpu` must show one version.

## 6. License policy

Run `cargo deny init`, then edit `deny.toml` so the licenses allowlist matches `.agents/rules/dependencies.md`. Run `cargo deny check licenses` and resolve anything flagged by asking the user, never by widening the list silently.

## 7. Verify

Run `cargo xtask verify` and fix until it is green. Expected warnings about missing `cargo-deny` or the wasm target mean step 3 is incomplete.

## 8. Record

- Tick M0.1 in `docs/roadmap.md`.
- Append an entry to `docs/progress.md` listing the crates, the wgpu and egui versions, the repository URL, and any decisions. If there was no `origin`, add the follow-up: set the remote, then replace the placeholder `repository` in `Cargo.toml`.

## 9. Commit

Only if the folder is a git repository:

1. Run `git status --porcelain` and check that only bootstrap files changed: `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `deny.toml`, `.cargo/`, `.github/`, `crates/`, `xtask/`, and the edited `AGENTS.md`, `LICENSE-MIT` and `docs/` files. Nothing under `target/` may appear; if it does, fix `.gitignore` first.
2. `git add -A`, then `git commit -m "chore: bootstrap cargo workspace"`.
3. Show the user the commit hash and the list of files.

## 10. Push, only after asking

If an `origin` exists, ask: "Push the bootstrap commit to origin/<branch>? This starts the first CI run." Push only on an explicit yes:

- `git push -u origin <branch>` if the branch has no upstream yet, otherwise `git push`.
- If the push is rejected because the remote has commits (for example a README created on GitHub), stop and explain. Never force-push, rebase or merge on your own.
- After a successful push, point the user to `<repository URL>/actions` to watch CI. If the GitHub CLI is installed, `gh run watch` shows the run in the terminal.

Never create, rename or reconfigure remotes, and never change git settings, without the user asking.

Finally, suggest `/make-test-data` (task M0.2) as the next step.
