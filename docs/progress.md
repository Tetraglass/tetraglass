# Progress log

Newest entries at the bottom. `/next-task` appends one entry per finished task using this template:

```
## YYYY-MM-DD: <task id> <title>
- Done:
- Tests:
- Decisions:
- Dependencies added:
- Follow-ups:
```

## 2026-10-22: Planning package imported
- Done: design plan, decisions, roadmap, scene format draft, Antigravity rules and skills added to the repository.
- Tests: none yet.
- Decisions: D-001 to D-017 recorded, including the name Tetraglass.
- Dependencies added: none.
- Follow-ups: run `/bootstrap-workspace` (M0.1).

## 2026-10-23: M0.1 Bootstrap
- Done: confirmed all 10 `tetraglass` / `tetraglass-*` crate names are free on crates.io; created Cargo workspace (`Cargo.toml`, `rust-toolchain.toml`, `.cargo/config.toml`), CI workflow (`.github/workflows/ci.yml`), `xtask`, license policy (`deny.toml`), and initial crate skeletons (`tetraglass-core`, `tetraglass-io`, `tetraglass-filters`, `tetraglass-scene`, `tetraglass-render`, `tetraglass-egui`, `tetraglass-cli`) with repository URL `https://github.com/Tetraglass/tetraglass` and `rust-version = "1.95"`.
- Tests: `cargo xtask verify` (format, clippy, unit/doc tests, `wasm32-unknown-unknown` check on `-core`, `-io`, `-filters`, `-scene`, and `cargo deny check`).
- Decisions: confirmed D-017 crate names remain available; pinned GPU/UI stack to `egui = "0.37.0"`, `egui-wgpu = "0.37.0"`, `wgpu = "30.0.1"`.
- Dependencies added: `egui` (0.37.0), `egui-wgpu` (0.37.0), `wgpu` (30.0.1), `bytemuck` (1, `derive`), `log` (0.4), `thiserror` (2).
- Follow-ups: run `/make-test-data` (M0.2); reserve crate names on crates.io before first public release.
