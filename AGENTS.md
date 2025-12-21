# Agent guidelines for working in this repo

Build / Lint / Test
- Build: `cargo build` (debug) and `cargo build --release` (release)
- Run all tests: `cargo test`
- Run a single test by name: `cargo test <pattern> -- --nocapture` (use `--exact` for exact matches: `cargo test --exact test_name`)
- Run library tests only: `cargo test --lib`
- Format check: `cargo fmt --all -- --check` (auto-fix: `cargo fmt`)
- Lint: `cargo clippy --all-targets -- -D warnings`

Code style (applies to agents editing files)
- Formatting: always run `cargo fmt`; follow Rust 2024 edition idioms.
- Imports: group and order as `std` -> external crates -> crate-local; avoid glob imports (`use foo::*`).
- Naming: snake_case for functions/variables/modules, CamelCase for types/structs/enums, UPPER_SNAKE_CASE for constants.
- Types: prefer explicit types on public APIs; use type inference locally when readable.
- Visibility: keep items private by default; use `pub(crate)` before `pub` when appropriate.
- Error handling: prefer `anyhow::Result` at application boundaries (present in Cargo.toml), use `?` for propagation, avoid `unwrap`/`expect` in library code; only panic on invariant violations.
- Tests: write small, deterministic unit tests; prefer table-driven tests for variants.

Agent behavior notes
- Respect AGENTS.md scope rules when modifying files in subdirectories.
- No Cursor or Copilot special rules found in repo. If such files are added, include them here.
- Keep edits minimal and focused; run `cargo test` after substantial changes and fix failures you introduce.
