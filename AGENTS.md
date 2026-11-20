# AGENTS.md

Guidance for AI agents working in this repository. See also `README.md`,
`ARCHITECTURE.md`, `CONTRIBUTING.md`, and `doc/src/development.md` for fuller
context. When these conflict, trust the configs and CI over prose.

## What this is

Rust workspace of `hoomd-rs` simulation crates (Monte Carlo / molecular
dynamics of hard and interacting shapes). **Default branch is `trunk`** (not `main`).

Crates are deliberately split so consumers compile only what they need; most
releases update several crates at once and all crates share one version.

## Canonical commands

Commands match `doc/src/development.md` and the CI workflows
(`.github/workflows/{style,test}.yaml`); `--quiet` keeps output short.

- **Build:** `cargo build --quiet --workspace`.
- **Test the workspace:** `cargo test --quiet` (excludes `hoomd-bevy`).
- **Single test**: `cargo test --quiet -p <crate> <test_name>`.
- **Docs:** `cd doc && mdbook build` for the book; `./build_api_documentation.sh`
  for rustdoc (builds crates one at a time so the sidebar order is correct).
- **Examples:** name = filename without path/extension.
  - batch (GSD output): `cargo run --quiet --release --example <name>`
  - interactive desktop: `cargo run --quiet --release --features bevy --example <name>`
  - browser/wasm: `cargo run --quiet --features bevy --target wasm32-unknown-unknown --example <name>`
  Agents should not run examples without permission.

Lints, format, and prek checks are not needed after every task, only once when the
complete pull request is finalized.
- **Lint:** `cargo clippy --all-targets --all-features --quiet`.
  Single crate: `cargo clippy --all-targets --all-features --quiet -p <crate>`.
- **Format:** `cargo +nightly-2025-09-17 fmt --quiet` — use the pinned nightly.
  `.rustfmt.toml` uses `unstable_features = true` and 2024 style options, so
  `cargo fmt` on **stable** silently warns and does not format doctests.
- **Pre-commit hooks:** `prek run --quiet --show-diff-on-failure --files <modified files>`

## Design guidelines

Design priority (see `ARCHITECTURE.md`): flexibility > reusability > readability >
simplicity > efficient algorithms > hardware micro-optimizations. No GPU / domain
decomposition; array-of-structures over structure-of-arrays. `hoomd-rs` writes
nothing to stdout by default — it uses the `log` crate.

## Tests and documentation

- Unit tests live in the same file as the implementation in `mod tests`.
- Use `approxim::assert_relative_eq` for floating point number comparisons when
  values are not exact.
- Doctests serve as minimal examples to users. Provide one doc test for each method,
  type, and trait. Doc tests are allowed to have unused variables for demonstration
  purposes.
- Inline LaTeX math in documentation is dollar-outside, backtick-inside: $`<math>`$.
- Display LaTex math in documentation is:
  ```math
  <math>
  ```

