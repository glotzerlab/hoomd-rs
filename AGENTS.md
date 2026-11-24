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
  - browser/wasm: `cargo run --quiet --no-default-features --features bevy --target wasm32-unknown-unknown --example <name>`
  Agents should not run examples without permission.

Lints, format, and prek checks are not needed after every task, only once when the
complete pull request is finalized.
- **Lint Rust code:**
  - Single crate (check only): `cargo clippy --all-targets --all-features --quiet  --no-deps -p <crate>`.
  - Whole workspace (check only): `cargo clippy --all-targets --all-features --quiet  --quiet --no-deps`.
- **Format Rust code:**
  - Single crate (check only): `cargo +nightly-2025-09-17 --quiet fmt --check -p <crate>`
  - Whole workspace (check only): `cargo +nightly-2025-09-17 --quiet fmt --check`
  - Use the pinned nightly. Remove `--check` when it is appropriate to apply the
    formatting changes.
- **Other lints:**
  - Specific files: `prek run --quiet --show-diff-on-failure --files <modified files>`
  - Whole workspace: `prek run --quiet --show-diff-on-failure --all-files`
  - There is no check-only variant for `prek`.

## Design guidelines

Design priority (see `ARCHITECTURE.md`): flexibility > reusability > readability >
simplicity > efficient algorithms > hardware micro-optimizations. No GPU / domain
decomposition; array-of-structures over structure-of-arrays. `hoomd-rs` writes
nothing to stdout by default — it uses the `log` crate.

## Tests

- Unit tests live in the same file as the implementation in `mod tests`.
- Use `approxim::assert_relative_eq` for floating point number comparisons when
  values are not exact.
