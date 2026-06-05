# Development

## Code Style

All contributions to *hoomd-rs* must follow the established code style:

* All *hoomd-rs* code should be [Idiomatic Rust].
* Follow standard [API guidelines] when implementing new types and traits.
* [Spell check] your documentation comments **AND CODE!**
* Name crates, types, traits, fields, methods, and variables with **complete words**,
  even for internal names that are not exposed in the public API. Additionally,
  the chosen word(s) should match those in the most common usage, such as
  defined in a textbook.

[Idiomatic Rust]: https://github.com/mre/idiomatic-rust
[API guidelines]: https://rust-lang.github.io/api-guidelines/
[Spell check]: #spell-checking

## Tools

### prek

Run
```shell
prek run --all-files
```
to perform a number of style checks and fixes.

### Code formatting

Run
```shell
cargo +nightly-2025-09-17 fmt
```
to automatically format the code. Use the shown Rust nightly version to obtain
the same results as the CI checks.

### Code Linting

Run
```shell
cargo clippy --all-targets --all-features
```
to ensure that the code follows established best practices.

### Spell Checking

Use [codebook] to check for spelling errors in arguments, variable names,
comments, etc... *hoomd-rs* includes a codebook dictionary file exempting
words commonly used throughout the repository.

[codebook]: https://github.com/blopker/codebook

### Build the documentation

#### mdBook

This documentation is built with [mdBook] using the following plugins:
* [mdbook-katex]

[mdBook]: https://rust-lang.github.io/mdBook/
[mdbook-katex]: https://github.com/lzanini/mdbook-katex

> [!IMPORTANT]
> Install the [mdbook-katek v0.10.0 alpha] for compatibility with [mdBook] 0.5.x.

[mdbook-katex v0.10.0 alpha]: https://github.com/lzanini/mdbook-katex/releases/tag/0.10.0-alpha-binaries

To preview the documentation locally:
```shell
$ cd doc
$ mdbook serve --open
```

#### rustdoc

To build the API documentation from source, execute:
```shell
./build_api_documentation.sh
```
Open `target/doc/hoomd-vector/index.html` in your web browser to view the
documentation.

### WASM

This documentation contains example scripts built for WASM. To build these,
you need to install the following tools:
* The `wasm` Rust target:
  ```shell
  $ rustup target install wasm32-unknown-unknown
  ```
* [wasm-bindgen] and [wasm-server-runner]
  ```shell
  $ cargo install wasm-bindgen-cli wasm-server-runner
  ```
* [wasm-opt]. Install with `micromamba`:
  ```
  $ micromamba install binaryen
  ```
  or by another method of your choice.

[wasm-bindgen]: https://github.com/rustwasm/wasm-bindgen
[wasm-server-runner]: https://github.com/jakobhellermann/wasm-server-runner
[wasm-opt]: https://github.com/WebAssembly/binaryen

For more information, see [the WASM chapter in the bevy cheat book].

To add a new web example to this documentation:
* Add the `.rs` code in the appropriate subdirectory under `examples/`.
* Add the example target in `examples/Cargo.toml`, including the metadata section.
* Add `.md` text in the appropriate subdirectory under `doc` and list it in
  `SUMMARY.md`.

In all cases, see the existing examples for the needed syntax.

To build and test an individual example with WASM (requires
[wasm-server-runner]), run:
```shell
$ cargo run --features bevy --target wasm32-unknown-unknown --example {example}
```
and open the printed `http://localhost` URL in your browser.

To build all the interactive examples, run:
```shell
$ cargo run -p build-wasm-doc-examples
```
It will build the examples listed in the package metadata in
`examples/Cargo.toml` and place them in `doc/src`. You can view the interactive
example scripts with `mdbook serve` and navigating to the `http://localhost`
URL. Due to browser security protocols, the examples will not run if you open
via the `file://` URL.

> [!CAUTION]
> **DO NOT COMMIT** the generated `.wasm` or `.js` files to the repository.
> These will be compiled by CI when needed.

> [!TIP]
> You can locally test how an example will appear with:
> ```js
> import init from './{example-name}.js'
> ```

> [!IMPORTANT]
> The import line must be
> ```js
> import init from 'https://glotzerlab.github.io/hoomd-rs/{example-directory}/{example-name}.js'
> ```
> to appear in the published documentation.

[the WASM chapter in the bevy cheat book]: https://bevy-cheatbook.github.io/platforms/wasm.html

## Release Process

To make a new release:

- [ ] Run `./diff_public_api.sh` (which uses [cargo-public-api]) to help determine
  whether this should be a *patch*, *minor*, or *major* release.
  * *patch* release: no changes at all.
  * *minor* release: Only added changes (including new crates).
  * *major* release: Often needed when APIs are changed and removed, though
    [it is very complicated]. Our user base is small enough that conservatively
    making a *major* release is fine when there is any doubt.
- [ ] Make a new `release-{X.Y.Z}` branch (where `{X.Y.Z}` is the new version).

On that branch, take the following steps (committing after each step when needed):

- [ ] Run `prek autoupdate --freeze`.
- [ ] Check for new or duplicate contributors since the last release:
  ```shell
  comm -13 (git log $(git describe --tags --abbrev=0) --format="%aN <%aE>" | sort | uniq | psub) (git log --format="%aN <%aE>" | sort | uniq | psub)
  ````
  Add entries to `.mailmap` to remove duplicates.
- [ ] Review `release-notes.md` and revise if needed.
- [ ] Add highlights to release notes (*if needed*).
- [ ] Run `./check_links.sh` and fix any broken links. If `README.md` has changed,
      this command will copy it to all the crates. Commit the updated README files.
- [ ] Run `bump-my-version bump {type}`. Set `{type}` to `patch`, `minor`, or `major`.
- [ ] Run `cargo check`
- [ ] Run `cargo update`
- [ ] Run `cargo bundle-licenses --format yaml --output THIRDPARTY.yaml`
- [ ] Push the branch and open a pull request.
- [ ] Check that readthedocs builds the docs correctly in the pull request checks.
- [ ] Merge the pull request after all tests pass.
- [ ] Make a new tag on the trunk branch:
  ```
  git switch trunk
  git pull
  git tag -a {X.Y.Z}
  git push origin --tags
  ```

> [!IMPORTANT]
> Make sure to **exclude** `v` from the tag name!

- [ ] Add a blank release notes entry for the next release:
  ```
  ## Next release

  *Added:*

  *Changed:*

  *Deprecated:*

  *Removed:*

  *Fixed:*
  ```

> [!NOTE]
> Paste `Next release` exactly as shown. `bump-my-version` will replace that
> string with the version number and date of the *next* release.

GitHub Actions will trigger on the tag and upload the release to crates.io and create a
GitHub release.

- [ ] [Check that the GitHub release posted correctly](https://github.com/glotzerlab/hoomd-rs/releases/).
- [ ] [Check that all crates.io uploads succeeded](https://crates.io/users/joaander).

[cargo-public-api]: https://github.com/cargo-public-api/cargo-public-api
[it is very complicated]: https://doc.rust-lang.org/cargo/reference/semver.html
