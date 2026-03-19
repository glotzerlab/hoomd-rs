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
* [mdbook-alerts]
* [mdbook-katex]

Install these with Cargo:
```shell
$ cargo install mdbook mdbook-alerts mdbook-katex
```

[mdBook]: https://rust-lang.github.io/mdBook/
[mdbook-alerts]: https://github.com/lambdalisue/rs-mdbook-alerts
[mdbook-katex]: https://github.com/lzanini/mdbook-katex

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
