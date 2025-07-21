# Developer Tools

## Spell checking

Use [codebook] to check for spelling errors in arguments, variable names,
comments, etc... *hoomd-rs* includes a codebook dictionary file exempting
words commonly used throughout the repository.

[codebook]: https://github.com/blopker/codebook

## mdBook

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
$ mdbook serve
```

## WASM

This documentation contains example scripts built with WASM. To build these,
you need to install the following tools:
* The wasm Rust target:
  ```shell
  $ rustup target install wasm32-unknown-unknown
  ```
* [wasm-bindgen]
  ```shell
  $ cargo install wasm-bindgen-cli
  ```
* [wasm-opt]. Can be installed e.g. with `micromamba`
  ```
  $ micromamba install binaryen
  ```

[wasm-bindgen]: https://github.com/rustwasm/wasm-bindgen
[wasm-opt]: https://github.com/WebAssembly/binaryen

To build and test individual examples for WASM, see [the WASM chapter in the bevy cheat book].

To add a new web example to this documentation:
* Add the `.rs` code in the appropriate subdirectory under `examples/`.
* Add the example target in `examples/Cargo.toml`, including the metadata section.
* Add `.md` text in the appropriate subdirectory under `doc` and list it in
  `SUMMARY.md`.

In all cases, see the existing examples for the needed syntax.

To build all the interactive examples, run:
```shell
$ TODO
```

It will build the examples listed in the package metadata in
`examples/Cargo.toml` and place them in `doc/src`. You can view the interactive
example scripts with `mdbook serve`. Due to browser security protocols, the
examples will not run if you use `mdbook build` and open the file locally.

> [!CAUTION]
> **DO NOT COMMIT** the generated `.wasm` or `.js` files to the repository.
> These will be compiled by CI when needed.

[the WASM chapter in the bevy cheat book]: https://bevy-cheatbook.github.io/platforms/wasm.html
