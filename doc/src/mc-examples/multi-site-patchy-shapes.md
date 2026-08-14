# Multi-site patchy shapes

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-examples/multi-site-patchy-shapes.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Model bodies with a hard shape core with attractive patches placed
  on its perimeter.
* File: `hoomd-rs/examples/mc-examples/multi-site-patchy-shapes.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example multi-site-patchy-shapes
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example multi-site-patchy-shapes
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/mc-examples/multi-site-patchy-shapes.rs:all}}
