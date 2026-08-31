# Binary hard shape systems

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-examples/binary-hard-shapes.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Show how to write a custom site pair interaction that checks for overlaps
  between two different types of hard shapes.
* File: `hoomd-rs/examples/mc-examples/binary-hard-shapes.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example binary-hard-shapes
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example binary-hard-shapes
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/mc-examples/binary-hard-shapes.rs:all}}
