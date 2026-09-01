# Melt an ideal hexagonal structure of hexagons

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-examples/crystal-stability.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Construct an ideal crystal structure of hexagons. Slowly decrease the
  packing fraction and observe at what point the structure melts.
* File: `hoomd-rs/examples/mc-examples/crystal-stability.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example crystal-stability
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example crystal-stability
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/mc-examples/crystal-stability.rs:all}}
