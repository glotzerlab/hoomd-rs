# Lennard-Jones Fluid

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/md-examples/lj-fluid.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Perform a molecular
* File: `hoomd-rs/examples/md-examples/lj-fluid.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example lj-fluid
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example lj-fluid
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/md-examples/lj-fluid.rs:all}}

