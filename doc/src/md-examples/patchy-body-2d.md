# Patchy Body (2D)

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/md-examples/patchy-body-2d.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Perform a molecular dynamics simulation of patchy rigid bodies.
* File: `hoomd-rs/examples/md-examples/patchy-body-2d.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example patchy-body-2d
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example patchy-body-2d
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/md-examples/patchy-body-2d.rs:all}}
