# Grand canonical ensemble simulation of ellipses (2D)

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-examples/ellipse-gcmc-2d.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Perform a grand canonical monte carlo simulation of ellipses at very high fugacity.
* File: `hoomd-rs/examples/mc-examples/ellipse-gcmc-2d.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example ellipse-gcmc-2d
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example ellipse-gcmc-2d
  ```

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/mc-examples/ellipse-gcmc-2d.rs:all}}
