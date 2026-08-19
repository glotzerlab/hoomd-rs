# 3D shapes confined to a 2D plane

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-examples/capsules-on-a-plane.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

* Objective: Self-assemble 3D capsules that are allowed to freely rotate while their
  center is confined to a 2D plane.
* File: `hoomd-rs/examples/mc-examples/capsules-on-a-plane.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example capsules-on-a-plane
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example capsules-on-a-plane
  ```

## Strategy

1. Bodies with `Cartesian<2>` positions and `Versor` orientations.
2. Define a custom `SiteProperties` with `Cartesian<3>` position and `Versor` orientation.
   Transform sites from the body frame to the simulation frame by lifting the body
   position into 3D: `(body_x, body_y, 0)`.
3. Define a custom `Boundary` type that wraps `Periodic<Rectangle>`.
   Implement `Wrap<BodyProperties>`, `Volume`, `MapPoint`, `Scale`, and `Distribution`
   for the custom boundary by calling the same method on the inner type.
   Implement `Wrap<SiteProperties>` and `GenerateGhosts` by projecting
   the 3D site position into 2D, calling the method on the wrapped type,
   then lift the result back into 3D.

See the example code for details. It implements all of these steps in a general fashion.
You can copy and paste this code and use it for Monte Carlo simulations of any 3D
interaction model (including multi-site rigid bodies) where the body centers should
be confined to the _xy_ plane.

## Complete Code
```rust,ignore
{{#rustdoc_include ../../../examples/mc-examples/capsules-on-a-plane.rs:all}}
