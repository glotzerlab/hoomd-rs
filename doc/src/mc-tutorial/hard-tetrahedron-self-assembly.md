# Hard Tetrahedron Self-Assembly

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-tutorial/hard-tetrahedron-self-assembly.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

There are many ways you can model **anisotropic bodies** in *hoomd-rs*. This
tutorial shows you how to represents **sites** with hard convex polyhedra.
When compressed to a sufficiently high packing fraction, systems of hard
tetrahedra **self-assemble** into a quasicrystal: [10.1038/nature08641].

[10.1038/nature08641]: http://doi.org/10.1038/nature08641

* Objectives:
  * Explain how to model system of hard convex polytopes.
  * Demonstrate the self-assembly of hard tetrahedra.
* File: `hoomd-rs/examples/mc-tutorial/hard-tetrahedron-self-assembly.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example hard-tetrahedron-self-assembly
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example hard-tetrahedron-self-assembly
  ```

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:use}}
```

## Type Aliases

Create type aliases for your model's *vector*, *body properties*, and *site
properties* types so that you don't need to repeat the full generic type
names throughout the code:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:type_aliases}}
```

The **sites** are in this tutorial are represented by tetrahedra with both
position and orientation. Therefore, use `OrientedPoint` for both the **body**
and **site** properties. Use `Versor` to represent rotations in 3D.

## Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:simulation_new}}
```

### Parameters

Assign all the model parameters in one code block so that they are easy to modify:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:parameters}}
```

`initial_packing_fraction` is the volume of the tetrahedra divided by the
volume of the simulation boundary in the initial state. Choose this value so
that tetrahedra can be placed easily in the microstate. During the `Initialize`
phase, the microstate will be compressed until it reaches the packing fraction
`target_packing_fraction`. `n_bodies` is the number of tetrahedra to add,
`maximum_distance` is the largest distance a translation trial move can
take, `maximum_rotation`controls the size of the rotation trial moves, and
`macrostate` holds the temperature set point (in units of energy).

### Hamiltonian

A `ConvexPolyhedron` is the shape given by the convex hull of the given vertices.
Wrap it in the `Convex` newtype for use with the `HardShape` Hamiltonian:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:hamiltonian}}
```

## Initialization and Simulation

See the [Hard Ellipse Self-Assembly] tutorial for a complete explanation of
remaining initialization and simulation code.
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:remainder}}
```

[Hard Ellipse Self-Assembly]: hard-ellipse-self-assembly.md

## Implement `main()`

To run the simulation, construct the `HardTetrahedronSelfAssembly` simulation model.
Then call `advance()` many times:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-ellipse-self-assembly.rs:main}}
```

Write the sites to a GSD file periodically so that you can inspect the results
of the simulation.

> [!NOTE]
> This `main()` function runs in batch mode. There is a different `main()` (not
> shown here) used in the interactive example.

## Conclusion

This tutorial showed you how to perform hard tetrahedron self-assembly simulations
using a shape overlap potential.

Navigate to the top of the page and refresh to see the simulation in action
again. Notice that tetrahedra are first added in a large batch. Once all the
overlaps are removed, another batch appears. After all all tetrahedra are in the
microstate and not overlapping, the simulation compresses to a higher packing
fraction. After that, it speeds up as it begins using the more efficient hard
particle overlap Hamiltonian. Watch the simulation long enough and you should
see dimers and pentamers form. These motifs organize to form a quasicrystal,
but only after very long simulation times with at least 4096 tetrahedra:
[10.1038/nature08641].

You can also run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example hard-tetrahedron-self-assembly
```

The next section will explain how to run self-assembly simulations of patchy
particles.

[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:all}}
