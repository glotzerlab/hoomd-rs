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

Assign all the model parameters in one code block:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:parameters}}
```

`initial_packing_fraction` is the volume of the tetrahedra divided by the
volume of the simulation boundary in the initial state. Choose this value so
that tetrahedra can be placed easily in the microstate. During the `Initialize`
phase, the microstate will be compressed until it reaches the packing fraction
`target_packing_fraction`. `n_bodies` is the number of tetrahedra to add,
`maximum_distance` is the largest distance a translation trial move can take
(initially), `maximum_rotation`controls the size of the rotation trial moves
(initially), and `macrostate` holds the temperature set point (in units of
energy).

### Hamiltonian

A `ConvexPolyhedron` is the shape given by the convex hull of the given vertices.
Wrap it in the `Convex` newtype for use with the `HardShape` Hamiltonian:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:hamiltonian}}
```

## Initialization and Simulation

See the [Hard Ellipse Self-Assembly] tutorial for a complete explanation of
remaining initialization and simulation code (see also the [complete code] below).

[Hard Ellipse Self-Assembly]: hard-ellipse-self-assembly.md
[complete code]: #complete-code

## Execute the Simulation in Batch Mode

### Implement `main()`

To run the simulation, construct the `HardTetrahedronSelfAssembly` simulation model.
Then call `advance()` many times and write the sites to a GSD file periodically so that
you can inspect the results of the simulation:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-ellipse-self-assembly.rs:main}}
```

See [Applying Interactions](applying-interactions.md) for a step-by-step explanation
of this code.

> [!NOTE]
> This `main()` function runs in batch mode. There is a different `main()` (not
> shown here) used in the interactive example. The interactive example does *not*
> write the GSD file.

### Run the Simulation

In a terminal, execute the following command to run the simulation in batch mode:
```shell
cargo run --release --example hard-tetrahedron-self-assembly
```

### Visualize the Simulation Results

Open the generated `hard-tetrahedron-self-assembly.gsd` in [Ovito] or another
visualization tool to see the simulation results. To render the tetrahedra in
[Ovito], set the *Particles* [Radius scaling] to 200% and import this [modifier
snippet] (use the copy button to copy the entire text to your clipboard):
```
{"description": "OVITO Modifier Snippet: Edit types (Particle Type)", "payload": "AAAHBnjafVU9aBRBFH57ufN+kzOJotHCExtJEaNBUBI5Y7wiRcgRzyA2YbI7uRvduz129xIOLCysBEVS2UgKwUIt0tgFUkkgpQpWgmhhYRGwMwg6b/bNZXNu8uDdzL7v/c17M+/yb9+/yfzZvgvQug4AKckDMAvzMA0VuRagDC44UtojOQ8B5WAMRuAiXJa/oyQzAsalJ0bfKVpjxI9ofUBrDnXKrtPkrt+W+xFoApPxfBBggg0cPLggM9gvK8jM2lKTS4tL/6EeFM6uqrwPttKEZ0oY9BGXnDZCaAKzVIIjaofbZGeX0rYoSnfbZroFWSyFQWeOU+xe2qP/vo5nrPKQUdzJXGu+/JzpXiV4FKMGdZ2Uv/2Sx8nTGFUWPafLzPWFaXNPfpw6tCADkk8eEnFQF0EF1HSMgiId1+2EumpCLaSHqe6Q7g9d7c0P7Vfvsvnzxa+BYHO7sHFleOB38Vsgn1gH48uLrY9arvU3u/Q31v7uTrPHaxPPAH5+2porfn/+ZPjM03E4lOLUOAjdUE1J6o6+xd1yo8uXlsN+P9jLWJn0s9wSPrcq7abqhrpWccL6PKflmjz0DjquTpg1Xhcms0s2r/OGv7Dg1ZjlrFCxg0dUZ54XBQxJUZPf9vgM92pTju24UVrnlNYNZt5fYiafatm2aFRLDbZocytK/3RNVGu2ZP8WGpasKo+M3qv8RiH9y9bKHLNEK9Kuzz0QGowILcUPNZ5WIfG0WFgj1MaEQsLNz/PoIyLWa0YVC5Fcg9V5pImz0uBudwsRSShnOk2UxNFJeMhkl4VHDZbnia8a9NoQywhLisWS4G74iv2iOzSE9SzJu6Vu1oxjac0kKeAbzelBgEpBEntYxRWsUbWpbBhTYfjis5QUmWkIh0XmJvPZ7OI9bvrBLIzpyZVEZF54wVRUYpxXqVBuWh0HV3qOL1WYW+U+na6T2KTpi2UejqEe2ejef42i+dD/C9IutZ1eZOwqjfo7NKTwIOs0srHOrxH8B4RtGuo="}
```

Alternately, you can:
1) Download [tetrahedron.obj].
2) Add an [Edit types] modification to the Ovito pipeline.
3) Set *Property Name* to [Aspherical Shape].
4) Choose the "Mesh/user-defined" *shape*.
5) Click *Load geometry file...* and choose `tetrahedron.obj`.
To save time when visualizing many similar systems, you can [save the session state]
or use [modifier templates].

[Ovito]: https://www.ovito.org/
[Radius scaling]: https://www.ovito.org/docs/current/reference/pipelines/visual_elements/particles.html#universal-settings
[Edit types]: https://www.ovito.org/docs/current/reference/pipelines/modifiers/edit_types.html
[tetrahedron.obj]: tetrahedron.obj
[save the session state]: https://docs.ovito.org/usage/miscellaneous.html#usage-saving-loading-scene
[modifier templates]: https://docs.ovito.org/reference/app_settings/modifier_templates.html#modifier-templates
[modifier snippet]: https://docs.ovito.org/advanced_topics/modifier_snippets.html#modifier-snippets

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

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-tetrahedron-self-assembly.rs:all}}
