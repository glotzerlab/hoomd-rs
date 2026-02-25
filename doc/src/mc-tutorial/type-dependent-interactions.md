# Type-dependent Interactions

<script type="module">
import init from 'https://glotzerlab.github.io/hoomd-rs/mc-tutorial/type-dependent-interactions.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

Some models label each site with **types** and apply interactions as a function
of type. For example, you can model coarse-grained phase separation with
attractive *A-A* and *B-B* interactions and purely repulsive *A-B* interactions.
This tutorial shows you how to assign **types** to sites and compute
type-dependent pairwise interactions.

* Objectives:
  * Show how to use an `enum` to name all the site types.
  * Define a custom **site properties** struct that includes the type.
  * Show how to compute type-dependent pairwise interactions.
  * Demonstrate phase separation of *A* and *B* types.
* File: `hoomd-rs/examples/mc-tutorial/type-dependent-interactions.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example type-dependent-interactions
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example type-dependent-interactions
  ```

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:use}}
```

## Type Aliases

Create type aliases for your model's *vector* and *body properties* so that you
don't need to repeat the full generic type names throughout the code:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:type_aliases}}
```

The **sites** are in this tutorial are placed at points in space and assigned a type.
Therefore, use `Point` for the **body** properties.

## Site Properties

You might be familiar with simulation tools where you assign types to sites
based on a numerical index or string name. In *hoomd-rs*, you can assign
type (and any other site-specific parameter(s)) using any Rust datatype.

### Type `enum`

Using an enum ensures that every site *always* has a well-defined type. If you
fail to assign a type or set an invalid one, Rust will issue an error *at compile
time*. In this example, `Type` enumerates all the site types:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:type}}
```

### Define `SiteProperties`

Previous tutorials used one of the built-in structures (`Point` or
`OrientedPoint`) to represent the **site properties**. These types are limited
as they represent only a site's position (or position and orientation). This
tutorial defines a new `SiteProperties` struct that gives each site a position
in space and a given **type** (named `type_` because `type` is a Rust keyword):
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:site_properties}}
```

Initialization, MC, and MD methods operate on a generic site properties type
`S` with *trait bounds* set as needed. The `derive` macro implements the
listed traits for `SiteProperties` automatically. All the types listed in the
above code block are required by at least one method in this example. Rust
provides the `Clone`, `Copy`, `Default` traits (and their `derive` macros).
`hoomd_microstate` defines `Position` and `Orientation` along with their
corresponding `derive` macros.

> [!NOTE]
> You can add any fields to your `SiteProperties` type and use those fields when
> computing interactions. `position` is the only required field.

### Transforming Sites

*Body* stores each of its **sites** in the *body frame* of reference. The
`Transform` trait (implemented for the **body properties** type) transforms
site properties from the *body frame* to the *system frame*. You must implement
`Transform<SiteProperties>` for the body properties type to use `Microstate`:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:site_transform}}
```

This implementation is suitable for point bodies with point sites in Euclidean
space as it transforms from one frame to the other by vector addition and copies
all other fields. See the `hoomd_microstate::property` module documentation for
code that works with oriented bodies and/or sites.

## Site-site Interactions

For demonstration purposes, this tutorial shows you how to implement a coarse-grained
model where *A-A* interactions attract via the Lennard-Jones potential, *B-B* interact
via the sum of a power law and a Gaussian, and *A-B* interact with the
Weeks-Chandler-Anderson potential.

### Define `SitePairInteraction`

Define a new struct to hold the interaction parameters. Use the provided
types for the `LennardJones` and `WeeksChandlerAnderson` interactions:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:interaction_type}}
```
Every site-site interaction type must implement `MaximumInteractionRange`
which sets the distance above which the interactions go to zero. You can implement
this trait directly, or add the `maximum_interaction_range` field and
`#derive[(MaximumInteractionRange)]` as shown here.

### Implement `SitePairEnergy`

`PairwiseCutoff` uses the `SitePairEnergy` trait to calculate the interaction
energy that each pair of sites contributes to the total. Implement the trait
and compute the type-dependent interactions:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:interaction_impl}}
```
This example uses isotropic interactions that depend only on the distance between
the two sites and their types. Use Rust's `match` expression to compute the desired
potential for every combination of types. Rust will produce a helpful compile error
should you miss one or more cases. You can compute interactions
that are not included in `hoomd-interaction` by writing the expression
directly, as demonstrated in the B-B case.

> [!IMPORTANT]
> Ensure that `site_pair_energy(i,j)` computes the same energy as
> `site_pair_energy(j,i)`. Rust does not enforce this symmetry and *hoomd-rs*
> cannot detect the problem.

## Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:simulation_new}}
```

### Parameters

Assign all the model parameters in one code block:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:parameters}}
```

`initial_packing_fraction` is the volume of the disks divided by the volume
of the simulation boundary in the initial state. Choose this value so
that disks can be placed easily in the microstate. During the `Initialize`
phase, the microstate will be compressed until it reaches the packing
fraction `target_packing_fraction`. `n_disks` is the number of disks to add,
`maximum_distance` is the largest distance a translation trial move can take
(initially), `sigma` is the disk diameter, and `macrostate` holds the current
temperature set point (in units of energy).

### Hamiltonian

Construct the `SitePairInteraction` type and use it with `PairwiseCutoff` to
form the system's Hamiltonian:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:hamiltonian}}
```

### Overlap Penalty Hamiltonian

Even though `SitePairInteraction` treats all sites as points with well-defined
potentials at all distances $` r \ne 0 `$, it is helpful to place sites where
$` r \ge \sigma `$. It can take many steps to relax high energy states.
Use the hard disk `OverlapPenalty` as a Hamiltonian when initializing:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:compress_hamiltonian}}
```

### Construct the Boundary

Place all bodies in periodic square boundary conditions at the chosen packing fraction:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:boundary}}
```

### Place A and B bodies with `QuickInsert`

One `QuickInsert` randomly places a number of copies of the given template body.
Use one `QuickInsert` to place half of the bodies with type *A* and a second
`QuickInsert` to place the remainder with type *B*:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:quick_insert}}
```

## Initialization and Simulation

The remaining initialization and simulation code is very similar to that
in the [Hard Ellipse Self-Assembly] tutorial. The differences are that
rotation moves are not present here, and there are two `quick_insert`
methods to apply instead of one (see also the [complete code] below).

[Hard Ellipse Self-Assembly]: hard-ellipse-self-assembly.md
[complete code]: #complete-code


## Implement `main()`

To run the simulation, construct the `TypeDependentInteractions` simulation model.
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

This tutorial showed you how to implement custom site properties and
site-site interaction types. Specifically, it demonstrated the addition
of a site type field and showed how you can use that when computing
the interaction energies.

Navigate to the top of the page and refresh to see the simulation in action
again. Notice that the sites quickly phase separate. Wait long enough and the
system should form two stripes. Due to the differently shaped potentials, the
*A* and *B* domains are distinct. The *A* sites form a hexagonal solid and *B*
form a lower density fluid.

You can also run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example type-dependent-interactions
```

[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/type-dependent-interactions.rs:all}}
