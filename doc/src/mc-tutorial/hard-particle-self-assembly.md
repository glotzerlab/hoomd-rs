# Hard Particle Self-Assembly

<script type="module">
import init from './hard-particle-self-assembly.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

There are many ways you can model **anisotropic bodies** in *hoomd-rs*. This
tutorial shows you how to represents **sites** with hard ellipses. You can apply
the same techniques to any hard shape. When compressed to a sufficiently high
packing fraction, systems of hard particles **self-assemble** into ordered
structures.

* Objectives:
  * Explain how to execute simulations with **periodic boundary conditions**.
  * Show how to quickly insert bodies into the microstate.
  * Demonstrate the self-assembly of hard ellipses into the nematic phase.
* File: `hoomd-rs/examples/mc-tutorial/hard-particle-self-assembly.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example hard-particle-self-assembly
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example hard-particle-self-assembly
  ```

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:use}}
```

## Type Aliases

Create type aliases for your model's *vector*, *body properties*, and *site
properties* types so that you don't need to repeat the full nested generic type
names throughout the code:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:type_aliases}}
```

The **sites** are in this tutorial are represented by ellipses with both
position and orientation. Therefore, use `OrientedPoint` for both the **body**
and **site** properties.

## The Simulation Model

Here is the type that holds the simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:simulation_struct}}
```

### Construct the Simulation Model

The `new()` method constructs a new simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:simulation_new}}
```

#### Parameters

Assign all the model parameters in one code block so that they are easy to modify:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:parameters}}
```

`box_length` is the side length of the square simulation box, `n_bodies` is
the number of ellipses to add, `maximum_distance` is the largest distance
a translation trial move can take, `maximum_rotation` is the largest angle
possible in a rotation trial move, `sigma` is the major axis of the ellipse,
`aspect` is the ellipse aspect ratio and `kt` is the temperature set point (in
units of energy).

To ensure that `sigma` is the major axis, `aspect` must be greater than or equal
to 1.0.

#### Hamiltonian

`CutoffPairOverlap` represents each site with the given shape. Overlapping
pairs of sites produce an infinite energy. The energy is 0 when the sites do not
overlap. Use `CutoffPairOverlap` as the Hamiltonian:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:hamiltonian}}
```

As with `CutoffPair`, you must provide `$ r_\mathrm{cut} $`. All pairs separated
by a distance larger than `$ r_\mathrm{cut} $` are assumed to be non-overlapping.
For the case of hard ellipses, the largest distance between the centers of two
potentially overlapping ellipses is `sigma` -- when two ellipses a distance
`sigma` apart rotated so their their long axes just touching.

#### Periodic Boundary Conditions

Use **periodic boundary conditions** via the `Periodic` type to represent an
infinitely repeating system. To construct a `Periodic`, you need to provide the
underlying shape and the **maximum interaction range** between sites:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:periodic}}
```

`Periodic` uses this distance to generate **ghost sites** *outside* the
boundary that are periodic images of **sites** *inside*. Methods like
`CutoffPairOverlap` will compute interactions between **sites** inside the
boundary with *all* other sites *whether they are ghosts or not*. When using
`CutoffPairOverlap`, `CutoffPair`, or any method that utilizes
`$ r_\mathrm{cut} $`, `maximum_interaction_range` should be set to the maximum
of all the  `$ r_\mathrm{cut} $` values.

> [!IMPORTANT]
> In *hoomd-rs*, it is *YOUR responsibility* to determine the appropriate
> `maximum_interaction_range`. You might be used to other simulation codes,
> HOOMD-blue for example, that *automatically* determine this maximum for
> you. That is not possible in *hoomd-rs* as your model's interactions
> and/or any analysis methods could be *any arbitrary code*.

> [!WARNING]
> If you set `maximum_interaction_range` too small, `CutoffPair` (and similar
> methods) will *miss interactions that they should have computed*.

#### Microstate

Construct a microstate with the periodic boundary conditions:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:microstate}}
```
Start with no bodies in the microstate.

#### Trial Moves

Apply both `Translate` and `Rotate` trial moves to the bodies:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:trial_moves}}
```

In 2D simulations, `Rotate` uniformly selects a random angle between
`-maximum_rotation` and `maximum_rotation`

#### Add Bodies with `QuickInsert`

`QuickInsert` will add *up to* `n_bodies` new bodies to the microstate
drawn randomly from the given distribution. `UniformIn` generates bodies
with positions uniformly distributed in the given `boundary` and orientations
uniformly distributed among all possible orientations:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:quick_insert}}
```

#### Hamiltonian (For `QuickInsert`)

`QuickInsert` will only add a body when the change in energy due to the addition
is finite. You *could* use `hamiltonian` with `QuickInsert` and ensure that no
ellipses in the microstate overlap. However, *random* body insertions do not pack
densely. Used this way, `QuickInsert` is typically not able to achieve densities
high enough to drive self-assembly.

One way around this problem is to allow inserted bodies to overlap *a little*
and allow the trial moves to remove the overlap. The `OverlapPenalty` potential
consists of an infinite energy core followed by a harmonic potential added to a
step function. The infinite core prevents inserted bodies from overlapping too
much, the harmonic potential encourages the trial moves to separate bodies, and
the step function prevents the trial moves from introducing new overlaps between
sites.

Express this computation using `CutoffPair` with an `Anisotropic` evaluator:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:insert_hamiltonian}}
```

`ApproximateShapeOverlap` computes the *approximate* amount of overlap between
a pair of shapes, `OverlapPenalty` applies the potential describe above,
and the `Anisotropic` `CutoffPair` computes this potential on all pairs of
sites.

> [!IMPORTANT]
> Use `ApproximateShapeOverlap` *only* to remove overlaps during initialization.
> Tt does not compute the *exact* amount of overlap and is therefore not
> appropriate for use in production sampling.

## Implement `Simulation`

The `Simulation` implementation closely follows that in [Applying Interactions].
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:impl_simulation}}
```

### Advance the Simulation

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:advance}}
```

#### Add New hard-particle-self-assembly

The code that adds hard-particle-self-assembly is more complex than that for disks:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:add}}
```
It first chooses a random tetronimo from the `template_sites`, then it adds the
body near the top of the boundary with a default orientation of `$ \theta = 0 $`
and clone of the chosen sites. Each body has four sites in this example.

*hoomd-rs* uses a *counter based random number generator*. Whenever you need to
use random numbers in your code, you can get a `Rng` to generate them by calling
`microstate.counter().make_rng()`.

> [!IMPORTANT]
> Whenever you use `counter.make_rng`, You *MUST* indicate that your substep is
> complete by calling `microstate.increment_substep()` so that the next substep
> will use a different set of random numbers.

#### Apply Trial Moves

Apply the custom trial move to each body in the microstate:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:apply}}
```

#### Reset the Simulation

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:reset}}
```

### Get the Simulation Step

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:step}}
```

## Implement `main()`

To run the simulation, construct the `hard-particle-self-assembly` simulation model.
Then call `advance()` many times:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:main}}
```

Write the sites to a GSD file periodically so that you can inspect the results
of the simulation.

> [!NOTE]
> This `main()` function runs in batch mode. There is a different `main()` (not
> shown here) used in the interactive example.

## Conclusion

This tutorial showed you how to add bodies with multiple sites and how they
can be translated and rotated by trial moves.

Navigate to the top of the page and refresh to see the simulation in
action again. Notice how the randomly generated hard-particle-self-assembly fall to the
bottom while randomly rotating.

Alternately, you can run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example hard-particle-self-assembly
```

The next section will explain how to run self-assembly simulations of hard
particles.

[Applying Interactions]: applying-interactions.md
[Custom Random Walk]: custom-random-walk.md
[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/hard-particle-self-assembly.rs:all}}
