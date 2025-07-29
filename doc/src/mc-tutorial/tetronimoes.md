# Tetronimoes

<!-- <script type="module"> -->
<!-- import init from './tetronimoes.js' -->
<!-- {{#include ../../scripts/init-wasm-canvas.js}} -->
<!-- </script> -->
<!-- {{#include ../../scripts/canvas.html}} -->

## Overview

There are many ways you can make **anisotropic bodies** in *hoomd-rs*.
This tutorial shows you how to place multiple **sites** in **bodies** that
can translate *and rotate*. It uses the same *site-site* and *site-field*
interactions as the [Applying Interactions] tutorial and the lattice moves of
the [Custom Random Walk] tutorial. Combine those with tetronimo-shaped bodies,
and you get something very interesting.

* Objective: Show how to add multiple sites to a body.
* File: `hoomd-rs/examples/mc-tutorial/tetronimoes.rs`
* To build and run: `cargo run --release --features "bevy" --example tetronimoes`

## Use Declarations

```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:use}}
```

## Type Aliases

Create type aliases for our model's *vector*, *body properties*, and *site
properties* types so that you don't need to repeat the full nested generic types
throughout the code:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:type_aliases}}
```
The `OrientedPoint` type gives the tetronimo bodies in this tutorial both a
position in space and an orientation that rotates about the origin of the body.
These tetronimoes have disks at each site and therefore only need `Point`
properties.

## Custom Trial Move

Let's implement a custom trial move to make the tetronimoes move in the way
you might expect. Like in the [Custom Random Walk], tetronimoes take discrete
steps left, right, down, or up. Tetronimoes can also rotate by `$ \pm \pi/2 $`.
Here is the complete code:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:local_trial_all}}
```

`DiscreteRotateOrTransLate` implements `LocalTrial` by first enumerating the
possible moves:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:local_trial_steps}}
```

Then it chooses a random move and mutates the body properties accordingly:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:local_trial_mut}}
```

Using `random_bool()`, this code proposes translate moves more often than rotate
moves because the result is more visually interesting.

## The Simulation Model

Construct the simulation model as in the [Applying Interactions] tutorial:

```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:simulation_struct}}
```
but with a few differences.

### Trial Moves

Apply sweeps of the custom `DiscreteRotateOrTranslate` trial move:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:trial_moves}}
```

### Prepare the Tetronimo Shapes

`new()` also prepares a list of tetronimo shapes for later use. There are
five types of tetronimoes that are each represented by a vector of
four points:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:template_sites}}
```

### The Tetronimoes Struct

Here is the type that `new()` constructs:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:simulation_struct}}
```

## Advancing the Simulation

To advance the tetronimo simulation forward one step, follow the same procedure
used in [Applying Interactions]:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:impl_simulation}}
```

The code that adds tetronimoes is more complex than that for disks:
```rust,ignore
{{#include ../../../examples/mc-tutorial/tetronimoes.rs:add}}
```
It first chooses a random tetronimo from the `template_sites`, then it adds the
body near the top of the boundary with a default orientation of `$ \theta = 0 $`
and clone of the chosen sites.

*hoomd-rs* uses a *counter based random number generator*. Whenever you need to
use random numbers in your code, you can get a `Rng` to generate them by calling
`microstate.counter().make_rng()`. You *MUST* indicate that your substep is
complete by calling `microstate.increment_substep()` so that the next substep
will use a different set of random numbers.

## Conclusion

This tutorial showed you how to add bodies with multiple sites and how they
can be both translated and rotated.

Scroll back to the top of the page and refresh to see the simulation in
action again. Notice how the randomly generated tetronimoes fall to the
bottom while randomly rotating.

The next section will explain how to run self-assembly simulations of hard
particles.

[Applying Interactions]: applying-interactions.md
[Custom Random Walk]: custom-random-walk.md
