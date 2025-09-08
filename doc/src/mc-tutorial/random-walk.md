# Random Walk

<script type="module">
import init from './random-walk.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

A random walk describes the motion of a point over a series of steps. At each
step, the point translates by a random vector. This tutorial shows you how to
implement a random walk using *hoomd-rs*. There are certainly easier ways to
write a random walk code, but the purpose of this tutorial is to explain how
you can express the components of a MC simulation using *hoomd-rs*.

* Objective: Demonstrate a minimal MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/random-walk.rs`
* To build and run: `cargo run --release --features "bevy" --example random-walk`

## Bring Used Names Into Scope

The first lines of any Rust code typically bring all the used names into scope:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:use}}
```

Rust's `use` is similar to Python's `import`. See [The Rust Programming
Language] for more information.

`std` is the [standard library] included with Rust. [`hoomd-interaction`],
[`hoomd-mc`], [`hoomd-microstate`], and [`hoomd-vector`] are *crates* that each
implement a part of the simulation. The [API documentation] provides a for a
full reference for all *hoomd-rs* crates.

## The Simulation Model

In the random walk simulation, the **microstate** contains the positions of `$N$`
points, the **sweep** applies a **trial move** to each point, the Hamiltonian
is always 0 and the temperature `$kT$` is not relevant.

Define a type that collects all these:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:simulation_struct}}
```

## Constructing the Simulation Model

The `new()` function constructs a `RandomWalk` simulation. Here is the
complete function:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:simulation_new}}
```

### The new() Function

The first part **implements** the associated function `new()`
for `RandomWalk`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:new_fn}}
```
The purpose of the `new()` function is to construct a new simulation. However,
an error might occur when doing so, so it returns a `Result` type
(see [The Rust Programming Language] for more information).

### Parameters

Next, set the parameters of the simulation:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:params}}
```
`$kT$` (`kt`) is the temperature of the simulation (in units of energy), `$d$`
(`d`) is the maximum distance to move a point during a trial move, and `$N$`
(`n`) is the number of points to add to the **microstate**.
It is a good idea to collect all your model parameters into one place in the
code, especially those that are used in multiple places.

### Microstate

The **microstate** describes all of the degrees of freedom in the simulation. In
this example, it consists of `$N$` **bodies**, each of which is a single point
(later tutorials will show how you can create other types of bodies).
This code builds a microstate and adds `$N$` points at the origin:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:microstate}}
```
`try_build()` may fail with an error. The `?` will return an error or unpack a
valid `Result`. Read more about `?` in [The Rust Programming Language].

### Trial Moves

A random walk is defined entirely by the **trial moves** applied to each **body**.
The `Translate` trial move describes a displacement by a random vector
drawn uniformly inside the sphere with radius `maximum_distance`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:local_trial}}
```
The `try_into()?` ensures that the given `f64` value is a positive
real value.

`translate` describes how **trial moves** should be applied to *individual
bodies*. Now you need to describe how to apply these trial moves to the *whole
microstate* using `Sweep`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:sweep}}
```
`Sweep` applies the given **trial move** to each of the **microstate's
bodies** in sequence and accepts or rejects each move based on the Metropolis
criterion.

### Hamiltonian

The points in a random walk do not interact. Set the Hamiltonian `$H = 0$`, so
that `Sweep` will accept every **trial move**. The [`hoomd-interaction`] crate
provides the `Zero` type that expresses `$H = 0$`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:hamiltonian}}
```

### Returning the Simulation

The code has now constructed all the elements of the simulation. Construct
a `RandomWalk` containing these fields for later use:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:return}}
```
Read more about instantiating structs in [The Rust Programming Language].

## Advancing the Simulation

The `Simulation` **trait** includes two methods: `advance()` moves the
simulation forward one step and `step()` returns the current step. The
`advance()` method may return an error via `anyhow::Result`.

Here is the full implementation of `Simulation` for `RandomWalk`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:impl_simulation}}
```

### Apply Translation Moves

The `translate_sweep.apply()` method step applies a translate move to each body
in the microstate:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:apply}}
```

### Increment the Step

*hoomd-rs* makes no assumptions about your simulation model. One step in your
model may involve many types of MC **trial moves**, or a mixture of MD and
MC calculations. Therefore, you *must* explicitly call `increment_step()` to
indicate that this step is complete:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:increment_step}}
```

> [!TIP]
> To simplify your scripts, omit the `struct` that holds the `Simulation`.
> You can instead combine the contents of `new()` and `advance()` (wrapped in
> a loop) directly in your `main()` function. These examples use the struct to
> facilitate the interactive display.

# Conclusion

Now you have learned how to create a **microstate** and apply random translation
**trial moves** to the points in it.

Navigate to the top of the page and refresh to see the simulation in action
again. Try pausing the simulation and advancing one step at a time. See if you
can identify individual local trial moves. Notice that every particle moves
at least a little bit on every step. You should also notice that the particles
can move without bounds. By default, a **Microstate** has **open** boundary
conditions.

The next section will show you how to apply custom boundary conditions
and custom trial moves to the random walk.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[standard library]: https://doc.rust-lang.org/std/
[`hoomd-interaction`]: ../api/hoomd_interaction/index.html
[`hoomd-mc`]: ../api/hoomd_mc/index.html
[`hoomd-microstate`]: ../api/hoomd_microstate/index.html
[`hoomd-vector`]: ../api/hoomd_vector/index.html
[API documentation]: ../api.md
[`anyhow`]: https://docs.rs/anyhow/
