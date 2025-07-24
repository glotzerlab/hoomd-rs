# Random Walk

<script type="module">
import init from './random-walk.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>

<canvas id="hoomd-example" width="750" height="421" style="width: 100%; height: 100%; min-width: 180px; min-height: 120px;"></canvas>
*Pres `tab` or click to focus. Refresh the page to restart the simulation.*

## Overview

A random walk describes the motion of a point over a series of steps. At each
step, the point translates by a random vector. This tutorial shows you how to
implement a random walk using *hoomd-rs*. There are certainly easier ways to
write a random walk code, but the purpose of this tutorial is to explain how
you can express the components of a MC simulation using *hoomd-rs*. You will use
these same concepts in more complex simulations.

* Objective: Demonstrate a minimal MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/random-walk.rs`
* To build and run: `cargo run --release --features "bevy" --example random-walk`

## Bring Used Names Into Scope

The first lines of any Rust code typically bring all the used names into scope:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:use}}
```

If you are familiar with Python, `use` is similar to `import`. In Rust, `use`
is optional. After you add `hoomd-vector` as a dependency, names such as
`hoomd_vector::Cartesian` are available throughout your code. The declaration
`use hoomd_vector::Cartesian` makes the type available under the shorter name
`Cartesian`. See [The Rust Programming Language] for more information.

`std` is the [standard library] included with Rust.
[`hoomd-mc`], [`hoomd-microstate`], and [`hoomd-vector`] are *crates* that
each implement a part of the simulation:
* [`hoomd-mc`] implements trial moves and sweeps that apply them to microstates.
* [`hoomd-microstate`] provides types and traits that describe the microstate.
* [`hoomd-vector`] describes vector types and operations.

The [API documentation] completely describes every crate *hoomd-rs*. As you
read these tutorials, open the API documentation for the types and traits you
encounter to see how else they can be used.

## The Simulation Model

In the random walk simulation, the **Microstate** contains the positions of `$N$`
points, the **Sweep** applies **trial moves** to all `$N$` points, the Hamiltonian
is always 0 and temperature `$kT$` is not relevant.

Define a struct that collects all these:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:simulation_struct}}
```

## Constructing the Simulation

The `new()` function constructs a `RandomWalk` simulation. Here is the
complete function:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:simulation_new}}
```

Let's look at it in more detail. The first part **implements** (`impl`) the
`new()` associated function of `RandomWalk`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:new_fn}}
```
The purpose of the `new()` function is to construct a new simulation. However,
an error might occur when doing so. Rust lacks exceptions by design. Every
function that might error *should* return a `Result` type that will be (in
this case) `Ok(RandomWalk)` when the simulation is constructed successfully
or `Err(anyhow::Error)` when there is an error. [`anyhow`] is not part of
the Rust standard library. It is a commonly used crate that can accept any
kind of error will conveniently format the error chain when `main` returns
`Err(anyhow::Error)`.

### Parameters

Next, set the parameters of the simulation:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:params}}
```
`$kT$` (`kt`) is the temperature of the simulation (in units of energy), `$d$`
(`d`) is the maximum distance to move a point during a trial move, and `$N$`
(`n`) is the number of points to add to the **microstate**.

### Microstate

The **microstate** describes all of the degrees of freedom in the simulation.
It consists of `$N$` **bodies** that are each a single point
(a later tutorial will explain `Microstate` in more detail):
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:microstate}}
```
The random walk example constructs a microstate and adds `$N$` points at the
origin (`Cartesian::default()` constructs the 0 vector).

*By default*, a **Microstate** has **Open** boundary conditions that allow
bodies to be placed anywhere in space. `try_build` returns a `Result` that will
be an `Err` when a body is placed outside the boundaries. That is not possible
with **Open** boundary conditions, but you must check for the error regardless.
That is what the `?` does. If `try_build()` returns an error, `try_build()?`
will stop executing `new()` and return the error. When `try_build()` returns
`Ok(microstate)`, `try_build()?` unpacks the `microstate`. To summarize, the `?`
is a useful shorthand so that you don't need to write an if/else every time after
calling a function that returns a `Result`. Read more about `?` in [The Rust
Programming Language].

### Trial moves

A random walk is defined entirely by the trial moves applied to each **body**.
The `Translate` trial move describes a displacement by a random vector
drawn uniformly inside the sphere with radius `maximum_distance`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:local_trial}}
```
What does the `try_into()?` accomplish? Go look at the [API documentation]
for `Translate`. When you find it, you will see that `maximum_distance` is
a `PositiveReal` -- which is a `f64` value guaranteed to be not infinity,
not NaN, and greater than 0. Even though you didn't bring `PositiveReal` in
with a use declaration, `d.try_into()?` can still attempt to convert `d` to a
`PositiveReal`. If the conversion fails, `try_into` will fail and the resulting
error will propagate all the way up to `main` at which point [`anyhow`] will
print the error message.

> [!TIP]
> To find the reference for anything in *hoomd-rs*: open the documentation for
> *any* of the crates, press `s` to open the search bar, type a few characters
> of the name, press the down arrow, then press return.

When the conversion succeeds, `translate` will describe how trial moves should
be applied to *individual bodies*. Now you need to describe how to apply these
trial moves to the *whole microstate*. `Sweep` does just that, applying the
given local trial move to each of the bodies in sequence:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:sweep}}
```
Sweep accepts or rejects trial moves based on the Metropolis criterion.

### Hamiltonian

The points in a random walk do not interact, so the Hamiltonian `$H = 0$`,
and `Sweep` will accept every trial move.

The [`hoomd-mc`] crate provides a convenient type that expresses `$H = 0$`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:hamiltonian}}
```
Later tutorials will introduce non-zero interactions.

### Returning the Simulation

The code has now constructed all the elements of the simulation and there is
no longer the possibility for error. All that remains is to package up
the local variables into a `RandomWalk` struct for later use:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:return}}
```
Remember that the `new()` function returns a `Result<RandomWalk>`. Therefore, it
needs to return `Ok(RandomWalk { ... })` to indicate that the result is *OK* and
not an error.

## Advancing the Simulation

In `Rust` a **trait** is a generic interface. When you **implement a trait**
for a **type**, you provide a specific set of methods implemented that operate
on that type. The `Simulation` trait includes two methods: `advance()` moves the
simulation forward one step and `step()` returns the current step. The
`advance()` method may return an error via `anyhow::Result`.

Here is the full implementation of `Simulation` for `RandomWalk`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:impl_simulation}}
```
Notice that `advance()` takes a mutable reference to *self* (`&mut self`). This
is similar to the `self` argument in Python or the implicit `this` variable
in C++.

The `translate_sweep.apply()` method step applies a translate move to each body
in the microstate:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:apply}}
```

*hoomd-rs* makes no assumptions about your simulation model. One step in your
simulation may involve many types of MC trial moves, or maybe a mixture of MD
and MC calculations. Therefore, you must explicitly call `increment_step()` to
indicate that this step is complete:
```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:increment_step}}
```
If you fail to call `increment_step()`, your simulation will still run, but
`step()` will always return 0. All frames you write to a trajectory file will
also be recorded at step 0.

> [!TIP]
> In simple scripts, you can skip both the `struct` and the `Simulation` trait.
> You can instead construct and use the `microstate`, `translate_sweep`, and
> `hamiltonian`, variables directly in your `main` function. These examples must
> use a struct to facilitate the interactive display.

# Conclusion

Now you have learned how to create a **microstate** with open boundary
conditions and apply random translation **trial moves** to the points in it.

Now that you know how it works, scroll back to the top of the page and refresh
to see the simulation in action again. Try pausing the simulation and advancing
one step at a time. See if you can identify individual local trial moves. Notice
that every particle moves at least a little bit on every step.

The next section will show you how to apply custom boundary conditions
and trial moves to the random walk.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[standard library]: https://doc.rust-lang.org/std/
[`hoomd-mc`]: ../api/hoomd_mc/index.html
[`hoomd-microstate`]: ../api/hoomd_microstate/index.html
[`hoomd-vector`]: ../api/hoomd_vector/index.html
[API documentation]: ../api.md
[`anyhow`]: https://docs.rs/anyhow/
