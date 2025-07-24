# Custom Random Walk

<script type="module">
import init from './custom-random-walk.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

*hoomd-rs* allows you to customize your simulation model at **all** levels. This
tutorial takes the previous random walk and shows you how to implement a custom
**boundary condition** and a custom **trial move**. It sets a closed boundary
condition that confines bodies inside a circle, and applies trial moves that
take discrete steps left, right, down, or up.

* Objective: Demonstrate the customization of a MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/custom-random-walk.rs`
* To build and run: `cargo run --release --features "bevy" --example custom-random-walk`

## Use Declarations

```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:use}}
```
*hoomd-rs* uses [rand] crate to generate pseudorandom numbers.

## Custom Boundary Condition

The [Random Walk] tutorial used the *default* **open** boundary condition. This
tutorial shows how you can create a custom **closed** boundary condition.

First, you need to define a type that describes the boundary. In this case, the
boundary is a `Circle` that has a radius:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_struct}}
```

Then, implement the `Boundary` trait for `Circle`. The only required method
is `is_inside()` which should return `true` for points inside the boundary
and `false` for points outside the boundary:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_all}}
```
This implementation of `Boundary` is only for the `Cartesian<2>` vector type
(`V` in the `Boundary` definition), but is generic over any *body property* `B`
and any *site property* `S`. You can read [The Rust Programming Language] to
learn more about **generic types** and **traits**.

When you implement only `is_inside()`, the boundary becomes **closed**.
A later tutorial will demonstrate periodic boundary conditions.

## Custom Trial Move

The first [Random Walk] tutorial moved points by a random distance (up to a
maximum) and in a random direction. Look up "random walk" in a text book and the
first example you are likely to find will make random hops on a square lattice.
Let's implement that with a custom trial move. Here is the code:

```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_all}}
```

Similar to the custom boundary, you need to implement a **trait**
(`LocalTrial`) for a **type**. Here is the type:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_struct}}
```
Discrete trials always take one step in one direction. In this case, there
are no parameters and therefore the `Discrete` struct needs no fields.

`LocalTrial` has one method, `propose()`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_fn}}
```
`propose()` takes in the properties (`body_properties`) of a body in the current
microstate of the system. It returns the **trial** body properties. In this
tutorial, bodies are points and have only the position property.

> [!IMPORTANT]
> Local trial moves **MUST** satisfy *local detailed balance*,
> as defined in [Manousiouthakis & Deem](https://doi.org/10.1063/1.477973).

To implement a local trial move that can only step left, right, down, or up,
you first need to list the possible moves:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_steps}}
```

Then use the [rand] crate to choose one of the steps randomly and add
it to the body's position:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_mut}}
```

`steps.choose(rng)` chooses one of the elements of the array at random (with
uniform probability). Arrays in Rust do not have the `choose()` method by
default, it is provided by the `rand::seq::IndexedRandom` **trait**. `choose()`
returns an option that will never be `None` (unless all elements of `steps` are
removed), so you can unpack the option's value with `expect()`.

## The Simulation Model

Here is the `Simulation` struct, which is similar to `RandomWalk` in the [Random
Walk] tutorial:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:simulation_struct}}
```
The `Microstate` type now sets the `Circle` boundary condition as the third
generic type. Also, `translate_sweep` now has the type `Sweep<Discrete>`
indicating that it applies a sweep of the custom `Discrete` trial moves.

## Constructing the Custom Simulation

The `new()` method that constructs the simulation is also very similar to
that in the [Random Walk] example:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:simulation_new}}
```

There are two differences.

First, the `Microstate` is constructed with the custom
`Circle` boundary condition using `with_boundary()` instead of `new()`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:microstate}}
```

Second, the `Sweep` is constructed wrapping the custom `Discrete` type instead
of `Translate`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:sweep}}
```
`Sweep` can wrap any type that implements `LocalTrial`.

## Advancing the Simulation

The code to advance the simulation is identical to that in the [Random Walk]
tutorial:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:impl_simulation}}
```
All the complexity of the customizations is contained in the implementation
of `Boundary` for `Circle` and `LocalTrial` for `Discrete`.

## Conclusion

Now you know how to customize the random walk simulation with new
boundary conditions and apply your own trial moves to the points in it. Rust
compiles your customizations into machine code and can inline them into the main
simulation loop. This means that your custom simulations run *just as fast* as
using the built-in types.

Scroll back to the top of the page and refresh to see the simulation in action
again. Notice that no points leave the boundary. Try pausing the simulation and
advancing one step at a time. You should see that every particle moves left,
right, down, or up on every step.

The next section shows how to use the Hamiltonian to describe how the bodies
interact with each other and with an external field.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[Random Walk]: random-walk.md
[rand]: https://docs.rs/rand
