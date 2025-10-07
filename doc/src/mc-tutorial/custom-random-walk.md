# Custom Random Walk

<script type="module">
import init from './custom-random-walk.js'
{{#include ../../scripts/init-wasm-canvas.js}}
</script>
{{#include ../../scripts/canvas.html}}

## Overview

*hoomd-rs* allows you to customize your simulation model at *all* levels. This
tutorial you how to implement a custom **boundary condition** and a custom
**trial move**. It sets a closed boundary condition that confines bodies inside
a circle, and applies trial moves that take discrete steps left, right, down,
or up.

* Objective: Demonstrate the customization of a MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/custom-random-walk.rs`
* Run (interactively):
  ```shell
  cargo run --release --features "bevy" --example custom-random-walk
  ```
* Run (in batch mode):
  ```shell
  cargo run --release --example custom-random-walk
  ```

## Use Declarations

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:use}}
```
*hoomd-rs* uses [`rand`] crate to generate pseudorandom numbers.

## Custom Boundary Condition

The [Random Walk] tutorial used the *default* **open** boundary condition. This
tutorial shows how you can create a custom **closed** boundary condition.

### Define the `Circle` Type

First, define a type that describes the boundary. In this case, the boundary is a
`Circle` that has a radius:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_struct}}
```

### Implement `IsPointInside`

Then, implement the `IsPointInside` trait for `Circle`. The only required method
is `is_point_inside()` which should return `true` for points inside the boundary
and `false` for points outside the boundary:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_impl}}
```
This implementation of `IsPointInside` is only for the `Cartesian<2>` vector
type (`V` in the `IsPointInside` definition). You can read [The Rust Programming
Language] to learn more about **generic types** and **traits**.

> [!TIP]
> Many shapes in `hoomd_geometry` implement `IsPointInside` and can be used for
> closed boundary conditions. This tutorial creates a new `Circle` type to teach
> you about customization, but `hoomd_geometry::shape::Circle` would work just
> as well.

## Custom Trial Move

The [Random Walk] tutorial moved points by a random distance (up to a
maximum) and in a random direction. Look up *"random walk"* in a text book and
you find will a model that makes random hops on a square lattice.
You can implement that in *hoomd-rs* with a custom trial move.

### Define the Discrete Type

Similar to the custom boundary, you need to implement the **trait**
`LocalTrial` for a **type**. Here is the type:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_struct}}
```
`Discrete` trials always take one step in one direction. Therefore, the
`Discrete` struct needs no fields.

### Implement `LocalTrial`

`LocalTrial` has one method, `propose()`:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_impl}}
```
`propose()` takes in the properties (`body_properties`) of a body in the current
microstate of the system. It returns the **trial** body properties randomly
generated using the given `rng`. In this tutorial, the only body property is
`position`.

#### Enumerate Possible Steps

First, place the possible moves (down, up, left, right) in an array:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_steps}}
```

#### Randomly Select a Step

Then choose one of the steps randomly and add it to the body's position:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_mut}}
```

`steps.choose(rng)` chooses one of the elements of the array at random (with
uniform probability). Arrays in Rust do not have the `choose()` method by
default, it is provided by the `IndexedRandom` **trait** in the [`rand`]
crate. `choose()` will return `None` when the input array is empty. That can
never happen in this code, so you can safely unpack the option's value with
`expect()`.

> [!IMPORTANT]
> Local trial moves **MUST** satisfy *local detailed balance*,
> as defined in [Manousiouthakis & Deem](https://doi.org/10.1063/1.477973).

## The Simulation Model

The [Random Walk] tutorial used local variables in the `main()` function to
store all the elements of the simulation model. You can certainly continue that
practice in your applications. As you build more complex codes, however, you
will need to move those elements to a struct so that the whole simulation model
can be accessed in different modules.

The custom random walk model consists of the **microstate**, the **Hamiltonian**, the
translation **trial moves**, and the **temperature** set point (in units of energy
$`kT`$):
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:simulation_struct}}
```

Notice that this definition explicitly names the generic types of each of these fields. The
[Random Walk] tutorial did not do this because the Rust compiler *automatically
determined* which generic types were used. Structs can be made generic as well, but
the details are beyond the scope of this tutorial. Consult the [The Rust Programming
Language] to learn more.

### Construct the Simulation Model

The `new()` method that constructs the `CustomRandomWalk` simulation model:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:simulation_new}}
```

One or more steps might fail, so return a `Result<CustomRandomWalk>`.

#### Parameters

Assign all the model parameters in one code block so that they are easy to
modify:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:parameters}}
```

#### Microstate

Construct the `Microstate` with the `circle` boundary condition and place `n`
bodies at the origin:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:microstate}}
```

The newtype `Closed` can wrap *any* type that implements `IsPointInside`
(like the custom [`Circle`]) so that it can be used as a boundary condition.

[`Circle`]: #define-the-circle-type

#### Trial Moves

Apply the custom `Discrete` trial move to all bodies in the microstate:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:sweep}}
```
`Sweep` can wrap any type that implements `LocalTrial`.

#### Hamiltonian

As in the [Random Walk] tutorial, set $`H = 0`$ so that bodies do not interact:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:hamiltonian}}
```

#### Initialize the Struct

Now that all of the elements of the simulation model have been constructed,
package them in a `CustomRandomWalk` struct:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:initialize_struct}}
```

The struct is wrapped in `Ok` to indicate that the `Result` of this function is
not an error.

### Implement `Simulation`

The `Simulation` **trait** provides a common interface that all simulation
models can follow:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:impl_simulation}}
```

#### Advance the Simulation

The first method that all simulation models must have is an `advance()` method
that moves the model forward one step. The implementation is identical to that
in the [Random Walk] tutorial:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:advance}}
```
All the complexity of the customizations is contained in the implementation
of `IsPointInside` for `Circle` and `LocalTrial` for `Discrete`.

#### Get the Simulation Step

All simulation models must also implement a method that provides the current
simulation step:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:step}}
```

## Implement `main()`

To run the simulation, construct the `CustomRandomWalk` simulation model.
Then call `advance()` many times:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:main}}
```

Write the sites to a GSD file periodically so that you can inspect the results
of the simulation.

> [!NOTE]
> This `main()` function runs in batch mode. There is a different `main()` (not
> shown here) used in the interactive example.

## Conclusion

Now you know how to customize the random walk simulation with new
boundary conditions and apply your own trial moves to the points in it. Rust
compiles your customizations into machine code and can inline them into the main
simulation loop. This means that your custom simulations run *just as fast* as
they do when using the built-in types.

Navigate to the top of the page and refresh to see the simulation in action
again. Notice that no points leave the boundary. Try pausing the simulation and
advancing one step at a time. You should see that every particle moves left,
right, down, or up on every step.

Alternately, you can run the example in batch mode and then open
the generated `trajectory.gsd` in [Ovito] or another visualization tool:
```shell
cargo run --release --example custom-random-walk
```

The next section shows how to use the Hamiltonian to describe how the bodies
interact with each other and with an external field.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[Random Walk]: random-walk.md
[`rand`]: https://docs.rs/rand
[API documentation]: ../api.md
[Ovito]: https://www.ovito.org/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/custom-random-walk.rs:all}}
