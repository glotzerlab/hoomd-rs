# Random Walk

## Overview

A random walk describes the motion of a point over a series of steps. At each
step, the point translates by a random vector. This tutorial shows you how to
implement a random walk using *hoomd-rs*. There are certainly easier ways to
write a random walk code, but the purpose of this tutorial is to explain how
you can express the components of a MC simulation using *hoomd-rs*.

* Objective: Demonstrate a minimal MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/random-walk.rs`
* To build and run:
  ```shell
  cargo run --release --example random-walk
  ```

## Bring Used Names Into Scope

The first lines of any Rust code typically bring all the used names into scope:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:use}}
```

Rust's `use` is similar to Python's `import`. See [The Rust Programming
Language] for more information.

[`hoomd-interaction`], [`hoomd-mc`], [`hoomd-microstate`], and [`hoomd-vector`]
are *crates* that each implement a part of the simulation. The [API
documentation] provides a for a full reference for all *hoomd-rs* crates.

## The `main()` Function

When executed, every Rust application starts in the `main()` function:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:main}}
```

An error might occur while running the simulation, so `main()` returns a
`Result` type. See [The Rust Programming Language] for more information. When
you use `anyhow::Result<()>` as the return value for `main()`, the [`anyhow`]
crate will nicely print any errors that occur in a human-readable format.

## The Simulation Model

In the random walk simulation, the **microstate** contains the position of the
point, the **sweep** applies a **trial move** to each point, the Hamiltonian
is always 0 and the temperature is not relevant.

### Microstate

The **microstate** describes all of the degrees of freedom in the simulation. In
this example, it consists of one **body** represented by a single point.
This code builds a microstate and adds one point at the origin:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:microstate}}
```
`try_build()` may fail with an error. The `?` will return an error or unpack a
valid `Result`. Read more about `?` in [The Rust Programming Language].

### Trial Moves

A random walk is defined entirely by the **trial moves** applied to each **body**.
The `Translate` trial move describes a displacement by a random vector
drawn uniformly inside the sphere with radius `maximum_distance`:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:local_trial}}
```
The `try_into()?` ensures that the given `f64` value is a positive
real value.

`translate` describes how **trial moves** should be applied to *individual
bodies*. Now you need to describe how to apply these trial moves to the *whole
microstate* using `Sweep`:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:sweep}}
```
`Sweep` applies the given **trial move** to each of the **microstate's
bodies** in sequence and accepts or rejects each move based on the Metropolis
criterion.

### Hamiltonian

Set the Hamiltonian $`H = 0`$, so that `Sweep` will accept every random walk
move **trial move**. The [`hoomd-interaction`] crate provides the `Zero` type
that expresses $`H = 0`$:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:hamiltonian}}
```

### Macrostate

Local trial moves are accepted or rejected based on the temperature of the
simulation. You need to supply a macrostate even though this simulation has
no interactions. Set the temperature to 1.0 as a placeholder:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:macrostate}}
```

## Advancing the Simulation

Use a `for` loop to execute the simulation a given number of steps:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:steps}}
```

### Apply Translation Moves

The `translate_sweep.apply()` method step applies a translate move to each body
in the microstate:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:apply}}
```

### Increment the Step

*hoomd-rs* makes no assumptions about your simulation model. One step in your
model may involve many types of MC **trial moves**, or a mixture of MD and
MC calculations. Therefore, you *must* explicitly call `microstate.increment_step()` to
indicate that this step is complete:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:increment}}
```

### Produce Output

To see some output, print the coordinates of the point at each step:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:print}}
```

Future tutorials will show interactive graphical representations of the
simulations and/or write simulation trajectory files.

## Return from the `main()` Function

Return `Ok(())` to exit the application with no errors:
```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:end_main}}
```

## Conclusion

Now you have learned how to create a **microstate** and apply random translation
**trial moves** to the points in it. Change to the `hoomd-rs` directory and
execute `cargo run --release --example random-walk` to run this tutorial. You
should  see output similar to:
```
[0.10107592706479421, 0.03942734821995694]
[0.0655227510362725, -0.02209445404554057]
[0.04227546664585716, 0.10644967595431494]
[0.1150968339945419, 0.07985960466968363]
[0.09117191688791704, 0.20862494597769088]
[0.19179261557730518, 0.15143176807959824]
[0.08739701966674704, 0.17504549059226845]
[0.16662973228106845, 0.16482677512825672]
...
```

Notice the particle moves a little bit on every step. Run the simulation for
many steps and notice that the particles can move without bounds. By default,
a **Microstate** has **open** boundary conditions. The next section will show
you how to apply custom boundary conditions and custom trial moves to the random
walk.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[`hoomd-interaction`]: ../api/hoomd_interaction/index.html
[`hoomd-mc`]: ../api/hoomd_mc/index.html
[`hoomd-microstate`]: ../api/hoomd_microstate/index.html
[`hoomd-vector`]: ../api/hoomd_vector/index.html
[API documentation]: ../api.md
[`anyhow`]: https://docs.rs/anyhow/

## Complete Code

```rust,ignore
{{#rustdoc_include ../../../examples/mc-tutorial/random-walk.rs:all}}
```
