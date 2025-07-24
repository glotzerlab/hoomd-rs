# Custom Random Walk

<script type="module">
import init from './custom-random-walk.js'

init().catch((error) => {
  if (!error.message.startsWith("Using exceptions for control flow, don't mind me. This isn't actually an error!")) {
    throw error;
  }
});
document.getElementById('hoomd-example').addEventListener("keydown", function(e) {
  e.stopPropagation();
});
</script>

<canvas id="hoomd-example" width="750" height="421" style="width: 100%; height: 100%; min-width: 180px; min-height: 120px;"></canvas>
*Refresh the page to restart the simulation.*

## Overview

*hoomd-rs* allows you to customize your simulation model at **all** levels. This
tutorial takes the previous random walk and shows you how to implement a custom
**boundary condition** and **trial move**. It sets a closed boundary condition
that confines bodies inside a circle, and applies trial moves that take discrete
steps left, right, down, or up.

* Objective: Demonstrate the customization of a MC simulation.
* File: `hoomd-rs/examples/mc-tutorial/custom-random-walk.rs`
* To build and run: `cargo run --release --features "bevy" --example custom-random-walk`

## Use Declarations

```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:use}}
```

Compared to the [Random Walk] code, the customized random walk
uses some new traits: `rand::Rng`, `rand::seq::IndexRandom`,
`hoomd_microstate::boundary::Boundary`, and `hoomd_vector::Vector`.

While these use declarations are optional for **types**, they are *required*
for many **traits**. You cannot call a trait's method(s) unless you include that
trait in a use declaration. When you forget, the Rust compiler can often tell
you what change will fix the error.

Here is an example of the compile error Rust gives when you omit the `Vector`
trait:

```text
error[E0599]: no method named `distance` found for reference `&Cartesian<2>` in the current scope
  --> examples/mc-tutorial/custom-random-walk.rs:28:15
   |
28 |         point.distance(&[0.0, 0.0].into()) < self.radius
   |               ^^^^^^^^
   |
   = help: items from traits can only be used if the trait is in scope
help: trait `Vector` which provides `distance` is implemented but not in scope; perhaps you want to import it
   |
2  + use hoomd_vector::Vector
```

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

There is a lot going on in these few lines. Let's examine the more complicated
parts. This line implements the `Boundary` trait for the type `Circle`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_impl}}
```
`impl<B, S>` states that this implementation has the **generic types** `B` and
`S`. `Boundary<Cartesian<2>, B, S>` states that the trait being implemented is
`Boundary` where it's **generic types** are `Cartesian<2>`, `B`, and `S` (in
that order). Find the documentation for the `Boundary` trait and you will see
that the first (`Cartesian<2>`) is the vector space in which the simulation
exists, the second (`B`) stands for body properties, and the third (`S`) for
site properties. The next section will discuss body and site properties in
more detail. The `is_inside()` check doesn't use `B` or `S`, so these can be left
*completely generic*. The `for Circle` states that this implementation adds the
`is_inside()` method to the `Circle` type.

In other words, this `impl` allows our code to call `circle.is_inside(point)`,
as long as `point` has the `Cartesian<2>` type. You will get a compiler error
if `point` is `Cartesian<3>` or any other type.

The body of `is_inside()`
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:boundary_check}}
```
returns `true` when the distance from `point` to the origin is less
than the radius and `false` otherwise. The default implementations of
`Boundary::wrap_body()` and `Boundary::wrap_site()` use this `is_inside()` check
to prevent any body or site from going outside the boundary.
Specifically, Monte Carlo simulations will reject any trial move that places
the point outside the boundary.

You probably have many questions at this point, such as "What about periodic
(and partially periodic) boundaries?" and "Wait, what are generic traits
again?". A later tutorial will cover periodic boundaries. You can read [The
Rust Programming Language] to learn more about **generic types** and **traits**.

## Custom Trial Move

The first [Random Walk] tutorial moved points by a random distance (up to a
maximum) and in a random direction. Look up "random walk" in a text book and the
first example you are likely to find will make random hops on a square lattice.
Let's implement that with a custom trial move. Here is the code:

```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_all}}
```

Similar to the custom boundary, you need to implement a **trait**
(`LocalTrial`) for a **type**. First, you need to define the type:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_struct}}
```
Discrete trials always take one step in one direction. In this case, there
are no parameters and therefore the `Discrete` struct needs no fields. In Rust,
this is called a **unit struct**.

`LocalTrial` has one method, `propose()`:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_fn}}
```
Here, the `<R: Rng>` makes `propose()` a **generic function** that can be called
with any type `R` that satisfies the trait bound `Rng`. In other words, `propose()`
will work with any random number generator from the [rand] crate.

`propose()` takes in the properties (`body_properties`) of a body in the current
microstate of the system. It returns the **trial** body properties. In this
tutorial, bodies are points and have only the position property. In your code,
you could give your bodies any properties and modify them with trial moves.

> [!IMPORTANT]
> Local trial moves **MUST** satisfy *local detailed balance*,
> as defined in [Manousiouthakis & Deem](https://doi.org/10.1063/1.477973).

To implement a local trial move that can only step left, right, down, or up,
you first need to list the possible moves:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_steps}}
```
Notice here that the type `Cartesian<2>` is not mentioned. Instead the
elements are values like `[0.0, 1.0].into()`. The *Into* **trait** is part
of the Rust standard library and it allows you to convert a value of one type
*into* another, provided that the conversion exists. You, the programmer, do
not necessarily need to know what type it is converting into. Rust examines
the entire function, sees how the values are used, and determines the type
automatically. This process succeeds when there is only *one* possible type.
If more than one type could be used, Rust will issue a compile error and ask you
to set the type explicitly.

Then you can use the [rand] crate to choose one of the steps randomly and add
it to the body's position:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:local_trial_mut}}
```

The first line creates a **mutable** variable `trial` with the current body
properties (variables are **immutable** by default in Rust). `steps.choose(rng)`
chooses one of the elements of the array at random (with uniform probability).
Arrays in Rust do not come with the `choose` method by default. The
`rand::seq::IndexedRandom` **trait** implements it. This is one very powerful
capability in Rust. In an object-oriented language `ran` would have to create
a whole class structure and invent purpose-built array types to do the same. In
Rust, *any* trait may be implemented for *any* type, including built-in types.

Now, what if the array had 0 elements? What vector should `choose()` return?
That is a trick question, there is no vector that it can. In some languages,
`choose()` might return a references to a vector, leading to a segmentation
fault if the method is ever called with 0 elements. Rust doesn't allow that.
`choose()` is implemented in a *safe* manner. It returns an `Option<T>` that
is `Some(T)` when the array has 1 or more elements and `None` when the array
is empty. You need to unpack the `Option` and handle the `None` appropriately.
In this case, we hard-coded the array and know it *should* always have 4
elements. However, what if some other programmer starts modifying this function
and without realizing it comments out all of the array elements? It would
be polite to give that programmer (who may be your future self) a nice error
message indicating what assumption is violated when `None` is chosen. That is
what `.expect("message")` does. It unpacks the vector when `choose()` returns
`Some(vector)` and **panics** with the given error message when it returns
`None`. Read more about **panics** and `expect()` in [The Rust Programming
Language]. The reference documentation for `expect()` also has excellent
guidance on how to phrase the message.

`trial.position += ...` uses invokes the `SumAssign` vector math operation
implemented in the `hoomd_vector::Vector` trait to displace the body's
current position. It is *this point* where the compiler finds that
(via `expect()`, via `choose()`) that the elements of `steps` must have
the type `Cartesian<2>` because `trial.position` is a `Cartesian<2>`.

## The Simulation Model

The customizations modify the types of the objects in the `CustomRandomWalk`
struct which is otherwise very similar to `RandomWalk` in the [Random Walk]
tutorial:
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
`Sweep` can wrap any type that implements `LocalTrial`. It allows you,
the researcher, to think about the important part of your model
(what trial move to apply to a single body in this case) and reuse the
common code that loops over all bodies, applies the trial move,
and accepts or rejects it (which are all handled by `Sweep`).

## Advancing the Simulation

The code to advance the simulation is identical to that in the [Random Walk]
tutorial:
```rust,ignore
{{#include ../../../examples/mc-tutorial/custom-random-walk.rs:impl_simulation}}
```
All the complexity of the customizations is contained in the implementations
of the `Boundary` and `LocalTrial` traits.

## Conclusion

Now you know how to customize the random walk simulation with circular
boundary conditions and apply discrete trial moves to the points in it. Rust
compiles your customizations into machine code and can inline them into the main
simulation loop. This means that your custom simulations run *just as fast* as
using the built-in types.

Now that you know how it works, scroll back to the top of the page and refresh
to see the simulation in action again. Notice that no points leave the boundary.
Try pausing the simulation and advancing one step at a time. See if you can
identify individual local trial moves. Notice that every particle moves left,
right, down, or up on every step.

The next section will explain how **bodies** and **sites** relate to each
other in the **Microstate**.

[The Rust Programming Language]: https://doc.rust-lang.org/stable/book/
[Random Walk]: random-walk.md
[rand]: https://docs.rs/rand
