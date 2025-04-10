# hoomd_mc

The `hoomd_mc` crate implements Monte Carlo (MC) simulations for systems
of interacting bodies. There are two main components to MC simulations: The
energy of the system and the trial moves that evolve the microstate.

## DeltaEnergy

The **Energy** of the system defines how the bodies interact. The `Energy` trait
itself is more general and therefore lives in another crate (`hoomd_model`)
so that it can be used by MD simulations or for offline analysis. What MC
simulations need is a way to evaluate the change in the Energy between two
states: the `DeltaEnergy`.

There are many types of `DeltaEnergy` to facilitate efficient evaluation of
different types of trial moves. The most general type is `DeltaEnergy` which
defines a method that evaluates the delta-H between two microstates: `fn
delta_hamiltonian(a: &Microstate, b: &Microstate) -> f64`. This method will
be called when the entire simulation box is scaled, for example. However, it
requires a full O(N) computation which is inefficient when only one particle
is moved.

The second type is `DeltaEnergyOne` that can efficiently compute the change
in energy when a single body is moved. The function signature will look
something like: `fn delta_hamiltonian_one(microstate: &Microstate, new_body:
&Tagged<Body>) -> f64`. The `new_body` defines the properties and sites of the
new body after the trial move and the method computes the change in energy from
moving the body with the given tag (ignoring self-interactions). TODO: consider
whether this fully general approach is fine, or if a more optimized version that
only allows changing the body properties would be helpful.

Two more types compute the energy delta when inserting and removing bodies from
the microstate.

`DeltaEnergy` and friends are defined as traits so that users can implement
custom interactions in their MC simulations. `hoomd_mc` will implement these
traits for very commonly used interactions (e.g. cutoff pair potentials) in the
`hoomd_interaction` crate. `hoomd_mc` should also implement some solution for
summing several `DeltaEnergy` types together, for example via an implementation
on a tuple of different types, each implementing the appropriate traits.

### Energy return type

TODO: Evaluate the implications (performance an otherwise) of returning f64
results that may be infinity for hard interactions. Is `exp(-inf)` expensive to
compute? What about `isinf` checks to avoid unnecessary calculations? Would it
be cleaner or more complicated to return an enum that differentiates between
infinity and valid floats?

## Trial moves

Each trial move is implemented in its own type. The struct fields hold the
parameters of the move (such as the maximum move size). The
apply method (which takes a mutable reference to a microstate) attempts a trial move and
modifies the microstate when accepted given the Hamiltonian. The trial move type
has no internal state and is not specifically associated with any simulation
or microstate object. One trial move can be reused on many different systems,
provided they have the same generic types and the move size parameters are
meaningful to set the same. This design therefore requires an auxiliary type
to track the trial move counts for use with monitoring and tuning moves. It
is the responsibility of the caller to accumulate counter values (if desired).
The method signature will look something like: `fn apply(&self, &mut microstate:
M, &hamiltonian: H) -> Counter` To make accumulation easy, the `Counter` types
should implement the necessary arithmetic traits.

## Model parameters

There are many model parameters, and different trial moves use different ones.
For example, kT will be used by practically every type of trial. Pressure will
be used by box moves, and fugacity by insert/remove moves. The Hamiltonian
itself is also a model parameter.

Should MC define a `Model` type that collects all these together?

There are problems with a catch-all `Model` type.
* NVT simulations have no set pressure. Therefore, pressure should be an option.
  However, it is an error to apply a volume-changing move to a system with no
  pressure set. The validation that `pressure` matches `Some(value)` will occur
  at runtime, where errors like this would be best detected at compile time.
* New types of trial moves in the future may require adding new model parameters.
  This would be an API breaking change, and also impossible for users to achieve
  with custom implementations of `Trial`.

The only advantage to a `Model` type is that these values would all be held
in one place. Interoperation with MD is a possibility, but only if the `Model`
were to also include parameters like `delta_t` that are meaningless to MC
simulations.

Whether the model parameters are held in a `Model` type or as separate
variables, the second question is this: Who owns (and/or holds references) to
the parameters? Should a `Trial` object copy or clone parameters given to it?
Probably not, as that can easily lead to errors where a parameter is changed in
one place but not others. To avoid this, should a `Trial` hold a reference to the
parameters? Doing so would avoid the possibility of accidentally passing one
temperature to the local moves and another to the box moves. However, doing this
will tie the lifetime of the `Trial` to the lifetime of the parameters - which
prevents `Trial` from being reused on different models or generally standing on
its own.

One solution for this is to implement `apply` separately for each trial move
type that accepts the model parameters it needs. This is a simple and clean
approach with the disadvantage that we can no longer have an overarching
`Trial` trait with a common `apply` method signature!

JAA- On balance, I think that `Trial` should hold reference to the parameters.
It really cannot stand on its own and other parameters (such as move sizes)
are inherently related to the model parameters (move sizes are smaller at lower
temperatures). I will leave the other proposal here in case we want to revisit
this design after testing.

### Tuning move sizes

Users commonly need to tune trial move sizes. Based on the structure above, it
is not possible for tuners to operate in the same way as they do in HOOMD-blue.
The trial move carries no internal state and the caller manages the counters.
One way it could be done is via a recipe -- a generic method that users can call
that runs several batches of steps, counts the moves, and adjusts the move size
accordingly. This solution would only allow tuning one trial move type at a time
and excludes other operations that the user inserts into the simulation loop
(e.g. trajectory file writes). However, it does force users to not continuously
tune (which breaks detailed balance).

Tuning box move sizes requires local trial moves. Otherwise, box moves
of a given size will always be accepted and tuning requires sampling the
probability of acceptance. Therefore, the recipe needs some way to identify
additional move types. Users might find missing trajectory frames or other
operations during tuning is not acceptable. To allow this, the tuner should be
implemented as a basic building block that users could incorporate into their
own simulation loops as desired. The move size tuner recipe should be built on
that implementation.

TODO: Implement for_each to automatically increment the time step.
