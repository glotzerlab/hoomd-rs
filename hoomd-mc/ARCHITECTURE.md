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
in energy when a single body is moved. One possible function signature is:
`fn delta_energy_one(microstate: &Microstate, new_body: &Tagged<Body>) -> f64`
where `new_body` defines the properties and sites of the new body after the
trial move and the method computes the change in energy from moving the body
with the given tag (ignoring self-interactions). This general scheme allows the
evaluation of energy deltas when trial moves mutate sites in the body, and in
general add/remove sites. Typical trial moves mutate only the body properties.
However, there is no reason to limit `DeltaEnergyOne` to this subset as the
implementation is essentially the same. Due to limitations in `Microstate`,
changing the number of sites in the body requires removing and then adding it
back in, but that is a problem for the `Trial` implementation.

Two more types compute the energy delta when inserting and removing bodies from
the microstate.

`DeltaEnergy` and friends are defined as traits so that users can implement
custom interactions in their MC simulations. `hoomd_mc` will implement these
traits for very commonly used interactions (e.g. cutoff pair potentials) in the
`hoomd_interaction` crate. `hoomd_mc` should also implement some solution for
summing several `DeltaEnergy` types together, for example via an implementation
on a tuple of different types, each implementing the appropriate traits.

### Handling infinite energies

Hard particle simulations model systems where overlapping particles yield an
infinite energy. In an ideal world, the `Energy` traits could simply return
the floating point representation of `inf`: `exp(-inf)` is correctly evaluated
as 0, so these trial moves will be rejected. The problem is that floating point
math is not perfect, nor can user inputs be trusted. For example, a user might
provide an initial condition where particles overlap. Or, MC accepts a move
with no overlaps, then wraps the particle around the periodic boundary. Due to
round-off errors, the new ghost particle ends up in a slightly different position
than the trial causing an overlap. In the latter case, these miniscule overlaps
can safely be ignored. In the former, MC trial moves may or may not be able to
remove the overlaps present in the initial condition.

In both cases, `DeltaEnergy` would effectively be computing `inf - inf` which
is undefined. HOOMD-blue avoids this issue (and also boosts performance for
hard particle simulations) by _skipping_ the overlap checks in the current
configuration. This is equivalent to assuming that the current configuration is
always valid, leading to an effective delta E of `0 - inf` and a rejection of
the move.

`inf - inf` is `NaN` which correctly evaluates to a rejected move in the
Metropolis check. Therefore, potentials that return `inf` are technically
well-defined and should not be prevented. To boost performance, specialized
implementations of `DeltaEnergy` for hard overlaps will skip the initial
energy calculation.

In some cases, such as Frenkel-Ladd integration, the move sizes may be so small
that the phantom overlaps cannot be resolved. In such cases, potentials can
compute an energy of 1000 instead of inf. In double precision `exp(-1000) == 0`,
so this is effectively infinite. What changes is that `E_final - E_initial` is
now computed as `1000 - 1000 == 0` when going from a microstate with overlaps
to another with overlaps. In a narrow set of cases, this may be desirable. In
the Frenkel-Ladd case, it could allow a phantom overlap to be resolved over the
course of a few trial moves. Users of this mode should be careful, as competing
energies on the same order of magnitude could cancel out. For example, a high
pressure simulation could easily collapse all particles onto a single point.

## Overlap counts and early exit conditions

The hard potential types themselves can exit early after finding the first overlap
when producing a value for `DeltaEnergy`. However, some algorithms (like `QuickCompress`
need to know the full count. TODO: Determine how to opt-out of early exit conditions.

When using `DeltaEnergy` for trial moves, there is no need to evaluate the soft
potentials if the hard potential has already found an overlap. An opt-out
of this early exit should also be implemented in a composite hamiltonian.

## Trial moves

Each trial move is implemented in its own type. The struct fields hold the
parameters of the move (such as the maximum move size). The apply method (which
takes a mutable reference to a microstate) attempts a trial move and modifies
the microstate when accepted given the Hamiltonian. The trial move type has
no internal state and is not specifically associated with any simulation or
microstate object. One trial move can be reused on many different systems,
provided they have the same generic types and the move size parameters are
meaningful to set the same. This design therefore requires an auxiliary type
to track the trial move counts for use with monitoring and tuning moves. It is
the responsibility of the caller to accumulate counter values (if desired). The
method signature will look something like: `fn apply(&self, &mut microstate: M,
hamiltonian: &H, state: &Macrostate ) -> Counter` To make accumulation easy, the
`Counter` types should implement the necessary arithmetic traits. 

## Model parameters

There are many model parameters, and different trial moves use different ones.
For example, kT will be used by practically every type of trial. Pressure will
be used by box moves, and fugacity by insert/remove moves. The Hamiltonian
itself is also a model parameter.

The caller owns the model parameters. This way, the caller can change the
parameters at will. Rust purposefully makes sharing state difficult (it is
possible with `Arc`). `Trial` will not take a shared state, but will borrow
the state on every call to `apply`. This makes the flow of information clear
to the user, and makes it obvious when and where that state can be modified.
Additionally, it allows a single trial move object to be reused on many
microstates with different Hamiltonians and/or at different macrostates.

The `Macrostate` associated type on `Trial` describes which parameters a
particular trial move needs. It should be set to a scalar or a tuple of the
state parameters (`(kt, pressure)` for example) used by the trial move.

### Tuning move sizes

Users commonly need to tune trial move sizes. Based on the structure above, it
is not possible for tuners to operate in the same way as they do in HOOMD-blue.
The trial move carries no internal state and the caller manages the counters.
One way it could be done is via a recipe -- a generic method that users can call
that runs several batches of steps, counts the moves, and adjusts the move size
accordingly. This solution would only allow tuning one trial move type at a time
and excludes other operations that the user inserts into the simulation loop
(e.g. trajectory file writes). However, it does force users to not continuously
tune (which breaks detailed balance). Additionally, this type of tuning could
be much more efficient for local moves. Instead of tuning after N full sweeps
(which can be expensive in large simulations), it could perform N individual
trial moves. The cost of tuning would be fixed regardless of system size.

Tuning box move sizes requires local trial moves. Otherwise, box moves
of a given size will always be accepted and tuning requires sampling the
probability of acceptance. Therefore, the recipe needs some way to identify
additional move types. Users might find missing trajectory frames or other
operations during tuning is not acceptable. To allow this, the tuner should be
implemented as a basic building block that users could incorporate into their
own simulation loops as desired. The move size tuner recipe should be built on
that implementation.

The tuning recipe should take ownership of the microstate, and return the
modified microstate back. This way, users could even opt to clone their
microstate so that the moves introduced by the tuner do not appear in the
trajectory.

