# hoomd_model

The `hoomd_model` crate combines interactions from `hoomd_interactions` with
systems from `hoomd_microstate` and to define physical models of how those
systems behave.

## Energy

The **energy** of the system defines how the bodies of a microstate interact.
`hoomd_model` describes a `Energy` trait that computes the energy of a given
microstate. The crate also implements commonly used energies, such as external
potentials, cutoff pair potentials, and kinetic energy. Users can write custom
types that implement the `Energy` trait.

## Hamiltonian

The **Hamiltonian** is the sum of all the energies that apply to a given system.
`hoomd_model` should provide some convenient syntax to achieve this. One idea
would be to implement the `Hamiltonian` trait on tuples of heterogeneous types,
where each element of the tuple also implements `Hamiltonian`. There should
also be a blanket implementation that allows a single `Energy` to be used as a
`Hamiltonian` as this is commonly used in MC simulations.

TODO: Will this idea work for MD and MC? MD may need to separate potential
energy from kinetic (and further separate translational and rotational kinetic
energies). Furthermore, one can run MD simulations on non-Hamiltonian systems
(e.g. active matter) where some forces are not associated with a potential
energy.

## Infinite energies

The `Energy` and `Hamiltonian` traits must serve the needs of both MD and MC
simulations. MD simulations will call `energy` only for logging and relies
mainly on `force` and `torque` (see below). MC simulations use `energy` directly
in the evaluation of trial move acceptance. For finite potentials, there is
no practical difference. But MC simulations can (and often do) operate with
infinite potentials. To improve performance, the `Energy` and `Hamiltonian`
implementations should exit early after encountering the first infinity as there
is no need to spend time computing values that will not change the total.

## Forces and torques

TODO: Determine how to implement force and torque calculations on these types for MD.
