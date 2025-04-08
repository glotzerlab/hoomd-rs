# hoomd_model

The `hoomd_model` crate combines interactions from `hoomd_interactions` with
systems from `hoomd_microstate` and to define physical models of how those
systems behave.

## Hamiltonian

In physics, the **Hamiltonian** is the sum of all the energies (potential and kinetic) that apply to a given system.
It turns out to be problematic to expose the concept of a **Hamiltonian** as a trait in `hoomd_model`:

* MC simulations operate purely on potential energy. Should one include kinetic terms
  in the Hamiltonian, MC simulations will waste time computing them.
* MD simulations need to be aware of translational kinetic and rotational kinetic terms
  separately. MD simulations are also not strictly Hamiltonian as they can be run with
  non-conservative forces.
* Furthermore, many MD simulations compute the kinetic terms on a subset of the
  system.

Therefore, `hoomd_model` will not provide a `Hamiltonian` trait. Instead, it provides
several traits that allow users to implement types that compute the various
terms of the Hamiltonian. The concept of a Hamiltonian will appear, but in more
focused cases. MC trial moves will take a `hamiltonian` argument that implements
`Energy`. It will appear more implicitly in MD.

## Energy

The **energy** of the system defines how the bodies of a microstate interact.
`hoomd_model` describes a `Energy` trait that computes the energy of a given
microstate. The crate also implements commonly used energies, such as external
potentials and cutoff pair potentials. Users can write custom types that
implement the `Energy` trait.

TODO: Provide a convenient mechanism to add multiple energies together.
One solution would be to implement `Energy` for a heterogeneous tuple
where each element implements `Energy`.

## Forces and torques

TODO: Determine how to implement force and torque calculations on these types
for MD.

## Infinite energies

The `Energy` trait must serve the needs of both MD and MC simulations. MD
simulations will call `energy` only for logging and relies mainly on `force` and
`torque` (see below). MC simulations use `energy` directly in the evaluation of
trial move acceptance. For finite potentials, there is no practical difference.
But MC simulations can (and often do) operate with infinite potentials. To
improve performance, the `Energy` implementations should exit early after
encountering the first infinity as there is no need to spend time computing
values that will not change the total.
