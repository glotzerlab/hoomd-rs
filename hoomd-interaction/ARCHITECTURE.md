# hoomd_interaction

The `hoomd_interaction` crate defines traits for and implements the most common
particle-particle, particle-field, and other types of interactions.

## Design goals:

* Require minimal information to compute interactions in base types.
* Provide extended traits that operate on sites and microstates.
* The API should be equally usable by HOOMD internals and by other libraries.
* Users should be able to implement custom types to evaluate interactions.
* Allow customization at all levels.
* Batteries are **NOT** included. `hoomd_interaction` implements only those
  interactions that are in extremely wide usage. Users are expected to provide
  custom interactions in most cases.

## Base types

### Traits

Not all interactions are differentiable, so `Energy` and `Force` are computed by
separate traits. Similarly, anisotropic interactions require more information to
compute than isotropic ones. This leads to a number of possible traits that each
interaction type can implement (or not) as appropriate:

* `pairwise::IsotropicForce`
* `pairwise::IsotropicEnergy`
* `pairwise::AnisotropicForce`
* `pairwise::AnisotropicPairwiseEnergy`

Here are some sketches of what this might look like.
```
let lj = LennardJones { epsilon: 1.0, sigma: 1.0 };
let e = lj.energy(2.0);
let f = lj.force(2.0);

let boxcar = Boxcar { epsilon: -1.0, left: 0.0, right: 2.0 };
let f = step.force(0.5); // compile error, does not implement force trait
let e = step.energy(0.5);
```

```
let hard_core = HardCore(shape_i, shape_j);
hard_core.energy(r_ij, orientation_ij)
```

```
fn get_interaction(p_i: Particle&, p_j: Particle&) {
let lj = hoomd_interaction::LennardJones((p_i.epsilon + p_j.epsilon) / 2.0, 2.0)
}
```

### Arguments

Trait methods accept the minimum number of arguments needed to compute the
relevant quantity. For example `IsotropicPairEnergy::energy` is a function of
`r` alone. Anisotropic interactions are defined in the coordinate system of
the *i* particle and take a single displacement vector `r_ij` and rotation
`orientation_ij`. Note: This is a departure from HOOMD-blue which passes
`orientation_i` and `orientation_j` separately.

### Transformations

There are many common transformations that users apply to potentials,
including shifting (`U(r) = f(r) - f(r_shift)`), expanding
(`U(r) = f(r - delta)`), XPLOR smoothing, and others. `hoomd-rs` implements
these transformations as their own type allowing users to combine
any transformation with any potential.

```
let shifted_lj = Shifted { f: LennardJones { epsilon: 1.0, sigma: 1.0 }, r_shift: 2.5 };
```

This greatly simplifies the design used in HOOMD-blue where each
transformed potential needed to be separately implemented and
documented.

### Cutoff potentials

The goal is that the base types in `hoomd_interaction` cleanly define the
functional form of the interaction. In most cases, this implies that it
evaluates potentials without any notion of a cutoff radius. The use of a cutoff
is a detail left up to the caller and is not present in `hoomd_interaction`'s
API.

In some specific cases (e.g. Weeks-Chandler-Anderson), the notion of a cutoff
is baked into the definition of the potential. `hoomd_interaction` evaluates
these potentials and forces appropriate to their definition. It is the
responsibility of the user to choose `r_cut` values that are commensurate
with the functional form of the potential they choose, whether that potential
has a built-in cutoff or not.

## Models

A model combines interactions from `hoomd_interactions` with systems from
`hoomd_microstate` to define a physical model of how those systems behave.
Models were originally intended for a separate crate, but it is much easier
to handle the orphan rule with both base interactions and models in the
same crate/module. Callers are still free to use the base type implementations
without the models.

### Hamiltonian

In physics, the **Hamiltonian** is the sum of all the energies (potential and
kinetic) that apply to a given system. It turns out to be problematic to expose
the concept of a **Hamiltonian** as a trait:

* MC simulations operate purely on potential energy. Should one include kinetic
  terms in the Hamiltonian, MC simulations will waste time computing them.
* MD simulations need to be aware of translational kinetic and rotational
  kinetic terms separately. MD simulations are also not strictly Hamiltonian as
  they can employ non-conservative forces.
* Furthermore, many MD simulations compute the kinetic terms on a subset of
  the system.

Therefore, `hoomd_interaction` will not provide a `Hamiltonian` trait. Instead,
it provides several traits that allow users to implement types that compute the
various terms of the Hamiltonian. The concept of a Hamiltonian will appear, but
in more focused cases. MC trial moves will take a `hamiltonian` argument that
implements `TotalEnergy`. It will appear more implicitly in MD.

### TotalEnergy

The **energy** of the system defines how the bodies of a microstate interact.
The `TotalEnergy` trait describes a type that computes the energy of a given
microstate. The crate also implements commonly used energies, such as external
potentials and cutoff pair potentials. Users can write custom types that
implement the `TotalEnergy` trait.

Tuples of types that implement `TotalEnergy` (and similar traits) sum the
contributions from all elements of the tuple.

### Forces and torques

TODO: Determine how to implement force and torque calculations on these types
for MD.

### Infinite energies

The `TotalEnergy` trait must serve the needs of both MD and MC simulations.
MD simulations will call `total_energy` only for logging and rely mainly
on `force` and `torque` (see below). MC simulations use `total_energy` directly
in the evaluation of trial move acceptance. For finite potentials, there is
no practical difference. But MC simulations can (and often do) operate with
infinite potentials. To improve performance, the `TotalEnergy` implementations
should exit early after encountering the first infinity as there is no need to
spend time computing values that will not change the total.

### Layers of abstraction

`hoomd_interaction` breaks each energy/force computation up into multiple layers.
For example, the external potential module defines the `SiteEnergy` trait
that computes the energy of a single site. The `External` type wraps a
generic `SiteEnergy` type, implements `TotalEnergy` over the microstate, and
reimplements `SiteEnergy`. This way, one variable (e.g. `linear`) implements
multiple methods that both the MD and MC engines can use.
* `site_energy(&self, site_properties: &S)` evaluates the energy of a given site.
* `total_energy(&self, microstate: &M)` evaluates the total
  energy of the microstate.
* ... and later forces

The wrapper type is needed to constrain the general implementation to a single
type. Rust does not allow multiple blanket implementations (even if they do not
overlap). In other potentials, such as cutoff pair energies, the wrapper type
will also hold parameters (e.g. r_cut).

In this way, a user can customize force and energy computations at any
level. To use a new functional form for an external potential, the user could
implement `SiteEnergy` for their type. To apply site-specific potentials
(e.g. type-dependent parameters), they could implement `SiteEnergy` with the
appropriate logic (possibly using types internally). Lastly, the user could
implement an entirely custom implementation of `TotalEnergy`.

Similarly, `SitePairEnergy` describes a type that can compute an energy between
a pair of sites (as a function of their properties). The wrappers `Isotropic`
and `Anisotropic` implement this for the implement `SitePairEnergy` for the
potentials described above. `PairwiseCutoff` implements `TotalEnergy` for pairwise
interactions that are cutoff to 0 at a given distance `r_cut`.

## Type dependent parameters

TODO: Sketch out how enums and `SiteEnergy`/`SitePairEnergy` work with match expressions.

## Cutoff potentials, spatial data structures, and periodic boundary conditions

`PairwiseCutoff` needs to find all sites (including ghost sites with periodic
boundaries) that are close to a given point in space. The `r_cut` value
thus needs to be consistent between 3 different structs, the boundary,
the spatial data structure, and `PairwiseCutoff` itself.

At this point, the best `hoomd-rs` can do is check for this consistency at
runtime and panic when it is not met. Perhaps a cleaner solution will present
itself.
