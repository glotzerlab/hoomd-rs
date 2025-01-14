# hoomd_interaction

The `hoomd_interaction` crate defines traits for and implements the most common
particle-particle, particle-field, and other types of interactions.

## Design goals:

* Require minimal information to compute interactions.
* The API should be equally usable by HOOMD internals and by users.
* Users should be able to implement custom types to evaluate interactions.
* When possible, provide blanket implementations on Fn (such as for isotropic
  pairwise energies) allow customization with a minimal amount of code.
* Batteries are **NOT** included. `hoomd_interaction` implements only those
  interactions that are in extremely wide usage. Users are expected to provide
  custom interactions in most cases.

## Non-goals

`hoomd_interaction` works directly with interaction parameters, separation
vectors (or their magnitude) and orientations only. By design, it does not
depend on particle data structures directly. The `hoomd_md` and `hoomd_mc`
crates introduce new types and traits that interface particles in the
microstate with interactions in `hoomd_interaction` - likely through
the construction of a `hoomd_interaction` type given a particle (or a pair
of particles).

## Traits

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
let lj = LennardJones::new(1.0, 1.0);
let e = lj.energy(2.0);
let f = lj.force(2.0);

let step = Step::new(1.0);
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

## Arguments

Trait methods accept the minimum number of arguments needed to compute the
relevant quantity. For example `IsotropicPairEnergy::energy` is a function of
`r` alone. Anisotropic interactions are defined in the coordinate system of
the *i* particle and take a single displacement vector `r_ij` and rotation
`orientation_ij`. Note: This is a departure from HOOMD-blue which passes
`orientation_i` and `orientation_j` separately.
