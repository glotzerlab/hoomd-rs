# hoomd_interaction

The `hoomd_interaction` crate defines traits for and implements the most common
particle-particle, particle-field, and other types of interactions.

## Design goals:

* Require minimal information to compute interactions.
* The API should be equally usable by HOOMD internals and by users.
* Users should be able to implement custom types to evaluate interactions.
* When possible, provide blanket implementations on Fn (such as for isotropic
  pairwise energies) allow customization with a minimal amount of code.
* Batteries are **NOT** included. Implement only those interactions that
  are in extremely wide usage. Users are expected to provide custom
  interactions in most cases.
* 

## Traits

Not all interactions are differentiable, so `Energy` and `Force` are computed by
separate traits. Similarly, anisotropic interactions require more information to
compute than isotropic ones. This leads to a number of possible traits that each
interaction type can implement (or not) as appropriate:

* `IsotropicPairwiseForce`
* `IsotropicPairwiseEnergy`
* `AnisotropicPairwiseForceTorque`
* `AnisotropicPairwiseEnergy`
* `IsotropicExternalForce`
* `IsotropicExternalEnergy`
* `AnisotropicExternalForceTorque`
* `AnisotropicExternalEnergy`

Questions: Is this too many traits? The alternative is to make everything anisotropic,
but then callers need to pass in needless `identity()` rotations and ignore torques
for isotropic potentials.

## Arguments

Trait methods accept the minimum number of arguments needed to compute the
relevant quantity. For example `IsotropicPairEnergy::energy` is a function of
`r` alone. Anisotropic interactions are defined in the coordinate system of
the *i* particle and take a single displacement vector `r_ij` and rotation
`orientation_ij`. Note: This is a departure from HOOMD-blue which passes
`orientation_i` and `orientation_j` separately.
