// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! Pairwise interactions.
 */

use hoomd_vector::{Rotate, Vector};

pub mod angular_mask;
#[doc(inline)]
pub use angular_mask::AngularMask;

mod boxcar;
pub use boxcar::Boxcar;

mod lennard_jones;
pub use lennard_jones::LennardJones;

mod shifted;
pub use shifted::Shifted;

mod xplor;
pub use xplor::Xplor;

mod weeks_chandler_anderson;
pub use weeks_chandler_anderson::WeeksChandlerAnderson;

mod chimes_cheby2b;
pub use chimes_cheby2b::Chimes2b;

mod isotropic;
pub use isotropic::Isotropic;


/** Computes pairwise energies between point particles.

An isotropic pairwise energy is function only of the distances between the
particles: $`U(r)`$

Implement [`IsotropicEnergy`] on a custom type or use one of the provided
potentials in [`pairwise`](crate::pairwise) in MD or MC simulations.
Use an [`IsotropicEnergy`] in combination with [`Isotropic`] and
[`CutoffPair`](crate::CutoffPair).

# Examples

Set a custom potential using a closure:
```
use hoomd_interaction::pairwise::IsotropicEnergy;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let a = 2.0;
let custom = |r: f64| a / (r.powi(12));

let energy = custom.energy(1.0);
assert_eq!(energy, 2.0);
# Ok(())
# }
```

Implement a custom potential via a type:
```
use hoomd_interaction::pairwise::IsotropicEnergy;

struct Custom {
    a: f64,
}

impl IsotropicEnergy for Custom {
    fn energy(&self, r: f64) -> f64 {
        self.a / r.powi(12)
    }
}

let custom = Custom { a: 2.0 };

let energy = custom.energy(1.0);
assert_eq!(energy, 2.0);
```
*/
pub trait IsotropicEnergy {
    /** Compute the pairwise energy between two point particles.
    ```math
    U(r)
    ```
    */
    #[must_use]
    fn energy(&self, r: f64) -> f64;
}

impl<F> IsotropicEnergy for F
where
    F: Fn(f64) -> f64,
{
    #[inline]
    fn energy(&self, r: f64) -> f64 {
        self(r)
    }
}

/** Computes pairwise forces between point particles.

An isotropic pairwise force is function only of the distances between the
particles and acts along the vector separating the particles.

Implement [`IsotropicForce`] on a custom type or use one of the provided
forces in [`pairwise`](crate::pairwise) in MD simulations.
*/
pub trait IsotropicForce {
    /** Compute the radial component of the pairwise force between two point
    particles.

    The direction of the force is along the unit vector between the two
    particles.

    When the force is associated with a potential energy [`IsotropicEnergy`],
    it must follow:
    ```math
    -\frac{\mathrm{d} U}{\mathrm{d} r}
    ```
    */
    #[must_use]
    fn force(&self, r: f64) -> f64;
}

/** Computes pairwise energies between oriented particles.

An anisotropic pairwise energy is function of the relative position and
orientation of the *j* particle in *i's* reference frame:
```math
U(\vec{r}_{ij}, \mathbf{o}_{ij})
```

Implement [`AnisotropicEnergy`] on a custom type or use one of the provided
potentials in [`pairwise`](crate::pairwise) in MD or MC simulations.
*/
pub trait AnisotropicEnergy<V: Vector, R: Rotate<V>> {
    /** Compute the pairwise energy between two oriented particles.
    ```math
    U(\vec{r}_{ij}, \mathbf{o}_{ij})
    ```
    */
    #[must_use]
    fn energy(&self, r_ij: &V, o_ij: &R) -> f64;
}

// TODO: determine how to express the torque return type in a general way. Possibly use
// an associated type of Rotation.
// pub trait AnisotropicForce<V: Vector, R: Rotation+Rotate<V>> {
//     /** Compute the pairwise force and torque between two oriented particles.
//     TODO: math expression.
//     */
//     #[must_use]
//     fn energy(&self, r_ij: &V, o_ij: &R) -> f64;
// }

// TODO: Implement Harmonic and HarmonicRepulsion based on that (Harmonic cut off at r_0)
// TODO: Implement Expanded as an adapter (like shifted)
