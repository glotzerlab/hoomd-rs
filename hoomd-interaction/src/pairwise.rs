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

mod isotropic;
pub use isotropic::Isotropic;

/** Computes pairwise energies between point particles.

An isotropic pairwise energy is function only of the distances between the
particles.
<!-- U(r) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo></mrow></math>

Implement [`IsotropicEnergy`] on a custom type or use one of the provided
potentials in [`pairwise`](crate::pairwise) in MD or MC simulations.
*/
pub trait IsotropicEnergy {
    /** Compute the pairwise energy between two point particles.
    <!-- U(r) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo></mrow></math>
    */
    #[must_use]
    fn energy(&self, r: f64) -> f64;
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
    <!-- -\frac{\mathrm{d} U}{\mathrm{d} r} -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mo>−</mo><mfrac><mrow><mrow><mi mathvariant="normal">d</mi></mrow><mi>U</mi></mrow><mrow><mrow><mi mathvariant="normal">d</mi></mrow><mi>r</mi></mrow></mfrac></mrow></math>
    */
    #[must_use]
    fn force(&self, r: f64) -> f64;
}

/** Computes pairwise energies between oriented particles.

An anisotropic pairwise energy is function of the relative position and
orientation of the *j* particle in *i's* reference frame:
<!-- U(\vec{r}_{ij}, \mathbf{o}_{ij}) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo separator="true">,</mo><msub><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo></mrow></math>

Implement [`AnisotropicEnergy`] on a custom type or use one of the provided
potentials in [`pairwise`](crate::pairwise) in MD or MC simulations.
*/
pub trait AnisotropicEnergy<V: Vector, R: Rotate<V>> {
    /** Compute the pairwise energy between two oriented particles.
    <!-- U(\vec{r}_{ij}, \mathbf{o}_{ij}) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><msub><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo separator="true">,</mo><msub><mi>𝐨</mi><mrow><mi>i</mi><mi>j</mi></mrow></msub><mo form="postfix" stretchy="false">)</mo></mrow></math>    */
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

// TODO: Implement Xplor smoothing
// TODO: Implement Harmonic and HarmonicRepulsion based on that (Harmonic cut off at r_0)
// TODO: Implement Expanded as an adapter (like shifted)
// TODO: Consider implementing IsotropicEnergy for Fn(f64) -> f64 to allow the user to directly
//       use a closure in place of an IsotropicEnergy. It isn't clear how to do the same for
//       both energy and force.
