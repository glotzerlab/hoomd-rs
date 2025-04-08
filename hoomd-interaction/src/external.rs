// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/*! External interactions.
 */

mod linear;
pub use linear::Linear;

/** Computes external energies on point particles.

An isotropic external energy is function only of the position of the particle.
<!-- U(\vec{r}) -->
<math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mover><mi>r</mi><mo stretchy="false" style="transform:scale(0.75) translate(10%, 30%);">→</mo></mover><mo form="postfix" stretchy="false">)</mo></mrow></math>

Implement [`IsotropicEnergy`] on a custom type or use one of the built-in
potentials in [`external`](crate::external) in MD or MC simulations.
*/
pub trait IsotropicEnergy<V> {
    /** The energy of a point particle in an external field.
    <!-- U(r) -->
    <math display="block" class="tml-display" style="display:block math;"><mrow><mi>U</mi><mo form="prefix" stretchy="false">(</mo><mi>r</mi><mo form="postfix" stretchy="false">)</mo></mrow></math>
    */
    #[must_use]
    fn energy(&self, r: &V) -> f64;
}

// TODO: Isotropic Force
