// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Apply the Metropolis Monte Carlo simulation method to systems of particles.

TODO: Expand documentation.
 */

use rand::Rng;
use hoomd_microstate::Tagged;

mod sweep;
mod translate;
mod external;

pub use translate::Translate;
pub use sweep::Sweep;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Count {
    pub accepted: usize,
    pub rejected: usize,
}

/** Propose a trial microstate, evaluate the change in energy and accept or reject accordingly.

*/
pub trait Trial<M> {
    type Count;

    fn apply(&self, microstate: &mut M) -> Count;
}

/** Propose a new configuration for given body properties.
*/
pub trait LocalTrial<B> {
    #[must_use]
    fn propose<R: Rng>(&self, rng: &mut R, body_properties: B) -> B;
}

/** Compute the change in energy after making changes to a single body in the microstate.
*/
pub trait DeltaEnergyOne {
    #[must_use]
    fn delta_energy_one<M, B>(&self, microstate: &M, new_body: &Tagged<B>) -> f64;
}

/** Set the energy of any system to 0.

[`Zero`] is useful for trivial code examples that demonstrate MC simulations.
It returns 0 for all delta energies.
*/
pub struct Zero;

impl DeltaEnergyOne for Zero {
    #[inline]
    fn delta_energy_one<M, B>(&self, _microstate: &M, _new_body: &Tagged<B>) -> f64 {
        0.0
    }
}
