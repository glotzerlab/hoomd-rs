// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use crate::thermalizer::TranslationalMomentumModifier;
use hoomd_microstate::{
    Microstate, SiteKey, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{Mass, Momentum, NetForce, Position},
};
use hoomd_spatial::PointUpdate;
use hoomd_vector::Cartesian;

/// Remove the center-of-mass momentum.
pub struct ComMomentumRemover;

impl<const N: usize, B, S, X, C> TranslationalMomentumModifier<N, B, S, X, C> for ComMomentumRemover
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Momentum = Cartesian<N>>
        + NetForce<NetForce = Cartesian<N>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    X: PointUpdate<Cartesian<N>, SiteKey>,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Remove the center-of-mass momentum.
    ///
    /// The function modifies the system's momentum by zeroing the
    /// center-of-mass momentum as
    /// ```math
    /// \mathbf{p}_{k,\; \mathrm{new}} = \mathbf{p}_{k,\; \mathrm{old}} - \frac{\sum_i \mathbf{p}_{i,\; \mathrm{old}}}{\sum_i m_i} m_k
    /// ```
    /// where $`k`$ is the index of each body in a system, $`\mathbf{p}_{k,\; \mathrm{old}}`$
    /// and $`\mathbf{p}_{k,\; \mathrm{new}}`$ are the momentum vector before and after
    /// modification of $`k`$-th body, and $`m_k`$ is the mass of $`k`$-th body.
    fn modify(&self, microstate: &mut Microstate<B, S, X, C>) {
        let mut total_mass = 0.0;
        let mut total_momentum = Cartesian::<N>::default();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();
            let momentum = body_properties.momentum();
            total_mass += mass;
            total_momentum += *momentum;
        }

        let center_of_mass_velocity = total_momentum / total_mass;
        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();
            let mass = body_properties.mass();
            let mut momentum = body_properties.momentum().clone();

            momentum -= center_of_mass_velocity * mass;

            *body_properties.momentum_mut() = momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}
