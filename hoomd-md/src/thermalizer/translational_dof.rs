// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
use hoomd_microstate::{
    Microstate, Transform,
    boundary::{GenerateGhosts, Wrap},
    property::{
        Mass, Momentum, NetForce, Position,
    },
};
use hoomd_vector::Cartesian;
use rand_distr::{Distribution, Normal};
use crate::thermalizer::{TranslationalThermalizer, Thermalizer};


impl<const N: usize, B, S, C> TranslationalThermalizer<N, B, S, C> for Thermalizer
where
    B: Position<Position = Cartesian<N>>
        + Momentum<Vector = Cartesian<N>>
        + NetForce<Vector = Cartesian<N>>
        + Mass
        + Transform<S>
        + Clone,
    S: Position<Position = Cartesian<N>> + Default,
    C: Wrap<B> + Wrap<S> + GenerateGhosts<S>,
{
    /// Draw random momentum from Gaussian.
    /// 
    /// The function thermalizes the system's translational montion given $`k_BT`$ by
    /// drawing random momentum from Gaussians.
    ///
    /// According to the Maxwell–Boltzmann statistics, each
    /// component of momentum $`p_i,\; i=x,y,z`$ with the mass $`m`$ distributes
    /// as a Gaussian with the probability density function
    /// with mean of 0 and the standard deviation of $`\sqrt{m k_B}`$ as:
    /// ```math
    ///    f(p_i) = \sqrt{ \frac{1}{2 \pi m k_B T} } \exp{\left( -\frac{p_i^2}{2 m k_B T} \right)}
    /// ```
    ///
    /// It is equivalent to sample the Maxwell-Boltzmann distribution $`f_\mathrm{Maxwell-Boltzmann}(p)`$,
    /// which can be obtained from the relation to the joint Gaussian
    /// $`f_\mathrm{Maxwell-Boltzmann}(p) dp = f(p_x)f(p_y)f(p_z) dp_x dp_y dp_z`$ and express it in terms of the
    /// magnitude of momentum $`p = \sqrt{(p_x^2+p_y^2+p_z^2)}`$ as:
    /// ```math
    ///    f_\mathrm{Maxwell-Boltzmann}(p) = \left[ \frac{1}{2 m \pi k_B T} \right]^\frac{3}{2} (4 \pi p^2) \exp{\left( -\frac{p^2}{2 m k_B T} \right)}
    /// ```
    fn thermalize_translation(&self, microstate: &mut Microstate<B, S, C>) {
        let mut rng = microstate.counter().make_rng();

        for body_index in 0..microstate.bodies().len() {
            // Get the important information from the body
            let mut body_properties = microstate.bodies()[body_index].item.properties.clone();

            let mass = body_properties.mass();

            let sigma_momentum = (self.kT * mass).sqrt();

            let normal = Normal::new(0.0, sigma_momentum).unwrap();

            let random_momentum: Cartesian<N> =
                Cartesian::from([(); N].map(|_| normal.sample(&mut rng)));
            *body_properties.momentum_mut() = random_momentum;

            // Update the microstate with new body properties, wrapping automatically
            microstate
                .update_body_properties(body_index, body_properties)
                .expect("Bodies and sites should remain in simulation boundary.");
        }
    }
}