// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for thermalizing or modifying the momenta.
//!
use hoomd_microstate::{Body, Microstate, Tagged};

// mod remove_com_angular_momentum;
// mod remove_com_momentum;
// mod rotational_dof;
mod thermalize_momentum;

// pub use remove_com_angular_momentum::ComAngularMomentumRemover;
// pub use remove_com_momentum::ComMomentumRemover;

/// Draw random momenta from a thermal distribution.
///
/// In the [Maxwell–Boltzmann distribution], each component of the momentum $` p_i `$
/// is normally distributed with mean 0 and variance $` \sigma^2 = m k T`$:
/// ```math
///    f(p_i) = \frac{1}{\sqrt{2 \pi m k T}} \exp{\left( -\frac{p_i^2}{2 m k T} \right)}
/// ```
///
/// Set a new random momentum for every body in the microstate with [`thermalize_momentum`].
/// Call [`thermalize_momentum_with_filter`] to assign new momenta to selected bodies.
///
/// `ThermalizeMomentum` gives the system's center of mass a non-zero momentum.
/// TODO: reference momentum removal method.
///
/// [Maxwell–Boltzmann distribution]: https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
/// [`thermalize_momentum`]: Self::thermalize_momentum
/// [`thermalize_momentum_with_filter`]: Self::thermalize_momentum_with_filter
///
/// # Example
///
/// ```
/// use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// use hoomd_vector::Cartesian;
/// use hoomd_md::ThermalizeMomentum;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .bodies([
///         Body { properties: DynamicPoint {
///           position: Cartesian::from([1.0, 2.0]),
///           ..Default::default()
///           },
///           sites: vec![Point::default()],
///           },
///         Body { properties: DynamicPoint {
///           position: Cartesian::from([-2.0, 3.0]),
///           ..Default::default()
///           },
///           sites: vec![Point::default()],
///           },
///     ])
///     .try_build()?;
///
/// microstate.thermalize_momentum(1.5);
/// # Ok(())
/// # }
/// ```
pub trait ThermalizeMomentum<B, S, X, C> {
    /// Assign thermally distributed random momenta to all bodies.
    #[inline]
    fn thermalize_momentum(&mut self, temperature: f64) {
        self.thermalize_momentum_with_filter(temperature, |_| true);
    }

    /// Assign thermally distributed random momenta to selected bodies.
    ///
    /// Does not modify unselected bodies.
    fn thermalize_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, temperature: f64, should_thermalize: F);
}

/// Thermalize the rotational motion of [`Microstate`].
///
/// Implement [`RotationalThermalizer`] on a custom type
/// or use one of the provide method in
/// [`thermalizer`](crate::thermalizer) in MD simulations.
pub trait RotationalThermalizer<const N: usize, B, S, X, C> {
    /// Thermalize the rotational motion.
    fn thermalize_rotation(&self, microstate: &mut Microstate<B, S, X, C>);
}

/// Modify the translational momenta of [`Microstate`].
///
/// Implement [`TranslationalMomentumModifier`] on a custom type
/// or use one of the provide method in
/// [`thermalizer`](crate::thermalizer) in MD simulations.
pub trait TranslationalMomentumModifier<const N: usize, B, S, X, C> {
    /// Modify the translational momenta.
    fn modify(&self, microstate: &mut Microstate<B, S, X, C>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use approxim::assert_abs_diff_eq;
    use hoomd_microstate::{
        Body,
        property::{AngularMomentum, DynamicPoint, Momentum, DynamicOrientedPoint, Point},
    };
    use hoomd_vector::{Cartesian, Versor, Wedge};
    use rstest::*;

    // When draw n samples from gaussian of N(mu, sigma)
    // the error in the sample mean is ~ sigma * sqrt(1/n)
    // the error in the sample variance is ~ sigma^2 * sqrt(2/(n-1))
    // here I use 4 * sigma * sqrt(1/n) and 4 * sigma^2 * sqrt(2/(n-1)) as the testing tolerance
    // which should cover 99.99% cases.
    // sqrt(2/9999) ~ 0.01414284
    const N_BODIES: usize = 10000;
    const EPSILON_MEAN_SCALE: f64 = 4.0 * 0.01;
    const EPSILON_VARIANCE_SCALE: f64 = 4.0 * 0.014_142_840;

    // #[rstest]
    // fn test_init_zero_mom() {
    //     // Instantiation
    //     let _ = ComMomentumRemover {};
    // }

    // #[rstest]
    // fn test_init_zero_angularmom() {
    //     // Instantiation
    //     let _ = ComAngularMomentumRemover {};
    // }

    mod momentum {
        use super::*;

        // #[rstest]
        // fn test_mom_removal(
        //     #[values(1.0, 2.0)] mass1: f64,
        //     #[values([1.0, -1.0, 0.0], [-1.0, 1.0, 1.0])] mom1: [f64; 3],
        //     #[values(1.0, 0.5)] mass2: f64,
        //     #[values([-1.0, 0.0, 1.0], [0.5, 1.5, 3.0])] mom2: [f64; 3],
        // ) -> anyhow::Result<()> {
        //     two_particles_mom_removal(mass1, mom1, mass2, mom2)
        // }

        // #[rstest]
        // fn test_angmom_removal(
        //     #[values([1.0, 0.0, 0.0])] pos1: [f64; 3],
        //     #[values(1.0, 2.0)] mass1: f64,
        //     #[values([1.0, -1.0, 0.0], [-1.0, 1.0, 1.0])] mom1: [f64; 3],
        //     #[values([-1.0, 0.0, 0.0])] pos2: [f64; 3],
        //     #[values(1.0, 0.5)] mass2: f64,
        //     #[values([-1.0, 0.0, 1.0], [0.5, 1.5, 3.0])] mom2: [f64; 3],
        // ) -> anyhow::Result<()> {
        //     two_particles_stop_rotation(pos1, mass1, mom1, pos2, mass2, mom2)
        // }

        fn create_point_body_3d(
            position: Cartesian<3>,
            mass: f64,
            momentum: Cartesian<3>,
        ) -> Body<DynamicPoint<Cartesian<3>>, Point<Cartesian<3>>> {
            Body {
                properties: DynamicPoint {
                    position,
                    momentum,
                    net_force: Cartesian::default(),
                    mass,
                },
                sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
            }
        }

        #[rstest]
        fn test_distribution(#[values(0.5, 1.0, 2.0)] mass: f64,
            #[values(0.5, 1.5)] temperature: f64,
            #[values(42, 123, 999)] seed: u32) -> anyhow::Result<()> {
            let mut microstate = Microstate::builder().seed(seed).try_build()?;
            let expected_variance = temperature * mass;

            for _ in 0..N_BODIES {
                microstate
                    .add_body(create_point_body_3d(
                        Cartesian::default(),
                        mass,
                        Cartesian::default(),
                    ))
                    .expect("body should be inside boundary");
            }

            microstate.thermalize_momentum(temperature);

            let momenta: Vec<[f64; 3]> = microstate
                .bodies()
                .iter()
                .map(|b| b.item.properties.momentum().coordinates)
                .collect();

            for dim in 0..3 {
                let components: Vec<f64> = momenta.iter().map(|m| m[dim]).collect();
                let mean = components.iter().sum::<f64>() / N_BODIES as f64;
                let variance = components.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                    / (N_BODIES - 1) as f64;

                assert_abs_diff_eq!(mean, 0.0, epsilon = expected_variance.sqrt() * EPSILON_MEAN_SCALE);
                assert_abs_diff_eq!(variance, expected_variance, epsilon = expected_variance * EPSILON_VARIANCE_SCALE);
            }

            Ok(())
        }

        #[rstest]
        fn test_filter() -> anyhow::Result<()> {
            let temperature = 1.5;
            let mass = 2.0;
            let n_bodies = N_BODIES*2;
            
            let mut microstate = Microstate::builder().try_build()?;
            let expected_variance = temperature * mass;

            for _ in 0..n_bodies {
                microstate
                    .add_body(create_point_body_3d(
                        Cartesian::default(),
                        mass,
                        Cartesian::default(),
                    ))
                    .expect("body should be inside boundary");
            }

            microstate.thermalize_momentum_with_filter(temperature, |b| b.tag < n_bodies / 2);

            let momenta: Vec<[f64; 3]> = microstate
                .bodies()
                .iter()
                .take(N_BODIES)
                .map(|b| b.item.properties.momentum().coordinates)
                .collect();

            for dim in 0..3 {
                let components: Vec<f64> = momenta.iter().map(|m| m[dim]).collect();
                let mean = components.iter().sum::<f64>() / N_BODIES as f64;
                let variance = components.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                    / (N_BODIES - 1) as f64;

                assert_abs_diff_eq!(mean, 0.0, epsilon = expected_variance.sqrt() * EPSILON_MEAN_SCALE);
                assert_abs_diff_eq!(variance, expected_variance, epsilon = expected_variance * EPSILON_VARIANCE_SCALE);
            }

            for i in n_bodies/2..n_bodies {
                check!(microstate.bodies()[i].item.properties.momentum == Cartesian::from([0.0, 0.0, 0.0]));
            }

            Ok(())
        }

        // fn two_particles_mom_removal(
        //     mass1: f64,
        //     mom1: [f64; 3],
        //     mass2: f64,
        //     mom2: [f64; 3],
        // ) -> anyhow::Result<()> {
        //     // Use two point body with arbitrary mass and momenta
        //     // to test the com momentum remover
        //     let mom1 = Cartesian::from(mom1);
        //     let mom2 = Cartesian::from(mom2);
        //     let com_velocity = (mom1 + mom2) / (mass1 + mass2);

        //     let mut microstate = Microstate::builder().try_build()?;
        //     let com_remover = ComMomentumRemover {};
        //     microstate
        //         .add_body(create_point_body_3d(Cartesian::default(), mass1, mom1))
        //         .expect("body should be inside boundary");
        //     microstate
        //         .add_body(create_point_body_3d(Cartesian::default(), mass2, mom2))
        //         .expect("body should be inside boundary");

        //     com_remover.modify(&mut microstate);

        //     let modified_mom1 = microstate.bodies()[0].item.properties.momentum;
        //     let modified_mom2 = microstate.bodies()[1].item.properties.momentum;

        //     let expected_mom1 = mom1 - com_velocity * mass1;
        //     let expected_mom2 = mom2 - com_velocity * mass2;

        //     assert_abs_diff_eq!(
        //         modified_mom1 + modified_mom2,
        //         Cartesian::default(),
        //         epsilon = 1e-15
        //     );
        //     assert_eq!(modified_mom1, expected_mom1);
        //     assert_eq!(modified_mom2, expected_mom2);
        //     Ok(())
        // }

        // fn two_particles_stop_rotation(
        //     pos1: [f64; 3],
        //     mass1: f64,
        //     mom1: [f64; 3],
        //     pos2: [f64; 3],
        //     mass2: f64,
        //     mom2: [f64; 3],
        // ) -> anyhow::Result<()> {
        //     // Use two point body with arbitrary positions, masses and momenta
        //     // to test the com angular momentum remover
        //     let pos1 = Cartesian::from(pos1);
        //     let mom1 = Cartesian::from(mom1);
        //     let pos2 = Cartesian::from(pos2);
        //     let mom2 = Cartesian::from(mom2);
        //     let com_pos = (pos1 * mass1 + pos2 * mass2) / (mass1 + mass2);

        //     let mut microstate = Microstate::builder().try_build()?;
        //     let com_remover = ComAngularMomentumRemover {};
        //     microstate
        //         .add_body(create_point_body_3d(pos1, mass1, mom1))
        //         .expect("body should be inside boundary");
        //     microstate
        //         .add_body(create_point_body_3d(pos2, mass2, mom2))
        //         .expect("body should be inside boundary");

        //     com_remover.modify(&mut microstate);

        //     let modified_mom1 = microstate.bodies()[0].item.properties.momentum;
        //     let modified_mom2 = microstate.bodies()[1].item.properties.momentum;

        //     let modified_angmom1 = (pos1 - com_pos).wedge(&modified_mom1);
        //     let modified_angmom2 = (pos2 - com_pos).wedge(&modified_mom2);

        //     assert_abs_diff_eq!(
        //         modified_angmom1 + modified_angmom2,
        //         Cartesian::default(),
        //         epsilon = 1e-15
        //     );
        //     Ok(())
        // }
    }

    // mod rotational_dof {
    //     use super::*;

    //     #[rstest]
    //     fn test_mom_distribution(
    //         #[values([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [4.0, 2.0, 0.5])]
    //         inertia: [f64; 3],
    //         #[values(0.5, 1.5)] kt: f64,
    //         #[values(42, 123, 999)] seed: u32,
    //     ) -> anyhow::Result<()> {
    //         test_distribution(inertia, kt, seed)
    //     }

    //     fn create_body_3d(
    //         inertia: Cartesian<3>,
    //         angmom: Cartesian<3>,
    //     ) -> Body<DynamicOrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>> {
    //         Body {
    //             properties: DynamicOrientedPoint {
    //                 position: Cartesian::<3>::default(),
    //                 orientation: Versor::default(),
    //                 momentum: Cartesian::<3>::default(),
    //                 net_force: Cartesian::<3>::default(),
    //                 moment_of_inertia: inertia,
    //                 angular_momentum: angmom,
    //                 net_torque: Cartesian::<3>::default(),
    //                 mass: 1.0,
    //             },
    //             sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
    //         }
    //     }

    //     fn test_distribution(inertia: [f64; 3], kt: f64, seed: u32) -> anyhow::Result<()> {
    //         let mut microstate = Microstate::builder().seed(seed).try_build()?;
    //         let thermalizer = Thermalizer { kT: kt };
    //         let expected_var = Cartesian::from(inertia) * kt;

    //         for _ in 0..N_PARTICLES {
    //             microstate
    //                 .add_body(create_body_3d(
    //                     Cartesian::from(inertia),
    //                     Cartesian::default(),
    //                 ))
    //                 .expect("body should be inside boundary");
    //         }

    //         thermalizer.thermalize_rotation(&mut microstate);

    //         // Collect momentum into a nested structure for easier iteration
    //         let momenta: Vec<[f64; 3]> = microstate
    //             .bodies()
    //             .iter()
    //             .map(|b| b.item.properties.angular_momentum().coordinates)
    //             .collect();

    //         // Check X, Y, and Z dimensions
    //         for dim in 0..3 {
    //             let components: Vec<f64> = momenta.iter().map(|m| m[dim]).collect();

    //             // 1. Calculate Mean
    //             let mean = components.iter().sum::<f64>() / N_PARTICLES as f64;

    //             // 2. Calculate Variance
    //             let variance = components.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
    //                 / (N_PARTICLES - 1) as f64;

    //             assert_abs_diff_eq!(mean, 0.0, epsilon = expected_var[dim].sqrt() * EPSILON_MEAN);
    //             assert_abs_diff_eq!(
    //                 variance,
    //                 expected_var[dim],
    //                 epsilon = expected_var[dim] * EPSILON_VARIANCE
    //             );
    //         }

    //         Ok(())
    //     }
    // }
}
