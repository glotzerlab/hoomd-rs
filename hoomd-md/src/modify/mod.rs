// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods for thermalizing and zeroing the momenta.

use hoomd_microstate::{Body, Tagged};

mod thermalize_angular_nomentum;
mod thermalize_momentum;
mod zero_center_angular_momentum;
mod zero_center_momentum;

/// Draw random momenta from a thermal distribution.
///
/// In the [Maxwell–Boltzmann distribution], each component of the momentum $` p_i `$
/// is normally distributed with mean 0 and variance $` \sigma^2 = m k T`$:
/// ```math
///    f(p_i) = \frac{1}{\sqrt{2 \pi m k T}} \exp{\left( -\frac{p_i^2}{2 m k T} \right)}
/// ```
///
/// [`ThermalizeMomentum`] gives the system's center of mass a non-zero momentum.
/// Use [`ZeroCenterMomentum`] to remove it.
///
/// [Maxwell–Boltzmann distribution]: https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
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
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([1.0, 2.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([-2.0, 3.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///     ])
///     .try_build()?;
///
/// microstate.thermalize_momentum(1.5);
/// # Ok(())
/// # }
/// ```
pub trait ThermalizeMomentum<B, S> {
    /// Assign thermally distributed random momenta to all bodies in the microstate.
    #[inline]
    fn thermalize_momentum(&mut self, temperature: f64) {
        self.thermalize_momentum_with_filter(temperature, |_| true);
    }

    /// Assign thermally distributed random momenta to a subset of the bodies in the microstate.
    fn thermalize_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, temperature: f64, should_thermalize_body: F);
}

/// Remove translational motion from the system's center of mass.
///
/// [`ZeroCenterMomentum`] subtracts the average momentum from every body's momentum:
/// ```math
/// \vec{p}_{i,\mathrm{new}} = \vec{p}_{i,\mathrm{old}} - \langle \vec{p}_\mathrm{old} \rangle
/// ```
///
/// # Example
///
/// ```
/// use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// use hoomd_vector::Cartesian;
/// use hoomd_md::ZeroCenterMomentum;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .bodies([
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([1.0, 2.0]),
///           momentum: Cartesian::from([-2.0, 4.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([-2.0, 3.0]),
///           momentum: Cartesian::from([3.0, -6.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///     ])
///     .try_build()?;
///
/// microstate.zero_center_momentum();
/// # Ok(())
/// # }
/// ```
pub trait ZeroCenterMomentum<B, S> {
    /// Subtract the average momentum from every body's momentum.
    #[inline]
    fn zero_center_momentum(&mut self) {
        self.zero_center_momentum_with_filter(|_| true);
    }

    /// Subtract the average momentum from each selected body's momentum.
    fn zero_center_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, should_zero_body: F);
}

/// Remove angular motion about the system's center of mass.
///
/// [`ZeroCenterAngularMomentum`] adjusts the translational momentum of every body to zero
/// out the total angular momentum of the system about the center of mass (ignoring
/// periodic boundary conditions).
///
/// # 2D
///
/// In 2D, [`ZeroCenterAngularMomentum`] applies:
/// ```math
/// \vec{p}_{i,\mathrm{new}} = \vec{p}_{i,\mathrm{old}} - \left( [-r_{ci}^{y}, r_{ci}^{x}] \right) \frac{L_c}{I_c} m_i
/// ```
/// where $`i`$ is the index of each body in a system, $`L_c`$ is the
/// angular momentum about the center of mass, $`I_c`$ is the moment of
/// inertia about the center of mass, and $`\vec{r}_{ci}`$ is the position of body *i*
/// relative to the center of mass.
///
/// # 3D
///
/// In 3D, [`ZeroCenterAngularMomentum`] applies:
/// ```math
/// \vec{p}_{i,\mathrm{new}} = \vec{p}_{i,\mathrm{old}} - \left( \vec{\omega}_c \times \vec{r}_{ci} \right) m_k
/// ```
/// where $`i`$ is the index of each body in a system,
/// $`\vec{\omega}_c`$ is angular velocity about the center of mass,
/// $`\vec{r}_{ci}`$ is the position of body *i* relative to the center of mass.
///
/// $`\vec{\omega}_c`$ is obtained by solving the following linear system:
/// ```math
/// \mathbf{I}_c \vec{\omega}_c = \vec{L}_c
/// ```
/// where $`\mathbf{I}_c`$ moment of inertia about the center of mass,
/// and $`\vec{L}_c`$ is the angular momentum about the center of mass.
///
/// # Example
///
/// ```
/// use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// use hoomd_vector::Cartesian;
/// use hoomd_md::ZeroCenterAngularMomentum;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .bodies([
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([1.0, 2.0]),
///           momentum: Cartesian::from([-2.0, 4.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///         Body::single_site(DynamicPoint {
///           position: Cartesian::from([-2.0, 3.0]),
///           momentum: Cartesian::from([3.0, -6.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///     ])
///     .try_build()?;
///
/// microstate.zero_center_angular_momentum();
/// # Ok(())
/// # }
/// ```
pub trait ZeroCenterAngularMomentum<B, S> {
    /// Subtract the average momentum from every body's momentum.
    #[inline]
    fn zero_center_angular_momentum(&mut self) {
        self.zero_center_angular_momentum_with_filter(|_| true);
    }

    /// Subtract the average momentum from each selected body's momentum.
    fn zero_center_angular_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, should_zero_body: F);
}

/// Draw random angular momenta from a thermal distribution.
///
/// In the [Maxwell–Boltzmann distribution], each component of the angular momentum $` L_i `$
/// (aligned to the principal axes) is normally distributed with mean 0 and variance
/// $` \sigma^2 = I_i k T`$:
/// ```math
///    f(L_i) = \frac{1}{\sqrt{2 \pi I_i k T}} \exp{\left( -\frac{L_i^2}{2 I_i k T} \right)}
/// ```
///
/// [Maxwell–Boltzmann distribution]: https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution
///
/// # Example
///
/// ```
/// use hoomd_microstate::{Body, Microstate, property::{DynamicOrientedPoint, Point}};
/// use hoomd_vector::{Angle, Cartesian};
/// use hoomd_md::ThermalizeAngularMomentum;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::builder()
///     .bodies([
///         Body::single_site(DynamicOrientedPoint {
///           position: Cartesian::from([1.0, 2.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///         Body::single_site(DynamicOrientedPoint {
///           position: Cartesian::from([-2.0, 3.0]),
///           ..Default::default()
///           },
///           Point::default(),
///           ),
///     ])
///     .try_build()?;
///
/// microstate.thermalize_angular_momentum(1.5);
/// # Ok(())
/// # }
/// ```
pub trait ThermalizeAngularMomentum<B, S> {
    /// Assign thermally distributed random angular momenta to all bodies in the microstate.
    #[inline]
    fn thermalize_angular_momentum(&mut self, temperature: f64) {
        self.thermalize_angular_momentum_with_filter(temperature, |_| true);
    }

    /// Assign thermally distributed random angular momenta to a subset of bodies in the microstate.
    fn thermalize_angular_momentum_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&mut self, temperature: f64, should_thermalize_body: F);
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::{assert_abs_diff_eq, assert_relative_eq};
    use hoomd_microstate::{
        Body,
        Microstate,
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

    mod momentum {
        use super::*;

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
                    ))?;
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
        fn zero_center_momentum(
            #[values(1.0, 2.0)] mass_a: f64,
            #[values([1.0, -1.0, 0.0], [-1.0, 1.0, 1.0])] momentum_a: [f64; 3],
            #[values(1.0, 0.5)] mass_b: f64,
            #[values([-1.0, 0.0, 1.0], [0.5, 1.5, 3.0])] momentum_b: [f64; 3],
        ) -> anyhow::Result<()> {
            let momentum_1 = Cartesian::from(momentum_a);
            let momentum_2 = Cartesian::from(momentum_b);
            let center_momentum = momentum_1 + momentum_2;

            let mut microstate = Microstate::builder().try_build()?;
            microstate
                .add_body(create_point_body_3d(Cartesian::default(), mass_a, momentum_1))?;
            microstate
                .add_body(create_point_body_3d(Cartesian::default(), mass_b, momentum_2))?;

            microstate.zero_center_momentum();

            let modified_momentum_1 = microstate.bodies()[0].item.properties.momentum;
            let modified_momentum_2 = microstate.bodies()[1].item.properties.momentum;

            let expected_momentum_1 = momentum_1 - center_momentum / 2.0;
            let expected_momentum_2 = momentum_2 - center_momentum / 2.0;

            assert_abs_diff_eq!(
                modified_momentum_1 + modified_momentum_2,
                Cartesian::default(),
                epsilon = 1e-15
            );
            assert_relative_eq!(modified_momentum_1, expected_momentum_1);
            assert_relative_eq!(modified_momentum_2, expected_momentum_2);
            Ok(())
        }

        #[rstest]
        fn two_particles_stop_rotation(
            #[values([1.0, 0.0, 0.0])] position_a: [f64; 3],
            #[values(1.0, 2.0)] mass_a: f64,
            #[values([1.0, -1.0, 0.0], [-1.0, 1.0, 1.0])] momentum_a: [f64; 3],
            #[values([-1.0, 0.0, 0.0])] position_b: [f64; 3],
            #[values(1.0, 0.5)] mass_b: f64,
            #[values([-1.0, 0.0, 1.0], [0.5, 1.5, 3.0])] momentum_b: [f64; 3],
        ) -> anyhow::Result<()> {
            let position_a = Cartesian::from(position_a);
            let momentum_a = Cartesian::from(momentum_a);
            let position_b = Cartesian::from(position_b);
            let momentum_b = Cartesian::from(momentum_b);
            let position_center = (position_a * mass_a + position_b * mass_b) / (mass_a + mass_b);

            let mut microstate = Microstate::builder().try_build()?;
            microstate
                .add_body(create_point_body_3d(position_a, mass_a, momentum_a))?;
            microstate
                .add_body(create_point_body_3d(position_b, mass_b, momentum_b))?;

            microstate.zero_center_angular_momentum();

            let modified_momentum_a = microstate.bodies()[0].item.properties.momentum;
            let modified_momentum_b = microstate.bodies()[1].item.properties.momentum;

            let modified_angular_momentum_a = (position_a - position_center).wedge(&modified_momentum_a);
            let modified_angular_momentum_b = (position_b - position_center).wedge(&modified_momentum_b);

            assert_abs_diff_eq!(
                modified_angular_momentum_a + modified_angular_momentum_b,
                Cartesian::default(),
                epsilon = 1e-15
            );
            Ok(())
        }
    }

    mod angular_momentum {
        use super::*;

        fn create_body_3d(
            moment_of_inertia: [f64; 3],
            angular_momentum: Cartesian<3>,
        ) -> Body<DynamicOrientedPoint<Cartesian<3>, Versor>, Point<Cartesian<3>>> {
            Body {
                properties: DynamicOrientedPoint {
                    position: Cartesian::<3>::default(),
                    orientation: Versor::default(),
                    momentum: Cartesian::<3>::default(),
                    net_force: Cartesian::<3>::default(),
                    moment_of_inertia,
                    angular_momentum,
                    net_torque: Cartesian::<3>::default(),
                    mass: 1.0,
                },
                sites: vec![Point::new(Cartesian::from([0.0, 0.0, 0.0]))],
            }
        }

        #[rstest]
        fn test_distribution(#[values([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [4.0, 2.0, 0.5])] inertia: [f64; 3],
            #[values(0.5, 1.5)] temperature: f64,
            #[values(42, 123, 999)] seed: u32) -> anyhow::Result<()> {
            let mut microstate = Microstate::builder().seed(seed).try_build()?;
            let expected_variance = Cartesian::from(inertia) * temperature;

            for _ in 0..N_BODIES {
                microstate
                    .add_body(create_body_3d(
                        inertia,
                        Cartesian::default(),
                    ))?;
            }

            microstate.thermalize_angular_momentum(temperature);

            let angular_momenta: Vec<[f64; 3]> = microstate
                .bodies()
                .iter()
                .map(|b| b.item.properties.angular_momentum().coordinates)
                .collect();

            for dim in 0..3 {
                let components: Vec<f64> = angular_momenta.iter().map(|m| m[dim]).collect();

                let mean = components.iter().sum::<f64>() / (N_BODIES as f64);
                let variance = components.iter().map(|&v| (v - mean).powi(2)).sum::<f64>()
                    / (N_BODIES- 1) as f64;

                assert_abs_diff_eq!(mean, 0.0, epsilon = expected_variance[dim].sqrt() * EPSILON_MEAN_SCALE);
                assert_abs_diff_eq!(
                    variance,
                    expected_variance[dim],
                    epsilon = expected_variance[dim] * EPSILON_VARIANCE_SCALE
                );
            }

            Ok(())
        }
    }
}
