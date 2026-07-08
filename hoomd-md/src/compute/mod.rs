// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods that compute properties of microstates.

use hoomd_microstate::{Body, Tagged};

mod rotational_kinetic_energy;
mod translational_kinetic_energy;

/// Compute the translational kinetic energy of bodies in a microstate.
///
/// `TranslationalKineticEnergy` is implemented for `Microstate`. Call
/// `microstate.translational_kinetic_energy` to compute the total translational
/// kinetic energy (and degrees of freedom) of all bodies in the microstate.
/// Energy can be calculated for a subset of bodies using the companion method
/// `translational_kinetic_energy_with_filter`.
///
/// Sum the per-body kinetic energies:
/// ```math
/// K = \sum_{i \in \mathrm{selection}} \frac{\vec{p}_i \cdot \vec{p}_i}{2m_i}
/// ```
///
/// Count the degrees of freedom of each selected body:
/// ```math
/// \mathrm{degrees\_of\_freedom} = \sum_{i \in \mathrm{selection}} D
/// ```
/// where `D` is the dimensionality of momentum vector space.
///
/// # Example
///
/// ```
/// use hoomd_md::TranslationalKineticEnergy;
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{DynamicPoint, Point},
/// };
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let microstate = Microstate::builder()
///     .bodies([
///         Body::single_site(
///             DynamicPoint {
///                 mass: 2.0,
///                 momentum: Cartesian::<2>::from([2.0, 0.0]),
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///         Body::single_site(
///             DynamicPoint {
///                 mass: 4.0,
///                 momentum: Cartesian::<2>::from([1.0, 1.0]),
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///         Body::single_site(
///             DynamicPoint {
///                 mass: 3.0,
///                 momentum: Cartesian::<2>::from([-4.0, -2.0]),
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///     ])
///     .try_build()?;
/// let (translational_kinetic_energy, translational_degrees_of_freedom) =
///     microstate.translational_kinetic_energy();
/// # Ok(())
/// # }
/// ```
pub trait TranslationalKineticEnergy<B, S> {
    /// Compute the total translational kinetic energy and degrees of freedom over all bodies
    /// in the microstate.
    #[inline]
    fn translational_kinetic_energy(&self) -> (f64, usize) {
        self.translational_kinetic_energy_with_filter(|_| true)
    }

    /// Compute the total translational kinetic energy and degrees of freedom over selected
    /// bodies in the microstate.
    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize);
}

/// Compute the rotational kinetic energy of bodies in a microstate.
///
/// `RotationalKineticEnergy` is implemented for `Microstate`. Call
/// `microstate.rotational_kinetic_energy` to compute the total rotational
/// kinetic energy (and degrees of freedom) of all bodies in the microstate.
/// Energy can be calculated for a subset of bodies using the companion method
/// `rotational_kinetic_energy_with_filter`.
///
/// # 2D
///
/// In 2D, each body has only 0 or 1 rotational degree of freedom. Set $` L = 0 `$ to deactivate
/// rotations for a body. The total number of degrees of freedom is then:
/// ```math
/// \mathrm{degrees\_of\_freedom} = \sum_{i \in \mathrm{selection}} \left| L_i \ne 0 \right|
/// ```
/// where $` \left| \right| `$ is the Iverson bracket.
///
/// The kinetic energy is
/// ```math
/// K = \sum_{i \in \mathrm{selection}} \frac{L_i^2}{2I}
/// ```
/// (ignoring terms where the moment of inertia is zero).
///
/// # 3D
///
/// In 3D, there are 0 to 3 degrees of freedom per body.
/// Set $` I_{xx}=0 `$, $` I_{yy}=0 `$, and/or $` I_{zz}=0 `$ to deactivate
/// rotations one or more axes. The total number of degrees of freedom is then:
/// ```math
/// \mathrm{degrees\_of\_freedom} = \sum_{i \in \mathrm{selection}} \left| I_{xx,i} \ne 0 \right| + \left| I_{yy,i} \ne 0 \right| + \left| I_{zz,i} \ne 0 \right|
/// ```
///
/// The kinetic energy is
/// ```math
/// K = \sum_{i \in \mathrm{selection}}\frac{L_{x,i}(t)^2}{2I_{xx,i}} + \frac{L_{y,i}(t)^2}{2I_{yy,i}} + \frac{L_{z,i}(t)^2}{2I_{zz,i}}
/// ```
/// (ignoring terms where the moment of inertia is zero).
///
/// # Example
///
/// ```
/// use hoomd_md::RotationalKineticEnergy;
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{DynamicOrientedPoint, Point},
/// };
/// use hoomd_vector::{Angle, Cartesian};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let microstate: Microstate<
///     DynamicOrientedPoint<Cartesian<2>, Angle>,
///     _,
///     _,
///     _,
/// > = Microstate::builder()
///     .bodies([
///         Body::single_site(
///             DynamicOrientedPoint {
///                 moment_of_inertia: 0.0,
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///         Body::single_site(
///             DynamicOrientedPoint {
///                 moment_of_inertia: 2.0,
///                 angular_momentum: 8.0,
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///         Body::single_site(
///             DynamicOrientedPoint {
///                 moment_of_inertia: 4.0,
///                 angular_momentum: 3.0,
///                 ..Default::default()
///             },
///             Point::default(),
///         ),
///     ])
///     .try_build()?;
///
/// let (rotational_kinetic_energy, rotational_degrees_of_freedom) =
///     microstate.rotational_kinetic_energy();
/// # Ok(())
/// # }
/// ```
pub trait RotationalKineticEnergy<B, S> {
    /// Compute the total rotational kinetic energy and degrees of freedom over all bodies
    /// in the microstate.
    #[inline]
    fn rotational_kinetic_energy(&self) -> (f64, usize) {
        self.rotational_kinetic_energy_with_filter(|_| true)
    }

    /// Compute the total rotational kinetic energy and degrees of freedom over selected
    /// bodies in the microstate.
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize);
}
