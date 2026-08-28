// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods that compute properties of microstates.

use hoomd_microstate::{Body, Tagged};

mod translational_kinetic_energy;

pub mod rotational_kinetic_energy;

/// Compute the translational kinetic energy of bodies in a microstate.
///
/// This trait is implemented directly on [`Microstate`]. Call
/// [`microstate.translational_kinetic_energy()`] to compute the total
/// translational kinetic energy (and degrees of freedom) of all bodies in the
/// microstate. Calculations can be performed on a subset of bodies using the
/// companion method [`translational_kinetic_energy_with_filter`].
/// 
/// [`Microstate`]: hoomd_microstate::Microstate
/// [`microstate.translational_kinetic_energy()`]: Self::translational_kinetic_energy
/// [`translational_kinetic_energy_with_filter`]: Self::translational_kinetic_energy_with_filter
/// 
/// The procedure for computing translational kinetic energy (and degrees of
/// freedom) is bound to a [`Microstate`] type that uses a specific type for
/// momentum. (For details on existing implementations, see the implementation
/// section below.) To extend this functionality to a new spatial
/// representation, implement `TranslationalKineticEnergy` on a [`Microstate`]
/// that uses your momentum type.
/// 
/// [`Microstate`]: hoomd_microstate::Microstate
/// 
/// [`InnerProduct`]: hoomd_vector::InnerProduct
/// [implementation section]: Self::translational_kinetic_energy_with_filter
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
    /// Compute the total translational kinetic energy (and degrees of freedom) over all bodies in the microstate.
    #[inline]
    fn translational_kinetic_energy(&self) -> (f64, usize) {
        self.translational_kinetic_energy_with_filter(|_| true)
    }

    /// Compute the total translational kinetic energy (and degrees of freedom) over selected bodies in the microstate.
    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize);
}

/// Compute the rotational kinetic energy of bodies in a microstate.
///
/// This trait is implemented directly on [`Microstate`]. Call
/// [`microstate.rotational_kinetic_energy()`] to compute the total
/// rotational kinetic energy (and degrees of freedom) of all bodies in the
/// microstate. Calculations can be performed on a subset of bodies using the
/// companion method [`rotational_kinetic_energy_with_filter`].
/// 
/// [`Microstate`]: hoomd_microstate::Microstate
/// [`microstate.rotational_kinetic_energy()`]: Self::rotational_kinetic_energy
/// [`rotational_kinetic_energy_with_filter`]: Self::rotational_kinetic_energy_with_filter
/// 
/// The procedure for computing rotational kinetic energy (and degrees of
/// freedom) is bound directly to the type that represents orientation. (For
/// details on existing implementations, see the implementation section for
/// [`AggregateEnergyRotation`].) To extend this functionality to a new spatial
/// representation, implement [`AggregateEnergyRotation`] on your orientation
/// type.
/// 
/// [`AggregateEnergyRotation`]: crate::compute::rotational_kinetic_energy::AggregateEnergyRotation
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
    /// Compute the total rotational kinetic energy (and degrees of freedom) over all bodies in the microstate.
    #[inline]
    fn rotational_kinetic_energy(&self) -> (f64, usize) {
        self.rotational_kinetic_energy_with_filter(|_| true)
    }

    /// Compute the total rotational kinetic energy (and degrees of freedom) over selected bodies in the microstate.
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &self,
        should_sum_body: F,
    ) -> (f64, usize);
}
