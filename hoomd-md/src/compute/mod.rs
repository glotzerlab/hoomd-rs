// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Methods that compute properties of microstates.

use hoomd_microstate::{Body, Tagged};

mod translational_kinetic_energy;
mod rotational_kinetic_energy;

/// Compute the translational kinetic energy of bodies in a microstate.
///
/// `TranslationalKineticEnergy` is implemented for `Microstate`. Call
/// `microstate.translational_kinetic_energy` to compute the total translational
/// kinetic energy (and degrees of freedom) of all bodies in the microstate.
///
/// # Example
///
/// ```
/// use hoomd_microstate::{
///     Microstate,
///     Body,
///     property::{DynamicPoint, Point},
/// };
/// use hoomd_vector::Cartesian;
/// use hoomd_md::TranslationalKineticEnergy;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///       let microstate = Microstate::builder()
///           .bodies([
///               Body::single_site(
///                   DynamicPoint {
///                       mass: 2.0,
///                       momentum: Cartesian::<2>::from([2.0, 0.0]),
///                       ..Default::default()
///                   },
///                   Point::default(),
///               ),
///               Body::single_site(
///                   DynamicPoint {
///                       mass: 4.0,
///                       momentum: Cartesian::<2>::from([1.0, 1.0]),
///                       ..Default::default()
///                   },
///                   Point::default(),
///               ),
///               Body::single_site(
///                   DynamicPoint {
///                       mass: 3.0,
///                       momentum: Cartesian::<2>::from([-4.0, -2.0]),
///                       ..Default::default()
///                   },
///                   Point::default(),
///               ),
///           ])
///           .try_build()?;
///       let (translational_kinetic_energy, translational_degrees_of_freedom) =
///           microstate.translational_kinetic_energy();
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
    fn translational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&self,
        should_sum_body: F) -> (f64, usize);
}

/// Compute the translational kinetic energy of bodies in a microstate.
///
/// `RotationalKineticEnergy` is implemented for `Microstate`. Call
/// `microstate.rotational_kinetic_energy` to compute the total rotational
/// kinetic energy (and degrees of freedom) of all bodies in the microstate.
///
/// # Example
///
/// ```
/// use hoomd_microstate::{
///     Microstate,
///     Body,
///     property::{DynamicOrientedPoint, Point},
/// };
/// use hoomd_vector::{Angle, Cartesian};
/// use hoomd_md::RotationalKineticEnergy;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let microstate: Microstate<DynamicOrientedPoint<Cartesian<2>, Angle>, _, _, _> =
///         Microstate::builder()
///             .bodies([
///                 Body::single_site(
///                     DynamicOrientedPoint {
///                         moment_of_inertia: 0.0,
///                         ..Default::default()
///                     },
///                     Point::default(),
///                 ),
///                 Body::single_site(
///                     DynamicOrientedPoint {
///                         moment_of_inertia: 2.0,
///                         angular_momentum: 8.0,
///                         ..Default::default()
///                     },
///                     Point::default(),
///                 ),
///                 Body::single_site(
///                     DynamicOrientedPoint {
///                         moment_of_inertia: 4.0,
///                         angular_momentum: 3.0,
///                         ..Default::default()
///                     },
///                     Point::default(),
///                 ),
///             ])
///             .try_build()?;
///
///       let (rotational_kinetic_energy, rotational_degrees_of_freedom) =
///           microstate.rotational_kinetic_energy();
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
    fn rotational_kinetic_energy_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(&self,
        should_sum_body: F) -> (f64, usize);
}
