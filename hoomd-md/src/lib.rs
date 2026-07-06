// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Apply the molecular dynamics simulation method to systems of bodies.
//!
//! `hoomd-md` provides building blocks that you can use to create a molecular dynamics
//! simulation model. Start with a [`Microstate`] to represent the properties of all the
//! bodies and sites. Form an interaction model using types from [`hoomd_interaction`]
//! that implement [`NetBodyForce`] or [`NetBodyForceAndTorque`] and set the macrostate
//! using one of the types from [`hoomd_simulation`].
//!
//! [`Microstate`]: hoomd_microstate::Microstate
//! [`NetBodyForce`]: hoomd_interaction::NetBodyForce
//! [`NetBodyForceAndTorque`]: hoomd_interaction::NetBodyForceAndTorque
//! [`DeltaEnergyRemove`]: hoomd_interaction::DeltaEnergyRemove
//! [`TotalEnergy`]: hoomd_interaction::TotalEnergy
//!
//! # Integration methods
//!
//! The [`TranslationalMotion`] and [`RotationalMotion`] traits describe types that
//! can integrate the translational and/or rotational degrees of freedom in the
//! microstate, respectively. Most users will call [`integrate_translation`]
//! or [`integrate_translation_and_rotation`] to advance all bodies in the microstate
//! forward one time step. See the trait documentation for details on how to pin some bodies
//! in place and/or apply different integration methods to different bodies.
//!
//! [`integrate_translation`]: TranslationalMotion::integrate_translation
//! [`integrate_translation_and_rotation`]: RotationalMotion::integrate_translation_and_rotation
//!
//! The [`ConstantVolume`] method integrates the equations of motion for the model
//! while keeping the volume of the simulation boundary fixed. [`ConstantVolume`]
//! can sample the microcanonical (NVE) or canonical (NVT) ensembles based on the
//! choice of thermostat (see below).
//!
//! [`ConstantVolume`]: crate::method::ConstantVolume
//!
//! ## Body and site properties
//!
//! Currently, *hoomd-rs* implements [`TranslationalMotion`] for any [`InnerProduct`] vector
//! space for bodies with [`Mass`], [`Momentum`], and [`NetForce`] properties in the same
//! vector space as [`Position`]. For systems with only translational degrees of freedom,
//! most users will choose [`DynamicPoint<Cartesian<N>>`] body properties and
//! [`Point<Cartesian<N>>`] site properties.
//!
//! [`InnerProduct`]: hoomd_vector::InnerProduct
//! [`Mass`]: hoomd_microstate::property::Mass
//! [`Momentum`]: hoomd_microstate::property::Momentum
//! [`NetForce`]: hoomd_microstate::property::NetForce
//! [`Position`]: hoomd_microstate::property::Position
//! [`DynamicPoint<Cartesian<N>>`]: hoomd_microstate::property::DynamicPoint
//! [`Point<Cartesian<N>>`]: hoomd_microstate::property::Point
//!
//! Due to the mathematical nature of rotational degrees of freedom, *hoomd-rs* implements
//! [`RotationalMotion`] specifically for [`DynamicOrientedPoint<Cartesian<2>, Angle>`] for
//! 2D simulations and [`DynamicOrientedPoint<Cartesian<3>, Versor>`] for 3D. You must use
//! one of these two types for body properties to integrate rotational degrees of freedom.
//! There are fewer restrictions on the site properties type. Most users will choose
//! [`Point<Cartesian<N>>`] or [`OrientedPoint<Cartesian<N>>`] site properties for
//! models with rotational degrees of freedom, while some will need custom types.
//! The choice for site properties is driven by the interaction model, not the integration
//! method.
//!
//! [`DynamicOrientedPoint<Cartesian<2>, Angle>`]: hoomd_microstate::property::DynamicOrientedPoint
//! [`DynamicOrientedPoint<Cartesian<3>, Versor>`]: hoomd_microstate::property::DynamicOrientedPoint
//! [`OrientedPoint<Cartesian<N>>`]: hoomd_microstate::property::OrientedPoint
//!
//! ## Thermostats
//!
//! Some of the integration methods sample constant temperature ensembles using velocity
//! rescaling thermostats. There are many algorithms to choose from. Find them in the
//! [`thermostat`] module. Use [`NoThermostat`] to sample constant energy (or enthalpy)
//! ensembles.
//!
//! [`NoThermostat`]: thermostat::NoThermostat
//! 
//! # The `Rigid` interaction model
//!
//! All integration methods in *hoomd-rs* model bodies as rigid bodies. The net force and
//! torque on each body results from the forces and torques applied to its sites.
//! The [`Rigid`] type implements [`NetBodyForce`] and [`NetBodyForceAndTorque`] when
//! it wraps a type that computes forces ([`NetSiteForce`]) and torques
//! ([`NetSiteForceAndTorque`]) on sites. For example:
//! `Rigid<PairwiseCutoff<Isotropic<LennardJones>>>` is a valid interaction model
//! for use with molecular dynamics integration methods.
//!
//! [`Rigid`]: hoomd_interaction::Rigid
//! [`NetSiteForce`]: hoomd_interaction::NetSiteForce
//! [`NetSiteForceAndTorque`]: hoomd_interaction::NetSiteForceAndTorque
//!
//! Most differentiable interaction models implement both [`NetSiteForce`] and all the
//! Hamiltonian traits needed for Monte Carlo simulations in *hoomd-mc*. With these
//! interaction models, you can freely swap between MD and MC simulation steps.
//! Non-differentiable energies, such as [`Boxcar`] implement energy traits,
//! but not forces and can therefore only be used with MC. Others, like active forces,
//! might implement the force traits but not energy and can only be used with MD.
//! Rust will validate the trait bounds and issue a compile error for invalid
//! combinations.
//!
//! [`Boxcar`]: hoomd_interaction::univariate::Boxcar
//!
//! # Microstate modifiers
//!
//! Use [`ThermalizeMomentum`] and [`ThermalizeAngularMomentum`] to sample random
//! momenta from a thermal distribution. Use [`ZeroCenterMomentum`] and
//! [`ZeroCenterAngularMomentum`] to remove motion of the center of mass.
//!
//! All of these modifier traits are implemented for [`Microstate`] itself:
//! e.g. `microstate.zero_center_momentum()`.
//!
//! # Compute properties of the microstate
//!
//! Use [`TranslationalKineticEnergy`] to compute the translational kinetic
//! energy and count the corresponding translational degrees of freedom
//! in the microstate. [`RotationalKineticEnergy`] does the same for
//! rotational degrees of freedom.
//!
//! As with the modifies, the compute traits are implemented for [`Microstate`].

use rand::Rng;

use hoomd_microstate::{Body, Tagged};
use hoomd_microstate::Microstate;

pub mod thermostat;
pub mod method;

mod compute;
pub use compute::TranslationalKineticEnergy;
pub use compute::RotationalKineticEnergy;

mod modify;
pub use modify::ThermalizeAngularMomentum;
pub use modify::ThermalizeMomentum;
pub use modify::ZeroCenterMomentum;
pub use modify::ZeroCenterAngularMomentum;

mod update_net_force;
pub use update_net_force::UpdateNetForceAndVirial;
pub use update_net_force::UpdateNetForceVirialAndTorque;

/// Scale momenta to hold the system at constant temperature.
///
/// Use any of the thermostats in the [`thermostat`] module along with the
/// integration method of your choice.
///
/// The [`ConstantVolume`] integration method rescales every momentum in the
/// system following the given [`Thermostat`] to sample trajectories from the
/// canonical ensemble.
///
/// [`ConstantVolume`]: crate::method::ConstantVolume
pub trait Thermostat<M> {
    /// Integrate the thermostat one half step forward in time.
    ///
    /// Returns the momentum scaling factor to use during the first half step.
    fn integrate_half_step_one<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;

    /// Integrate the thermostat one half step forward in time.
    ///
    /// Returns the momentum scaling factor to use during the second half step.
    fn integrate_half_step_two<R: Rng + ?Sized>(
        &mut self,
        rng: &mut R,
        macrostate: &M,
        delta_t: f64,
        kinetic_energy: f64,
        degrees_of_freedom: usize,
    ) -> f64;
}

/// Integrate translational degrees of freedom.
///
/// [`TranslationalMotion`] integrates the [`Position`] and [`Momentum`] degrees of
/// freedom for selected bodies.
///
/// To integrate the whole system forward one step, call [`integrate_translation`]:
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// # use hoomd_vector::Cartesian;
/// # use hoomd_md::{ThermalizeMomentum, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method.integrate_translation(&mut microstate, &macrostate, &interaction_model);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// To integrate only some bodies, call [`integrate_translation_with_filter`]:
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// # use hoomd_vector::Cartesian;
/// # use hoomd_md::{ThermalizeMomentum, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method.integrate_translation_with_filter(&mut microstate, &macrostate, &interaction_model, |b| b.tag < 2);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// To integrate some bodies with one integration method and other bodies with another,
/// call [`integrate_translation_half_step_one_with_filter`] for all methods, then call
/// `update_net_force`, and finish with [`integrate_translation_half_step_one_with_filter`].
/// The filters must select distinct subsets of bodies. The filters must also select
/// the same bodies in half step one and half step two.
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicPoint, Point}};
/// # use hoomd_vector::Cartesian;
/// # use hoomd_md::{UpdateNetForceAndVirial, ThermalizeMomentum, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method_1 = ConstantVolume::builder(0.001).build();
/// # let mut integration_method_2 = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method_1.integrate_translation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_2.integrate_translation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// microstate.update_net_force_and_virial(&interaction_model);
/// integration_method_1.integrate_translation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_2.integrate_translation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `X`: The spatial data structure type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
/// * `M`: The [`macrostate`](hoomd_simulation::macrostate) type.
///
/// [`integrate_translation`]: Self::integrate_translation
/// [`integrate_translation_with_filter`]: Self::integrate_translation_with_filter
/// [`integrate_translation_half_step_one_with_filter`]: Self::integrate_translation_half_step_one_with_filter
/// [`integrate_translation_half_step_two_with_filter`]: Self::integrate_translation_half_step_two_with_filter
/// [`Position`]: hoomd_microstate::property::Position
/// [`Momentum`]: hoomd_microstate::property::Momentum
pub trait TranslationalMotion<B, S, X, C, M> {
    /// Integrate all body positions forward a full step and the momenta forward a half step.
    #[inline]
    fn integrate_translation_half_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body positions forward a full step and the momenta forward a half step.
    fn integrate_translation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate all body momenta forward a half step.
    #[inline]
    fn integrate_translation_half_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body momenta forward a half step.
    fn integrate_translation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate selected body translational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_with_filter<E, F>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
        should_integrate_body: F,
    ) where
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: UpdateNetForceAndVirial<E> {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        microstate.update_net_force_and_virial(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
    }

    /// Integrate all body translational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation<E>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
    ) where
    Microstate<B, S, X, C>: UpdateNetForceAndVirial<E> {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
        microstate.update_net_force_and_virial(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }
}

/// Integrate rotational degrees of freedom.
///
/// [`RotationalMotion`] integrates the [`Orientation`] and [`AngularMomentum`] degrees of
/// freedom for selected bodies. 
/// 
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `X`: The spatial data structure type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
/// * `M`: The [`macrostate`](hoomd_simulation::macrostate) type.
///
/// To integrate the whole system forward one step, call [`integrate_translation_and_rotation`]:
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicOrientedPoint, Point}};
/// # use hoomd_vector::Cartesian;
/// # use hoomd_md::{ThermalizeMomentum, RotationalMotion, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method.integrate_translation_and_rotation(&mut microstate, &macrostate, &interaction_model);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// To integrate only some bodies, call [`integrate_translation_and_rotation_with_filter`]:
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicOrientedPoint, Point}};
/// # use hoomd_vector::{Angle, Cartesian};
/// # use hoomd_md::{ThermalizeMomentum, RotationalMotion, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           orientation: Angle::default(),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method.integrate_translation_with_filter(&mut microstate, &macrostate, &interaction_model, |b| b.tag < 2);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// To integrate some bodies with one integration method and other bodies with another,
/// call `integrate_translation_half_step_one_with_filter`
/// [`integrate_rotation_half_step_one_with_filter`] for all methods, then call
/// `update_net_force_and_torque`, and finish with `integrate_translation_half_step_one_with_filter`
/// [`integrate_rotation_half_step_one_with_filter`].
/// The filters must select distinct subsets of bodies. The filters must also select
/// the same bodies in half step one and half step two.
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicOrientedPoint, Point}};
/// # use hoomd_vector::Cartesian;
/// # use hoomd_md::{UpdateNetForceVirialAndTorque, ThermalizeMomentum, RotationalMotion, TranslationalMotion, method::ConstantVolume};
/// # use hoomd_interaction::{Rigid, Zero};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let mut microstate = Microstate::builder()
/// #     .bodies([
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([1.0, 2.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #         Body::single_site(DynamicOrientedPoint {
/// #           position: Cartesian::from([-2.0, 3.0]),
/// #           ..Default::default()
/// #           },
/// #           Point::default(),
/// #           ),
/// #     ])
/// #     .try_build()?;
/// # microstate.thermalize_momentum(1.5);
/// # let mut integration_method_1 = ConstantVolume::builder(0.001).build();
/// # let mut integration_method_2 = ConstantVolume::builder(0.001).build();
/// # let interaction_model = Rigid(Zero);
/// # let macrostate = ();
/// integration_method_1.integrate_translation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_1.integrate_rotation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_2.integrate_translation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// integration_method_2.integrate_rotation_half_step_one_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// microstate.update_net_force_virial_and_torque(&interaction_model);
/// integration_method_1.integrate_translation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_1.integrate_rotation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag < 2);
/// integration_method_2.integrate_translation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// integration_method_2.integrate_rotation_half_step_two_with_filter(&mut microstate, &macrostate, |b| b.tag >= 2);
/// microstate.increment_step();
/// # Ok(())
/// # }
/// ```
///
/// [`integrate_translation_and_rotation`]: Self::integrate_translation_and_rotation
/// [`integrate_translation_and_rotation_with_filter`]: Self::integrate_translation_and_rotation_with_filter
/// [`integrate_rotation_half_step_one_with_filter`]: Self::integrate_rotation_half_step_one_with_filter
/// [`integrate_rotation_half_step_two_with_filter`]: Self::integrate_rotation_half_step_two_with_filter
/// [`Orientation`]: hoomd_microstate::property::Orientation
/// [`AngularMomentum`]: hoomd_microstate::property::AngularMomentum
pub trait RotationalMotion<B, S, X, C, M> {
    /// Integrate all body orientations forward a full step and their angular momenta forward a half step.
    #[inline]
    fn integrate_rotation_half_step_one(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, |_| true);
    }
    
    /// Integrate selected body orientations forward a full step and their angular momenta forward a half step.
    fn integrate_rotation_half_step_one_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate all body angular momenta forward a half step.
    #[inline]
    fn integrate_rotation_half_step_two(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
    ) {
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }

    /// Integrate selected body angular momenta forward a half step.
    fn integrate_rotation_half_step_two_with_filter<F: Fn(&Tagged<Body<B, S>>) -> bool>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        should_integrate_body: F,
    );

    /// Integrate selected body translational and rotational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_and_rotation_with_filter<E, F>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
        should_integrate_body: F,
    ) where
    F: Fn(&Tagged<Body<B, S>>) -> bool,
    Microstate<B, S, X, C>: UpdateNetForceVirialAndTorque<E>,
    Self: TranslationalMotion<B, S, X, C, M> 
    {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, &should_integrate_body);
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, &should_integrate_body);
    }

    /// Integrate all body translational and rotational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation_and_rotation<E>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
    ) where
    Microstate<B, S, X, C>: UpdateNetForceVirialAndTorque<E>,
    Self: TranslationalMotion<B, S, X, C, M> 
    {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, |_| true);
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }
}
