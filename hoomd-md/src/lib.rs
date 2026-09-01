// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Apply molecular dynamics simulation methods to systems of bodies.
//!
//! `hoomd-md` provides building blocks that you can use to create molecular
//! dynamics simulations. Start with a [`Microstate`] containing bodies and
//! sites with the required properties. Build an interaction model using types
//! from [`hoomd_interaction`] that implement the required traits. Finally,
//! choose a macrostate using one of the types from [`hoomd_simulation`], as
//! well as a suitable thermostat. With these building blocks, you are ready
//! to use an integration method to evolve your microstate forward in time.
//!
//! [`Microstate`]: hoomd_microstate::Microstate
//!
//! # Required body and site properties
//! 
//! **Translational motion** is implemented only for bodies with scalar [`Mass`]
//! and vector [`Position`], [`Momentum`], and [`NetForce`]. (Some integration
//! methods also require [`Drag`] and [`NetVirial`].) The vector properties must
//! all be embedded in the same [`InnerProduct`] vector space. (Some integration
//! methods further require that the vector space be [`Cartesian`].) The body
//! properties type must additionally implement [`Transform`] on the site
//! properties type.
//! 
//! For systems in cartesian space with only translational degrees of freedom,
//! most users will use [`DynamicPoint<Cartesian<N>>`] for their body
//! properties. Types representing custom body properties can also be used,
//! provided that they implement the required traits. A macro is provided for
//! this purpose: [`derive_dynamic_point`].
//! 
//! [`InnerProduct`]: hoomd_vector::InnerProduct
//! [`Mass`]: hoomd_microstate::property::Mass
//! [`Position`]: hoomd_microstate::property::Position
//! [`Momentum`]: hoomd_microstate::property::Momentum
//! [`NetForce`]: hoomd_microstate::property::NetForce
//! [`Drag`]: hoomd_microstate::property::Drag
//! [`NetVirial`]: hoomd_microstate::property::NetVirial
//! [`Cartesian`]: hoomd_vector::Cartesian
//! [`Transform`]: hoomd_microstate::Transform
//! [`DynamicPoint<Cartesian<N>>`]: hoomd_microstate::property::DynamicPoint
//! [`derive_dynamic_point`]: hoomd_derive::derive_dynamic_point
//! 
//! **Rotational motion** is implemented for bodies with [`Position`],
//! [`Orientation`], [`MomentOfInertia`], [`AngularMomentum`], and
//! [`NetTorque`]. (Some integration methods also require [`RotationalDrag`]).
//! These properties are subject to the following restrictions:
//! 
//! - the [`Position`] type must be embedded in [`Wedge`] vector space
//! - the [`Orientation`] type must
//!   - implement the relevant [integration trait]---see
//!     [Extending the integration schemes]
//!   - implement [`RotationalMotionTypes`] to define associated types for
//!     [`MomentOfInertia`], [`AngularMomentum`], and [`RotationalDrag`]
//! - the [`MomentOfInertia`], [`AngularMomentum`], and [`RotationalDrag`] types
//!   must match the types defined through [`RotationalMotionTypes`]
//! - the [`NetTorque`] type must be a bivector embedded in the same vector
//!   space as [`Position`]
//! 
//! [`Wedge`]: hoomd_vector::Wedge
//! [`Orientation`]: hoomd_microstate::property::Orientation
//! [Extending the integration schemes]: crate#extending-the-integration-schemes
//! [`MomentOfInertia`]: hoomd_microstate::property::MomentOfInertia
//! [`AngularMomentum`]: hoomd_microstate::property::AngularMomentum
//! [`NetTorque`]: hoomd_microstate::property::NetTorque
//! [`RotationalDrag`]: hoomd_microstate::property::RotationalDrag
//! [`RotationalMotionTypes`]: hoomd_microstate::property::RotationalMotionTypes
//! 
//! For systems in cartesian space with translational and rotational degrees of
//! freedom, most users will use [`DynamicOrientedPoint`] for their body
//! properties. Types representing custom body properties can also be used,
//! provided that they implement the required traits. A macro is provided for
//! this purpose: [`derive_dynamic_oriented_point`].
//! 
//! [`DynamicOrientedPoint`]: hoomd_microstate::property::DynamicOrientedPoint
//! [`derive_dynamic_oriented_point`]: hoomd_derive::derive_dynamic_oriented_point
//! 
//! **Site properties** are subject to fewer restrictions. Translational and
//! rotational motion are implemented for any site with [`Position`] embedded in
//! the same vector space as its body. Most users will use
//! [`Point<Cartesian<N>>`] or [`OrientedPoint`], but custom site properties
//! types may also be used, provided that the derive macro for [`Position`] is
//! added above the type definition. The choice for site properties is driven by
//! the interaction model, not the integration method.
//! 
//! [`Point<Cartesian<N>>`]: hoomd_microstate::property::Point
//! [`OrientedPoint`]: hoomd_microstate::property::OrientedPoint
//! 
//! # The interaction model
//! 
//! The system for constructing an interaction model relies on Rust's
//! [newtype] idiom, in which types are wrapped around each other to flexibly
//! construct complex functionality from simpler atomic types. This pattern
//! allows [`hoomd_interaction`] to provide a single interaction system that is
//! compatible with both MD and MC simulation methods.
//! 
//! [newtype]: https://doc.rust-lang.org/rust-by-example/generics/new_types.html
//! 
//! MD integration methods operate on bodies, not individual sites, so the
//! interaction model must be a type that implements [`NetBodyForceAndVirial`]
//! (for purely translational motion) or [`NetBodyForceVirialAndTorque`] (for
//! translational and rotational motion). These traits sum the net forces,
//! virials, and torques for individual sites to determine the net quantities
//! for each body. These traits are implemented on [`Rigid`].
//! 
//! [`NetBodyForceAndVirial`]: hoomd_interaction::NetBodyForceAndVirial
//! [`NetBodyForceVirialAndTorque`]: hoomd_interaction::NetBodyForceVirialAndTorque
//! [`Rigid`]: hoomd_interaction::Rigid
//! 
//! ```ignore (pseudocode)
//! let model = Rigid( ... );
//! ```
//! 
//! [`Rigid`] is a newtype, meaning that it is designed to be wrapped around
//! another type to build on its functionality. To sum the net forces,
//! virials, and torques on individual sites, [`Rigid`] needs to be able to
//! get those net quantities for each site; this means that it must be
//! wrapped around a type that implements [`NetSiteForceAndVirial`] or
//! [`NetSiteForceVirialAndTorque`]. These traits sum the individual forces,
//! virials, and torques on each site to determine the net quantities. These
//! traits are implemented on [`External`] and [`PairwiseCutoff`].
//! 
//! [`NetSiteForceAndVirial`]: hoomd_interaction::NetSiteForceAndVirial
//! [`NetSiteForceVirialAndTorque`]: hoomd_interaction::NetSiteForceVirialAndTorque
//! [`External`]: hoomd_interaction::External
//! [`PairwiseCutoff`]: hoomd_interaction::PairwiseCutoff
//! 
//! ```ignore (pseudocode)
//! let external_model = Rigid( External( ... ) );
//! let pairwise_model = Rigid( PairwiseCutoff( ... ) );
//! ```
//! 
//! [`External`] and [`PairwiseCutoff`] are also both newtypes. To sum the
//! individual forces, virials, and torques on each site, they need to be able
//! to get those individual quantities. [`External`] needs to be able to
//! obtain interaction quantities between a single standalone object (e.g.
//! an external field) and every site, while [`PairwiseCutoff`]
//! needs to obtain interaction quantities between every pair of sites within
//! some cutoff radius. This means that...
//! 
//! * [`External`] must be wrapped around a type that implements
//! [`SiteForceAndVirial`] or [`SiteForceVirialAndTorque`].
//! * [`PairwiseCutoff`] must be wrapped around a type that implements
//! [`SitePairForceAndVirial`] or [`SitePairForceVirialAndTorque`].
//! 
//! [`SiteForceAndVirial`]: hoomd_interaction::SiteForceAndVirial
//! [`SiteForceVirialAndTorque`]: hoomd_interaction::SiteForceVirialAndTorque
//! [`SitePairForceAndVirial`]: hoomd_interaction::SitePairForceAndVirial
//! [`SitePairForceVirialAndTorque`]: hoomd_interaction::SitePairForceVirialAndTorque
//! 
//! The `SiteForce...` traits are implemented on types in the [`external`]
//! module (e.g. [`ConstantForce`]), which terminate the chain of newtypes that
//! goes through [`External`].
//! 
//! [`ConstantForce`]: hoomd_interaction::external::ConstantForce
//! 
//! ```ignore (pseudocode)
//! let external_model = Rigid( External( ConstantForce { ... } ) );
//! ```
//! 
//! The `SitePairForce...` traits are implemented on another type,
//! [`Isotropic`], which calculates interaction quantities based solely on the
//! positions of two sites. [`Isotropic`] needs to be able to get the force
//! magnitude for a given distance, which means it must be wrapped around a type
//! that implements [`UnivariateForce`]. This trait is implemented on types in
//! the [`univariate`] submodule (e.g. [`LennardJones`]), which terminate the
//! chain of newtypes and types that goes through [`PairwiseCutoff`]. Notably,
//! [`UnivariateForce`] is only implemented on *differentiable* interactions,
//! excluding e.g. [`Boxcar`].
//! 
//! [`Isotropic`]: hoomd_interaction::pairwise::Isotropic
//! [`UnivariateForce`]: hoomd_interaction::univariate::UnivariateForce
//! [`univariate`]: hoomd_interaction::univariate
//! [`LennardJones`]: hoomd_interaction::univariate::LennardJones
//! [`Boxcar`]: hoomd_interaction::univariate::Boxcar
//! 
//! ```ignore (pseudocode)
//! let pairwise_model = Rigid( PairwiseCutoff( Isotropic {
//!     interaction: LennardJones::<12, 6> { ... },
//!     r_cut: ...
//! })));
//! ```
//!  
//! **A multi-interaction model** can be constructed by creating a new custom
//! type that is adjacent to the newtypes that implement the `NetSiteForce...`
//! traits, wrapping it around any number of those newtypes, and using the
//! derive macro for the necessary `NetSiteForce...` trait. The custom type may
//! be a named struct or a tuple struct.
//! 
//! ```ignore (pseudocode)
//! #[derive(NetSiteForceAndVirial)]
//! struct MultiInteraction(
//!     PairwiseCutoff<Isotropic<WeeksChandlerAnderson>>,
//!     External<ConstantForce<Cartesian<3>>>,
//! );
//! 
//! let multi_model = Rigid(
//!     MultiInteraction(
//!         PairwiseCutoff( Isotropic {
//!             interaction: WeeksChandlerAnderson{ ... },
//!             r_cut: ...
//!         }),
//!         External( ConstantForce { ... } )
//!     )
//! );
//! ```
//! 
//! **A site-specific interaction model** can be constructed by creating a new
//! site properties type to distinguish between sites and a new interaction type
//! that is adjacent to types that implement the `SiteForce...` or
//! `SitePairForce...` traits. Wrap the new interaction type around any number
//! of the types it is adjacent to, and then implement the `SiteForce..` or
//! `SitePairForce...` trait that they have in common. If you are wrapping types
//! with `SitePairForce...`, you must additionally re-implement
//! [`MaximumInteractionRange`].
//! 
//! [`MaximumInteractionRange`]: hoomd_interaction::MaximumInteractionRange
//! 
//! ```ignore (pseudocode)
//! #[derive(Clone, Copy, Default, PartialEq)]
//! enum SiteType {
//!     A,
//!     B,
//! }
//! 
//! #[derive(Clone, Copy, Default, Position)]
//! struct SiteProperties {
//!     position: Cartesian<3>,
//!     site_type: SiteType,
//! }
//! 
//! struct SiteSpecificPairwise {
//!     aa: Isotropic<WeeksChandlerAnderson>,
//!     bb: Isotropic<LennardJones::<12,6>>,
//! }
//! 
//! impl MaximumInteractionRange for SiteSpecificPairwise {
//!     fn maximum_interaction_range(&self) -> f64 {
//!         self.aa
//!             .maximum_interaction_range()
//!             .max(self.bb.maximum_interaction_range())
//!     }
//! }
//! 
//! impl SitePairForceAndVirial<SiteProperties> for SiteSpecificPairwise {
//!     type Force = Cartesian<3>;
//! 
//!     fn site_pair_force_and_virial(
//!         &self,
//!         site_properties_i: &SiteProperties,
//!         site_properties_j: &SiteProperties,
//!     ) -> (Self::Force, <Self::Force as Outer>::Tensor) {
//!         let (force, virial) =
//!             match (site_properties_i.site_type, site_properties_j.site_type) {
//!                 (SiteType::A, SiteType::A) => {
//!                     self.aa.site_pair_force_and_virial(
//!                         site_properties_i,
//!                         site_properties_j
//!                     )
//!                 }
//!                 (SiteType::B, SiteType::B) => {
//!                     self.bb.site_pair_force_and_virial(
//!                         site_properties_i,
//!                         site_properties_j
//!                     )
//!                 }
//!                 _ => (Cartesian::default(), Matrix::<3, 3>::default())
//!             };
//!         (force, virial)
//!     }
//! }
//! 
//! let site_specific_model = Rigid(PairwiseCutoff(SiteSpecificPairwise {
//!     aa: Isotropic {
//!         interaction: WeeksChandlerAnderson { epsilon: 1.0, sigma: 1.0 },
//!         r_cut: 5.0
//!     },
//!     bb: Isotropic {
//!         interaction: LennardJones::<12, 6>{ epsilon: 1.0, sigma: 1.0 },
//!         r_cut: 5.0
//!     },
//! }));
//! ```
//! 
//! # The macrostate
//! 
//! Just like in statistical mechanics, system-wide quantities are collectively
//! represented by a *macrostate*. Scientifically, many different microstates
//! correspond to the same macrostate. The same is true in *hoomd-rs*.
//! 
//! The difference between the scientific macrostate and the *hoomd-rs*
//! macrostate is that the latter stores the *set points* for macroscopic
//! quantities, rather than their instantaneous values. In Rust terms, the
//! macrostate "owns" these set points, which are "borrowed" by various other
//! objects that operate on the microstate. For example, the [`Isothermal`]
//! macrostate type owns a temperature set point. During integration that
//! set point is borrowed by the thermostat, which rescales quantities in the
//! microstate to achieve an instantaneous temperature that matches the
//! set point.
//! 
//! Currently, `hoomd-md` only provides integration methods that sample NVE and
//! NVT ensembles. Therefore, users will only use [`Isoenergy`] (which stores no
//! set points) or [`Isothermal`] (which stores a temperature set point).
//! 
//! [`Isoenergy`]: hoomd_simulation::macrostate::Isoenergy
//! [`Isothermal`]: hoomd_simulation::macrostate::Isothermal
//! 
//! # The thermostat
//!
//! To sample a constant temperature ensemble, the user must provide a
//! thermostat, which rescales velocities to match a temperature [set point]
//! during the integration scheme. `hoomd-md` provides implementations of many
//! momentum-rescaling thermostats, which can be found in the [`thermostat`]
//! module. Use [`NoThermostat`] to prevent momentum-rescaling, which samples
//! constant energy (or enthalpy) ensembles.
//!
//! [set point]: crate#the-macrostate
//! [`NoThermostat`]: thermostat::NoThermostat
//!
//! # Prepare the microstate
//!
//! Use [`ThermalizeMomentum`] and [`ThermalizeAngularMomentum`] to set body
//! momenta to match a thermal distribution. Then use [`ZeroCenterMomentum`] and
//! [`ZeroCenterAngularMomentum`] to remove effective motion of the system's
//! center of mass.
//!
//! These traits are implemented directly on [`Microstate`], so their methods
//! are called on the microstate object itself, e.g.
//! `microstate.zero_center_momentum()`.
//! 
//! # Integration methods
//!
//! The traits [`TranslationalMotion`] and [`RotationalMotion`] describe types
//! that can integrate, respectively, the translational and/or rotational
//! degrees of freedom in the microstate. The implementations of these traits on
//! the integration method types are where those methods' integration schemes
//! are encoded. Translational and rotational integration are encoded
//! differently---see [Extending the integration schemes].
//! 
//! [Extending the integration schemes]: crate#extending-the-integration-schemes
//! 
//! Currently, `hoomd-md` provides the following integration methods:
//! 
//! - [`ConstantVolume`] symplectically integrates the equations of motion for
//!   the model while keeping the volume of the simulation boundary fixed. This
//!   method can sample the microcanonical (NVE) or canonical (NVT) ensembles
//!   based on the choice of [thermostat].
//! - [`Langevin`] symplectically integrates the equations of motion for the
//!   model while adding additional terms for drag and random thermal forces and
//!   torques.
//! - [`Brownian`] integrates the equations of motion for the model in the
//!   overdamped limit of Langevin dynamics. Position and orientation are still
//!   coupled to force and torque, but momentum and angular momentum are not.
//! 
//! [`ConstantVolume`]: crate::method::ConstantVolume
//! [thermostat]: crate#the-thermostat
//! [`Langevin`]: crate::method::langevin::Langevin
//! [`Brownian`]: crate::method::brownian::Brownian
//! 
//! When using integration method types, most users will call
//! [`integrate_translation`] or [`integrate_translation_and_rotation`] to
//! evolve all bodies in the microstate forward one time step. See the trait
//! documentation for details on how to pin some bodies in place and apply
//! different integration methods to different bodies.
//!
//! [`integrate_translation`]: TranslationalMotion::integrate_translation
//! [`integrate_translation_and_rotation`]: RotationalMotion::integrate_translation_and_rotation
//!
//! # Compute properties of the microstate
//!
//! Use [`TranslationalKineticEnergy`] to compute the translational kinetic
//! energy and count the corresponding translational degrees of freedom
//! in the microstate. [`RotationalKineticEnergy`] does the same for
//! rotational degrees of freedom.
//!
//! As with the [modifiers], the compute traits are implemented directly on
//! [`Microstate`].
//! 
//! [modifiers]: crate#prepare-the-microstate
//! 
//! # Extending the integration schemes
//! 
//! The integration schemes for translational and rotational motion are bound
//! to the types that represent position and orientation. Extending these
//! integration schemes to new forms of spatial representation, such as
//! [`Minkowski`] space, is made possible by Rust's system of trait bounds.
//! 
//! [`Minkowski`]: hoomd_manifold::Minkowski
//! 
//! **For translational integration**, the given implementations are suitable
//! for any [`InnerProduct`] vector space (in the case of [`ConstantVolume`]),
//! or [`Cartesian`] vector space (in the case of [`Langevin`] and
//! [`Brownian`]).
//! 
//! To extend a specific method's translational integration scheme to a new
//! spatial representation, Rust's trait implementation rules require you to
//! create a custom body properties type embedded in your space, and then
//! implement [`TranslationalMotion`] for your chosen integration method type,
//! substituting your custom body properties type for the type parameter `B`.
//! The exact steps you must follow depend on whether the properties of your
//! space.
//! 
//! If your space is a vector space with a mathematically defined outer product,
//! your body properties type definition can use a convenience macro to
//! automatically add the required property trait implementations. The steps
//! are as follows.
//! 
//! 1. Create a struct representing your vector type.
//! 2. Implement all of the supertraits for [`Vector`] on your vector type, then
//!    implement [`Vector`] on it.
//! 3. Implement [`Outer`] on your vector type.
//!    * If your space also has a mathematically defined wedge product and you
//!      intend to integrate rotational degrees of freedom, you must
//!      additionally implement [`Wedge`] on your vector type.
//! 4. Create a struct representing your body properties, using a convenience
//!    macro to give it the required trait implementations.
//!    * If you only implemented [`Outer`], use [`derive_dynamic_point`].
//!    * If you also implemented [`Wedge`], use
//!      [`derive_dynamic_oriented_point`] along with your orientation type. See
//!      the macro documentation for additional restrictions on the orientation
//!      type.
//! 5. Implement [`TranslationalMotion`] as described above.
//! 
//! [`Vector`]: hoomd_vector::Vector
//! [`Outer`]: hoomd_vector::Outer
//! [`Wedge`]: hoomd_vector::Wedge
//! 
//! If your space does not satisfy the criteria above, you cannot use a
//! convenience macro. The steps are then as follows.
//! 
//! 1. Create all the structs needed for representing position, momentum,
//!    net force, and mass in your space.
//! 2. Create a struct representing your body properties, using the types
//!    defined in step 1.
//! 3. Implement the required traits on your custom body property type.
//! 4. Implement [`TranslationalMotion`] as described above.
//! 
//! **For rotational integration**, the given implementations are suitable only
//! for 2D and 3D cartesian space. To extend a specific method's rotational
//! integration scheme to a new spatial representation, you must follow the same
//! process as for translational integration, but with several additional steps.
//! 
//! 1. You must implement [`RotationalMotionTypes`] on your orientation type.
//! 2. You must implement additional required properties on your body properties
//!    type: [`Orientation`], [`AngularMomentum`], [`MomentOfInertia`],
//!    [`NetTorque`], and depending on the method, [`RotationalDrag`]. The types
//!    of these properties should reflect those chosen in step 1.
//! 3. You must implement the relevant rotational integration trait for your
//!    integration method on your orientation type.
//!    * For [`ConstantVolume`] and [`Langevin`], implement
//!      [`SymplecticIntegrateRotation`].
//!    * For [`Brownian`], implement [`BrownianIntegrateRotation`].
//! 4. If you are extending [`Langevin`], you must additionally implement
//!    [`DragAndRandomTorque`] on your orientation type.
//! 
//! [`SymplecticIntegrateRotation`]: crate::method::SymplecticIntegrateRotation
//! [`BrownianIntegrateRotation`]: crate::method::brownian::BrownianIntegrateRotation
//! [`DragAndRandomTorque`]: crate::method::langevin::DragAndRandomTorque

use rand::Rng;

use hoomd_microstate::{Body, Microstate, Tagged};

pub mod method;
pub mod thermostat;

pub mod compute;
pub use compute::{RotationalKineticEnergy, TranslationalKineticEnergy};

pub mod modify;
pub use modify::{
    ThermalizeAngularMomentum, ThermalizeMomentum, ZeroCenterAngularMomentum, ZeroCenterMomentum,
};

mod update_net_force;
pub use update_net_force::{UpdateNetForceAndVirial, UpdateNetForceVirialAndTorque};

/// Scale momenta to hold the system at constant temperature.
///
/// Momentum scaling algorithms are implemented for the various types in the
/// [`thermostat`] module.
/// 
/// Any thermostat can be used with any integration method that accepts one. For
/// example, construct [`ConstantVolume`] with a translational---and if
/// relevant, rotational---thermostat to sample  trajectories from the canonical
/// (NVT) ensemble.
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
        Microstate<B, S, X, C>: UpdateNetForceAndVirial<E>,
    {
        self.integrate_translation_half_step_one_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
        microstate.update_net_force_and_virial(interaction_model);
        self.integrate_translation_half_step_two_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
    }

    /// Integrate all body translational degrees of freedom forward one step.
    #[inline]
    fn integrate_translation<E>(
        &mut self,
        microstate: &mut Microstate<B, S, X, C>,
        macrostate: &M,
        interaction_model: &E,
    ) where
        Microstate<B, S, X, C>: UpdateNetForceAndVirial<E>,
    {
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
/// #           orientation: Angle::default(),
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
/// follow these steps:
/// 
/// 1. call [`integrate_rotation_half_step_one_with_filter`] for all methods
/// 2. call [`update_net_force_virial_and_torque`],
/// 3. call [`integrate_rotation_half_step_one_with_filter`] for all methods.
/// 
/// [`update_net_force_virial_and_torque`]: crate::UpdateNetForceVirialAndTorque::update_net_force_virial_and_torque
/// 
/// The filters must select distinct subsets of bodies. The filters must also select
/// the same bodies in half step one and half step two.
/// 
/// ```
/// # use hoomd_microstate::{Body, Microstate, property::{DynamicOrientedPoint, Point}};
/// # use hoomd_vector::{Angle, Cartesian};
/// # use hoomd_md::{UpdateNetForceVirialAndTorque, ThermalizeMomentum, RotationalMotion, TranslationalMotion, method::ConstantVolume};
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
/// #           orientation: Angle::default(),
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
pub trait RotationalMotion<R, B, S, X, C, M> {
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
        Self: TranslationalMotion<B, S, X, C, M>,
    {
        self.integrate_translation_half_step_one_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
        self.integrate_rotation_half_step_one_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
        self.integrate_rotation_half_step_two_with_filter(
            microstate,
            macrostate,
            &should_integrate_body,
        );
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
        Self: TranslationalMotion<B, S, X, C, M>,
    {
        self.integrate_translation_half_step_one_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_one_with_filter(microstate, macrostate, |_| true);
        microstate.update_net_force_virial_and_torque(interaction_model);
        self.integrate_translation_half_step_two_with_filter(microstate, macrostate, |_| true);
        self.integrate_rotation_half_step_two_with_filter(microstate, macrostate, |_| true);
    }
}
