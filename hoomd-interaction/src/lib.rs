// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Particle interactions and physical models that apply to microstates.
//!
//! # Hamiltonian
//!
//! A type that describes a Hamiltonian (or a single term in a multi-part Hamiltonian)
//! implements one or more of the following traits: [`TotalEnergy`], [`DeltaEnergyOne`],
//! [`DeltaEnergyInsert`], and [`DeltaEnergyRemove`]. Given a microstate, the
//! [`total_energy`] method computes the total energy of the Hamiltonian. The various
//! `delta_energy_*` methods compute the *change* in total energy when updating, inserting,
//! or removing a body. Total energy computations *typically* cost $` O(N) `$ while
//! `delta_energy` methods typically cost $` O(1) `$. These costs may vary based
//! on the specific interaction type and/or the microstate's spatial data structure.
//!
//! As a convenience, most Hamiltonian types also implement [`MaximumInteractionRange`],
//! so that callers can easily determine the maximum site-site interaction range in a
//! given model.
//!
//! All the Hamiltonian traits can be automatically derived using a `#[derive()]` macro
//! of the same name. The derived implementation sums over all the fields in the struct.
//!
//! [`total_energy`]: TotalEnergy::total_energy
//!
//! # Univariate interactions
//!
//! Many interaction potentials are a function of one variable, typically the
//! distance between two sites but sometimes the distance between a site and
//! surface, or an angle. `hoomd-interaction` implements the most commonly
//! used univariate potentials, such as [`LennardJones`] in [`univariate`].
//! These types are not Hamiltonian terms on their own, but can be combined
//! with other types to create interaction models.
//!
//! To implement your own univariate interactions, implement the
//! [`UnivariateEnergy`] and/or [`UnivariateForce`] traits.
//!
//! [`LennardJones`]: univariate::LennardJones
//! [`UnivariateEnergy`]: univariate::UnivariateEnergy
//! [`UnivariateForce`]: univariate::UnivariateForce
//!
//! # Interactions between sites and external objects
//!
//! The [`SiteEnergy`] trait describes a type that computes the contribution
//! of a single site to the total energy as a function only of that site's
//! properties along with fixed external parameters. The [`External`] type
//! implements all the Hamiltonian traits. It applies the wrapped [`SiteEnergy`]
//! to all the sites in the microstate. See [`external`] for a list of built-in
//! [`SiteEnergy`] implementations.
//!
//! # Interactions between all pairs of sites
//!
//! The [`SitePairEnergy`] trait describes a type that computes the energy
//! that a pair of sites contributes to the Hamiltonian as a function of
//! the properties of the two sites. The [`PairwiseCutoff`] type implements
//! all the Hamiltonian traits. It applies the wrapped [`SitePairEnergy`]
//! to all pairs of sites that are within the maximum interaction range.
//!
//! The [`pairwise`] module provides numerous types that implement
//! [`SitePairEnergy`], including [`Isotropic`] (which wraps any
//! univariate potential), [`HardShape`] (which wraps a shape from
//! [`hoomd_geometry`], and many others.
//!
//! # Zero
//!
//! [`Zero`] implements all Hamiltonian traits and represents $` H=0 `$.
//!
//! [`Isotropic`]: pairwise::Isotropic
//! [`HardShape`]: pairwise::HardShape
//!
//! # Complete documentation
//!
//! `hoomd-interaction` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://glotzerlab-hoomd-rs.readthedocs-hosted.com

use hoomd_microstate::{Body, Microstate};

pub mod external;
pub mod pairwise;
pub mod univariate;

mod external_type;
mod pairwise_cutoff;
mod zero;

pub use external_type::External;
pub use hoomd_derive::{
    DeltaEnergyInsert, DeltaEnergyOne, DeltaEnergyRemove, MaximumInteractionRange, SitePairEnergy,
    TotalEnergy,
};
pub use pairwise_cutoff::PairwiseCutoff;
pub use zero::Zero;

/// Compute the total energy of a potential applied to the microstate.
///
/// The `TotalEnergy` trait describes a type that can compute the energy of a
/// given microstate. Depending on the type, `total_energy` might compute the
/// total potential energy of the whole Hamiltonian or a single term, such as
/// the Lennard-Jones potential energy.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, SitePairEnergy, TotalEnergy, pairwise::Isotropic,
///     univariate::LennardJones,
/// };
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{Point, Position},
/// };
/// use hoomd_vector::{Cartesian, Vector};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([0.0, 0.0])),
///     Body::point(Cartesian::from([1.0, 0.0])),
///     Body::point(Cartesian::from([0.0, 5.0])),
///     Body::point(Cartesian::from([-1.0, 5.0])),
/// ])?;
///
/// let lennard_jones: LennardJones = LennardJones {
///     epsilon: 1.5,
///     sigma: 1.0 / 2.0_f64.powf(1.0 / 6.0),
/// };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: lennard_jones,
///     r_cut: 2.5,
/// });
///
/// let total_energy = pairwise_cutoff.total_energy(&microstate);
/// assert_eq!(total_energy, -3.0);
/// # Ok(())
/// # }
/// ```
///
/// # Derive macro
///
/// Use the [`TotalEnergy`](macro@TotalEnergy) derive macro to automatically implement
/// the `TotalEnergy` trait on a type. The derived implementation sums the result of
/// `total_energy` over all fields in the struct (in the order in which fields
/// are named in the struct definition). The sum short circuits and returns
/// `f64::INFINITY` when any one field returns infinity.
/// ```
/// use hoomd_interaction::{
///     External, PairwiseCutoff, TotalEnergy, external::Linear,
///     pairwise::Isotropic, univariate::Boxcar,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// #[derive(TotalEnergy)]
/// struct Hamiltonian {
///     linear: External<Linear<Cartesian<2>>>,
///     pairwise_cutoff: PairwiseCutoff<Isotropic<Boxcar>>,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([0.0, 4.0])),
///     Body::point(Cartesian::from([1.0, 4.0])),
/// ])?;
///
/// let epsilon = 2.0;
/// let (left, right) = (0.0, 1.5);
/// let boxcar = Boxcar {
///     epsilon,
///     left,
///     right,
/// };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: boxcar,
///     r_cut: right,
/// });
///
/// let linear = External(Linear {
///     alpha: 1.0,
///     plane_origin: Cartesian::default(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// });
///
/// let hamiltonian = Hamiltonian {
///     pairwise_cutoff,
///     linear,
/// };
///
/// let total_energy = hamiltonian.total_energy(&microstate);
/// assert_eq!(total_energy, 10.0);
/// # Ok(())
/// # }
/// ```
pub trait TotalEnergy<M> {
    /// Compute the energy.
    #[must_use]
    fn total_energy(&self, microstate: &M) -> f64;

    /// Compute the difference in energy between two microstates.
    ///
    /// Returns $` E_\mathrm{final} - E_\mathrm{initial} `$.
    #[inline]
    #[must_use]
    fn delta_energy_total(&self, initial_microstate: &M, final_microstate: &M) -> f64 {
        self.total_energy(final_microstate) - self.total_energy(initial_microstate)
    }
}

/// Compute the energy contribution of a single site.
///
/// The `SiteEnergy` trait describes a type that can compute the energy contribution
/// of a site to the system's total energy *as a function only of that site's
/// properties*.
///
/// The [`external`] module provides a number of commonly used implementations.
/// Combine them with [`External`] newtype for use with MC and MD simulations or to
/// compute system-wide properties.
///
/// The generic type names are:
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
///
/// ## Examples
///
/// Implement a custom site energy function:
///
/// ```
/// use hoomd_interaction::{External, SiteEnergy, TotalEnergy};
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{Point, Position},
/// };
/// use hoomd_vector::Cartesian;
///
/// struct Custom {
///     a: f64,
///     b: f64,
/// }
///
/// impl<S> SiteEnergy<S> for Custom
/// where
///     S: Position<Position = Cartesian<2>>,
/// {
///     fn site_energy(&self, site_properties: &S) -> f64 {
///         self.a * (site_properties.position()[0] / self.b).cos()
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([1.0, 0.0])),
///     Body::point(Cartesian::from([-1.0, 2.0])),
/// ])?;
///
/// let custom = Custom { a: 1.0, b: 10.0 };
/// let site_energy = custom.site_energy(&microstate.sites()[0].properties);
///
/// let custom = External(custom);
/// let total_energy = custom.total_energy(&microstate);
/// # Ok(())
/// # }
/// ```
///
/// Custom method that checks for overlaps of a disk with a circular boundary.
///
/// ```
/// use hoomd_interaction::{External, SiteEnergy, TotalEnergy};
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{Point, Position},
/// };
/// use hoomd_vector::{Cartesian, Metric};
///
/// struct Custom {
///     r: f64,
/// }
///
/// impl<S> SiteEnergy<S> for Custom
/// where
///     S: Position<Position = Cartesian<2>>,
/// {
///     fn site_energy(&self, site_properties: &S) -> f64 {
///         if site_properties.position().distance(&Cartesian::default())
///             > self.r - 0.5
///         {
///             f64::INFINITY
///         } else {
///             0.0
///         }
///     }
///
///     fn is_only_infinite_or_zero() -> bool {
///         true
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([Body::point(Cartesian::from([9.6, 0.0]))])?;
///
/// let custom = Custom { r: 10.0 };
/// let site_energy = custom.site_energy(&microstate.sites()[0].properties);
/// assert_eq!(site_energy, f64::INFINITY);
///
/// let custom = External(custom);
/// let total_energy = custom.total_energy(&microstate);
/// assert_eq!(total_energy, f64::INFINITY);
/// # Ok(())
/// # }
/// ```
pub trait SiteEnergy<S> {
    /// Evaluate the energy contribution of a single site.
    #[must_use]
    fn site_energy(&self, site_properties: &S) -> f64;

    /// Evaluate the energy contribution of a single site *in the initial state*.
    ///
    /// Override this method in potentials that have both infinite or zero
    /// terms and finite terms, such as the sum of a hard site-wall interaction
    /// plus an attractive well. `site_energy` should compute both terms and
    /// `site_energy_initial` should compute only the finite terms.
    ///
    /// [`External`] calls `site_energy_initial` when evaluating the energy of
    /// the initial state in a trial move. The infinite interaction term can be
    /// assumed 0 in the initial state because no site will ever be placed in an
    /// infinite energy configuration.
    #[must_use]
    #[inline]
    fn site_energy_initial(&self, site_properties: &S) -> f64 {
        self.site_energy(site_properties)
    }

    /// Does this potential only ever return infinity or zero?
    ///
    /// Override this method and return `true` for e.g. hard site-wall
    /// interactions that always return infinity or zero and **never** any other
    /// value. When this method returns `true`, [`External`] skips the initial
    /// energy computation and assumes it is zero.
    #[must_use]
    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        false
    }
}

/// Compute the energy contribution from a pair of sites.
///
/// The `SitePairEnergy` trait describes a type that can compute the energy
/// contribution from a pair of sites to the system's total energy *as a function
/// only of those site's properties*.
///
/// The [`pairwise`] module provides a number of commonly used implementations,
/// such as [`Isotropic`], [`Anisotropic`], and [`HardShape`]. Combine any
/// of them with the [`PairwiseCutoff`] for use with MC and MD simulations or to
/// compute system-wide properties.
///
/// The generic type names are:
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
///
/// [`Isotropic`]: pairwise::Isotropic
/// [`Anisotropic`]: pairwise::Anisotropic
/// [`HardShape`]: pairwise::HardShape
///
/// ## Examples
///
/// Implement a custom site energy method:
/// ```
/// use hoomd_interaction::{
///     MaximumInteractionRange, PairwiseCutoff, SitePairEnergy, TotalEnergy,
/// };
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{Point, Position},
/// };
/// use hoomd_vector::{Cartesian, InnerProduct};
///
/// #[derive(MaximumInteractionRange)]
/// struct Custom {
///     epsilon: f64,
///     maximum_interaction_range: f64,
/// }
///
/// impl<S> SitePairEnergy<S> for Custom
/// where
///     S: Position<Position = Cartesian<2>>,
/// {
///     fn site_pair_energy(
///         &self,
///         site_properties_i: &S,
///         site_properties_j: &S,
///     ) -> f64 {
///         self.epsilon
///             * site_properties_i
///                 .position()
///                 .dot(&site_properties_j.position())
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([1.0, 0.0])),
///     Body::point(Cartesian::from([0.0, 1.0])),
/// ])?;
///
/// let custom = Custom {
///     epsilon: 1.0,
///     maximum_interaction_range: 2.5,
/// };
/// let site_pair_energy = custom.site_pair_energy(
///     &microstate.sites()[0].properties,
///     &microstate.sites()[1].properties,
/// );
///
/// let custom = PairwiseCutoff(custom);
/// let total_energy = custom.total_energy(&microstate);
/// # Ok(())
/// # }
/// ```
///
/// Implement a custom site overlap method:
/// ```
/// use hoomd_interaction::{
///     MaximumInteractionRange, PairwiseCutoff, SitePairEnergy, TotalEnergy,
/// };
/// use hoomd_microstate::{
///     Body, Microstate, Transform,
///     property::{Point, Position},
/// };
/// use hoomd_utility::valid::PositiveReal;
/// use hoomd_vector::{Cartesian, Metric};
///
/// #[derive(Default, Position)]
/// struct CircleSiteProperties {
///     position: Cartesian<2>,
///     radius: PositiveReal,
/// }
///
/// impl Transform<CircleSiteProperties> for Point<Cartesian<2>> {
///     fn transform(
///         &self,
///         site_properties: &CircleSiteProperties,
///     ) -> CircleSiteProperties {
///         CircleSiteProperties {
///             position: self.position + site_properties.position,
///             ..*site_properties
///         }
///     }
/// }
///
/// #[derive(MaximumInteractionRange)]
/// struct PolydisperseCircleOverlap {
///     maximum_interaction_range: f64,
/// }
///
/// impl SitePairEnergy<CircleSiteProperties> for PolydisperseCircleOverlap {
///     fn site_pair_energy(
///         &self,
///         a: &CircleSiteProperties,
///         b: &CircleSiteProperties,
///     ) -> f64 {
///         let r = a.position().distance(b.position());
///
///         if r < a.radius.get() + b.radius.get() {
///             f64::INFINITY
///         } else {
///             0.0
///         }
///     }
///
///     fn is_only_infinite_or_zero() -> bool {
///         true
///     }
///
///     fn site_pair_energy_initial(
///         &self,
///         _a: &CircleSiteProperties,
///         _b: &CircleSiteProperties,
///     ) -> f64 {
///         0.0
///     }
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body {
///         properties: Point::new(Cartesian::from([0.0, 0.0])),
///         sites: vec![CircleSiteProperties {
///             position: Cartesian::from([0.0, 0.0]),
///             radius: 0.5.try_into()?,
///         }],
///     },
///     Body {
///         properties: Point::new(Cartesian::from([1.4, 0.0])),
///         sites: vec![CircleSiteProperties {
///             position: Cartesian::from([0.0, 0.0]),
///             radius: 1.0.try_into()?,
///         }],
///     },
/// ])?;
///
/// let overlap = PolydisperseCircleOverlap {
///     maximum_interaction_range: 1.5,
/// };
/// let site_pair_energy = overlap.site_pair_energy(
///     &microstate.sites()[0].properties,
///     &microstate.sites()[1].properties,
/// );
/// assert_eq!(site_pair_energy, f64::INFINITY);
///
/// let pairwise_cutoff = PairwiseCutoff(overlap);
/// let total_energy = pairwise_cutoff.total_energy(&microstate);
/// assert_eq!(total_energy, f64::INFINITY);
/// # Ok(())
/// # }
/// ```
///
/// # Derive macro
///
/// Use the [`SitePairEnergy`](macro@SitePairEnergy) derive macro to
/// automatically implement the `SitePairEnergy` trait on a type.
/// The implemented `site_pair_energy` sums the result of `site_pair_energy`
/// over all fields. The implementation returns early when any one field returns
/// infinity. The implemented `site_pair_energy_initial` behaves similarly.
/// The derived `is_only_infinite_or_zero` returns true only when all fields
/// also return true for the same method.
///
/// ```
/// use hoomd_interaction::{
///     MaximumInteractionRange, SitePairEnergy,
///     pairwise::{AngularMask, Anisotropic, HardSphere},
///     univariate::Boxcar,
/// };
/// use hoomd_vector::Cartesian;
///
/// #[derive(MaximumInteractionRange, SitePairEnergy)]
/// struct SitePairInteraction {
///     hard_disk: HardSphere,
///     angular_mask: Anisotropic<AngularMask<Boxcar, Cartesian<2>>>,
/// }
/// ```
pub trait SitePairEnergy<S> {
    /// Evaluate the energy contribution from a pair of sites.
    fn site_pair_energy(&self, site_properties_i: &S, site_properties_j: &S) -> f64;

    /// Evaluate the energy contribution from a pair of sites *in the initial state*.
    ///
    /// Override this method in potentials that have both infinite or zero terms
    /// and finite terms, such as the sum of a hard site-wall interaction plus
    /// an attractive well. `site_pair_energy` should compute both terms and
    /// `site_pair_energy_initial` should compute only the finite terms.
    ///
    /// [`PairwiseCutoff`] calls `site_pair_energy_initial` when evaluating the
    /// energy of the initial state in a trial move. The infinite interaction
    /// term can be assumed 0 in the initial state because no site will ever be
    /// placed in an infinite energy configuration.
    #[must_use]
    #[inline]
    fn site_pair_energy_initial(&self, site_properties_i: &S, site_properties_j: &S) -> f64 {
        self.site_pair_energy(site_properties_i, site_properties_j)
    }

    /// Does this potential only ever return infinity or zero?
    ///
    /// Override this method and return `true` for e.g. hard particle
    /// interactions that always return infinity or zero and **never** any other
    /// value. When this method returns `true`, [`PairwiseCutoff`] skips the
    /// initial energy computation and assumes it is zero.
    #[must_use]
    #[inline]
    fn is_only_infinite_or_zero() -> bool {
        false
    }
}

/// Largest distance between two sites where the pairwise interaction may be non-zero.
///
/// [`PairwiseCutoff`] uses the provided maximum interaction range to
/// efficiently compute only the needed interactions. All types that implement
/// `SitePair*` traits must also implement [`MaximumInteractionRange`].
///
/// # Derive macro
///
/// Use the [`MaximumInteractionRange`](macro@MaximumInteractionRange) derive macro to
/// automatically implement the `MaximumInteractionRange` trait on a type.
///
/// When the type has a field named `maximum_interaction_range`, the derived implementation
/// returns it. When there is no such field, the derived implementation takes
/// the maximum of the `maximum_interaction_range` over all fields in the struct.
/// The former case is intended for use with custom site pair potentials and
/// the latter is intended for use with multi-term Hamiltonian types.
/// ```
/// use hoomd_interaction::{
///     External, MaximumInteractionRange, PairwiseCutoff, external::Linear,
/// };
/// use hoomd_vector::Cartesian;
///
/// #[derive(MaximumInteractionRange)]
/// struct SitePairInteraction {
///     // ...
///     maximum_interaction_range: f64,
/// }
///
/// #[derive(MaximumInteractionRange)]
/// struct Hamiltonian {
///     linear: External<Linear<Cartesian<2>>>,
///     pairwise_cutoff: PairwiseCutoff<SitePairInteraction>,
/// }
/// ```
pub trait MaximumInteractionRange {
    /// The largest distance between two sites where the pairwise interaction may be non-zero.
    fn maximum_interaction_range(&self) -> f64;
}

/// Compute the change in energy as a function of a single modified body.
///
/// Some trial moves apply to a single body at a time and use a Hamiltonian that
/// implements `DeltaEnergyOne` to efficiently compute the change in energy.
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
///
/// See the [Implementors](#implementors) section below for examples.
///
/// # Derive macro
///
/// Use the [`DeltaEnergyOne`](macro@DeltaEnergyOne) derive macro to
/// automatically implement the `DeltaEnergyOne` trait on a type. The derived
/// implementation sums the result of `delta_energy_one` over all fields in the
/// struct (in the order in which fields are named in the struct definition).
/// The sum short circuits and returns `f64::INFINITY` when any one field
/// returns infinity.
/// ```
/// use hoomd_interaction::{
///     DeltaEnergyOne, External, PairwiseCutoff, external::Linear,
///     pairwise::Isotropic, univariate::Boxcar,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// #[derive(DeltaEnergyOne)]
/// struct Hamiltonian {
///     linear: External<Linear<Cartesian<2>>>,
///     pairwise_cutoff: PairwiseCutoff<Isotropic<Boxcar>>,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([
///     Body::point(Cartesian::from([0.0, 0.0])),
///     Body::point(Cartesian::from([1.0, 0.0])),
/// ])?;
///
/// let epsilon = 2.0;
/// let (left, right) = (0.0, 1.5);
/// let boxcar = Boxcar {
///     epsilon,
///     left,
///     right,
/// };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: boxcar,
///     r_cut: right,
/// });
///
/// let linear = External(Linear {
///     alpha: 10.0,
///     plane_origin: Cartesian::default(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// });
///
/// let hamiltonian = Hamiltonian {
///     pairwise_cutoff,
///     linear,
/// };
///
/// let delta_energy = hamiltonian.delta_energy_one(
///     &microstate,
///     0,
///     &Body::point([-1.0, 0.0].into()),
/// );
/// assert_eq!(delta_energy, -2.0);
/// # Ok(())
/// # }
/// ```
pub trait DeltaEnergyOne<B, S, X, C> {
    /// Compute the change in energy.
    ///
    /// `initial_microstate` describes the initial configuration and `final_body`
    /// describes the new body configuration. In the final configuration, the
    /// body may have changed properties and/or sites. The index `body_index`
    /// identifies which body in `initial_microstate` is changing.
    ///
    /// Returns:
    /// ```math
    /// \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    /// ```
    #[must_use]
    fn delta_energy_one(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
        final_body: &Body<B, S>,
    ) -> f64;
}

/// Compute the change in energy when a single body is inserted.
///
/// Some trial moves insert a single body at a time and use a Hamiltonian that
/// implements `DeltaEnergyInsert` to efficiently compute the change in energy.
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `X`: The spatial data structure type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
///
/// See the [Implementors](#implementors) section below for examples.
///
/// # Derive macro
///
/// Use the [`DeltaEnergyInsert`](macro@DeltaEnergyInsert) derive macro to
/// automatically implement the `DeltaEnergyInsert` trait on a type. The derived
/// implementation sums the result of `delta_energy_insert` over all fields in the
/// struct (in the order in which fields are named in the struct definition).
/// The sum short circuits and returns `f64::INFINITY` when any one field
/// returns infinity.
/// ```
/// use hoomd_interaction::{
///     DeltaEnergyInsert, External, PairwiseCutoff, external::Linear,
///     pairwise::Isotropic, univariate::Boxcar,
/// };
/// use hoomd_microstate::{Body, Microstate, property::Point};
/// use hoomd_vector::Cartesian;
///
/// #[derive(DeltaEnergyInsert)]
/// struct Hamiltonian {
///     linear: External<Linear<Cartesian<2>>>,
///     pairwise_cutoff: PairwiseCutoff<Isotropic<Boxcar>>,
/// }
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut microstate = Microstate::new();
/// microstate.extend_bodies([Body::point(Cartesian::from([0.0, 4.0]))])?;
///
/// let epsilon = 2.0;
/// let (left, right) = (0.0, 1.5);
/// let boxcar = Boxcar {
///     epsilon,
///     left,
///     right,
/// };
/// let pairwise_cutoff = PairwiseCutoff(Isotropic {
///     interaction: boxcar,
///     r_cut: right,
/// });
///
/// let linear = External(Linear {
///     alpha: 1.0,
///     plane_origin: Cartesian::default(),
///     plane_normal: [0.0, 1.0].try_into()?,
/// });
///
/// let hamiltonian = Hamiltonian {
///     pairwise_cutoff,
///     linear,
/// };
///
/// let new_body = Body::point(Cartesian::from([1.0, 4.0]));
/// let delta_energy = hamiltonian.delta_energy_insert(&microstate, &new_body);
/// assert_eq!(delta_energy, 6.0);
/// # Ok(())
/// # }
/// ```
pub trait DeltaEnergyInsert<B, S, X, C> {
    /// Compute the change in energy.
    ///
    /// `initial_microstate` describes the initial configuration and `new_body`
    /// describes the new body configuration. The final configuration includes
    /// all bodies in the initial microstate and `new_body`.
    ///
    /// Returns:
    /// ```math
    /// \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    /// ```
    #[must_use]
    fn delta_energy_insert(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        new_body: &Body<B, S>,
    ) -> f64;
}

/// Compute the change in energy when a single body is removed.
///
/// Some trial moves remove a single body at a time and use a Hamiltonian that
/// implements `DeltaEnergyRemove` to efficiently compute the change in energy.
///
/// The generic type names are:
/// * `B`: The [`Body::properties`](hoomd_microstate::Body) type.
/// * `S`: The [`Site::properties`](hoomd_microstate::Site) type.
/// * `C`: The [`boundary`](hoomd_microstate::boundary) condition type.
///
/// See the [Implementors](#implementors) section below for examples.
///
/// # Derive macro
///
/// Use the [`DeltaEnergyRemove`](macro@DeltaEnergyRemove) derive macro to
/// automatically implement the `DeltaEnergyRemove` trait on a type. The derived
/// implementation sums the result of `delta_energy_remove` over all fields in the
/// struct (in the order in which fields are named in the struct definition).
/// The sum short circuits and returns `f64::INFINITY` when any one field
/// returns infinity.
/// ```
/// use hoomd_interaction::{
///     DeltaEnergyRemove, External, PairwiseCutoff, external::Linear,
///     pairwise::Isotropic, univariate::Boxcar,
/// };
/// use hoomd_vector::Cartesian;
///
/// #[derive(DeltaEnergyRemove)]
/// struct Hamiltonian {
///     linear: External<Linear<Cartesian<2>>>,
///     pairwise_cutoff: PairwiseCutoff<Isotropic<Boxcar>>,
/// }
/// ```
pub trait DeltaEnergyRemove<B, S, X, C> {
    /// Compute the change in energy.
    ///
    /// `initial_microstate` describes the initial configuration and `body_index` is
    /// the index of the body to remove. The final configuration includes all bodies
    /// in the initial microstate except the body previously at `body_index`.
    ///
    /// Returns:
    /// ```math
    /// \Delta E = E_\mathrm{final} - E_\mathrm{initial}
    /// ```
    #[must_use]
    fn delta_energy_remove(
        &self,
        initial_microstate: &Microstate<B, S, X, C>,
        body_index: usize,
    ) -> f64;
}
