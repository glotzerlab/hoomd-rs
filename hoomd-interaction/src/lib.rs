// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! Particle interactions and physical models that apply to microstates.
//!
//! TODO: Expand documentation.

use hoomd_microstate::{Body, Microstate};

pub mod external;
pub mod pairwise;

mod pairwise_cutoff;
mod external_type;
mod hamiltonian;
mod zero;

pub use pairwise_cutoff::PairwiseCutoff;
pub use external_type::External;
pub use zero::Zero;

/// Compute the total energy of a potential applied to the microstate.
///
/// The `TotalEnergy` trait describes a type that can compute the energy of a
/// given microstate. Depending on the type, `total_energy` might compute the total
/// potential energy of the system or a single term, such as the Lennard-Jones
/// potential energy.
///
/// # Example
///
/// ```
/// use hoomd_interaction::{
///     PairwiseCutoff, SitePairEnergy, TotalEnergy,
///     pairwise::{Isotropic, LennardJones},
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
/// let lennard_jones = Isotropic(lennard_jones);
/// let cutoff_pair = PairwiseCutoff {
///     r_cut: 2.5,
///     evaluator: lennard_jones,
/// };
///
/// let total_energy = cutoff_pair.total_energy(&microstate);
/// assert_eq!(total_energy, -3.0);
/// # Ok(())
/// # }
/// ```
pub trait TotalEnergy<M> {
    /// Compute the energy.
    #[must_use]
    fn total_energy(&self, microstate: &M) -> f64;
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
/// let custom_evaluator = Custom { a: 1.0, b: 10.0 };
/// let site_energy =
///     custom_evaluator.site_energy(&microstate.sites()[0].properties);
///
/// let custom = External(custom_evaluator);
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
/// let custom_evaluator = Custom { r: 10.0 };
/// let site_energy =
///     custom_evaluator.site_energy(&microstate.sites()[0].properties);
/// assert_eq!(site_energy, f64::INFINITY);
///
/// let custom = External(custom_evaluator);
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
/// use hoomd_interaction::{PairwiseCutoff, SitePairEnergy, TotalEnergy};
/// use hoomd_microstate::{
///     Body, Microstate,
///     property::{Point, Position},
/// };
/// use hoomd_vector::{Cartesian, InnerProduct};
///
/// struct Custom {
///     epsilon: f64,
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
/// let evaluator = Custom { epsilon: 1.0 };
/// let site_pair_energy = evaluator.site_pair_energy(
///     &microstate.sites()[0].properties,
///     &microstate.sites()[1].properties,
/// );
///
/// let custom = PairwiseCutoff {
///     r_cut: 2.5,
///     evaluator,
/// };
/// let total_energy = custom.total_energy(&microstate);
/// # Ok(())
/// # }
/// ```
///
/// Implement a custom site overlap method:
/// ```
/// use hoomd_geometry::{IntersectsAt, shape::Circle};
/// use hoomd_interaction::{PairwiseCutoff, SitePairEnergy, TotalEnergy};
/// use hoomd_microstate::{
///     Body, Microstate, Transform,
///     property::{Point, Position},
/// };
/// use hoomd_utility::valid::PositiveReal;
/// use hoomd_vector::{self, Angle, Cartesian};
///
/// #[derive(Default)]
/// struct CircleSiteProperties {
///     position: Cartesian<2>,
///     radius: PositiveReal,
/// }
///
/// impl Position for CircleSiteProperties {
///     type Position = Cartesian<2>;
///
///     fn position(&self) -> &Cartesian<2> {
///         &self.position
///     }
///
///     fn position_mut(&mut self) -> &mut Cartesian<2> {
///         &mut self.position
///     }
/// }
///
/// impl Transform<CircleSiteProperties> for Point<Cartesian<2>> {
///     fn transform(
///         &self,
///         site_properties: &CircleSiteProperties,
///     ) -> CircleSiteProperties {
///         CircleSiteProperties {
///             position: self.position + site_properties.position,
///             radius: site_properties.radius,
///         }
///     }
/// }
///
/// struct PolydisperseCircleOverlap;
///
/// impl SitePairEnergy<CircleSiteProperties> for PolydisperseCircleOverlap {
///     fn site_pair_energy(
///         &self,
///         a: &CircleSiteProperties,
///         b: &CircleSiteProperties,
///     ) -> f64 {
///         let circle_a = Circle { radius: a.radius };
///         let circle_b = Circle { radius: b.radius };
///         let (v_ij, o_ij) = hoomd_vector::pair_system_to_local(
///             a.position(),
///             &Angle::default(),
///             b.position(),
///             &Angle::default(),
///         );
///         if circle_a.intersects_at(&circle_b, &v_ij, &o_ij) {
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
/// let evaluator = PolydisperseCircleOverlap;
/// let site_pair_energy = evaluator.site_pair_energy(
///     &microstate.sites()[0].properties,
///     &microstate.sites()[1].properties,
/// );
/// assert_eq!(site_pair_energy, f64::INFINITY);
///
/// let cutoff_pair = PairwiseCutoff {
///     r_cut: 1.5,
///     evaluator,
/// };
/// let total_energy = cutoff_pair.total_energy(&microstate);
/// assert_eq!(total_energy, f64::INFINITY);
/// # Ok(())
/// # }
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

/// Compute the change energy as a function of a single modified body.
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

/// Compute the change energy when a single body is inserted.
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

/// Compute the change energy when a single body is removed.
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

// TODO: More doc examples for all implementors.
