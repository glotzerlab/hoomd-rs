// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Order parameters and other analysis methods.
//!
//! `hoomd-order` implements order parameter, correlation function, and other
//! analysis methods along with the infrastructure that makes them possible.
//! `hoomd-order` is similar to the [`freud`] Python package.
//!
//! # Local order parameters
//!
//! A *local order parameter* is a function only of the sites in the space near
//!  a given point $` \vec{r} `$. That point may or may not be the position of
//! another site. Local order parameters typically have a function signature
//! like this:
//!
//! ```ignore
//! fn local_order_parameter<I>(r: &Cartesian<3>, neighbors: I) -> Output
//! where
//!     I: IntoIterator<Item = Cartesian<3>>
//! ```
//!
//! The properties of the point $` \vec{r} `$ are passed as arguments and the
//! neighborhood is given by an iterator. The iterator should produce all the
//! sites of interest in the local space. The signature for some order parameters
//! might include additional parameters. For example, some order parameters are
//! a function of both position and orientation. In that case, the iterator would
//! produce tuples: `(position, orientation)`.
//!
//! Each type's documentation includes complete examples:
//! * [`KAtic`]: 2D hexatic, tetratic, etc... order parameter.
//! * [`Steinhardt`]: 3D Steinhardt order parameter.
//! * [`RotationalAutocorrelation`]: 3D rotational autocorrelation function.
//!
//! # Neighbor queries
//!
//! You provide neighbors as iterators, so you have infinite flexibility to
//! compute the local order parameters however you need. You can provide a
//! fixed list, pull neighbors from a list of bonds, use a Voronoi
//! construction, or provide them with any code you can write.
//!
//! `hoomd-order` implements the standard neighbor query methods.
//! Each type's documentation includes complete examples:
//! * [`SitesInBall`]: Find all sites in a [`Microstate`] with positions inside
//!   a ball of radius *r*.
//!
//! You can fine tune these queries by calling `iter_sites` and apply further
//! filtering, mapping, or otherwise manipulating the iterator chain.
//!
//! # Handling results
//!
//! Wherever possible, `hoomd-order` operates directly on points in space
//! without reference to a a specific microstate or any notion of site
//! identity. This leaves you free to handle the results optimally for your
//! application. For example, a metadynamics simulation might compute an
//! order parameter before and after a trial move.
//!
//! In other situations, you might want to store the resulting order parameter
//! for every site in the system. [`freud`] always stores results in numpy
//! arrays with a length equal to the number of particles. You could
//! follow that practice with `hoomd-order` and store results in a [`Vec`].
//! However, there is a fundamental problem with this approach: it loses
//! track of site identity from one frame to the next. When a site is removed
//! from a microstate, its tag is no longer present yet a `Vec<f64>` must store
//! a value at every index.
//!
//! To fix this problem, you could instead store results in a [`HashMap`] using
//! site tags as the keys. The various algorithms (see below) in `hoomd-order`
//! will work with either method. When you work with [`Vec`], the tags *must*
//! be identical to the [`Vec`] indices.
//!
//! Of course, you are not limited to just [`Vec`] and [`HashMap`]. For
//! example, you could store results in a [`polars`] data frame.
//!
//! [`polars`]: https://docs.pola.rs
//!
//! # Algorithms
//!
//! `hoomd-order` implements some commonly used algorithms that operate on
//! systems as a whole:
//! * [`AverageOverNeighbors`]: Average a quantity over the neighbors around a site.
//!
//! # Tips
//!
//! When using a microstate for both simulation and analysis, set the maximum
//! interaction range to the larger of the model's maximum interaction range and
//! the largest neighbor distance you will use when computing order parameters.
//! You may get better performance setting `nominal_search_radius` to the
//! model's interaction range (when more time is spent in evaluating the model)
//! or the overall maximum range (when more time is spent evaluating order
//! parameters).
//!
//! [`freud`]: https://freud.readthedocs.io
//! [`Microstate`]: hoomd_microstate::Microstate
//! [`HashMap`]: std::collections::HashMap

use hoomd_vector::Cartesian;
use thiserror::Error;

/// Enumerate possible sources of error in fallible hoomd-order methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error<V = Cartesian<3>> {
    /// The point is outside the microstate's boundary.
    #[error("The point {0} is outside the microstate's boundary")]
    PointOutsideBoundary(V),

    /// `r_j` - `r` is invalid.
    #[error("{0} - {1} is an invalid delta r")]
    InvalidDeltaR(V, V, #[source] hoomd_vector::Error),

    /// Neighbor site tag missing in value map.
    #[error(
        "the site with tag {0} (a neighbor of the site tag {1}) is not present in the value map"
    )]
    SiteTagMissing(usize, usize),
}

pub mod math;

mod k_atic;
pub use k_atic::KAtic;

mod steinhardt;
pub use steinhardt::Steinhardt;

mod rotational_autocorrelation;
pub use rotational_autocorrelation::RotationalAutocorrelation;

mod sites_in_ball;
pub use sites_in_ball::SitesInBall;

mod average_over_neighbors;
pub use average_over_neighbors::AverageOverNeighbors;
