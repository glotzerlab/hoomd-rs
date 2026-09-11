// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO: Overview and examples
//! # Tips
//!
//! When using a microstate for both simulation and analysis, set the maximum
//! interaction range to the larger of the model's maximum interaction range and
//! the largest neighbor distance you will use when computing order parameters.
//! You may get better performance setting `nominal_search_radius` to the
//! model's interaction range (when more time is spent in evaluating the model)
//! or the overall maximum range (when more time is spent evaluating order
//! parameters).

use hoomd_vector::Cartesian;
use thiserror::Error;

/// Enumerate possible sources of error in fallible hoomd-order methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Failed to convert `r_j` - `r` to a unit vector.
    #[error("{0} - {1} could not be converted to a unit vector")]
    InvalidDeltaR3(Cartesian<3>, Cartesian<3>, #[source] hoomd_vector::Error),
    /// Neighbor site tag missing in value map.
    #[error(
        "the site with tag {0} (a neighbor of the site tag {1}) is not present in the value map"
    )]
    SiteTagMissing(usize, usize),
}

pub mod math;

mod k_atic_psi;
pub use k_atic_psi::k_atic_psi;

mod steinhardt;
pub use steinhardt::Steinhardt;

mod sites_in_ball;
pub use sites_in_ball::SitesInBall;

mod average_over_neighbors;
pub use average_over_neighbors::average_over_neighbors;
