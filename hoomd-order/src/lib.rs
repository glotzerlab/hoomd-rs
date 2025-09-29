// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! TODO

use thiserror::Error;

/// Enumerate possible sources of error in fallible order parameter calculations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// The two point sets do not have the same length.
    #[error("The two point sets do not have the same length.")]
    MismatchedPointSetSize,
    /// A matrix that was expected to be unitary was not.
    #[error("A matrix that was expected to be unitary was not.")]
    NonUnitaryMatrix,
}

mod cross_covariance;
pub use cross_covariance::CrossCovariance;

pub mod template_matching;
// pub use template_matching:;
