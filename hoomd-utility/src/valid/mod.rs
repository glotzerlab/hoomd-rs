// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Ensure that values are in well-defined ranges.

mod open_unit_interval_number;
mod positive_real;

use thiserror::Error;

pub use open_unit_interval_number::OpenUnitIntervalNumber;
pub use positive_real::PositiveReal;

/// Enumerate possible sources of error in fallible validation methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// A positive value greater than 0 is required.
    #[error("{0} is not greater than 0")]
    NotPositive(f64),

    /// A finite value is required.
    #[error("{0} is not finite")]
    NotFinite(f64),

    /// An value in (0,1) is required.
    #[error("{0} is not in (0,1)")]
    NotInOpenUnitInterval(f64),
}
