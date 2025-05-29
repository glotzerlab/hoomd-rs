// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Utilities

Common utility code used by many other hoomd-rs crates.
*/

pub mod random;
pub mod valid;

use thiserror::Error;

/// Enumerate possible sources of error in fallible utility methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// A positive value greater than 0 is required.
    #[error("{0} is not greater than 0")]
    NotPositive(f64),

    /// A finite value is required.
    #[error("{0} is not finite")]
    NotFinite(f64),
}
