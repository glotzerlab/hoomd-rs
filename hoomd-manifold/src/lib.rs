// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

/*! Tools for non-Euclidean geometries

Description of `hoomd-manifold`

## Sphere

Description of 2-sphere embedded in three-dimensional cartesian space.

## Hyperboloid

Description of 1-sheeted hyperboloid embedded in three-dimensional cartesian
space.

## Minkowski

Description of N-dimensional Minkowski space. 
*/

mod sphere;
mod hyperboloid;
mod minkowski;

pub use {
    sphere::Sphere,
    hyperboloid::Hyperboloid,
    minkowski::Minkowski,
};

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use thiserror::Error;

/// Enumerate possible sources of error in fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Attempted to use sphere operations on vectors not confined to the sphere.
    #[error("coordinate outside of sphere")]
    InvalidSphereCoordinate,

    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("source length does not match the target dimensions")]
    InvalidVectorLength,
}

pub trait Geodesic: {
    /** Compute the length of a geodesic passing through two points on a metric space.

    # Example 
    ```
    use hoomd_manifold::{Sphere, Geodesic};

    ```
    */
    #[inline]
    fn geodesic_distance(&self, other: &Self) -> f64;
}