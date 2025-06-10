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
mod minkowski;

pub use {
    minkowski::Minkowski,
};

use thiserror::Error;
use hoomd_vector::{Vector, InnerProduct};

// / Enumerate possible sources of error in fallible vector math operations.
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

/** Operations defined on the upper sheet of a two-sheeted hyperboloid embedded in some metric 
vector space.
 */
pub trait Hyperboloid {
    /** Distance of the geodesic path passing through two points on the hyperboloid.
     */
    #[inline]
    fn hyperbolic_distance(&self, other: &Self, skirt: f64) -> f64;
}

/** Rotations in hyperbolic space
 */
pub trait HyperbolicRotate<V: Vector> {
    #[must_use]
    fn boost(&self, vector: &V) -> V;
}
/** Sphere
 */
pub trait Sphere: InnerProduct {
    /** Distance of the geodesic path passing through two points on the sphere.
     */
    #[inline]
    fn sphere_distance(&self, other: &Self, radius: f64) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_add_generic<T>(a: T, b: T) -> T
    where
        T: Vector,
    {
        a + b
    }

    #[test]
    fn add_generic() {
        let a = Minkowski::from([1.0, 2.0, 3.0]);
        let b = Minkowski::from([4.0, 5.0, 6.0]);
        let c = compute_add_generic(a, b);
        assert_eq!(c, [5.0, 7.0, 9.0].into());
    }
}