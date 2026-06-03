// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/glotzerlab/hoomd-rs/7352214172a490cc716492e9724ff42720a0018a/doc/theme/favicon.svg"
)]

//! Tools for non-Euclidean geometries.
//!
//! ## Spherical points
//!
//! [`Spherical<N>`] describes a point on an $`(N-1)`$-sphere of radius R embedded in
//! [`Cartesian<N>`]. The components of a point on an $`N`$-sphere
//! satisfy
//! ```math
//! \sum_{i=1}^{N+1}x_i^2 = R^2
//! ```
//! [`Spherical`] implements a distance metric through the trait
//! [`Metric`] which calculates the geodesic distance on the
//! surface of an $`N`$-sphere. Use [`Spherical`] to describe spaces with constant
//! positive curvature.
//!
//! [`Cartesian<N>`]: hoomd_vector::Cartesian
//! [`Metric`]: hoomd_vector::Metric
//!
//! ## Hyperbolic
//! [`Hyperbolic`] describes a point on the upper sheet of an $`N`$-dimensional
//! two-sheeted hyperboloid embedded in $`(N+1)`$-dimensional Minkowski
//! space. The components of a point on the Hyperbolic satisfy
//! ```math
//! x_1^2 + \cdots + x_{N-1}^2 - x_{N}^2 = -1
//! ```
//! [`Hyperbolic`] implements a distance metric through the trait
//! [`Metric`] which calculates the geodesic distance on the
//! surface of a Hyperboloid. Use [`Hyperbolic`] embedded in [`Minkowski`]
//! to implement hyperbolic space.
//!
//! ## Minkowski
//!
//! [`Minkowski<N>`] implements $`(N-1,1)`$-dimensional Minkowski space with the
//! metric signature $`(+ \;\cdots\; +\; -)`$. [`Minkowski`] supports
//! [`Vector`] operations such as vector addition and rescaling, but is not a
//! true inner product space. The distance metric on Minkowski space is given
//! by the "spacetime interval"
//! ```math
//! d_M^2(\vec{u},\vec{v}) = (\vec{u}-\vec{v})^T \eta (\vec{u}-\vec{v})
//! = (u_1-v_1)^2 +\cdots + (u_{N-1}-v_{N-1})^2 - (u_N - v_N)^2
//! ```
//!
//! ```
//! use hoomd_manifold::Minkowski;
//! use hoomd_vector::{Metric, Vector};
//!
//! let u = Minkowski::from([1.0, 0.0, 0.0, -1.0]);
//! let v = Minkowski::from([2.0, 0.0, 1.0, 1.0]);
//! let w = Minkowski::from([0.0, 0.0, 0.0, 3.0]);
//! let del = (u + w).distance(&v);
//! assert_eq!(1.0, del);
//! ```
//! ## Hyperbolic Rotations
//! "Hyperbolic rotations" describe elements of the group $`SO(N,1)`$, which
//! preserve hyperboloids embedded in [`Minkowski`]. Transformations include
//! pure spatial rotations as well as "boosts".
//!
//! For two-dimensional hyperbolic surfaces, use [`HyperbolicAngle`] to
//! implement elements of $`SO(2,1)`$ which rotate points about the z-axis or boost
//! points along the x- and y-axes. Use [`HyperbolicRotationMatrix`] to
//! generate the matrix from the values defined by [`HyperbolicAngle`].
//!
//! Rotation about the z axis:
//! ```
//! use hoomd_manifold::{
//!     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
//! };
//! use std::f64::consts::PI;
//!
//! let v = Minkowski::from([1.0, 0.0, 1.0]);
//! let rotation_about_z = HyperbolicAngle::from((PI / 2.0, 0.0_f64, 0.0_f64));
//! let matrix = HyperbolicRotationMatrix::from(rotation_about_z);
//! let rotated = matrix.hyperbolic_rotate(&v);
//! // rotated is approximately [0.0,1.0,1.0]);
//! ```
//!
//! Boost in the y direction:
//! ```
//! use hoomd_manifold::{
//!     HyperbolicAngle, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
//! };
//! use std::f64::consts::PI;
//!
//! let v = Minkowski::from([1.0, 0.0, 1.0]);
//! let boost_in_y = HyperbolicAngle::from((0.0_f64, 0.0_f64, 0.5_f64));
//! let matrix = HyperbolicRotationMatrix::from(boost_in_y);
//! let boosted = matrix.hyperbolic_rotate(&v);
//! // rotated is approximately [1.0,sinh(0.5),cosh(0.5)]);
//! ```
//!
//! For three-dimensional hyperbolic surfaces, use [`Biquaternion`].
//! Biquaternions are a generalization of quaternions which allow for complex
//! coefficients. Unit biquaternions give a representation of $`SO(3,1)`$; this can
//! either be done by converting the biquaternions to a
//! [`HyperbolicRotationMatrix`] or by using the [`UnitBiquaternion`] algebra
//! directly.
//!
//! Rotate point in 3D hyperbolic space about z axis using matrix
//! representation:
//! ```
//! use hoomd_manifold::{
//!     Biquaternion, HyperbolicRotate, HyperbolicRotationMatrix, Minkowski,
//!     UnitBiquaternion,
//! };
//! use num::complex::Complex;
//! use std::f64::consts::PI;
//!
//! let q = Biquaternion::from([
//!     Complex::new((PI / 4.0).sin(), 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new((PI / 4.0).cos(), 0.0),
//! ]);
//! let v = q.to_unit_unchecked();
//! let x = Minkowski::from([0.0, 1.0, 0.0, 1.0]);
//! let rotation_about_x = HyperbolicRotationMatrix::from(v);
//! let rotated = rotation_about_x.hyperbolic_rotate(&x);
//! // rotated vector is approximately [0.0, 0.0, 1.0, 1.0];
//! ```
//!
//! Boost point in 3D hyperbolic space in x direction using biquaternion
//! algebra:
//! ```
//! use approxim::assert_relative_eq;
//! use hoomd_manifold::{
//!     Biquaternion, HyperbolicRotate, Minkowski, UnitBiquaternion,
//! };
//! use num::complex::Complex;
//! use std::f64::consts::PI;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let x = Minkowski::from([0.0, 0.0, 0.0, 1.0]);
//! let q = Biquaternion::from([
//!     Complex::new(0.0, (0.2_f64).sinh()),
//!     Complex::new(0.0, 0.0),
//!     Complex::new(0.0, 0.0),
//!     Complex::new((0.2_f64).cosh(), 0.0),
//! ]);
//! let v = q.to_unit()?;
//! let boosted = v.hyperbolic_rotate(&x);
//! assert_relative_eq!(
//!     boosted,
//!     [(0.4_f64).sinh(), 0.0, 0.0, (0.4_f64).cosh()].into(),
//!     epsilon = 1e-12
//! );
//! # Ok(())
//! # }
//! ```
//!
//! # Complete documentation
//!
//! `hoomd-manifold` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://hoomd-rs.readthedocs.io

mod biquaternion;
mod hyperbolic_angle;
mod minkowski;
mod sphere;

pub use biquaternion::{Biquaternion, UnitBiquaternion};
pub use hyperbolic_angle::HyperbolicAngle;
pub use minkowski::{Hyperbolic, HyperbolicDisk, HyperbolicRotationMatrix, Minkowski};
pub use sphere::{Spherical, SphericalDisk};

use hoomd_vector::Vector;
use thiserror::Error;

/// Enumerate possible sources of error in fallible vector math operations.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Attempted to normalize a norm zero biquaternion
    #[error("Biquaternion with norm zero cannot be normalized")]
    InvalidBiquaternionMagnitude,

    /// Attempted converting a value to a vector with a dimension not equal to the value's length.
    #[error("source length does not match the target dimensions")]
    InvalidVectorLength,
}

/// Linear transformations preserving hyperboloids.
pub trait HyperbolicRotate<V: Vector> {
    /// Type of the related rotation matrix
    type Matrix: HyperbolicRotate<V>;
    /// Apply a SO(N-1,1) transformation to an N-dimensional Minkowski vector
    #[must_use]
    fn hyperbolic_rotate(&self, vector: &V) -> V;
}
