// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

#![doc(
    html_favicon_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]
#![doc(
    html_logo_url = "https://hoomd-blue.readthedocs.io/en/latest/_static/hoomdblue-logo-favicon.svg"
)]

//! General, performant computational geometry code.
//!
//! `hoomd_geometry` implements common operations for widely-used geometric
//! primitives, with additional functionality to accommodate hard-particle Monte
//! Carlo simulations.
//!
//! ## Geometric Primitives
//!
//! The [`Hypersphere`][shape::Hypersphere] demonstrates the design philosophy of
//! `hoomd_geometry`. The struct contains a single radius value, and immediately
//! provides access to a variety of methods. Hyperspheres are well defined in
//! arbitrary dimension, and therefore the implementation is parameterized with a
//! const generic `N` representing the embedding dimension:
//! ```
//! use approx::assert_relative_eq;
//! use hoomd_geometry::{Volume, shape::Hypersphere};
//! use std::f64::consts::PI;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! const N: usize = 3;
//! let s = Hypersphere::<N>::with_radius(1.0.try_into()?);
//! let volume = s.volume();
//! assert_relative_eq!(volume, (4.0 / 3.0 * PI));
//! # Ok(())
//! # }
//! ```
//!
//! ## Traits
//! [`Volume`] provides a notion of the amount of space a primitive
//! occupies, and indicates the N-hypervolume of a given struct. For a
//! [`Rectangle`][shape::Rectangle], for example, [`Volume`] returns the area in the
//! plane, and for a [`Sphere`][shape::Sphere] returns the three-dimensional volume.
//!
//! [`IntersectsAt`] determines if there is an intersection between two shapes,
//! where the second shape is placed in the coordinate system of the first.
//! This is the most efficient way to test for intersections in Monte Carlo
//! simulations as only the positions and orientations of the sites need to be
//! modified.
//!
//! [`IsPointInside`] checks if a point is inside or outside a shape.
//!
//! Many shapes implement the `Distribution` trait from **rand** to randomly sample
//! interior points.
//!
//! ## Intersection Tests
//!
//! For non-orientable shapes, or for bodies who have special intersection
//! tests for particular orientations, and inherent method `intersects` can be
//! implemented as well:
//! ```
//! use hoomd_geometry::{Convex, IntersectsAt, shape::Sphere};
//! use hoomd_vector::Versor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let s0 = Sphere {
//!     radius: 1.0.try_into()?,
//! };
//! let s1 = Sphere {
//!     radius: 1.0.try_into()?,
//! };
//!
//! let q_id = Versor::default();
//!
//! assert_eq!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id), true);
//! assert_eq!(s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id), false);
//! # Ok(())
//! # }
//! ```
//!
//! Any pair of shapes (with possibly different types) that both implement the
//! [`SupportMapping`] trait can be tested for overlaps through the  [`Convex`]
//! newtype. [`IntersectsAt`] uses the [`xenocollide`] algorithm, provided for
//! 2d and 3d shapes, to test for intersections between [`Convex`] shapes:
//! ```
//! use hoomd_geometry::{
//!     Convex, IntersectsAt,
//!     shape::{Cuboid, Sphere},
//! };
//! use hoomd_vector::Versor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sphere = Convex(Sphere {
//!     radius: 1.0.try_into()?,
//! });
//! let cuboid = Convex(Cuboid {
//!     edge_lengths: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?],
//! });
//!
//! assert_eq!(
//!     sphere.intersects_at(
//!         &cuboid,
//!         &[1.9, 0.0, 0.0].into(),
//!         &Versor::default()
//!     ),
//!     true
//! );
//! assert_eq!(
//!     sphere.intersects_at(
//!         &cuboid,
//!         &[2.1, 0.0, 0.0].into(),
//!         &Versor::default()
//!     ),
//!     false
//! );
//! # Ok(())
//! # }
//! ```

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::InnerProduct;
use thiserror::Error;

mod convex;
pub use convex::Convex;

pub mod shape;
pub mod xenocollide;

/// The N-hypervolume of a geometry. In 2D, this is area and in 3D this is Volume.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{Volume, shape::Hypersphere};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// const N: usize = 3;
/// let s = Hypersphere::<N>::with_radius(1.0.try_into()?);
/// let volume = s.volume();
/// # Ok(())
/// # }
/// ```
///
pub trait Volume {
    /// The N-hypervolume of a geometry.
    #[must_use]
    fn volume(&self) -> f64;
}

/// Find the point on a shape that is the furthest in a given direction.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{SupportMapping, shape::Cuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Cuboid {
///     edge_lengths: [3.0.try_into()?, 2.0.try_into()?],
/// };
///
/// let upper_right = cuboid.support_mapping(&[1.0, 1.0].into());
/// let lower_right = cuboid.support_mapping(&[1.0, -1.0].into());
///
/// assert_eq!(upper_right, [1.5, 1.0].into());
/// assert_eq!(lower_right, [1.5, -1.0].into());
/// # Ok(())
/// # }
/// ```
pub trait SupportMapping<V> {
    /// Return the furthest extent of a shape in the direction of `n`.
    fn support_mapping(&self, n: &V) -> V;
}

/// Test whether two shapes share any points in space.
///
/// # Examples
///
/// Some shapes implement [`IntersectsAt`] directly:
/// ```
/// use hoomd_geometry::{Convex, IntersectsAt, shape::Sphere};
/// use hoomd_vector::Versor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let s0 = Sphere {
///     radius: 1.0.try_into()?,
/// };
/// let s1 = Sphere {
///     radius: 1.0.try_into()?,
/// };
///
/// let q_id = Versor::default();
///
/// assert_eq!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id), true);
/// assert_eq!(s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id), false);
/// # Ok(())
/// # }
/// ```
///
/// Others must be wrapped in [`Convex`]:
/// ```
/// use hoomd_geometry::{
///     Convex, IntersectsAt,
///     shape::{Cuboid, Sphere},
/// };
/// use hoomd_vector::Versor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sphere = Convex(Sphere {
///     radius: 1.0.try_into()?,
/// });
/// let cuboid = Convex(Cuboid {
///     edge_lengths: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?],
/// });
///
/// assert_eq!(
///     sphere.intersects_at(
///         &cuboid,
///         &[1.9, 0.0, 0.0].into(),
///         &Versor::default()
///     ),
///     true
/// );
/// assert_eq!(
///     sphere.intersects_at(
///         &cuboid,
///         &[2.1, 0.0, 0.0].into(),
///         &Versor::default()
///     ),
///     false
/// );
/// # Ok(())
/// # }
/// ```
pub trait IntersectsAt<S, V, R> {
    /// Test whether the set of points in one shape intersects with the set of another.
    ///
    /// Each shape (`self` and `other`) remain unmodified in their own local
    /// coordinate systems. The intersection test is performed in the local
    /// coordinate system of `self`. The vector `v_ij` points from the local
    /// origin of `self` to the local origin of `other`. Similarly, `o_ij` is the
    /// orientation of `other` in the local coordinate system of `self`.
    ///
    /// # See Also
    ///
    /// Call [`pair_system_to_local`] to compute `v_ij` and `o_ij` from the
    /// system frame positions and orientations of two shapes.
    ///
    /// [`pair_system_to_local`]: hoomd_vector::pair_system_to_local
    fn intersects_at(&self, other: &S, v_ij: &V, o_ij: &R) -> bool;

    /// Approximate the amount of overlap between two shapes.
    ///
    /// Move `other` in along `v_ij` until the shapes no longer overlap. Return the
    /// approximate* distance needed to move `other` (which is 0 if the shapes are
    /// already separated). This is *not* the exact minimum separation distance and
    /// the method does *not* solve for an optimal direction.
    ///
    /// `resolution` sets the size of the steps between distances in the
    /// approximation.
    ///
    /// If `v_ij` has 0 norm, move `other` along the `V::default_unit()`.
    ///
    /// ```
    /// use hoomd_geometry::{Convex, IntersectsAt, shape::Cuboid};
    /// use hoomd_vector::Versor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Convex(Cuboid::with_equal_edges(2.0.try_into()?));
    ///
    /// let d = cuboid.approximate_separation_distance(
    ///     &cuboid,
    ///     &[1.8, 0.0, 0.0].into(),
    ///     &Versor::default(),
    ///     0.01.try_into()?,
    /// );
    ///
    /// assert!(d >= 0.2);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn approximate_separation_distance(
        &self,
        other: &S,
        v_ij: &V,
        o_ij: &R,
        resolution: PositiveReal,
    ) -> f64
    where
        V: InnerProduct,
    {
        let mut d = 0.0;

        let direction = v_ij.to_unit().unwrap_or((V::default_unit(), 1.0)).0;

        while self.intersects_at(other, &(*v_ij + *direction.get() * d), o_ij) {
            d += resolution.get();
        }

        d
    }
}

/// Radius of an N-dimensional hypersphere that bounds a shape.
///
/// The hypersphere has the same local origin as the shape `self`.
///
/// Some [`IntersectsAt`] tests use the bounding sphere radius as a first pass
/// before calling a more expensive intersection test. To improve performance,
/// the bounding sphere should be as tightly fitting as possible.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{BoundingSphereRadius, shape::Cuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Cuboid {
///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
/// };
/// let bounding_radius = cuboid.bounding_sphere_radius();
///
/// assert_eq!(bounding_radius.get(), 5.0);
/// # Ok(())
/// # }
/// ```
pub trait BoundingSphereRadius {
    /// Get the bounding radius.
    fn bounding_sphere_radius(&self) -> PositiveReal;
}

/// Test whether a point is inside or outside a shape.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{IsPointInside, shape::Cuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Cuboid {
///     edge_lengths: [6.0.try_into()?, 8.0.try_into()?],
/// };
///
/// assert!(cuboid.is_point_inside(&[2.5, -3.5].into()));
/// assert!(!cuboid.is_point_inside(&[4.0, -3.5].into()));
/// # Ok(())
/// # }
/// ```
pub trait IsPointInside<V> {
    /// Check if a point is inside the shape.
    fn is_point_inside(&self, point: &V) -> bool;
}

/// Enumerate possible sources of error in fallible geometry methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Polytopes require at least one vertex.
    #[error("a ConvexPolytope must have at least one vertex")]
    DegeneratePolytope,
}
