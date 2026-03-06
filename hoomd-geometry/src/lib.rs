// Copyright (c) 2024-2026 The Regents of the University of Michigan.
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
//! use approxim::assert_relative_eq;
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
//! assert!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id));
//! assert!(!s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id));
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
//!     shape::{Hypercuboid, Sphere},
//! };
//! use hoomd_vector::Versor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let sphere = Convex(Sphere {
//!     radius: 1.0.try_into()?,
//! });
//! let cuboid = Convex(Hypercuboid {
//!     edge_lengths: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?],
//! });
//!
//! assert!(sphere.intersects_at(
//!     &cuboid,
//!     &[1.9, 0.0, 0.0].into(),
//!     &Versor::default()
//! ));
//! assert!(!sphere.intersects_at(
//!     &cuboid,
//!     &[2.1, 0.0, 0.0].into(),
//!     &Versor::default()
//! ));
//! # Ok(())
//! # }
//! ```

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{InnerProduct, Rotate, Rotation, Vector};
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
/// use hoomd_geometry::{SupportMapping, shape::Hypercuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Hypercuboid {
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

/// Test whether the set of points in one shape intersects with the set of another
/// (in the global frame).
///
/// [`IntersectsAtGlobal`] supports hard-particle overlap checks for simulations
/// defined in arbitrary metric spaces.
pub trait IntersectsAtGlobal<S, P, R> {
    /// Test whether the set of points in one shape intersects with the set of another
    /// (in the global frame).
    ///
    /// Each shape (`self` and `other`) remain unmodified in their own local
    /// coordinate systems. The intersection test is performed in a global
    /// coordinate system where `self` has position/orientation `r_self`/`o_self`
    /// and other has position/orientation `r_other`/`o_other`.
    ///
    /// When starting with shapes in the global frame (such as in Monte Carlo
    /// simulations), `intersects_at_global` may be faster than `intersects_at`
    /// as it is able to check whether the bounding spheres of the shapes
    /// overlap *before* transforming into the local coordinate system about
    /// `self`.
    fn intersects_at_global(
        &self,
        other: &S,
        r_self: &P,
        o_self: &R,
        r_other: &P,
        o_other: &R,
    ) -> bool;
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
/// assert!(s0.intersects_at(&s1, &[1.9, 0.0, 0.0].into(), &q_id));
/// assert!(!s0.intersects_at(&s1, &[2.1, 0.0, 0.0].into(), &q_id));
/// # Ok(())
/// # }
/// ```
///
/// Others must be wrapped in [`Convex`]:
/// ```
/// use hoomd_geometry::{
///     Convex, IntersectsAt,
///     shape::{Hypercuboid, Sphere},
/// };
/// use hoomd_vector::Versor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sphere = Convex(Sphere {
///     radius: 1.0.try_into()?,
/// });
/// let cuboid = Convex(Hypercuboid {
///     edge_lengths: [2.0.try_into()?, 2.0.try_into()?, 2.0.try_into()?],
/// });
///
/// assert!(sphere.intersects_at(
///     &cuboid,
///     &[1.9, 0.0, 0.0].into(),
///     &Versor::default()
/// ));
/// assert!(!sphere.intersects_at(
///     &cuboid,
///     &[2.1, 0.0, 0.0].into(),
///     &Versor::default()
/// ));
/// # Ok(())
/// # }
/// ```
pub trait IntersectsAt<S, V, R>
where
    V: Vector,
    R: Rotation + Rotate<V>,
{
    /// Test whether the set of points in one shape intersects with the set of another
    /// (in the local frame).
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
    /// use hoomd_geometry::{Convex, IntersectsAt, shape::Hypercuboid};
    /// use hoomd_vector::Versor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let cuboid = Convex(Hypercuboid::with_equal_edges(2.0.try_into()?));
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
/// use hoomd_geometry::{BoundingSphereRadius, shape::Hypercuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Hypercuboid {
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
/// use hoomd_geometry::{IsPointInside, shape::Hypercuboid};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let cuboid = Hypercuboid {
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

/// Produce a new shape by uniform scaling.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{Scale, shape::Sphere};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let sphere = Sphere {
///     radius: 5.0.try_into()?,
/// };
///
/// let scaled_sphere = sphere.scale_length(0.5.try_into()?);
///
/// assert_eq!(scaled_sphere.radius.get(), 2.5);
/// # Ok(())
/// # }
/// ```
pub trait Scale {
    /// Produce a new shape by uniformly scaling length.
    #[must_use]
    fn scale_length(&self, v: PositiveReal) -> Self;

    /// Produce a new shape by uniformly scaling volume.
    #[must_use]
    fn scale_volume(&self, v: PositiveReal) -> Self;
}

/// Map a point from one shape to another.
///
/// # Example
///
/// ```
/// use hoomd_geometry::{MapPoint, shape::Circle};
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let closed_a = Circle {
///     radius: 10.0.try_into()?,
/// };
/// let closed_b = Circle {
///     radius: 20.0.try_into()?,
/// };
///
/// let mapped_point =
///     closed_a.map_point(Cartesian::from([-1.0, 1.0]), &closed_b);
///
/// assert_eq!(mapped_point, Ok(Cartesian::from([-2.0, 2.0])));
/// assert_eq!(
///     closed_a.map_point(Cartesian::from([-100.0, 1.0]), &closed_b),
///     Err(hoomd_geometry::Error::PointOutsideShape)
/// );
/// # Ok(())
/// # }
/// ```
pub trait MapPoint<P> {
    /// Map a point from one boundary to another.
    ///
    /// # Errors
    ///
    /// Returns [`Error::PointOutsideShape`] when `point` is outside the shape
    /// `self`.
    fn map_point(&self, point: P, other: &Self) -> Result<P, Error>;
}

/// Enumerate possible sources of error in fallible geometry methods.
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Polytopes require at least one vertex.
    #[error("a ConvexPolytope must have at least one vertex")]
    DegeneratePolytope,

    /// The point is outside the shape.
    #[error("cannot map a point that is outside the shape")]
    PointOutsideShape,
}
