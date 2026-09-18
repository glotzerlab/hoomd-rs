// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Convex polygon represented by vertices and edges.

use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::{
    BoundingSphereRadius, ConvexHull, Error, IntersectsAt, IntersectsAtGlobal, IsPointInside,
    Scale, SupportMapping, Volume, shape::ConvexPolytope,
};
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::{Cartesian, InnerProduct, Metric, Rotate, Rotation, RotationMatrix};



pub struct Simplex<const N: usize> {
    indices: [i32; N],
};

/// The vertices and edges that make up a convex polygon.
///
/// [`ConvexPolytope::<3>`] and [`ConvexSurfaceMesh3d`] can both represent
/// 3d convex polyhedra. The first is defined *implicitly* as the convex hull
/// of a set of points. It stores the given point set without any modification,
/// and can therefore be constructed quickly. The *implicit* convex hull is
/// formed by [`SupportMapping`] during intersection tests of
/// `Convex(ConvexPolygon)` with other `Convex(_)` types.
///
/// In contrast, [`ConvexSurfaceMesh3d`] *explicitly* computes the convex hull
/// on construction with [`ConvexHull`]. After construction, the [`vertices`] of
/// the shape include only the points on the convex hull, and the [`facets`] are
/// the triangular faces of the body. Using this representation, [`ConvexSurfaceMesh3d`]
/// is able to provide implementations of [`Volume`] and [`IsPointInside`].
///
/// [`vertices`]: Self::vertices
/// [`facets`]: Self::facets
///
/// # Examples
///
/// Construction:
/// ```
/// use hoomd_geometry::shape::ConvexSurfaceMesh3d;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let tetrahedron = ConvexSurfaceMesh3d::from_point_set([
///     [1.0, 1.0, 1.0].into(),
///     [1.0, -1.0, -1.0].into(),
///     [-1.0, 1.0, -1.0].into(),
///     [-1.0, -1.0, 1.0].into(),
/// ])?;
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConvexSurfaceMesh3d {
    /// The vertices of the polyhedron.
    vertices: Vec<Cartesian<3>>,
    /// .
    facets: Vec<Simplex<3>>,
    /// The radius of a bounding sphere of the geometry.
    bounding_radius: PositiveReal,
}
