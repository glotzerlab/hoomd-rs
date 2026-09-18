// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Convex polygon represented by vertices and edges.

use crate::{ConvexHull, Error, Facet, shape::ConvexPolytope};
use serde::{Deserialize, Serialize};

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

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
    /// The triangular facets of the polyhedron, as indices into `vertices`.
    faces: Vec<Facet<3>>,
    /// The radius of a bounding sphere of the geometry.
    bounding_radius: PositiveReal,
}

impl ConvexSurfaceMesh3d {
    /// Create a convex surface mesh from the convex hull of the given set of points.
    ///
    /// The resulting shape contains a subset of the given points, including only the
    /// non-degenerate points on the convex hull, in the order of the input. The faces
    /// of the result are the triangles of the hull, given as indices into its vertices.
    ///
    /// # Errors
    ///
    /// * [`Error::DegeneratePolytope`] when there are fewer than 4 non-coplanar points.
    ///
    /// # Example
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
    #[inline]
    pub fn from_point_set<I>(points: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = Cartesian<3>>,
    {
        let (vertices, faces) = Cartesian::<3>::convex_hull(points)?;

        Ok(Self {
            bounding_radius: ConvexPolytope::<3>::bounding_radius(&vertices),
            vertices,
            faces,
        })
    }
}
