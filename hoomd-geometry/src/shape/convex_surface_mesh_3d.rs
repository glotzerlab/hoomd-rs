// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Convex polyhedron represented by vertices and triangular facets.

use crate::{
    Error, Volume,
    hull::{Facet, incremental_hull},
    shape::ConvexPolytope,
};
use hoomd_linear_algebra::matrix::Matrix33;
use serde::{Deserialize, Serialize};

use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;

/// The vertices and triangular facets that make up a convex polyhedron.
///
/// [`ConvexPolytope::<3>`] and [`ConvexSurfaceMesh3d`] can both represent
/// 3d convex polyhedra. The first is defined *implicitly* as the convex hull
/// of a set of points. It stores the given point set without any modification,
/// and can therefore be constructed quickly. The *implicit* convex hull is
/// formed by the support mapping of `Convex` during intersection tests of
/// `Convex(ConvexPolyhedron)` with other `Convex(_)` types.
///
/// In contrast, [`ConvexSurfaceMesh3d`] *explicitly* computes the convex hull
/// on construction, with the same incremental algorithm that backs
/// [`ConvexHull`](crate::hull::ConvexHull). After construction, the [`vertices`] of
/// the shape include only the points on the convex hull, and the [`facets`] are
/// the triangular faces of the body. Using this representation, [`ConvexSurfaceMesh3d`]
/// is able to provide an implementation of [`Volume`].
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
    /// The triangular faces of the polyhedron, as indices into `vertices`.
    faces: Vec<[usize; 3]>,
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
    /// * [`Error::NotFinite`] when any coordinate input is not finite.
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
    #[expect(
        clippy::missing_panics_doc,
        reason = "Panic would occur due to a bug in hoomd-rs."
    )]
    pub fn from_point_set<I>(points: I) -> Result<Self, Error>
    where
        I: IntoIterator<Item = Cartesian<3>>,
    {
        let points: Vec<Cartesian<3>> = points.into_iter().collect();
        let (indices, faces) = incremental_hull(&points)?;

        // The faces index the input points, so remap them to the compacted list of
        // hull vertices the mesh stores. The vertex indices are sorted, so a binary
        // search locates the new position of each vertex.
        let position = |i: usize| {
            indices
                .binary_search(&i)
                .expect("the vertices of a face are vertices of the hull")
        };
        let faces = faces
            .iter()
            .map(|&[a, b, c]| [position(a), position(b), position(c)])
            .collect();
        let vertices: Vec<Cartesian<3>> = indices.iter().map(|&i| points[i]).collect();

        Ok(Self {
            bounding_radius: ConvexPolytope::<3>::bounding_radius(&vertices),
            vertices,
            faces,
        })
    }

    /// The vertices of the convex polyhedron, in the order of the input points.
    #[inline]
    #[must_use]
    pub fn vertices(&self) -> &[Cartesian<3>] {
        &self.vertices
    }

    /// The triangular facets of the convex polyhedron.
    ///
    /// Each facet is bound to [`Self::vertices`], the points its indices correspond to.
    #[inline]
    pub fn facets(&self) -> impl Iterator<Item = Facet<'_, 3>> + '_ {
        self.faces
            .iter()
            .map(|&indices| Facet::new(&self.vertices, indices))
    }
}

impl Volume for ConvexSurfaceMesh3d {
    /// Compute the volume of the polyhedron.
    ///
    /// # Example
    ///
    /// ```
    /// use approxim::assert_relative_eq;
    /// use hoomd_geometry::{Volume, shape::ConvexSurfaceMesh3d};
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// // Cube with edge length 2.0
    /// let cube = ConvexSurfaceMesh3d::from_point_set([
    ///     [-1.0, -1.0, -1.0].into(),
    ///     [1.0, -1.0, -1.0].into(),
    ///     [1.0, 1.0, -1.0].into(),
    ///     [-1.0, 1.0, -1.0].into(),
    ///     [-1.0, -1.0, 1.0].into(),
    ///     [1.0, -1.0, 1.0].into(),
    ///     [1.0, 1.0, 1.0].into(),
    ///     [-1.0, 1.0, 1.0].into(),
    /// ])?;
    ///
    /// assert_relative_eq!(cube.volume(), 8.0);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn volume(&self) -> f64 {
        self.faces
            .iter()
            .map(|face| {
                Matrix33 {
                    rows: face.map(|i| self.vertices[i].coordinates),
                }
                .determinant()
            })
            .sum::<f64>()
            / 6.0
    }
}
