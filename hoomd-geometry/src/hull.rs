// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Compute the convex hull of a set of points.
//!
//! A hull is returned as its vertices together with the [`Facet`]s that bound it:
//! each facet is the set of indices of the hull vertices on one of its bounding
//! hyperplanes, and [`Facet::orientation`] decides exactly on which [`Side`] of that
//! hyperplane any point lies.

use std::{borrow::Borrow, cmp::Ordering};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::Error;
use hoomd_vector::{Cartesian, Cross, InnerProduct};

/// The side of an oriented hyperplane a point lies on.
///
/// The vertices of a simplex $`(v_0, \ldots, v_{N-1})`$ in `N`-dimensional space,
/// along with a query point $`x`$, define a simplex. This simplex spans a hyperplane,
/// which divides the space into three regions: `Above`, `Below`, and `On` . `Above` is
/// the half-space in which the following determinant is positive:
///
/// ```math
/// \det(v_1 - v_0, \ldots, v_{N-1} - v_0, x - v_0)
/// ```
///
/// `Below` is the half-space in which the determinant is negative, and `On` are the
/// points for which the determinant is exactly zero.
///
/// In three dimensions, this construction is equivalent to the right-hand rule. In two
/// dimensions, `Above` lies to the left of a directed edge and `Below` lies to the
/// right. [`Side::On`] is exact up to floating-point precision, and indicates that the
/// point lies in the hyperplane itself, collinear with an edge or coplanar with a
/// a triangle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[expect(clippy::exhaustive_enums, reason = "Variants describe all of space.")]
pub enum Side {
    /// The point lies on the side the normal points to.
    Above,

    /// The point lies on the side opposite the normal.
    Below,

    /// The point lies exactly on the hyperplane.
    On,
}

/// The side of the hyperplane a finite determinant value indicates, as a [`Side`].
///
/// Signed zeros both map to [`Side::On`]. Any non-finite value maps to `None`.
#[inline]
fn sign(value: f64) -> Option<Side> {
    let ordering = value.partial_cmp(&0.0)?;
    if !value.is_finite() {
        return None;
    }
    Some(match ordering {
        Ordering::Greater => Side::Above,
        Ordering::Less => Side::Below,
        Ordering::Equal => Side::On,
    })
}

/// Whether every coordinate of every point in a set is finite.
#[inline]
fn finite_points<const N: usize>(points: &[Cartesian<N>]) -> bool {
    points
        .iter()
        .all(|p| p.coordinates.iter().all(|c| c.is_finite()))
}

/// A facet of a convex hull: the indices of the `N` vertices of the hull that bound it,
/// in the order given by [`ConvexHull::convex_hull`].
///
/// A facet of a two-dimensional hull is an edge (`Facet<2>`) and a facet of a
/// three-dimensional hull is a triangle (`Facet<3>`).
#[serde_as]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Facet<const N: usize> {
    /// The indices of the vertices of the facet.
    #[serde_as(as = "[_; N]")]
    indices: [usize; N],
}

impl<const N: usize> Facet<N> {
    /// Create a facet from the indices of its vertices.
    #[inline]
    #[must_use]
    pub const fn new(indices: [usize; N]) -> Self {
        Self { indices }
    }

    /// The indices of the vertices of the facet.
    #[inline]
    #[must_use]
    pub const fn indices(&self) -> [usize; N] {
        self.indices
    }

    /// The simplex the facet spans: its vertices, looked up in `vertices`.
    ///
    /// A facet of an `N`-dimensional hull is an `(N - 1)`-simplex bounded by
    /// `N` vertices of the hull. An edge of a polygon returns its two
    /// endpoints, and a triangle of a polyhedron its three corners.
    ///
    /// # Panics
    ///
    /// Panics when a vertex index is out of bounds, like indexing `points`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::Facet;
    /// use hoomd_vector::Cartesian;
    ///
    /// let points = [
    ///     Cartesian::from([1.0, 1.0]),
    ///     Cartesian::from([1.0, -1.0]),
    ///     Cartesian::from([-1.0, -1.0]),
    /// ];
    /// let edge = Facet::new([0, 2]);
    ///
    /// let [start, end] = edge.as_simplex(&points);
    /// assert_eq!(start, points[0]);
    /// assert_eq!(end, points[2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_simplex(&self, vertices: &[Cartesian<N>]) -> [Cartesian<N>; N] {
        self.indices.map(|i| vertices[i])
    }
}

/// Determine to which [`Side`] of a hyperplane a point lies.
///
/// If the [`Side`] cannot be unambiguously determined, this predicate reurns an
/// `Error` describing the specific failure mode, if know.
trait PointPlaneOrientation<const N: usize> {
    /// Determine the [`Side`] of `simplex` that `point` lies on.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate is not finite.
    fn orientation(simplex: &[Cartesian<N>; N], point: &Cartesian<N>) -> Result<Side, Error>;
}

impl PointPlaneOrientation<2> for Cartesian<2> {
    #[inline]
    fn orientation(simplex: &[Cartesian<2>; 2], point: &Cartesian<2>) -> Result<Side, Error> {
        sign(robust::orient2d(
            robust::Coord {
                x: simplex[0][0],
                y: simplex[0][1],
            },
            robust::Coord {
                x: simplex[1][0],
                y: simplex[1][1],
            },
            robust::Coord {
                x: point[0],
                y: point[1],
            },
        ))
        .ok_or(Error::NotFinite)
    }
}

impl PointPlaneOrientation<3> for Cartesian<3> {
    #[inline]
    fn orientation(simplex: &[Cartesian<3>; 3], point: &Cartesian<3>) -> Result<Side, Error> {
        // Shewchuk's orient3d is positive *below* the oriented plane, so we negate such
        // that Above is the side the vertex order's normal points to, as in 2D.
        let coord = |p: Cartesian<3>| robust::Coord3D {
            x: p[0],
            y: p[1],
            z: p[2],
        };
        sign(-robust::orient3d(
            coord(simplex[0]),
            coord(simplex[1]),
            coord(simplex[2]),
            coord(*point),
        ))
        .ok_or(Error::NotFinite)
    }
}

#[expect(
    private_bounds,
    reason = "Bound restricts the implementation to valid dimensions."
)]
impl<const N: usize> Facet<N>
where
    Cartesian<N>: PointPlaneOrientation<N>,
{
    /// Check on which side of the facet a point lies.
    ///
    /// Returns [`Side::Above`] when `point` lies strictly on the positive side of the
    /// the facet's vertex order, [`Side::Below`] when it lies strictly on the opposite
    /// side, and [`Side::On`] when it lies exactly in the hyperplane.
    ///
    /// The facets of a hull from [`Cartesian::<2>::convex_hull`] or
    /// [`Cartesian::<3>::convex_hull`] are oriented so that every point of the hull
    /// lies [`Side::Above`] or [`Side::On`] each of its facets.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate is not finite.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::{Facet, Side};
    /// use hoomd_vector::Cartesian;
    ///
    /// // The triangle spanned by the x and y axes, with a normal pointing toward +z.
    /// let vertices = [
    ///     Cartesian::from([0.0, 0.0, 0.0]),
    ///     Cartesian::from([1.0, 0.0, 0.0]),
    ///     Cartesian::from([0.0, 1.0, 0.0]),
    /// ];
    /// let facet = Facet::new([0, 1, 2]);
    ///
    /// assert_eq!(
    ///     facet.orientation(&vertices, &Cartesian::from([0.0, 0.0, 1.0])),
    ///     Ok(Side::Above)
    /// );
    /// assert_eq!(
    ///     facet.orientation(&vertices, &Cartesian::from([0.0, 0.0, -1.0])),
    ///     Ok(Side::Below)
    /// );
    /// assert_eq!(
    ///     facet.orientation(&vertices, &Cartesian::from([0.0, 0.0, 0.0])),
    ///     Ok(Side::On)
    /// );
    /// ```
    #[inline]
    pub fn orientation(
        &self,
        vertices: &[Cartesian<N>],
        point: &Cartesian<N>,
    ) -> Result<Side, Error> {
        Cartesian::orientation(&self.as_simplex(vertices), point)
    }
}

/// Compute the convex hull of a set of points.
///
/// [`ConvexHull`] is implemented by point types. The hull of a set of points is
/// the smallest convex body that contains them all. The points on the hull are
/// a subset of the given points.
///
/// The input may be any slice or iterator of points, e.g. a
/// `&[Cartesian<2>]`, a `Vec<Cartesian<2>>`, or the points generated on the
/// fly.
///
/// # Example
///
/// Compute the convex hull of a set of points in a plane:
/// ```
/// use hoomd_geometry::hull::ConvexHull;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let points = [
///     [1.0, 1.0].into(),
///     [1.0, -1.0].into(),
///     [-1.0, 1.0].into(),
///     [-1.0, -1.0].into(),
///     [0.0, 0.0].into(), // This point is in the interior of the hull.
/// ];
///
/// let (hull_vertices, edges) = Cartesian::<2>::convex_hull(&points)?;
///
/// assert_eq!(hull_vertices.len(), 4);
/// assert_eq!(edges.len(), 4); // A square is bounded by four edges.
/// # Ok(())
/// # }
/// ```
pub trait ConvexHull<const N: usize>: Sized {
    /// Compute the convex hull of a set of points.
    ///
    /// Returns the vertices of the hull together with the facets that bound
    /// it. The vertices are a subset of the given points, including only the
    /// non-degenerate points on the convex hull, arranged in a deterministic
    /// order: counter-clockwise in two dimensions, and in the order of the
    /// input in three dimensions and higher. Each facet is a set of indices
    /// into the returned vector of vertices: an edge (`Facet<2>`) in two
    /// dimensions and a triangle (`Facet<3>`) in three.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate of the input is not finite,
    /// and [`Error::DegeneratePolytope`] when the points do not span an `N`-dimensional
    /// convex body: fewer than `N + 1` points, or a set that lies in a hyperplane of
    /// fewer than `N` dimensions.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    ///
    /// Compute the hull from an iterator of points:
    /// ```
    /// use hoomd_geometry::hull::ConvexHull;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let (hull_vertices, edges) =
    ///     Cartesian::<2>::convex_hull((0..3).map(|i| {
    ///         let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 3.0;
    ///         Cartesian::from([angle.cos(), angle.sin()])
    ///     }))?;
    ///
    /// assert_eq!(hull_vertices.len(), 3);
    /// assert_eq!(edges.len(), 3); // A triangle is bounded by three edges.
    /// //
    /// # Ok(())
    /// # }
    /// ```
    fn convex_hull<I>(points: I) -> Result<(Vec<Self>, Vec<Facet<N>>), Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>;
}

impl ConvexHull<2> for Cartesian<2> {
    /// Compute the convex hull of points in 2D with the Graham scan algorithm.
    ///
    /// The orientation tests use robust adaptive predicates that compute the
    /// exact sign of the orientation determinant, so the resulting hull does
    /// not depend on where the point set lies relative to the origin (up to
    /// the precision of the coordinates themselves).
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate of the input is not finite,
    /// and [`Error::DegeneratePolytope`] when the points do not span the plane
    /// (fewer than three points, or all collinear).
    #[inline]
    fn convex_hull<I>(points: I) -> Result<(Vec<Self>, Vec<Facet<2>>), Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>,
    {
        let mut points: Vec<Self> = points.into_iter().map(|p| *p.borrow()).collect();

        // No need to try and triangulate if the hull is degenerate
        if points.len() < 3 {
            return Err(Error::DegeneratePolytope);
        }
        if !finite_points(&points) {
            return Err(Error::NotFinite);
        }

        let anchor_idx = find_lowest_leftmost(&points).ok_or(Error::DegeneratePolytope)?;

        // Move the anchor to the front of the list of vertices, as it is always in the hull
        points.swap(0, anchor_idx);
        let anchor = points[0];

        // Sort the remainder of the slice in-place
        points[1..].sort_unstable_by(|&a, &b| {
            let (a0, a1) = get_graham_key(a, anchor);
            let (b0, b1) = get_graham_key(b, anchor);
            a0.total_cmp(&b0).then(a1.total_cmp(&b1))
        });

        // Now vertices[..2] is an edge on the hull. Initialize counters for the hull length
        // and number of vertices on the hull
        let mut n_vertices_on_hull = 2;
        let mut next_candidate = 2;

        // Repeat until all interior points are gone
        while next_candidate < points.len() {
            let c = points[next_candidate];
            while n_vertices_on_hull >= 2 {
                let p = points[n_vertices_on_hull - 2];
                let n = points[n_vertices_on_hull - 1];

                if Cartesian::<2>::orientation(&[p, n], &c)? == Side::Above {
                    break;
                }
                // Point n is not to the left of the edge, so it lies inside the hull
                n_vertices_on_hull -= 1;
            }
            // Swap the vertex c onto the end of the hull, extending it by one
            points.swap(next_candidate, n_vertices_on_hull);
            n_vertices_on_hull += 1;
            next_candidate += 1;
        }

        points.truncate(n_vertices_on_hull);
        if points.len() < 3 {
            return Err(Error::DegeneratePolytope);
        }

        // The facets of a polygon are its edges, implied by the cyclic
        // counter-clockwise order of the vertices.
        let facets = (0..points.len())
            .map(|i| Facet::new([i, (i + 1) % points.len()]))
            .collect();

        Ok((points, facets))
    }
}

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
#[inline]
fn find_lowest_leftmost(vertices: &[Cartesian<2>]) -> Option<usize> {
    vertices.iter().position_min_by(|a, b| {
        // Compare y-coordinates, then x.
        a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0]))
    })
}

/// Get the key for a lexicographic order of points with respect to an anchor.
#[inline]
fn get_graham_key(p: Cartesian<2>, anchor: Cartesian<2>) -> (f64, f64) {
    let diff = p - anchor;
    (f64::atan2(diff[1], diff[0]), diff.dot(&diff))
}

impl ConvexHull<3> for Cartesian<3> {
    /// Compute the convex hull of points in 3D with an incremental algorithm.
    ///
    /// Orientation tests are evaluated with [`robust::orient3d`], Shewchuk's
    /// adaptive precision predicate, which computes the exact sign of the
    /// orientation determinant. No tolerance is needed, and the hull does not
    /// depend on where the point set lies relative to the origin (up to the
    /// precision of the coordinates themselves).
    ///
    /// This is an O(n*f) algorithm, where f is the number of faces, and is suited for
    /// high-symmetry solids and polyhedra with fewer than ~1000 vertices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate of the input is not finite, and
    /// [`Error::DegeneratePolytope`] when there are fewer than four points, or all
    /// points are coplanar.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::ConvexHull;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let cube = [
    ///     [-1.0, -1.0, -1.0].into(),
    ///     [1.0, -1.0, -1.0].into(),
    ///     [1.0, 1.0, -1.0].into(),
    ///     [-1.0, 1.0, -1.0].into(),
    ///     [-1.0, -1.0, 1.0].into(),
    ///     [1.0, -1.0, 1.0].into(),
    ///     [1.0, 1.0, 1.0].into(),
    ///     [-1.0, 1.0, 1.0].into(),
    /// ];
    ///
    /// let (hull_vertices, facets) = Cartesian::<3>::convex_hull(&cube)?;
    ///
    /// assert_eq!(hull_vertices.len(), 8);
    /// assert_eq!(facets.len(), 12); // Each square face is split into two triangles.
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn convex_hull<I>(points: I) -> Result<(Vec<Self>, Vec<Facet<3>>), Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>,
    {
        let points: Vec<Self> = points.into_iter().map(|p| *p.borrow()).collect();

        // No convex body without at least 4 points.
        if points.len() < 4 {
            return Err(Error::DegeneratePolytope);
        }
        if !finite_points(&points) {
            return Err(Error::NotFinite);
        }

        let (indices, faces) = incremental_hull(&points)?;

        // The faces index the input points, so remap them to the compacted list of hull
        // vertices this method returns. The vertex indices are sorted, so a binary
        // search locates the new position of each vertex.
        let position = |i: usize| {
            indices
                .binary_search(&i)
                .expect("the vertices of a face are vertices of the hull")
        };
        let facets = faces
            .iter()
            .map(|&[a, b, c]| Facet::new([position(a), position(b), position(c)]))
            .collect();
        let vertices = indices.into_iter().map(|i| points[i]).collect();

        Ok((vertices, facets))
    }
}

/// Compute the convex hull of points in 3D with an incremental algorithm.
///
/// Returns the indices of the points that are vertices of the hull in increasing order,
/// and the triangular faces of the hull as oriented index triples. A point lies
/// strictly outside a face when its orientation against it is [`Side::Below`], and
/// the faces are oriented so that all points of the hull lie [`Side::Above`] or
/// [`Side::On`] them.
///
/// Starting from an initial tetrahedron, each point is inserted in turn.
/// Points strictly outside the current hull see a connected set of faces,
/// which are removed and replaced by new faces joining the point to the
/// boundary (horizon) of that set. Points that see no face lie inside or on
/// the surface of the current hull and are skipped.
fn incremental_hull(points: &[Cartesian<3>]) -> Result<(Vec<usize>, Vec<[usize; 3]>), Error> {
    let (t0, t1, t2, t3) = initial_tetrahedron(points)?;

    // The four faces of the initial tetrahedron. Each face is oriented so the vertex
    // opposite it, and with it the hull, lies Above or On the face
    let mut faces: Vec<[usize; 3]> = vec![[t0, t1, t2], [t0, t2, t3], [t0, t3, t1], [t1, t3, t2]];
    let opposite = [t3, t1, t2, t0];
    for (face, o) in faces.iter_mut().zip(opposite) {
        if Cartesian::<3>::orientation(
            &[points[face[0]], points[face[1]], points[face[2]]],
            &points[o],
        )? == Side::Below
        {
            face.swap(0, 1);
        }
    }

    let in_initial = |i: usize| i == t0 || i == t1 || i == t2 || i == t3;

    // Reused scratch storage for the directed edges of the visible faces.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    let mut horizon: Vec<(usize, usize)> = Vec::new();

    for p in 0..points.len() {
        if in_initial(p) {
            continue;
        }

        // Remove the faces that p strictly sees and collect their directed edges
        edges.clear();
        faces.retain(|&face| {
            let outside = [points[face[0]], points[face[1]], points[face[2]]];
            if Cartesian::<3>::orientation(&outside, &points[p])
                .is_ok_and(|side| side == Side::Below)
            {
                edges.push((face[0], face[1]));
                edges.push((face[1], face[2]));
                edges.push((face[2], face[0]));
                false
            } else {
                true
            }
        });
        // p sees no face: it must be inside (or on) the hull
        if edges.is_empty() {
            continue;
        }

        // The boundary (horizon) of the visible region is formed by the directed edges
        // `ab` whose reverse `ba` is not also an edge of a visible face. Each becomes
        // a new face joined to point `p`, with consistent orientations.
        edges.sort_unstable();
        horizon.clear();
        horizon.extend(
            edges
                .iter()
                .copied()
                .filter(|&(u, v)| edges.binary_search(&(v, u)).is_err()),
        );

        for &(u, v) in &horizon {
            faces.push([u, v, p]);
        }
    }

    // The vertices of the hull are the vertices of the final faces.
    let mut on_hull = vec![false; points.len()];
    for face in &faces {
        for &i in face {
            on_hull[i] = true;
        }
    }

    Ok(((0..points.len()).filter(|&i| on_hull[i]).collect(), faces))
}

/// Find an initial tetrahedron for the incremental hull.
///
/// The four points are vertices of the hull: `a` is the lexicographically smallest point,
/// `b` is the furthest from that, `c` is the furthest noncolinear point from the line
/// `ab`, and `d` is the furthest noncoplanar point to the triangle `abc`.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when all points coincide, all are colinear, or
/// all are coplanar.
fn initial_tetrahedron(points: &[Cartesian<3>]) -> Result<(usize, usize, usize, usize), Error> {
    // a is the lowest, leftmost point in the set
    let a = points
        .iter()
        .position_min_by(|x, y| {
            x[0].total_cmp(&y[0])
                .then(x[1].total_cmp(&y[1]))
                .then(x[2].total_cmp(&y[2]))
        })
        .expect("the point set is not empty");

    // b is the point farthest from a.
    let b = (0..points.len())
        .filter(|&i| i != a)
        .max_by(|&i, &j| {
            (points[i] - points[a])
                .norm_squared()
                .total_cmp(&(points[j] - points[a]).norm_squared())
        })
        .expect("there are at least two points");
    if (points[b] - points[a]).norm_squared() == 0.0 {
        return Err(Error::DegeneratePolytope); // Every point coincides with a.
    }

    // c is the farthest noncolinear point from the segment ab.
    let direction = points[b] - points[a];
    let (c, _) = points
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != a && i != b && !collinear(points, a, b, i))
        .fold((None, 0.0), |(c, cd), (i, &p)| {
            let d = (p - points[a]).cross(&direction).norm_squared();
            if d > cd { (Some(i), d) } else { (c, cd) }
        });

    // Otherwise, all points are collinear.
    let c = c.ok_or(Error::DegeneratePolytope)?;

    // d is the point farthest from the plane through a, b and c.
    let face = [points[a], points[b], points[c]];
    let normal = (points[b] - points[a]).cross(&(points[c] - points[a]));
    let (mut d, mut rank) = (None, f64::NEG_INFINITY);
    for i in 0..points.len() {
        if i == a || i == b || i == c {
            continue;
        }
        if Cartesian::<3>::orientation(&face, &points[i])? != Side::On
            && normal.dot(&(points[i] - points[a])).abs() > rank
        {
            rank = normal.dot(&(points[i] - points[a])).abs();
            d = Some(i);
        }
    }

    // Otherwise, all points are coplanar.
    let d = d.ok_or(Error::DegeneratePolytope)?;

    Ok((a, b, c, d))
}

/// Whether three points given by index are exactly collinear.
///
/// The points are collinear when they are collinear in each of the three coordinate
/// plane projections, decided by the two-dimensional exact predicate.
#[inline]
fn collinear(points: &[Cartesian<3>], a: usize, b: usize, c: usize) -> bool {
    let projected =
        |p: usize, i: usize, j: usize| Cartesian::<2>::from([points[p][i], points[p][j]]);
    [(0, 1), (0, 2), (1, 2)].iter().all(|&(i, j)| {
        Cartesian::<2>::orientation(
            &[projected(a, i, j), projected(b, i, j)],
            &projected(c, i, j),
        ) == Ok(Side::On)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Volume,
        shape::{ConvexPolyhedron, Simplex3},
    };
    use approxim::assert_relative_eq;
    use assert2::check;
    use hoomd_vector::{Rotate, Versor};
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;
    use rstest_reuse::{self, apply, template};

    #[rstest]
    #[case::single_point(vec![[1.0, 2.0]], 0)]
    #[case::two_points_first_lower(vec![[1.0, 1.0], [2.0, 3.0]], 0)]
    #[case::two_points_second_lower(vec![[1.0, 3.0], [2.0, 1.0]], 1)]
    #[case::same_y_leftmost_wins(vec![[3.0, 1.0], [1.0, 1.0], [2.0, 1.0]], 1)]
    #[case::same_y_negative(vec![[0.0, -5.0], [-3.0, -5.0], [2.0, -5.0]], 1)]
    #[case::multiple_points(vec![[3.0, 5.0], [1.0, 1.0], [4.0, 2.0], [2.0, 3.0]], 1)]
    #[case::negative_coords(vec![[0.0, 0.0], [-1.0, -1.0], [1.0, -1.0]], 1)]
    #[case::same_y_all_negative_x(vec![[5.0, 0.0], [-10.0, 0.0], [-5.0, 0.0]], 1)]
    #[case::lowest_is_only_point(vec![[100.0, -100.0]], 0)]
    #[case::diagonal_tiebreak(vec![[5.0, 5.0], [4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]], 4)]
    fn test_find_lowest_leftmost(#[case] vertices: Vec<[f64; 2]>, #[case] expected_idx: usize) {
        let vertices: Vec<Cartesian<2>> = vertices.into_iter().map(Cartesian::from).collect();
        let idx = find_lowest_leftmost(&vertices).expect("returned None for non-empty input");
        assert_eq!(idx, expected_idx);
    }

    #[rstest]
    fn test_find_lowest_leftmost_empty() {
        let vertices: Vec<Cartesian<2>> = vec![];
        assert_eq!(find_lowest_leftmost(&vertices), None);
    }

    #[rstest]
    fn test_single_point_various_coords(
        #[values([0.0, 0.0], [-1.0, -1.0], [100.0, -50.0], [f64::MIN_POSITIVE, f64::MIN_POSITIVE])]
        coords: [f64; 2],
    ) {
        let vertices = vec![Cartesian::from(coords)];
        let idx = find_lowest_leftmost(&vertices).expect("returned None for non-empty input");
        assert_eq!(idx, 0);
    }

    #[rstest]
    fn test_square_corners() {
        let points: Vec<Cartesian<2>> = vec![[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_corners_big() {
        let points: Vec<Cartesian<2>> = vec![
            [101.0, 101.0],
            [101.0, 100.0],
            [100.0, 101.0],
            [100.0, 100.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [100.0, 100.0].into(),
            [101.0, 100.0].into(),
            [101.0, 101.0].into(),
            [100.0, 101.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_with_edge_points() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0], // bottom edge
            [1.0, 0.5],
            [1.0, 1.0], // right edge
            [0.5, 1.0],
            [0.0, 1.0], // top edge
            [0.0, 0.5], // left edge
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Hull should have 4 corners (interior edge points excluded)
        assert_eq!(vertices.len(), 4);
        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_square_dense_boundary() {
        let mut pts: Vec<[f64; 2]> = Vec::new();
        // Bottom edge
        for i in 0..20 {
            pts.push([f64::from(i) / 19.0, 0.0]);
        }
        // Right edge
        for i in 0..20 {
            pts.push([1.0, f64::from(i) / 19.0]);
        }
        // Top edge
        for i in 0..20 {
            pts.push([f64::from(i) / 19.0, 1.0]);
        }
        // Left edge
        for i in 0..20 {
            pts.push([0.0, f64::from(i) / 19.0]);
        }
        let points: Vec<Cartesian<2>> = pts.into_iter().map(Cartesian::from).collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_circle_uniform() {
        let n = 20;
        let points: Vec<Cartesian<2>> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // All points on circle should be in hull
        assert_eq!(vertices.len(), n);
    }

    #[rstest]
    fn test_circle_with_interior_points() {
        let n_boundary = 12;
        let mut points: Vec<Cartesian<2>> = (0..n_boundary)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        // Add interior points
        points.extend([[0.0, 0.0], [0.3, 0.3], [-0.2, 0.1], [0.1, -0.4]].map(Cartesian::from));
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Only boundary points should be in hull
        assert_eq!(vertices.len(), n_boundary);
    }

    #[rstest]
    fn test_circle_partial_arc() {
        let points: Vec<Cartesian<2>> = (0..10)
            .map(|i| {
                let angle = -std::f64::consts::FRAC_PI_4
                    + (std::f64::consts::FRAC_PI_2 * f64::from(i) / 9.0);
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // All points on partial arc should be in hull
        assert_eq!(vertices.len(), 10);
    }

    #[rstest]
    fn test_random_unit_square(#[values(0, 42, 64, 100_000)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let original: Vec<Cartesian<2>> = (0..50)
            .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
            .collect();
        let points = original.clone();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Hull should have at least 3 points
        assert!(vertices.len() >= 3);
        // All hull points should be in original set
        for h in &vertices {
            assert!(original.iter().any(|v| (*v - *h).norm() < 1e-10));
        }
    }

    #[rstest]
    fn test_random_gaussian() {
        let mut rng = StdRng::seed_from_u64(42);
        let points: Vec<Cartesian<2>> = (0..100)
            .map(|_| {
                Cartesian::from([
                    rng.random::<f64>() * 2.0 - 1.0,
                    rng.random::<f64>() * 2.0 - 1.0,
                ])
            })
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_random_deterministic_output() {
        for _ in 0..3 {
            let mut rng = StdRng::seed_from_u64(123);
            let points1: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let (vertices1, _) = Cartesian::<2>::convex_hull(&points1)
                .expect("hard-coded points should lie on a convex hull");

            let mut rng = StdRng::seed_from_u64(123);
            let points2: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let (vertices2, _) = Cartesian::<2>::convex_hull(&points2)
                .expect("hard-coded points should lie on a convex hull");

            assert_eq!(vertices1.len(), vertices2.len());
        }
    }

    #[rstest]
    fn test_duplicate_lowest_points() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0], // Three duplicates of lowest
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert!(vertices.len() >= 3);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_many_duplicates_few_unique() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Should handle duplicates gracefully
        assert!(vertices.len() >= 3);

        let hull = [[0.0, 0.0].into(), [2.0, 0.0].into(), [1.0, 2.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_collinear_bottom_edge() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],  // All at y=0
            [0.75, 1.0], // Apex
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Should pick leftmost of bottom points and include apex
        assert!(vertices.len() >= 3);

        let hull = [[0.0, 0.0].into(), [1.5, 0.0].into(), [0.75, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_leftmost_selected() {
        let points: Vec<Cartesian<2>> = vec![
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.0], // All y=0, but [0,0] is leftmost
            [0.5, 1.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // [0, 0] should be the anchor point
        let hull = [[0.0, 0.0].into(), [1.0, 0.0].into(), [0.5, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_many_bottom_points() {
        let points: Vec<Cartesian<2>> = (0..10)
            .map(|i| [f64::from(i) * 2.0 / 9.0, 0.0])
            .chain([[1.0, 1.0]])
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Leftmost [0,0] and rightmost [2,0] should be in hull with apex
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_collinear_from_anchor() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0], // Anchor (lowest leftmost)
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0], // Collinear at 45 degrees
            [1.0, 0.0],
            [0.0, 1.0], // Other corners
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Should only keep furthest point in each direction
        assert!(vertices.len() >= 3);

        let hull = [
            [0.0, 0.0].into(),
            [1.0, 0.0].into(),
            [3.0, 3.0].into(),
            [0.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_radial_collinear_multiple_directions() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0], // Anchor
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0], // Along x-axis
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0], // Along y-axis
            [1.0, 1.0],
            [2.0, 2.0], // Diagonal
            [-1.0, 0.0],
            [-2.0, 0.0], // Negative x-axis (but won't be picked due to anchor)
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Should keep only outermost points
        assert!(vertices.len() >= 3);
    }

    #[rstest]
    fn test_star_pattern() {
        let mut points: Vec<Cartesian<2>> = vec![Cartesian::from([0.0, 0.0])]; // Anchor
        // Create points along 8 rays
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 8.0;
            for r in [0.5, 1.0, 1.5] {
                points.push(Cartesian::from([r * angle.cos(), r * angle.sin()]));
            }
        }
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        // Outer points should form the hull
        assert!(vertices.len() >= 8); // At least 8 outer points
    }

    #[rstest]
    fn test_minimum_triangle() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 3);

        let hull = [[0.0, 0.0].into(), [1.0, 0.0].into(), [0.5, 1.0].into()];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_negative_coordinates() {
        let points: Vec<Cartesian<2>> = vec![[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [-1.0, -1.0].into(),
            [1.0, -1.0].into(),
            [1.0, 1.0].into(),
            [-1.0, 1.0].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_large_coordinates() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1e6, 0.0], [1e6, 1e6], [0.0, 1e6]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should lie on a convex hull");
        assert_eq!(vertices.len(), 4);

        let hull = [
            [0.0, 0.0].into(),
            [1e6, 0.0].into(),
            [1e6, 1e6].into(),
            [0.0, 1e6].into(),
        ];
        itertools::assert_equal(&vertices, &hull);
    }

    #[rstest]
    fn test_degenerate() {
        let points: Vec<Cartesian<2>> = vec![[0.0, 0.0], [0.5, 0.5], [0.25, 0.25], [1.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let result = Cartesian::<2>::convex_hull(&points);
        check!(result == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::one_point(vec![[0.0, 0.0]])]
    #[case::two_points(vec![[0.0, 0.0], [1.0, 0.0]])]
    fn test_too_few_points(#[case] points: Vec<[f64; 2]>) {
        let points: Vec<Cartesian<2>> = points.into_iter().map(Cartesian::from).collect();
        check!(Cartesian::<2>::convex_hull(&points) == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    #[case::infinity(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [f64::INFINITY, 0.5]])]
    #[case::nan(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [f64::NAN, 0.5]])]
    fn test_non_finite_points_rejected_2d(#[case] points: Vec<[f64; 2]>) {
        let points: Vec<Cartesian<2>> = points.into_iter().map(Cartesian::from).collect();
        check!(Cartesian::<2>::convex_hull(&points) == Err(Error::NotFinite));
    }

    #[rstest]
    #[case::infinity(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, f64::INFINITY], [0.0, 0.0, 1.0]])]
    #[case::nan(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0]])]
    fn test_non_finite_points_rejected_3d(#[case] points: Vec<[f64; 3]>) {
        let points: Vec<Cartesian<3>> = points.into_iter().map(Cartesian::from).collect();
        check!(Cartesian::<3>::convex_hull(&points) == Err(Error::NotFinite));
    }

    #[rstest]
    fn test_input_types() {
        let points: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [1.0, 1.0],
            [0.5, 0.25], // interior point
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();

        let hull = [[0.0, 0.0].into(), [2.0, 0.0].into(), [1.0, 1.0].into()];

        // A slice of points.
        let (from_slice, _) = Cartesian::<2>::convex_hull(&points[..])
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_slice, &hull);

        // An owned Vec of points.
        let (from_vec, _) = Cartesian::<2>::convex_hull(points.clone())
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_vec, &hull);

        // An iterator of points.
        let (from_iterator, _) = Cartesian::<2>::convex_hull(points.iter().copied())
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_iterator, &hull);
    }

    #[rstest]
    fn test_translation_invariance() {
        //  nearly collinear points far from the origin should classsify correctly
        for offset in [0.0, 1.0, 1e3, 1e5] {
            let offset = Cartesian::from([offset, offset]);
            let points: Vec<Cartesian<2>> = [[0.0, 0.0], [1.0, 1.0 - 1e-8], [2.0, 2.0], [0.0, 2.0]]
                .into_iter()
                .map(|p| Cartesian::from(p) + offset)
                .collect();

            let (vertices, _) = Cartesian::<2>::convex_hull(&points)
                .expect("hard-coded points should lie on a convex hull");

            let hull: Vec<Cartesian<2>> = [
                Cartesian::from([0.0, 0.0]),
                Cartesian::from([1.0, 1.0 - 1e-8]),
                Cartesian::from([2.0, 2.0]),
                Cartesian::from([0.0, 2.0]),
            ]
            .into_iter()
            .map(|p| p + offset)
            .collect();
            itertools::assert_equal(&vertices, &hull);
        }

        // When the point is exactly on the diagonal, the hull is the plain
        // triangle at every offset.
        for offset in [0.0, 1.0, 1e3, 1e5] {
            let offset = Cartesian::from([offset, offset]);
            let points: Vec<Cartesian<2>> = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [0.0, 2.0]]
                .into_iter()
                .map(|p| Cartesian::from(p) + offset)
                .collect();

            let (vertices, _) = Cartesian::<2>::convex_hull(&points)
                .expect("hard-coded points should lie on a convex hull");
            assert_eq!(vertices.len(), 3, "at offset {offset:?}");
        }
    }

    #[rstest]
    fn test_predicate_translation_invariance() {
        let p = Cartesian::from([0.0, 0.0]);
        let q = Cartesian::from([1.0, 1.0 - 1e-8]);
        let test = Cartesian::from([2.0, 2.0]);

        for offset in [0.0, 1.0, 1e3, 1e5, 1e8] {
            let offset = Cartesian::from([offset, offset]);
            check!(
                Cartesian::<2>::orientation(&[p + offset, q + offset], &(test + offset))
                    == Ok(Side::Above)
            );
        }

        // The same holds for collinear and right-turning triples.
        let q_collinear = Cartesian::from([1.0, 1.0]);
        let q_right = Cartesian::from([1.0, 1.0 + 1e-8]);
        for offset in [0.0, 1e3, 1e8] {
            let offset = Cartesian::from([offset, offset]);
            check!(
                Cartesian::<2>::orientation(&[p + offset, q_collinear + offset], &(test + offset),)
                    == Ok(Side::On)
            );
            check!(
                Cartesian::<2>::orientation(&[p + offset, q_right + offset], &(test + offset))
                    == Ok(Side::Below)
            );
        }
    }
    /// The 8 vertices of a cube with edge length 2.
    fn cube() -> Vec<Cartesian<3>> {
        let mut points = Vec::with_capacity(8);
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    points.push(Cartesian::from([x, y, z]));
                }
            }
        }
        points
    }

    /// The 6 vertices of an octahedron with edge length `sqrt(2)`.
    fn octahedron() -> Vec<Cartesian<3>> {
        let mut points = Vec::with_capacity(6);
        for axis in 0..3 {
            for sign in [-1.0, 1.0] {
                let mut p = [0.0; 3];
                p[axis] = sign;
                points.push(Cartesian::from(p));
            }
        }
        points
    }

    /// The vertex sets of the platonic solids, shared by the 3d hull tests.
    #[template]
    #[rstest]
    #[case::tetrahedron(Simplex3::default().vertices().to_vec())]
    #[case::cube(cube())]
    #[case::octahedron(octahedron())]
    #[case::dodecahedron(ConvexPolyhedron::dodecahedron().vertices().to_vec())]
    #[case::icosahedron(ConvexPolyhedron::icosahedron().vertices().to_vec())]
    fn platonic_solids(#[case] points: Vec<Cartesian<3>>) {}

    /// Validate a triangulated 3d hull.
    ///
    /// * Every face is a supporting plane
    /// * The faces form a closed oriented surface: every directed edge is unique and
    ///   matched by its reverse.
    /// * Euler's formula `V - E + F = 2` holds.
    /// * The reported vertices are exactly the vertices of the faces, in order
    fn validate_3d_hull(points: &[Cartesian<3>], vertices: &[usize], faces: &[[usize; 3]]) {
        // No point is strictly outside any face.
        for &face in faces {
            for q in 0..points.len() {
                let face_points = [points[face[0]], points[face[1]], points[face[2]]];
                let side = Cartesian::<3>::orientation(&face_points, &points[q])
                    .expect("the coordinates are resolvable");
                check!(
                    side != Side::Below,
                    "point {q} lies outside the face {face:?}"
                );
            }
        }

        // The surface is closed and consistently oriented.
        let mut edges: Vec<(usize, usize)> = faces
            .iter()
            .flat_map(|&[a, b, c]| [(a, b), (b, c), (c, a)])
            .collect();
        edges.sort_unstable();
        // Every directed edge appears exactly once ...
        for pair in edges.windows(2) {
            check!(
                pair[0] != pair[1],
                "the directed edge {:?} appears twice",
                pair[0]
            );
        }
        // ... and is matched by its reverse.
        for &(u, v) in &edges {
            check!(
                edges.binary_search(&(v, u)).is_ok(),
                "the edge ({u}, {v}) has no matching reverse"
            );
        }

        // Euler's formula.
        let n_edges = edges.len() / 2;
        check!(
            vertices.len() + faces.len() == n_edges + 2,
            "Euler's formula fails: {} - {n_edges} + {} != 2",
            vertices.len(),
            faces.len()
        );

        // The vertices are the face vertices in increasing order.
        let mut face_vertices: Vec<usize> = faces.iter().flatten().copied().collect();
        face_vertices.sort_unstable();
        face_vertices.dedup();
        check!(
            vertices == face_vertices,
            "vertices are not the face vertices"
        );
    }

    #[apply(platonic_solids)]
    fn test_3d_platonic_solids(#[case] points: Vec<Cartesian<3>>) {
        // The solids are in their axis-aligned orientations, where the facet
        // vertices are exactly coplanar.
        let (vertices, faces) =
            incremental_hull(&points).expect("platonic solid vertices should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        // Every vertex of a platonic solid is on the hull.
        check!(vertices == (0..points.len()).collect::<Vec<usize>>());
    }

    #[apply(platonic_solids)]
    fn test_3d_platonic_solids_random_orientations(#[case] points: Vec<Cartesian<3>>) {
        let mut rng = StdRng::seed_from_u64(27);

        for _ in 0..10_000 {
            let versor: Versor = rng.random();
            let rotated: Vec<Cartesian<3>> = points.iter().map(|p| versor.rotate(p)).collect();

            let (vertices, faces) = incremental_hull(&rotated)
                .expect("platonic solid vertices should form a convex body");

            validate_3d_hull(&rotated, &vertices, &faces);
            // A rotation maps vertices to vertices.
            check!(vertices == (0..points.len()).collect::<Vec<usize>>());
        }
    }

    #[rstest]
    #[case::one_point(vec![[0.0, 0.0, 0.0]])]
    #[case::two_points(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])]
    #[case::three_points(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    #[case::collinear(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])]
    #[case::coplanar(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])]
    #[case::identical(vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])]
    fn test_3d_degenerate(#[case] points: Vec<[f64; 3]>) {
        let points: Vec<Cartesian<3>> = points.into_iter().map(Cartesian::from).collect();
        check!(Cartesian::<3>::convex_hull(&points) == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    fn test_3d_non_vertex_points_dropped() {
        // The center, two face points and a duplicate corner of the cube are
        // not vertices of its hull.
        let mut points = cube();
        points.extend([
            Cartesian::from([0.0, 0.0, 0.0]),
            Cartesian::from([1.0, 0.0, 0.0]),
            Cartesian::from([1.0, 1.0, 0.0]),
            Cartesian::from([-1.0, -1.0, -1.0]),
        ]);

        let (vertices, faces) =
            incremental_hull(&points).expect("hard-coded points should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        // Only the corners of the cube remain, in input order.
        check!(vertices == (0..8).collect::<Vec<usize>>());

        let hull: Vec<Cartesian<3>> = vertices.iter().map(|&i| points[i]).collect();
        itertools::assert_equal(&hull, &cube());
    }

    #[rstest]
    fn test_3d_public_hull_facets() {
        // The public API remaps the facet indices to the compacted vertex
        // list, dropping the points interior to the hull.
        let mut points = cube();
        points.extend([
            Cartesian::from([0.0, 0.0, 0.0]),
            Cartesian::from([1.0, 0.0, 0.0]),
            Cartesian::from([1.0, 1.0, 0.0]),
            Cartesian::from([-1.0, -1.0, -1.0]),
        ]);

        let (vertices, facets) = Cartesian::<3>::convex_hull(&points)
            .expect("hard-coded points should form a convex body");

        assert_eq!(vertices.len(), 8);
        assert_eq!(facets.len(), 12); // Each square face is split into two triangles.

        let mut referenced = vec![false; vertices.len()];
        for facet in &facets {
            let indices = facet.indices();
            for &i in &indices {
                assert!(i < vertices.len(), "facet index {i} is out of bounds");
                referenced[i] = true;
            }
        }
        // Every returned vertex is part of a facet.
        assert!(referenced.iter().all(|&r| r));
    }

    #[rstest]
    fn test_3d_random_point_clouds(#[values(0, 1, 2, 3, 4)] seed: u64) {
        // The points are drawn from the uniform distribution over the cube [-1, 1]^3.
        let mut rng = StdRng::seed_from_u64(seed);
        let points: Vec<Cartesian<3>> = (0..30).map(|_| rng.random()).collect();

        let (vertices, faces) =
            incremental_hull(&points).expect("random points should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        check!(vertices.len() >= 4);
    }

    #[rstest]
    fn test_3d_fibonacci_sphere() {
        // The points of a Fibonacci lattice on the unit sphere are all on the hull
        let n = 20;
        let golden_angle = std::f64::consts::PI * (5.0_f64.sqrt() - 1.0);
        let points: Vec<Cartesian<3>> = (0..n)
            .map(|i| {
                let z = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
                let r = (1.0 - z * z).sqrt();
                let theta = golden_angle * i as f64;
                Cartesian::from([r * theta.cos(), r * theta.sin(), z])
            })
            .collect();

        let (vertices, faces) =
            incremental_hull(&points).expect("points on a sphere should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        check!(vertices == (0..n).collect::<Vec<usize>>());
    }

    #[rstest]
    fn test_3d_translation_and_scale_invariance(
        #[values(0.0, 1.0, 1e3, 1e5)] offset: f64,
        #[values(1e-6, 1.0, 1e6)] scale: f64,
    ) {
        // The exact predicates measure differences, so the hull is the same
        // wherever the point set lies and however it is scaled.
        let offset = Cartesian::from([offset, -offset, offset]);
        let points: Vec<Cartesian<3>> = ConvexPolyhedron::dodecahedron()
            .vertices()
            .iter()
            .map(|&p| p * scale + offset)
            .collect();

        let (vertices, faces) =
            incremental_hull(&points).expect("hard-coded points should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        check!(vertices == (0..20).collect::<Vec<usize>>());
    }

    #[rstest]
    fn test_3d_input_types() {
        let points = octahedron();

        let (from_slice, _) = Cartesian::<3>::convex_hull(&points)
            .expect("hard-coded points should form a convex body");
        let (from_vec, _) = Cartesian::<3>::convex_hull(points.clone())
            .expect("hard-coded points should form a convex body");
        let (from_iterator, _) = Cartesian::<3>::convex_hull(points.iter().copied())
            .expect("hard-coded points should form a convex body");

        assert_eq!(from_slice.len(), 6);
        itertools::assert_equal(&from_slice, &from_vec);
        itertools::assert_equal(&from_slice, &from_iterator);
    }

    #[rstest]
    fn test_3d_volume() {
        // The hull volume is the sum of the volumes of the tetrahedra formed
        // by the faces and any interior point. The origin (cube centroid) is interior.
        let points = cube();

        let (vertices, faces) =
            incremental_hull(&points).expect("hard-coded points should form a convex body");
        validate_3d_hull(&points, &vertices, &faces);

        let origin = Cartesian::<3>::default();
        let volume: f64 = faces
            .iter()
            .map(|&[a, b, c]| Simplex3::from([points[a], points[b], points[c], origin]).volume())
            .sum();

        assert_relative_eq!(volume, 8.0);
    }

    #[rstest]
    fn test_sign() {
        check!(sign(f64::NAN) == None);
        check!(sign(f64::INFINITY) == None);
        check!(sign(f64::NEG_INFINITY) == None);
        check!(sign(0.0) == Some(Side::On));
        check!(sign(-0.0) == Some(Side::On));
        check!(sign(1.0) == Some(Side::Above));
        check!(sign(-1.0) == Some(Side::Below));
        check!(sign(f64::MIN_POSITIVE) == Some(Side::Above));
        check!(sign(-f64::MIN_POSITIVE) == Some(Side::Below));
    }

    #[rstest]
    fn test_facet_orientation_2d() {
        let vertices = [Cartesian::from([0.0, 0.0]), Cartesian::from([1.0, 0.0])];
        let facet = Facet::new([0, 1]);

        check!(facet.orientation(&vertices, &Cartesian::from([0.5, 0.5])) == Ok(Side::Above));
        check!(facet.orientation(&vertices, &Cartesian::from([0.5, -0.5])) == Ok(Side::Below));
        check!(facet.orientation(&vertices, &Cartesian::from([2.0, 0.0])) == Ok(Side::On));

        // Non-finite coordinates are rejected.
        let nan = Cartesian::from([f64::NAN, 0.0]);
        check!(facet.orientation(&vertices, &nan) == Err(Error::NotFinite));

        let points: Vec<Cartesian<2>> = [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [0.5, 0.5], // interior
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let (hull_vertices, facets) = Cartesian::<2>::convex_hull(&points)
            .expect("hard-coded points should form a convex body");
        for facet in &facets {
            for q in &points {
                check!(
                    facet
                        .orientation(&hull_vertices, q)
                        .expect("finite coordinates resolve exactly")
                        != Side::Below
                );
            }
        }
    }

    #[rstest]
    fn test_facet_orientation_3d() {
        let vertices = [
            Cartesian::from([0.0, 0.0, 0.0]),
            Cartesian::from([1.0, 0.0, 0.0]),
            Cartesian::from([0.0, 1.0, 0.0]),
        ];
        let facet = Facet::new([0, 1, 2]);

        check!(facet.orientation(&vertices, &Cartesian::from([0.0, 0.0, -1.0])) == Ok(Side::Below));
        check!(facet.orientation(&vertices, &Cartesian::from([0.0, 0.0, 1.0])) == Ok(Side::Above));
        check!(facet.orientation(&vertices, &Cartesian::from([1.0, 1.0, 0.0])) == Ok(Side::On));

        let nan = Cartesian::from([f64::NAN, 0.0, 0.0]);
        check!(facet.orientation(&vertices, &nan) == Err(Error::NotFinite));

        let points = cube();
        let (hull_vertices, facets) = Cartesian::<3>::convex_hull(&points)
            .expect("hard-coded points should form a convex body");
        for facet in &facets {
            for q in &points {
                let orientation = facet
                    .orientation(&hull_vertices, q)
                    .expect("finite coordinates resolve exactly");
                check!(orientation != Side::Below);
            }
        }
    }
}
