// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Compute the convex hull of a set of points.
//!
//! A hull is returned as its vertices together with the [`Facet`]s that bound it:
//! each facet a set of indices into the given points. [`Facet::orientation`] decides
//! exactly on which [`Side`] of that hyperplane any point lies on. Facets are oriented
//! with outward normals, so the points of the hull lie [`Side::Below`] or [`Side::On`]
//! each of its facets.

use std::cmp::Ordering;

use itertools::Itertools;

use crate::{Error, orient4d::orient4d};
use hoomd_vector::{Cartesian, Cross, InnerProduct, Wedge};

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

/// A facet of a convex hull, given as `N` indices into a slice of vertices.
///
/// A facet of a two-dimensional hull is an edge (`Facet<2>`) and a facet of a
/// three-dimensional hull is a triangle (`Facet<3>`). A four-dimensional hull
/// has tetrahedral facets (`Facet<4>`).
///
/// The facet holds a reference to the points its indices refer to. A facet borrowed
/// from a hull remains valid as long as the points the hull was computed from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Facet<'a, const N: usize> {
    /// The points the facet's indices refer to.
    vertices: &'a [Cartesian<N>],
    /// The indices of the vertices of the facet.
    indices: [usize; N],
}

impl<'a, const N: usize> Facet<'a, N> {
    /// Create a facet from a slice of points and the indices of its vertices.
    ///
    /// # Panics
    ///
    /// Panics when an index is out of bounds for `vertices`.
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
    /// let edge = Facet::new(&points, [0, 2]);
    ///
    /// assert_eq!(edge.indices(), [0, 2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn new(vertices: &'a [Cartesian<N>], indices: [usize; N]) -> Self {
        assert!(
            indices.iter().all(|&i| i < vertices.len()),
            "facet indices {indices:?} are out of bounds for {} vertices",
            vertices.len()
        );
        Self { vertices, indices }
    }

    /// The points the facet's indices refer to.
    #[inline]
    #[must_use]
    pub const fn vertices(&self) -> &'a [Cartesian<N>] {
        self.vertices
    }

    /// The indices of the vertices of the facet, into [`Self::vertices`].
    #[inline]
    #[must_use]
    pub const fn indices(&self) -> [usize; N] {
        self.indices
    }

    /// The simplex the facet spans, as `Cartesian<N>` coordinates.
    ///
    /// A facet of an `N`-dimensional hull is an `(N - 1)`-simplex bounded by `N`
    /// vertices of the hull. An edge of a polygon returns its two endpoints, and a
    /// triangle of a polyhedron its three corners.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::Facet;
    /// use hoomd_vector::Cartesian;
    ///
    /// let triangle = [[1.0, 1.0].into(), [1.0, -1.0].into(), [-1.0, -1.0].into()];
    /// let edge = Facet::new(&triangle, [0, 2]);
    ///
    /// let [start, end] = edge.as_simplex();
    /// assert_eq!(start, triangle[0]);
    /// assert_eq!(end, triangle[2]);
    /// ```
    #[inline]
    #[must_use]
    pub fn as_simplex(&self) -> [Cartesian<N>; N] {
        self.indices.map(|i| self.vertices[i])
    }
}

/// Determine to which [`Side`] of a hyperplane a point lies.
///
/// If the [`Side`] cannot be unambiguously determined, this predicate returns an [`Error`].
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

impl PointPlaneOrientation<4> for Cartesian<4> {
    #[inline]
    fn orientation(simplex: &[Cartesian<4>; 4], point: &Cartesian<4>) -> Result<Side, Error> {
        let [a, b, c, d] = *simplex;
        if !finite_points(simplex) || !finite_points(std::slice::from_ref(point)) {
            return Err(Error::NotFinite);
        }
        Ok(match orient4d(a, b, c, d, *point)?.cmp(&0) {
            Ordering::Greater => Side::Above,
            Ordering::Less => Side::Below,
            Ordering::Equal => Side::On,
        })
    }
}

#[expect(
    private_bounds,
    reason = "Bound restricts the implementation to valid dimensions."
)]
impl<const N: usize> Facet<'_, N>
where
    Cartesian<N>: PointPlaneOrientation<N>,
{
    /// Check on which side of the facet a point lies.
    ///
    /// Returns [`Side::Above`] when `point` lies strictly on the positive side
    /// of the the facet (determined by vertex order), [`Side::Below`] when
    /// it lies strictly on the opposite side, and [`Side::On`] when it lies
    /// exactly in the hyperplane.
    ///
    /// The facets of a hull from [`ConvexHull::convex_hull`] are oriented with
    /// outward normals, so that
    /// every point of the hull lies [`Side::Below`] or [`Side::On`] each of its
    /// facets, and a point outside the hull lies [`Side::Above`] at least one facet.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate is not finite.
    /// In four dimensions, returns [`Error::NumericallyAmbiguousPolytope`] if the
    /// orientation cannot be resolved.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::{Facet, Side};
    /// use hoomd_vector::Cartesian;
    ///
    /// // The triangle spanned by the x and y axes, with a normal pointing toward +z.
    /// let vertices = [
    ///     [0.0, 0.0, 0.0].into(),
    ///     [1.0, 0.0, 0.0].into(),
    ///     [0.0, 1.0, 0.0].into(),
    /// ];
    /// let facet = Facet::new(&vertices, [0, 1, 2]);
    ///
    /// assert_eq!(
    ///     facet.orientation(&Cartesian::from([0.0, 0.0, 1.0])),
    ///     Ok(Side::Above)
    /// );
    /// assert_eq!(
    ///     facet.orientation(&Cartesian::from([0.0, 0.0, -1.0])),
    ///     Ok(Side::Below)
    /// );
    /// assert_eq!(
    ///     facet.orientation(&Cartesian::from([0.0, 0.0, 0.0])),
    ///     Ok(Side::On)
    /// );
    /// ```
    #[inline]
    pub fn orientation(&self, point: &Cartesian<N>) -> Result<Side, Error> {
        Cartesian::orientation(&self.as_simplex(), point)
    }
}

/// Compute the convex hull of a set of points.
///
/// [`ConvexHull`] is implemented by slices of points, e.g. `[Cartesian<2>]` and
/// `[Cartesian<3>]`. The hull of a set of points is the smallest convex body that
/// contains them all. The points on the hull are a subset of the given points.
///
/// # Example
///
/// Compute the convex hull of a set of points in a plane:
/// ```
/// use hoomd_geometry::hull::ConvexHull;
/// use hoomd_vector::Cartesian;
///
/// # fn main() -> Result<(), hoomd_geometry::Error> {
/// let points: [Cartesian<2>; 5] = [
///     [1.0, 1.0].into(),
///     [1.0, -1.0].into(),
///     [-1.0, 1.0].into(),
///     [-1.0, -1.0].into(),
///     [0.0, 0.0].into(),
/// ];
///
/// let (hull_vertices, edges) = points.convex_hull()?;
///
/// assert_eq!(hull_vertices.len(), 4);
/// assert_eq!(edges.len(), 4);
/// # Ok(())
/// # }
/// ```
pub trait ConvexHull<const N: usize> {
    /// Compute the convex hull of a set of points.
    ///
    /// Returns the vertices of the hull together with the facets that bound
    /// it. The vertices are a subset of the given points, ordered counter-clockwise
    /// in two dimensions and in input order in higher dimensions. In three dimensions
    /// and higher, the returned vertices are those used by the boundary triangulation,
    /// and can include non-extreme points on the hull boundary (for example, a point
    /// exactly on an edge or facet). Facets are edges (`Facet<2>`), triangles
    /// (`Facet<3>`), or tetrahedra (`Facet<4>`) depending on the hull dimension. The
    /// facets are bound to the given points; resolve the vertices of a facet with
    /// [`Facet::as_simplex`]. Facets are oriented with outward normals: every point of
    /// the hull lies [`Side::Below`] or [`Side::On`] each of them.
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
    /// Compute the hull from points generated on the fly, collected into a vector:
    /// ```
    /// use hoomd_geometry::hull::ConvexHull;
    /// use hoomd_vector::Cartesian;
    /// use std::f64::consts::PI;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let triangle: Vec<Cartesian<2>> = (0..3)
    ///     .map(|i| {
    ///         let angle = 2.0 * PI * f64::from(i) / 3.0;
    ///         Cartesian::from([angle.cos(), angle.sin()])
    ///     })
    ///     .collect();
    ///
    /// let (hull_vertices, edges) = triangle.convex_hull()?;
    ///
    /// assert_eq!(hull_vertices.len(), 3);
    /// assert_eq!(edges.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    fn convex_hull(&self) -> Result<(Vec<Cartesian<N>>, Vec<Facet<'_, N>>), Error>;
}

impl ConvexHull<2> for [Cartesian<2>] {
    /// Compute the convex hull of points in 2D with the Graham scan algorithm.
    ///
    /// The orientation tests use robust adaptive predicates that compute the
    /// exact sign of the orientation determinant, so the resulting hull does
    /// not depend on where the point set lies relative to the origin (up to
    /// the precision of the coordinates themselves).
    ///
    /// The vertices are ordered counter-clockwise. The facet edges are directed
    /// clockwise along that order, with outward normals, so every point of the
    /// hull lies [`Side::Below`] or [`Side::On`] each of them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate of the input is not finite,
    /// and [`Error::DegeneratePolytope`] when the points do not span the plane
    /// (fewer than three points, or all collinear).
    #[inline]
    fn convex_hull(&self) -> Result<(Vec<Cartesian<2>>, Vec<Facet<'_, 2>>), Error> {
        // Pair each point with its index, so that the facets can refer to the input pts
        let mut points: Vec<(Cartesian<2>, usize)> =
            self.iter().enumerate().map(|(i, &p)| (p, i)).collect();

        // Fewer than three points cannot span the plane.
        if points.len() < 3 {
            return Err(Error::DegeneratePolytope);
        }
        if !finite_points(self) {
            return Err(Error::NotFinite);
        }

        // Move the anchor (the lowest, leftmost point) to the front of the list,
        // as it is always on the hull.
        let anchor_idx = points
            .iter()
            .position_min_by(|(a, _), (b, _)| {
                // Compare y-coordinates, then x.
                a[1].total_cmp(&b[1]).then(a[0].total_cmp(&b[0]))
            })
            .expect("there are at least three points");
        points.swap(0, anchor_idx);
        let anchor = points[0].0;

        // Sort the remainder of the slice in-place
        points[1..].sort_unstable_by(|&(a, _), &(b, _)| {
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
            let c = points[next_candidate].0;
            while n_vertices_on_hull >= 2 {
                let p = points[n_vertices_on_hull - 2].0;
                let n = points[n_vertices_on_hull - 1].0;

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
        // counter-clockwise order of the vertices. Each edge is directed from
        // vertex i + 1 to vertex i, i.e. clockwise, so that its right-hand-rule
        // normal points outward and every point of the hull lies Below or On it.
        let n = points.len();
        let facets = (0..n)
            .map(|i| Facet::new(self, [points[(i + 1) % n].1, points[i].1]))
            .collect();
        let vertices = points.into_iter().map(|(p, _)| p).collect();

        Ok((vertices, facets))
    }
}

/// Get the key for a lexicographic order of points with respect to an anchor.
#[inline]
fn get_graham_key(p: Cartesian<2>, anchor: Cartesian<2>) -> (f64, f64) {
    let diff = p - anchor;
    (f64::atan2(diff[1], diff[0]), diff.dot(&diff))
}

impl ConvexHull<3> for [Cartesian<3>] {
    /// Compute the convex hull of points in 3D with an incremental algorithm.
    ///
    /// Orientation tests are evaluated with [`robust::orient3d`], Shewchuk's
    /// adaptive precision predicate, which computes the exact sign of the
    /// orientation determinant. No tolerance is needed, and the hull does not
    /// depend on where the point set lies relative to the origin (up to the
    /// precision of the coordinates themselves).
    ///
    /// This is an $` O(NF) `$ algorithm, where *F* is the number of faces, and is suited for
    /// high-symmetry solids and polyhedra with fewer than ~1000 vertices.
    ///
    /// The facets are oriented with outward normals: their vertices are ordered
    /// counter-clockwise when viewed from outside the hull, so every point of the
    /// hull lies [`Side::Below`] or [`Side::On`] each of them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NotFinite`] when any coordinate of the input is not finite, and
    /// [`Error::DegeneratePolytope`] when there are fewer than four points, or all
    /// points are coplanar.
    ///
    /// [`Error`]: enum@Error
    /// [`robust::orient3d`]: https://docs.rs/robust/latest/robust/fn.orient3d.html
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::hull::ConvexHull;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let cube: [Cartesian<3>; 8] = [
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
    /// let (hull_vertices, facets) = cube.convex_hull()?;
    ///
    /// assert_eq!(hull_vertices.len(), 8);
    /// // Each square face is split into two triangles:
    /// assert_eq!(facets.len(), 12);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn convex_hull(&self) -> Result<(Vec<Cartesian<3>>, Vec<Facet<'_, 3>>), Error> {
        let (indices, faces) = incremental_hull(self)?;

        // The faces index the given points, which the facets are bound to.
        // The compacted list of hull vertices is returned separately.
        let facets = faces
            .iter()
            .map(|&[a, b, c]| Facet::new(self, [a, b, c]))
            .collect();
        let vertices: Vec<Cartesian<3>> = indices.iter().map(|&i| self[i]).collect();

        Ok((vertices, facets))
    }
}

/// Compute the convex hull of points in 3D with an incremental algorithm.
///
/// Returns the indices of the points that are vertices of the hull in increasing order,
/// and the triangular faces of the hull as oriented index triples. The faces are
/// oriented with outward normals, so that all points of the hull lie [`Side::Below`]
/// or [`Side::On`] them, and a point lies strictly outside a face when its
/// orientation against it is [`Side::Above`].
///
/// Starting from an initial tetrahedron, each point is inserted in turn.
/// Points strictly outside the current hull see a connected set of faces,
/// which are removed and replaced by new faces joining the point to the
/// boundary (horizon) of that set. Points that see no face lie inside or on
/// the surface of the current hull and are skipped.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when there are fewer than four points,
/// and [`Error::NotFinite`] when any coordinate of the input is not finite.
pub(crate) fn incremental_hull(
    points: &[Cartesian<3>],
) -> Result<(Vec<usize>, Vec<[usize; 3]>), Error> {
    // No convex body without at least 4 points.
    if points.len() < 4 {
        return Err(Error::DegeneratePolytope);
    }
    if !finite_points(points) {
        return Err(Error::NotFinite);
    }

    let (t0, t1, t2, t3) = initial_tetrahedron(points)?;

    // The four faces of the initial tetrahedron. Each face is oriented with an
    // outward normal, so the vertex opposite it, and with it the hull, lies
    // Below or On the face
    let mut faces: Vec<[usize; 3]> = vec![[t0, t1, t2], [t0, t2, t3], [t0, t3, t1], [t1, t3, t2]];
    let opposite = [t3, t1, t2, t0];
    for (face, o) in faces.iter_mut().zip(opposite) {
        if Cartesian::<3>::orientation(
            &[points[face[0]], points[face[1]], points[face[2]]],
            &points[o],
        )? == Side::Above
        {
            face.swap(0, 1);
        }
    }

    let in_initial = |i: usize| i == t0 || i == t1 || i == t2 || i == t3;

    // Reused scratch storage for the directed edges of the visible faces.
    let mut edges: Vec<(usize, usize)> = Vec::new();

    for p in 0..points.len() {
        if in_initial(p) {
            continue;
        }

        // Remove the faces that p strictly sees and collect their directed edges
        edges.clear();
        faces.retain(|&face| {
            let outside = [points[face[0]], points[face[1]], points[face[2]]];
            if Cartesian::<3>::orientation(&outside, &points[p])
                .is_ok_and(|side| side == Side::Above)
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
        // `uv` whose reverse `vu` is not also an edge of a visible face. Each becomes
        // a new face joined to point `p`, with consistent orientations.
        edges.sort_unstable();
        for (u, v) in edges
            .iter()
            .copied()
            .filter(|&(u, v)| edges.binary_search(&(v, u)).is_err())
        {
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

/// The first two vertices of a simplex in any dimension.
///
/// The first is the lexicographically smallest point of the set and the second is the
/// point farthest from it. Both are guaranteed to be vertices of the final hull.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when the set is empty or every point coincides
/// with the first.
fn first_two_vertices<const N: usize>(points: &[Cartesian<N>]) -> Result<(usize, usize), Error> {
    let a = points
        .iter()
        .position_min_by(|x, y| {
            x.coordinates
                .iter()
                .zip(&y.coordinates)
                .fold(Ordering::Equal, |order, (&x, &y)| {
                    order.then(x.total_cmp(&y))
                })
        })
        .ok_or(Error::DegeneratePolytope)?;
    let b = (0..points.len())
        .filter(|&i| points[i] != points[a])
        .max_by(|&i, &j| {
            (points[i] - points[a])
                .norm_squared()
                .total_cmp(&(points[j] - points[a]).norm_squared())
        })
        .ok_or(Error::DegeneratePolytope)?; // Every point coincides with a.
    Ok((a, b))
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
    let (a, b) = first_two_vertices(points)?;

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

impl ConvexHull<4> for [Cartesian<4>] {
    /// Compute the four-dimensional hull by inserting points into an initial
    /// simplex and triangulating the visible region of the boundary.
    ///
    /// # Errors
    /// Returns [`Error::NotFinite`] for nonfinite coordinates,
    /// [`Error::DegeneratePolytope`] if the points do not span four dimensions, or
    /// [`Error::NumericallyAmbiguousPolytope`] if an orientation cannot be resolved.
    #[inline]
    fn convex_hull(&self) -> Result<(Vec<Cartesian<4>>, Vec<Facet<'_, 4>>), Error> {
        let (indices, cells) = incremental_hull_4d(self)?;
        let vertices = indices.into_iter().map(|i| self[i]).collect();
        let facets = cells
            .into_iter()
            .map(|cell| Facet::new(self, cell))
            .collect();
        Ok((vertices, facets))
    }
}

/// Build the four-dimensional hull as vertex indices and outward-oriented cells.
fn incremental_hull_4d(points: &[Cartesian<4>]) -> Result<(Vec<usize>, Vec<[usize; 4]>), Error> {
    if points.len() < 5 {
        return Err(Error::DegeneratePolytope);
    }
    if !finite_points(points) {
        return Err(Error::NotFinite);
    }

    let initial = initial_fivecell(points)?;
    let outside = |[a, b, c, d]: [usize; 4], p| {
        orient4d(points[a], points[b], points[c], points[d], points[p]).map(|sign| sign > 0)
    };
    let mut cells = Vec::with_capacity(5);
    for (i, &opposite) in initial.iter().enumerate() {
        let mut cell = std::array::from_fn(|j| initial[j + usize::from(j >= i)]);
        if outside(cell, opposite)? {
            cell.swap(0, 1);
        }
        cells.push(cell);
    }

    let (mut retained, mut triangles) = (Vec::new(), Vec::new());
    for p in (0..points.len()).filter(|p| !initial.contains(p)) {
        triangles.clear();
        for [a, b, c, d] in cells.drain(..) {
            if outside([a, b, c, d], p)? {
                // The alternating boundary of the oriented tetrahedron.
                triangles.extend(
                    [[b, c, d], [a, d, c], [a, b, d], [a, c, b]].map(|[a, b, c]| {
                        // Rotate the smallest index first, preserving orientation.
                        match a.min(b).min(c) {
                            smallest if smallest == a => [a, b, c],
                            smallest if smallest == b => [b, c, a],
                            _ => [c, a, b],
                        }
                    }),
                );
            } else {
                retained.push([a, b, c, d]);
            }
        }
        std::mem::swap(&mut cells, &mut retained);

        // Interior triangles cancel with their reversed orientation.
        triangles.sort_unstable();
        for &[a, b, c] in &triangles {
            if triangles.binary_search(&[a, c, b]).is_err() {
                cells.push([p, a, b, c]);
            }
        }
    }

    let mut on_hull = vec![false; points.len()];
    for &i in cells.iter().flatten() {
        on_hull[i] = true;
    }
    Ok(((0..points.len()).filter(|&i| on_hull[i]).collect(), cells))
}

/// Find an initial five-cell for the incremental hull.
///
/// Start at the lexicographically smallest point. Choose each next point
/// by its distance from the affine span of the points already selected.
fn initial_fivecell(points: &[Cartesian<4>]) -> Result<[usize; 5], Error> {
    let (a, b) = first_two_vertices(points)?;
    let mut result = [a; 5];
    result[1] = b;
    let origin = points[a];

    // Find the independent coordinates in ab so we can re-use the vectors later.
    let pivot_coord = (0..4)
        .find(|&axis| origin[axis] != points[b][axis])
        .ok_or(Error::DegeneratePolytope)?;
    let mut basis = [origin; 5];
    basis[1] = points[b];

    for (i, axis) in (0..4).filter(|&axis| axis != pivot_coord).enumerate() {
        let coordinate = if origin[axis] == 0.0 {
            points.iter().map(|p| p[axis].abs()).fold(0.0, f64::max)
        } else {
            -origin[axis]
        };
        if coordinate == origin[axis] {
            return Err(Error::DegeneratePolytope);
        }
        basis[i + 2][axis] = coordinate;
    }

    let ab = points[b] - origin;
    result[2] = farthest_4d(points, &mut basis, &[a, b], |point| {
        ab.wedge(&(point - origin)).norm_squared()
    })?;

    let ac = points[result[2]] - origin;
    result[3] = farthest_4d(points, &mut basis, &[result[0], b, result[2]], |point| {
        Cartesian::counary_cross(&[ab, ac, point - origin]).norm_squared()
    })?;

    let normal = Cartesian::counary_cross(&[ab, ac, points[result[3]] - origin]);
    result[4] = farthest_4d(points, &mut basis, &[a, b, result[2], result[3]], |point| {
        normal.dot(&(point - origin)).abs()
    })?;
    Ok(result)
}

/// Find the farthest point from a set of up to 5 points in 4D.
///
/// This is a general helper, which allows for ranking of varied subdimensional
/// geometric components. Use `Cartesian::wedge` for planes, `Cartesian::counary_cross`
/// for hyperplanes, and a standard dot product for full-dimensional volumes.
fn farthest_4d(
    points: &[Cartesian<4>],
    basis: &mut [Cartesian<4>; 5],
    selected: &[usize],
    rank: impl Fn(Cartesian<4>) -> f64,
) -> Result<usize, Error> {
    let next = selected.len();
    let (mut best, mut error) = (None, Error::DegeneratePolytope);
    for (i, &point) in points.iter().enumerate() {
        if selected.contains(&i) {
            continue;
        }
        let distance = rank(point).max(0.0);
        if best.is_some_and(|(rank, _, _)| distance <= rank) {
            continue;
        }
        for slot in next..5 {
            let mut candidate = *basis;
            candidate[slot] = point;
            let [v0, v1, v2, v3, v4] = candidate;
            match orient4d(v0, v1, v2, v3, v4) {
                Ok(0) => (),
                Ok(_) => {
                    best = Some((distance, i, slot));
                    break;
                }
                Err(err) => error = err,
            }
        }
    }

    let (_, index, slot) = best.ok_or(error)?;
    basis[slot] = points[index];
    basis.swap(next, slot);
    Ok(index)
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

    #[test]
    fn test_4d_hull_prefers_farther_seed_vertices() -> Result<(), Error> {
        let points: [Cartesian<4>; 6] = [
            [0.0, 0.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0, 0.0].into(),
            [2.0, 0.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0, 0.0].into(),
            [0.0, 0.0, 1.0, 0.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
        ];
        let (vertices, facets) = points.convex_hull()?;
        assert_eq!(
            vertices,
            [points[0], points[2], points[3], points[4], points[5]]
        );
        assert_eq!(facets.len(), 5);
        Ok(())
    }

    #[test]
    fn test_4d_hull() -> Result<(), Error> {
        // The 16-cell has eight vertices and 16 tetrahedral facets.
        let points: [Cartesian<4>; 9] = std::array::from_fn(|i| {
            Cartesian::from(std::array::from_fn(|axis| {
                if i / 2 == axis {
                    if i % 2 == 0 { 1.0 } else { -1.0 }
                } else {
                    0.0
                }
            }))
        });
        let (vertices, facets) = points.convex_hull()?;
        assert_eq!(vertices, points[..8]);
        assert_eq!(facets.len(), 16);
        for facet in facets {
            assert_eq!(facet.vertices(), &points);
            assert_eq!(facet.orientation(&points[8])?, Side::Below);
            for point in &points {
                assert_ne!(facet.orientation(point)?, Side::Above);
            }
        }
        Ok(())
    }

    #[rstest]
    fn test_square_corners() {
        let points: Vec<Cartesian<2>> = vec![[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
            let (vertices1, _) = points1
                .convex_hull()
                .expect("hard-coded points should lie on a convex hull");

            let mut rng = StdRng::seed_from_u64(123);
            let points2: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let (vertices2, _) = points2
                .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let (vertices, _) = points
            .convex_hull()
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
        let result = points.convex_hull();
        check!(result == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    #[case::empty(vec![])]
    #[case::one_point(vec![[0.0, 0.0]])]
    #[case::two_points(vec![[0.0, 0.0], [1.0, 0.0]])]
    fn test_too_few_points(#[case] points: Vec<[f64; 2]>) {
        let points: Vec<Cartesian<2>> = points.into_iter().map(Cartesian::from).collect();
        check!(points.convex_hull() == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    #[case::infinity(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [f64::INFINITY, 0.5]])]
    #[case::nan(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [f64::NAN, 0.5]])]
    fn test_non_finite_points_rejected_2d(#[case] points: Vec<[f64; 2]>) {
        let points: Vec<Cartesian<2>> = points.into_iter().map(Cartesian::from).collect();
        check!(points.convex_hull() == Err(Error::NotFinite));
    }

    #[rstest]
    #[case::infinity(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, f64::INFINITY], [0.0, 0.0, 1.0]])]
    #[case::nan(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0]])]
    fn test_non_finite_points_rejected_3d(#[case] points: Vec<[f64; 3]>) {
        let points: Vec<Cartesian<3>> = points.into_iter().map(Cartesian::from).collect();
        check!(points.convex_hull() == Err(Error::NotFinite));
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
        let (from_slice, _) = points[..]
            .convex_hull()
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_slice, &hull);

        // An owned Vec of points.
        let (from_vec, _) = points
            .convex_hull()
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_vec, &hull);

        // An array of points.
        let points = [
            [0.0, 0.0].into(),
            [2.0, 0.0].into(),
            [1.0, 1.0].into(),
            [0.5, 0.25].into(), // interior point
        ];
        let (from_array, _) = points
            .convex_hull()
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_array, &hull);
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

            let (vertices, _) = points
                .convex_hull()
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

            let (vertices, _) = points
                .convex_hull()
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
    /// * Every face is a supporting plane, with its normal pointing outward
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
                    side != Side::Above,
                    "point {q} lies outside the face {face:?}"
                );
            }
        }

        // Every face normal points away from the hull. The centroid of the hull
        // vertices is an interior point, so it lies strictly below each face.
        let centroid = vertices
            .iter()
            .fold(Cartesian::from([0.0; 3]), |acc, &i| acc + points[i])
            / vertices.len() as f64;
        for &face in faces {
            let [a, b, c] = [points[face[0]], points[face[1]], points[face[2]]];
            let normal = (b - a).cross(&(c - a));
            check!(
                normal.dot(&(centroid - a)) < 0.0,
                "the normal of the face {face:?} points into the hull"
            );
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
        check!(points.convex_hull() == Err(Error::DegeneratePolytope));
    }

    #[test]
    fn test_3d_hull_can_retain_edge_midpoint() -> Result<(), Error> {
        let points: [Cartesian<3>; 6] = [
            [0.0, 0.0, 0.0].into(),
            [10.0, 0.0, 0.0].into(),
            [0.0, 10.0, 0.0].into(),
            [0.0, 0.0, 10.0].into(),
            [8.5, 1.0, 1.0].into(),
            [7.0, 2.0, 2.0].into(),
        ];
        let midpoint = (points[1] + points[5]) * 0.5;
        let (vertices, facets) = points.convex_hull()?;
        assert!(vertices.contains(&midpoint));
        assert!(
            facets
                .iter()
                .any(|facet| facet.as_simplex().contains(&midpoint))
        );
        Ok(())
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
        let mut points = cube();
        points.extend([
            Cartesian::from([0.0, 0.0, 0.0]),    // index 8: cube center
            Cartesian::from([1.0, 0.0, 0.0]),    // index 9: face center
            Cartesian::from([1.0, 1.0, 0.0]),    // index 10: face center
            Cartesian::from([-1.0, -1.0, -1.0]), // index 11: duplicate corner
        ]);

        let (vertices, facets) = points
            .convex_hull()
            .expect("hard-coded points should form a convex body");

        assert_eq!(vertices.len(), 8);
        assert_eq!(facets.len(), 12); // Each square face is split into two triangles.

        let mut referenced: Vec<Cartesian<3>> = Vec::new();
        for facet in &facets {
            // The facets are bound to the given points and index them.
            assert!(std::ptr::eq(facet.vertices().as_ptr(), points.as_ptr()));
            for &i in &facet.indices() {
                assert!(
                    i < 8,
                    "facet references the interior or duplicate point {i}"
                );
            }
            referenced.extend(facet.as_simplex());
        }
        // Every returned hull vertex is a corner of some facet.
        for v in &vertices {
            assert!(referenced.contains(v));
        }
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

        let (from_slice, _) = points[..]
            .convex_hull()
            .expect("hard-coded points should form a convex body");
        let (from_vec, _) = points
            .convex_hull()
            .expect("hard-coded points should form a convex body");
        let points = [
            [-1.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0].into(),
            [0.0, -1.0, 0.0].into(),
            [0.0, 1.0, 0.0].into(),
            [0.0, 0.0, -1.0].into(),
            [0.0, 0.0, 1.0].into(),
        ];
        let (from_array, _) = points
            .convex_hull()
            .expect("hard-coded points should form a convex body");

        assert_eq!(from_slice.len(), 6);
        itertools::assert_equal(&from_slice, &from_vec);
        itertools::assert_equal(&from_slice, &from_array);
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
        let facet = Facet::new(&vertices, [0, 1]);

        check!(facet.orientation(&Cartesian::from([0.5, 0.5])) == Ok(Side::Above));
        check!(facet.orientation(&Cartesian::from([0.5, -0.5])) == Ok(Side::Below));
        check!(facet.orientation(&Cartesian::from([2.0, 0.0])) == Ok(Side::On));

        // Non-finite coordinates are rejected.
        let nan = Cartesian::from([f64::NAN, 0.0]);
        check!(facet.orientation(&nan) == Err(Error::NotFinite));

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
        let (hull_vertices, facets) = points
            .convex_hull()
            .expect("hard-coded points should form a convex body");
        for facet in &facets {
            // Every point of the hull lies Below or On an outward-oriented edge.
            for q in &points {
                check!(
                    facet
                        .orientation(q)
                        .expect("finite coordinates resolve exactly")
                        != Side::Above
                );
            }

            // The normal of the edge's vertex order points away from the hull.
            let centroid = hull_vertices
                .iter()
                .fold(Cartesian::from([0.0; 2]), |acc, p| acc + *p)
                / hull_vertices.len() as f64;
            let [a, b] = facet.as_simplex();
            let edge = b - a;
            let normal = Cartesian::from([-edge[1], edge[0]]);
            let midpoint = (a + b) / 2.0;
            check!(normal.dot(&(midpoint - centroid)) > 0.0);
        }
    }

    #[rstest]
    fn test_facet_orientation_3d() {
        let vertices = [
            Cartesian::from([0.0, 0.0, 0.0]),
            Cartesian::from([1.0, 0.0, 0.0]),
            Cartesian::from([0.0, 1.0, 0.0]),
        ];
        let facet = Facet::new(&vertices, [0, 1, 2]);

        check!(facet.orientation(&Cartesian::from([0.0, 0.0, -1.0])) == Ok(Side::Below));
        check!(facet.orientation(&Cartesian::from([0.0, 0.0, 1.0])) == Ok(Side::Above));
        check!(facet.orientation(&Cartesian::from([1.0, 1.0, 0.0])) == Ok(Side::On));

        let nan = Cartesian::from([f64::NAN, 0.0, 0.0]);
        check!(facet.orientation(&nan) == Err(Error::NotFinite));

        let points = cube();
        let (_hull_vertices, facets) = points
            .convex_hull()
            .expect("hard-coded points should form a convex body");
        for facet in &facets {
            // Every point of the hull lies Below or On an outward-oriented facet.
            for q in &points {
                let orientation = facet
                    .orientation(q)
                    .expect("finite coordinates resolve exactly");
                check!(orientation != Side::Above);
            }

            // The normal of the facet's vertex order points away from the hull,
            // here centered on the origin: counter-clockwise seen from outside.
            let [a, b, c] = facet.as_simplex();
            let normal = (b - a).cross(&(c - a));
            let centroid = (a + b + c) / 3.0;
            check!(normal.dot(&centroid) > 0.0);
        }
    }

    #[rstest]
    fn test_hull_orientation_convention() {
        // In both dimensions, every point of the hull lies Below or On each of
        // its facets, and a point outside lies Above at least one of them.
        let square = [
            Cartesian::from([0.0, 0.0]),
            Cartesian::from([2.0, 0.0]),
            Cartesian::from([2.0, 2.0]),
            Cartesian::from([0.0, 2.0]),
        ];
        let (vertices, facets) = square.convex_hull().unwrap();
        for facet in &facets {
            for q in &vertices {
                check!(facet.orientation(q).unwrap() != Side::Above);
            }
        }
        check!(
            facets
                .iter()
                .any(|f| f.orientation(&Cartesian::from([1.0, 3.0])) == Ok(Side::Above))
        );

        let points = cube();
        let (vertices, facets) = points.convex_hull().unwrap();
        for facet in &facets {
            for q in &vertices {
                check!(facet.orientation(q).unwrap() != Side::Above);
            }
        }
        check!(
            facets
                .iter()
                .any(|f| f.orientation(&Cartesian::from([0.0, 0.0, 2.0])) == Ok(Side::Above))
        );
    }
}
