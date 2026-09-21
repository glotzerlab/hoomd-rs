// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Compute the convex hull of a set of points.

use std::{borrow::Borrow, cmp::Ordering};

use itertools::Itertools;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

use crate::{Error, orient4d::orient4d};
use hoomd_linear_algebra::{MatMul, matrix::Matrix};
use hoomd_vector::{Cartesian, InnerProduct};

/// A facet of a convex hull: the indices of the `N` vertices of the hull that bound it,
/// in the order given by [`ConvexHull::convex_hull`].
///
/// A facet of a two-dimensional hull is an edge (`Facet<2>`), a facet of a
/// three-dimensional hull is a triangle (`Facet<3>`), and a facet of a
/// four-dimensional hull is a tetrahedron (`Facet<4>`).
#[serde_as]
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
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

    /// The simplex the facet spans: its vertices, looked up in `points`.
    ///
    /// A facet of an `N`-dimensional hull is an `(N - 1)`-simplex bounded by
    /// `N` vertices of the hull. An edge of a polygon returns its two
    /// endpoints, a triangle of a polyhedron its three corners.
    ///
    /// # Panics
    ///
    /// Panics when a vertex index is out of bounds, like indexing `points`.
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::Facet;
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
    pub fn as_simplex(&self, points: &[Cartesian<N>]) -> [Cartesian<N>; N] {
        self.indices.map(|i| points[i])
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
/// use hoomd_geometry::ConvexHull;
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
/// //
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
    /// Returns [`Error`] if the input does not form a convex body with `>=N+1` points.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    ///
    /// Compute the hull from an iterator of points:
    /// ```
    /// use hoomd_geometry::ConvexHull;
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

                if predicate_orient2d((p, n), c) <= 0 {
                    // Point n is inside the hull, remove it by shrinking the hull
                    n_vertices_on_hull -= 1;
                } else {
                    break;
                }
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

/// Determines whether a point `test` is to the left, right, or collinear with `edge`.
///
/// The sign of the orientation determinant is computed with [`robust::orient2d`],
/// Shewchuk's adaptive precision predicate: the sign is therefore *exact* for the
/// coordinates as stored, and the resulting hull is exact up to the precision of the
/// cooridinates themselves.
///
/// Returns 1 when `test` lies to the left of the directed edge, -1 when it lies to the
/// right, and 0 when the three points are exactly collinear.
///
/// # Note
///
/// Because the sign is exact, it is antisymmetric in its arguments (swapping two points
/// points negates the sign) and invariant under cyclic permutation of the three inputs.
#[inline]
fn predicate_orient2d((p, q): (Cartesian<2>, Cartesian<2>), test: Cartesian<2>) -> i64 {
    let orientation = robust::orient2d(
        robust::Coord { x: p[0], y: p[1] },
        robust::Coord { x: q[0], y: q[1] },
        robust::Coord {
            x: test[0],
            y: test[1],
        },
    );

    match orientation.total_cmp(&0.0) {
        Ordering::Greater => 1,
        Ordering::Less => -1,
        Ordering::Equal => 0,
    }
}

/// Compact a set of points to the vertices on a hull, provided as indices.
///
/// Returns the hull vertices in the order of the input points, and remaps the
/// cells from indices of the input points to indices into the returned vertex
/// list. The vertex indices are sorted, so a binary search locates the new
/// position of each vertex.
fn compact_hull<const N: usize>(
    points: &[Cartesian<N>],
    indices: Vec<usize>,
    cells: &[Facet<N>],
) -> (Vec<Cartesian<N>>, Vec<Facet<N>>) {
    let position = |i: usize| {
        indices
            .binary_search(&i)
            .expect("the vertices of a cell are vertices of the hull")
    };
    let facets = cells
        .iter()
        .map(|cell| Facet::new(cell.indices().map(position)))
        .collect();
    let vertices = indices.into_iter().map(|i| points[i]).collect();

    (vertices, facets)
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
    /// Returns [`Error`] if the input points do not form a convex body with 4 or more points.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::ConvexHull;
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

        let (indices, faces) = incremental_hull(&points)?;

        Ok(compact_hull(&points, indices, &faces))
    }
}

/// Compute the convex hull of points in 3D with an incremental algorithm.
///
/// Returns the indices of the points that are vertices of the hull in increasing order,
/// and the triangular faces of the hull as oriented index triples. A point `p` lies
/// strictly outside a face `(a, b, c)` when `orient3d(a, b, c, p) > 0`. Faces are
/// oriented so that all points of the hull evaluate to zero or less against them.
///
/// Starting from an initial tetrahedron, each point is inserted in turn.
/// Points strictly outside the current hull see a connected set of faces,
/// which are removed and replaced by new faces joining the point to the
/// boundary (horizon) of that set. Points that see no face lie inside or on
/// the surface of the current hull and are skipped.
fn incremental_hull(points: &[Cartesian<3>]) -> Result<(Vec<usize>, Vec<Facet<3>>), Error> {
    let (t0, t1, t2, t3) = initial_tetrahedron(points)?;

    // The four faces of the initial tetrahedron. Each face is oriented so
    // that the vertex opposite it is on its negative side, which places
    // points strictly outside a face on its positive side.
    let mut faces: Vec<Facet<3>> = vec![
        Facet::new([t0, t1, t2]),
        Facet::new([t0, t2, t3]),
        Facet::new([t0, t3, t1]),
        Facet::new([t1, t3, t2]),
    ];
    let opposite = [t3, t1, t2, t0];
    for (face, o) in faces.iter_mut().zip(opposite) {
        if orient3d_at(points, *face, o) > 0.0 {
            face.indices.swap(0, 1);
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
            if orient3d_at(points, face, p) > 0.0 {
                let [a, b, c] = face.indices();
                edges.push((a, b));
                edges.push((b, c));
                edges.push((c, a));
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
            faces.push(Facet::new([u, v, p]));
        }
    }

    // The vertices of the hull are the vertices of the final faces.
    let mut vertices: Vec<usize> = faces.iter().flat_map(Facet::indices).collect();
    vertices.sort_unstable();
    vertices.dedup();

    Ok((vertices, faces))
}

/// Seed an initial simplex with two vertices of the hull.
///
/// The first vertex `a` is the lexicographically smallest point of the set,
/// and the second `b` is the point farthest from it.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when every point coincides with `a`.
fn find_two_points_on_hull<const N: usize>(
    points: &[Cartesian<N>],
) -> Result<(usize, usize), Error> {
    // a is the lowest, leftmost point in the set.
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
    if points[b] == points[a] {
        return Err(Error::DegeneratePolytope); // Every point coincides with a.
    }

    Ok((a, b))
}

/// Find the next vertex of an initial simplex.
///
/// The point returned is the one that maximizes `rank` among those that are
/// affinely independent of the `chosen` vertices, as `independent` decides
/// exactly: it is a vertex of the hull, whatever the ranking breaks ties over.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when every point of the set lies in
/// the span of the `chosen` vertices. Propagates the errors of `independent`.
fn find_next_point_on_hull<const N: usize>(
    points: &[Cartesian<N>],
    chosen: &[usize],
    mut is_independent: impl FnMut(usize) -> Result<bool, Error>,
    rank: impl FnMut(usize) -> f64,
) -> Result<usize, Error> {
    farthest_point(
        points,
        |i| Ok(!chosen.contains(&i) && is_independent(i)?),
        rank,
    )?
    .ok_or(Error::DegeneratePolytope)
}

/// Find the third vertex of an initial simplex.
///
/// The point returned is the farthest from the line through `a` and `b` among those not
/// exactly noncolinear with it, ranked by the squared area of the triangle they span.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when every point of the set is
/// colinear with `a` and `b`.
fn find_third_point_on_hull<const N: usize>(
    points: &[Cartesian<N>],
    a: usize,
    b: usize,
) -> Result<usize, Error> {
    let ab = points[b] - points[a];
    find_next_point_on_hull(
        points,
        &[a, b],
        |i| Ok(!collinear([points[a], points[b], points[i]])),
        |i| {
            let ap = points[i] - points[a];
            ab.norm_squared() * ap.norm_squared() - ap.dot(&ab).powi(2)
        },
    )
}

/// Find an initial tetrahedron for the incremental hull.
///
/// The four points are vertices of the hull: `a` is the lexographically smallest point,
/// `b` is the furthest from that, `c` is the furthest noncolinear point from the line
/// `ab`, and `d` is the furthest noncoplanar point to the triangle `abc`.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when all points coincide, all are colinear, or
/// all are coplanar.
fn initial_tetrahedron(points: &[Cartesian<3>]) -> Result<(usize, usize, usize, usize), Error> {
    let (a, b) = find_two_points_on_hull(points)?;
    let c = find_third_point_on_hull(points, a, b)?;

    // d is the farthest point from the plane through a, b and c, ranked by
    // the magnitude of the orientation determinant. Any nonzero value from
    // the exact predicate guarantees a valid tetrahedron.
    let abc = Facet::new([a, b, c]);
    let d = find_next_point_on_hull(
        points,
        &[a, b, c],
        |i| Ok(orient3d_at(points, abc, i) != 0.0),
        |i| orient3d_at(points, abc, i).abs(),
    )?;

    Ok((a, b, c, d))
}

/// The orientation determinant of four points given by index.
///
/// The sign of the returned value is positive when `d` lies strictly outside the face
/// `(a, b, c)` oriented as by [`incremental_hull`], negative when it lies strictly
/// inside, and zero when the four points are coplanar.
#[inline]
fn orient3d_at(points: &[Cartesian<3>], face: Facet<3>, test: usize) -> f64 {
    let [a, b, c] = face.as_simplex(points);
    let coord = |p: &Cartesian<3>| robust::Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    };

    robust::orient3d(coord(&a), coord(&b), coord(&c), coord(&points[test]))
}

impl ConvexHull<4> for Cartesian<4> {
    /// Compute the convex hull of points in 4D with an incremental algorithm.
    ///
    /// Orientation tests are evaluated with [`orient4d`], which computes the exact sign
    /// of the orientation determinant. No tolerance is needed, and the hull does not
    /// depend on where the point set lies relative to the origin (up to the precision
    /// of the coordinates themselves).
    ///
    /// This is an O(n*f) algorithm, where f is the number of tetrahedral cells of
    /// the simplicial hull. Unlike in three dimensions, f itself grows as O(n^2)
    /// in the worst case (e.g. for points on a hypersphere, where most points are
    /// vertices), so the total cost can reach O(n^3). It is suited for high-symmetry
    /// polychora and point sets with fewer than ~1000 vertices.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the input points do not form a convex body with 5 or more points.
    /// Returns [`Error::NumericallyAmbiguousPolytope`] when the orientation of any
    /// five points cannot be resolved exactly, which requires coordinates whose
    /// exponents span more than 72 powers of two.
    ///
    /// [`Error`]: enum@Error
    ///
    /// # Example
    ///
    /// ```
    /// use hoomd_geometry::ConvexHull;
    /// use hoomd_vector::Cartesian;
    ///
    /// # fn main() -> Result<(), hoomd_geometry::Error> {
    /// let hyperoctahedron = [
    ///     [1.0, 0.0, 0.0, 0.0].into(),
    ///     [-1.0, 0.0, 0.0, 0.0].into(),
    ///     [0.0, 1.0, 0.0, 0.0].into(),
    ///     [0.0, -1.0, 0.0, 0.0].into(),
    ///     [0.0, 0.0, 1.0, 0.0].into(),
    ///     [0.0, 0.0, -1.0, 0.0].into(),
    ///     [0.0, 0.0, 0.0, 1.0].into(),
    ///     [0.0, 0.0, 0.0, -1.0].into(),
    /// ];
    ///
    /// let (hull_vertices, cells) = Cartesian::<4>::convex_hull(&hyperoctahedron)?;
    ///
    /// assert_eq!(hull_vertices.len(), 8);
    /// assert_eq!(cells.len(), 16); // The 16-cell is bounded by 16 tetrahedra.
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn convex_hull<I>(points: I) -> Result<(Vec<Self>, Vec<Facet<4>>), Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>,
    {
        let points: Vec<Self> = points.into_iter().map(|p| *p.borrow()).collect();

        // No convex body without at least 5 points.
        if points.len() < 5 {
            return Err(Error::DegeneratePolytope);
        }

        let (indices, cells) = incremental_hull_4d(&points)?;

        Ok(compact_hull(&points, indices, &cells))
    }
}

/// Compute the convex hull of points in 4D with an incremental algorithm.
///
/// Returns the indices of the points that are vertices of the hull in increasing
/// order, and the tetrahedral cells of the hull as oriented index quadruples. A
/// point `p` lies strictly outside a cell `(a, b, c, d)` when
/// `orient4d(a, b, c, d, p) > 0`. Cells are oriented so that all points of the
/// hull evaluate to zero or less against them.
fn incremental_hull_4d(points: &[Cartesian<4>]) -> Result<(Vec<usize>, Vec<Facet<4>>), Error> {
    let pentachoron = initial_pentachoron(points)?;

    // The five cells of the initial pentachoron: the one that omits a vertex
    // collects the other four, in order. Each is oriented so that the vertex
    // opposite it is on its negative side, which places points strictly
    // outside a cell on its positive side. Note that unlike in three
    // dimensions, this orientation is the opposite of the one the boundary
    // chain of the pentachoron induces on its cells.
    let mut cells: Vec<Facet<4>> = pentachoron
        .iter()
        .enumerate()
        .map(|(omitted, &opposite)| {
            let mut cell = Facet::new(std::array::from_fn(|k| {
                pentachoron[if k < omitted { k } else { k + 1 }]
            }));
            if orient4d_at(points, cell, opposite)? > 0 {
                cell.indices.swap(1, 2);
            }
            Ok(cell)
        })
        .collect::<Result<_, Error>>()?;

    // Reused scratch storage for the oriented boundary triangles of the visible
    // cells and for the cells that survive an insertion; swapping the buffers
    // back keeps their allocations.
    let mut triangles: Vec<(Facet<3>, bool)> = Vec::new();
    let mut retained: Vec<Facet<4>> = Vec::new();

    // Insert every point of the set that is not a vertex of the initial 5cell
    for p in (0..points.len()).filter(|&p| !pentachoron.contains(&p)) {
        // Remove the cells that p strictly sees and collect their oriented boundary
        triangles.clear();
        retained.clear();
        for cell in cells.drain(..) {
            if orient4d_at(points, cell, p)? > 0 {
                triangles.extend(boundary_triangles(cell));
            } else {
                retained.push(cell);
            }
        }
        std::mem::swap(&mut cells, &mut retained);

        // p sees no cell: it must be inside (or on) the hull.
        if triangles.is_empty() {
            continue;
        }

        // The boundary (horizon) of the visible region is formed by the oriented
        // triangles whose orientation-reversed key is not also a boundary
        // triangle of a visible cell; triangles interior to the region appear
        // with both orientations. Each horizon triangle becomes a new cell
        // joined to p.
        triangles.sort_unstable();
        for &(triangle, flipped) in &triangles {
            if triangles.binary_search(&(triangle, !flipped)).is_err() {
                cells.push(build_cone(p, triangle, flipped));
            }
        }
    }

    // The vertices of the hull are the vertices of the final cells.
    let mut vertices: Vec<usize> = cells.iter().flat_map(Facet::indices).collect();
    vertices.sort_unstable();
    vertices.dedup();

    Ok((vertices, cells))
}

/// The cell that joins a point to a triangle of the horizon.
///
/// The new point comes first: the shared triangle then sits at index 0 of the
/// cell's boundary chain, where it keeps the orientation that the visible cell
/// it bounded induced on it, so the surface stays consistently oriented. The
/// triangle is given in canonical form; a flipped one is represented by a
/// transposition of its sorted vertices.
#[inline]
fn build_cone(p: usize, triangle: Facet<3>, flipped: bool) -> Facet<4> {
    let [s0, s1, s2] = triangle.indices();
    let (a, b) = if flipped { (s1, s0) } else { (s0, s1) };
    Facet::new([p, a, b, s2])
}

/// Find an initial pentachoron for the incremental hull.
///
/// The five points are vertices of the hull: `a` is the lexicographically
/// smallest point, `b` is the furthest from that, `c` is the furthest
/// noncolinear point from the line `ab`, `d` is the furthest noncoplanar point
/// from the plane `abc`, and `e` is the furthest point off the hyperplane
/// through `a`, `b`, `c` and `d`.
///
/// Exact predicates decide which points are colinear, coplanar and
/// hypercoplanar; the distances only rank the candidates.
///
/// # Errors
///
/// Returns [`Error::DegeneratePolytope`] when all points coincide, are
/// colinear, lie in a common plane or in a common hyperplane. Returns
/// [`Error::NumericallyAmbiguousPolytope`] when an orientation cannot be
/// resolved exactly.
#[expect(clippy::many_single_char_names, reason = "clarity")]
fn initial_pentachoron(points: &[Cartesian<4>]) -> Result<[usize; 5], Error> {
    let (a, b) = find_two_points_on_hull(points)?;
    let c = find_third_point_on_hull(points, a, b)?;
    let ab = points[b] - points[a];
    let ac = points[c] - points[a];

    // d is the farthest noncoplanar point from the plane abc, ranked by the
    // squared volume of the parallelepiped spanned by ab, ac and ad.
    let d = find_next_point_on_hull(
        points,
        &[a, b, c],
        |i| Ok(!coplanar_4d([points[a], points[b], points[c], points[i]])),
        |i| {
            let edges = Matrix {
                rows: [
                    ab.coordinates,
                    ac.coordinates,
                    (points[i] - points[a]).coordinates,
                ],
            };
            edges.matmul(&edges.transpose()).determinant()
        },
    )?;

    // e is the farthest point off the hyperplane through a, b, c and d, ranked
    // by the magnitude of the determinant of the difference matrix. The exact
    // predicate must filter the candidates: a point can have a larger spurious
    // filtered determinant than another that is exactly off the hyperplane.
    let cell = Facet::new([a, b, c, d]);
    let e = find_next_point_on_hull(
        points,
        &[a, b, c, d],
        |i| Ok(orient4d_at(points, cell, i)? != 0),
        |i| {
            Matrix {
                rows: cell.as_simplex(points).map(|p| (p - points[i]).coordinates),
            }
            .determinant()
            .abs()
        },
    )?;

    Ok([a, b, c, d, e])
}

/// The index of the point that maximizes `rank` among those for which
/// `eligible` holds, or `None` when no point is eligible.
///
/// The ranking only breaks ties between eligible points; the eligibility
/// test decides validity. An ineligible point ranks at negative infinity,
/// where the comparison is strict, so it never beats the running best. Ranks
/// are clamped at zero so that the NaN of an overflowing rank expression does
/// not discard an eligible point (any eligible point is valid, however
/// poorly it ranks).
///
/// # Errors
///
/// Propagates the errors of `eligible`.
fn farthest_point<const N: usize>(
    points: &[Cartesian<N>],
    mut eligible: impl FnMut(usize) -> Result<bool, Error>,
    mut rank: impl FnMut(usize) -> f64,
) -> Result<Option<usize>, Error> {
    (0..points.len())
        .try_fold((None, f64::NEG_INFINITY), |best, i| {
            let rank = if eligible(i)? {
                rank(i).max(0.0)
            } else {
                f64::NEG_INFINITY
            };
            Ok(if rank > best.1 { (Some(i), rank) } else { best })
        })
        .map(|best| best.0)
}

/// The orientation determinant of five points given by index.
///
/// The sign of the returned value is positive when `test` lies strictly outside the
/// cell `(a, b, c, d)` oriented as by [`incremental_hull_4d`], negative when it lies
/// strictly inside, and zero when the five points are hypercoplanar.
///
/// # Errors
///
/// Returns [`Error::NumericallyAmbiguousPolytope`] when the orientation cannot be
/// resolved exactly.
#[inline]
fn orient4d_at(points: &[Cartesian<4>], cell: Facet<4>, test: usize) -> Result<i64, Error> {
    let [a, b, c, d] = cell.as_simplex(points);
    orient4d(a, b, c, d, points[test])
}

/// Whether the three vertices of a simplex are exactly collinear.
///
/// The points are collinear when they are collinear in each coordinate plane
/// projection, decided by the two-dimensional exact predicate. The projected
/// coordinates are first divided by a common power of two, which is exact and
/// keeps the projected determinants from overflowing: unlike `orient4d`, the
/// lower-dimensional predicates are exact only while their products remain
/// finite.
#[inline]
fn collinear<const N: usize>(simplex: [Cartesian<N>; 3]) -> bool {
    let scale = coordinate_scale(&simplex);
    let scaled = simplex.map(|p| p / scale);
    let coord = |p: &Cartesian<N>, i: usize, j: usize| robust::Coord { x: p[i], y: p[j] };
    (0..N)
        .flat_map(|i| (i + 1..N).map(move |j| (i, j)))
        .all(|(i, j)| {
            robust::orient2d(
                coord(&scaled[0], i, j),
                coord(&scaled[1], i, j),
                coord(&scaled[2], i, j),
            ) == 0.0
        })
}

/// Whether the four vertices of a simplex are exactly coplanar.
///
/// The points are coplanar when they are coplanar in each coordinate triple
/// projection, decided by the three-dimensional exact predicate, again after
/// division by a common power of two (see [`collinear`]).
#[inline]
fn coplanar_4d(simplex: [Cartesian<4>; 4]) -> bool {
    let scale = coordinate_scale(&simplex);
    let scaled = simplex.map(|p| p / scale);
    let coord = |p: &Cartesian<4>, i: usize, j: usize, k: usize| robust::Coord3D {
        x: p[i],
        y: p[j],
        z: p[k],
    };
    [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        .iter()
        .all(|&(i, j, k)| {
            robust::orient3d(
                coord(&scaled[0], i, j, k),
                coord(&scaled[1], i, j, k),
                coord(&scaled[2], i, j, k),
                coord(&scaled[3], i, j, k),
            ) == 0.0
        })
}

/// A power of two that bounds the largest coordinate magnitude of the points.
///
/// Dividing the coordinates by `scale` is exact and brings them into `[-2, 2)`, so the
/// products inside the 2D and 3D predicates cannot overflow.
#[inline]
fn coordinate_scale<const N: usize>(points: &[Cartesian<N>]) -> f64 {
    /// The fraction bits of an `f64`; `MANTISSA_DIGITS` counts the implicit one.
    const FRACTION_BITS: u32 = f64::MANTISSA_DIGITS - 1;

    /// The mask of the exponent field: every bit above the fraction but the sign.
    const EXPONENT_MASK: u64 = u64::MAX >> FRACTION_BITS >> 1;

    let max = points
        .iter()
        .flat_map(|p| p.coordinates.iter())
        .fold(0.0_f64, |largest, &coordinate| {
            largest.max(coordinate.abs())
        });
    if !max.is_finite() || max == 0.0 {
        return 1.0; // Degenerate or non-finite inputs are rejected elsewhere
    }
    // The power of two that carries the exponent field of the largest magnitude, and
    // the smallest normal one stands in for subnormal magnitudes, where exponent==0
    let exponent = (max.to_bits() >> FRACTION_BITS) & EXPONENT_MASK;
    f64::from_bits(exponent << FRACTION_BITS).max(f64::MIN_POSITIVE)
}

/// The oriented boundary triangles of a cell, as canonical keys.
///
/// The oriented boundary of the cell `(c0, c1, c2, c3)` is the chain
/// `+[c1,c2,c3] - [c0,c2,c3] + [c0,c1,c3] - [c0,c1,c2]`: the term omitting `ci`
/// has the sign `(-1)^i`. Each triangle is keyed by its sorted vertices alongside a
/// flag that is set when the oriented triangle is an odd permutation of the order
#[inline]
fn boundary_triangles(cell: Facet<4>) -> [(Facet<3>, bool); 4] {
    let [c0, c1, c2, c3] = cell.indices();
    [
        (Facet::new([c1, c2, c3]), false),
        (Facet::new([c0, c2, c3]), true),
        (Facet::new([c0, c1, c3]), false),
        (Facet::new([c0, c1, c2]), true),
    ]
    .map(|(triangle, mut flag)| {
        // Sort the triangles
        let mut indices = triangle.indices();
        for (i, j) in [(0, 1), (1, 2), (0, 1)] {
            if indices[i] > indices[j] {
                indices.swap(i, j);
                flag = !flag;
            }
        }
        (Facet::new(indices), flag)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Volume,
        shape::{ConvexPolyhedron, Simplex3},
    };
    use std::iter::once;

    use approxim::assert_relative_eq;
    use assert2::check;
    use hoomd_vector::{Angle, Rotate, Versor};
    use itertools::iproduct;
    use rand::{RngExt, SeedableRng, rngs::StdRng, seq::SliceRandom};
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
            check!(predicate_orient2d((p + offset, q + offset), test + offset) == 1);
        }

        // The same holds for collinear and right-turning triples.
        let q_collinear = Cartesian::from([1.0, 1.0]);
        let q_right = Cartesian::from([1.0, 1.0 + 1e-8]);
        for offset in [0.0, 1e3, 1e8] {
            let offset = Cartesian::from([offset, offset]);
            check!(predicate_orient2d((p + offset, q_collinear + offset), test + offset) == 0);
            check!(predicate_orient2d((p + offset, q_right + offset), test + offset) == -1);
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
    fn validate_3d_hull(points: &[Cartesian<3>], vertices: &[usize], faces: &[Facet<3>]) {
        // No point is strictly outside any face.
        for &face in faces {
            for q in 0..points.len() {
                check!(
                    orient3d_at(points, face, q) <= 0.0,
                    "point {q} lies outside the face {face:?}"
                );
            }
        }

        // The surface is closed and consistently oriented.
        let mut edges: Vec<(usize, usize)> = faces
            .iter()
            .flat_map(|&face| {
                let [a, b, c] = face.indices();
                [(a, b), (b, c), (c, a)]
            })
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
        let mut face_vertices: Vec<usize> = faces.iter().flat_map(Facet::indices).collect();
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
            .map(|&face| {
                let [a, b, c] = face.as_simplex(&points);
                Simplex3::from([a, b, c, origin]).volume()
            })
            .sum();

        assert_relative_eq!(volume, 8.0);
    }

    /// TODO: replace when the Xenocollide 4d pr is merged
    /// The 5 vertices of a pentachoron with edge length `2*sqrt(2)`.
    fn pentachoron() -> Vec<Cartesian<4>> {
        vec![
            [1.0, 1.0, 1.0, 1.0].into(),
            [1.0, -1.0, -1.0, 1.0].into(),
            [-1.0, 1.0, -1.0, 1.0].into(),
            [-1.0, -1.0, 1.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0 + f64::sqrt(5.0)].into(),
        ]
    }

    /// A point with `sign` in coordinate `axis` and zero in the others.
    fn axis_point(axis: usize, sign: f64) -> Cartesian<4> {
        Cartesian::from(std::array::from_fn(|k| if k == axis { sign } else { 0.0 }))
    }

    /// The 16 vertices of a tesseract (hypercube) with edge length 2.
    fn tesseract() -> Vec<Cartesian<4>> {
        iproduct!([-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0])
            .map(|(x, y, z, w)| Cartesian::from([x, y, z, w]))
            .collect()
    }

    /// The 8 vertices of a 16-cell with edge length `sqrt(2)`.
    fn hexadecachoron() -> Vec<Cartesian<4>> {
        iproduct!(0..4, [-1.0, 1.0])
            .map(|(axis, sign)| axis_point(axis, sign))
            .collect()
    }

    /// The 24 vertices of a 24-cell: permutations of (+-1, +-1, 0, 0).
    fn icositetrachoron() -> Vec<Cartesian<4>> {
        iproduct!((0..4).array_combinations::<2>(), [-1.0, 1.0], [-1.0, 1.0])
            .map(|([i, j], a, b)| {
                let mut point = [0.0; 4];
                point[i] = a;
                point[j] = b;
                Cartesian::from(point)
            })
            .collect()
    }

    /// Rotate a point by the given angles in the xy and zw coordinate planes.
    fn rotate_in_planes(point: Cartesian<4>, xy: Angle, zw: Angle) -> Cartesian<4> {
        let rotate_plane =
            |angle: &Angle, [a, b]: [f64; 2]| angle.rotate(&Cartesian::from([a, b])).coordinates;
        let [x, y] = rotate_plane(&xy, [point[0], point[1]]);
        let [z, w] = rotate_plane(&zw, [point[2], point[3]]);
        Cartesian::from([x, y, z, w])
    }

    /// The hypervolume of a hull, as the sum of the hypervolumes of the
    /// 4-simplices formed by the cells and an interior point. The centroid of
    /// the vertices is interior.
    fn hull_volume(points: &[Cartesian<4>], vertices: &[usize], cells: &[Facet<4>]) -> f64 {
        let apex =
            vertices.iter().map(|&i| points[i]).sum::<Cartesian<4>>() / vertices.len() as f64;
        cells
            .iter()
            .map(|cell| {
                Matrix {
                    rows: cell.indices().map(|i| (points[i] - apex).coordinates),
                }
                .determinant()
            })
            .sum::<f64>()
            .abs()
            / 24.0 // volume of simplex is parallelepiped / N!
    }

    /// Validate a triangulated 4d hull.
    ///
    /// * Every cell is a supporting hyperplane: no point lies strictly outside any cell.
    /// * The cells form a closed, consistently oriented manifold: every oriented
    ///   boundary triangle appears exactly once and is matched by its opposite.
    /// * Euler's formula `V - E + F - C = 0` holds, the Euler characteristic of the
    ///   3-sphere the cells triangulate.
    /// * The reported vertices are exactly the vertices of the cells, in order.
    ///
    /// Together these conditions certify that the cells triangulate the boundary
    /// of the convex hull of the points.
    fn validate_4d_hull(points: &[Cartesian<4>], vertices: &[usize], cells: &[Facet<4>]) {
        // No point is strictly outside any cell.
        for (&cell, q) in cells.iter().cartesian_product(0..points.len()) {
            check!(
                orient4d_at(points, cell, q).expect("the coordinates are resolvable") <= 0,
                "point {q} lies outside the cell {cell:?}"
            );
        }

        // The surface is closed and consistently oriented: every oriented
        // boundary triangle appears exactly once and is matched by its reverse.
        let mut triangles: Vec<(Facet<3>, bool)> = cells
            .iter()
            .flat_map(|&cell| boundary_triangles(cell))
            .collect();
        triangles.sort_unstable();
        for (triangle, next) in triangles.iter().tuple_windows() {
            check!(
                triangle != next,
                "the oriented triangle {triangle:?} appears twice"
            );
        }
        for &(triangle, flipped) in &triangles {
            check!(
                triangles.contains(&(triangle, !flipped)),
                "the triangle {triangle:?} is not matched by its opposite orientation"
            );
        }

        // Euler's formula.
        let cell_vertices: Vec<usize> = cells
            .iter()
            .flat_map(Facet::indices)
            .sorted()
            .dedup()
            .collect();

        // Each triangle is in canonical (sorted) order, so each of its edges is sorted.
        let edges: Vec<[usize; 2]> = triangles
            .iter()
            .flat_map(|&(triangle, _)| {
                let [a, b, c] = triangle.indices();
                [[a, b], [a, c], [b, c]]
            })
            .sorted()
            .dedup()
            .collect();

        check!(
            cell_vertices.len() + triangles.len() / 2 == edges.len() + cells.len(),
            "Euler's formula fails: {} - {} + {} - {} != 0",
            cell_vertices.len(),
            edges.len(),
            triangles.len() / 2,
            cells.len()
        );

        // The vertices are the cell vertices in increasing order.
        check!(
            vertices == cell_vertices,
            "vertices are not the cell vertices"
        );
    }

    /// The vertex sets of the regular convex 4-polytopes, shared by the 4d hull tests.
    #[template]
    #[rstest]
    #[case::pentachoron(pentachoron())]
    #[case::tesseract(tesseract())]
    #[case::hexadecachoron(hexadecachoron())]
    #[case::icositetrachoron(icositetrachoron())]
    fn regular_polytopes(#[case] points: Vec<Cartesian<4>>) {}

    #[apply(regular_polytopes)]
    fn test_4d_regular_polytopes(#[case] points: Vec<Cartesian<4>>) {
        // The polytopes are in their axis-aligned orientations, where the facet
        // vertices are exactly hypercoplanar.
        let (vertices, cells) = incremental_hull_4d(&points)
            .expect("regular polytope vertices should form a convex body");

        validate_4d_hull(&points, &vertices, &cells);
        // Every vertex of a regular polytope is on the hull.
        check!(vertices == (0..points.len()).collect::<Vec<usize>>());
    }

    #[apply(regular_polytopes)]
    fn test_4d_regular_polytopes_rotated(#[case] points: Vec<Cartesian<4>>) {
        // A double rotation by irrational angles breaks the exact hypercoplanarity
        // of the facet vertices, exercising the exact fallback of the predicate.
        for iteration in 0..50 {
            let xy = Angle::from(0.3 + 0.17 * f64::from(iteration));
            let zw = Angle::from(0.7 + 0.11 * f64::from(iteration));
            let rotated: Vec<Cartesian<4>> = points
                .iter()
                .map(|&p| rotate_in_planes(p, xy, zw))
                .collect();

            let (vertices, cells) = incremental_hull_4d(&rotated)
                .expect("regular polytope vertices should form a convex body");

            validate_4d_hull(&rotated, &vertices, &cells);
            // A rotation maps vertices to vertices.
            check!(vertices == (0..points.len()).collect::<Vec<usize>>());
        }
    }

    #[rstest]
    #[case::tesseract(tesseract(), 16.0)]
    #[case::hexadecachoron(hexadecachoron(), 2.0 / 3.0)]
    #[case::icositetrachoron(icositetrachoron(), 8.0)]
    #[case::pentachoron(pentachoron(), 2.0 * f64::sqrt(5.0) / 3.0)]
    fn test_4d_volume(#[case] points: Vec<Cartesian<4>>, #[case] expected: f64) {
        let (vertices, cells) = incremental_hull_4d(&points)
            .expect("regular polytope vertices should form a convex body");
        validate_4d_hull(&points, &vertices, &cells);

        assert_relative_eq!(hull_volume(&points, &vertices, &cells), expected);
    }

    #[rstest]
    #[case::pentachoron(pentachoron(), 5)]
    #[case::hexadecachoron(hexadecachoron(), 16)]
    #[case::icositetrachoron(icositetrachoron(), 96)]
    fn test_4d_cell_counts(#[case] points: Vec<Cartesian<4>>, #[case] expected: usize) {
        // The simplicial polytopes have a unique triangulation, so their cell
        // counts are fixed. The count for the tesseract is not: its cubical
        // facets triangulate into different numbers of tetrahedra depending on
        // the insertion order.
        let (vertices, cells) = incremental_hull_4d(&points)
            .expect("regular polytope vertices should form a convex body");
        validate_4d_hull(&points, &vertices, &cells);

        check!(cells.len() == expected);
    }

    /// The grid `{0, 1}^3` embedded in the hyperplane `w = f(x, y, z)`.
    fn hyperplanar_grid(f: impl Fn(f64, f64, f64) -> f64) -> Vec<[f64; 4]> {
        iproduct!([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
            .map(|(x, y, z)| [x, y, z, f(x, y, z)])
            .collect()
    }

    #[rstest]
    #[case::four_points(vec![[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])]
    #[case::identical(vec![[1.0, 2.0, 3.0, 4.0]; 6])]
    #[case::collinear((0..8).map(|i| [f64::from(i), 2.0 * f64::from(i), -f64::from(i), 3.0 * f64::from(i)]).collect())]
    #[case::planar((0..3).flat_map(|i| (0..3).map(move |j| [f64::from(i), f64::from(j), 0.0, 0.0])).collect())]
    #[case::hyperplanar(hyperplanar_grid(|_, _, _| 0.0))]
    #[case::tilted_hyperplanar(hyperplanar_grid(|x, y, z| x + y + 2.0 * z))]
    fn test_4d_degenerate(#[case] points: Vec<[f64; 4]>) {
        let points: Vec<Cartesian<4>> = points.into_iter().map(Cartesian::from).collect();
        check!(Cartesian::<4>::convex_hull(&points) == Err(Error::DegeneratePolytope));
    }

    #[rstest]
    fn test_4d_numerically_ambiguous() {
        // All points lie in the hyperplane w = 0 except the last, which breaks
        // the degeneracy. The candidate orientations against the hypercoplanar
        // points combine coordinates whose exponents span 80 powers of two,
        // beyond the budget of the exact predicate, so the hull is rejected.
        let tiny = 2.0_f64.powi(-80);
        let points: Vec<Cartesian<4>> = vec![
            [0.0, 0.0, 0.0, 0.0].into(),
            [1.0, 0.0, 0.0, 0.0].into(),
            [0.0, 1.0, 0.0, 0.0].into(),
            [0.0, 0.0, 1.0, 0.0].into(),
            [tiny, tiny, tiny, 0.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
        ];

        check!(Cartesian::<4>::convex_hull(&points) == Err(Error::NumericallyAmbiguousPolytope));
    }

    #[rstest]
    #[case::overflowed_ranking(256, 256)]
    #[case::overflowed_projections(498, 448)]
    fn test_4d_huge_coordinates(#[case] exponent: i32, #[case] spacing: i32) {
        // A tesseract far from the origin, with coordinates of order 2^exponent
        // and edges of order 2^spacing. Two expressions of the initial
        // pentachoron overflow at these magnitudes, and used to reject this
        // full-dimensional input (whose orientations the exact predicates
        // resolve, the coordinate exponent span staying well below 72):
        //
        // * the candidate rankings: squared distances of order 2^512 and more
        //   overflow, and their indeterminate differences are NaN, which
        //   discarded every candidate for the third simplex vertex;
        //
        // * the collinearity and coplanarity filters: the products inside the
        //   lower-dimensional predicates overflow, which reported coplanar
        //   points as noncoplanar and produced a degenerate simplex.
        let offset = 2.0_f64.powi(exponent);
        let edge = 2.0_f64.powi(spacing);
        let points: Vec<Cartesian<4>> = tesseract()
            .into_iter()
            .map(|p| p * edge + Cartesian::from([offset; 4]))
            .collect();

        let (vertices, cells) =
            incremental_hull_4d(&points).expect("a far tesseract should form a convex body");
        validate_4d_hull(&points, &vertices, &cells);
        check!(vertices == (0..16).collect::<Vec<usize>>());
    }

    #[rstest]
    fn test_4d_non_vertex_points_dropped() {
        // The center, a facet point, an edge point and a duplicate corner of
        // the tesseract are not vertices of its hull.
        let mut points = tesseract();
        points.extend(
            [
                [0.0; 4],
                [1.0, 0.5, 0.25, 0.0],
                [1.0, 1.0, 0.5, 0.0],
                [-1.0; 4],
            ]
            .map(Cartesian::from),
        );

        let (vertices, cells) =
            incremental_hull_4d(&points).expect("hard-coded points should form a convex body");
        validate_4d_hull(&points, &vertices, &cells);
        // Only the corners of the tesseract remain, in input order.
        check!(vertices == (0..16).collect::<Vec<usize>>());
    }

    #[rstest]
    fn test_4d_boundary_points_random_orders() {
        // The exact facet centers and edge midpoints of the tesseract, two interior
        // points and a duplicate corner, shuffled into random insertion orders.
        // A boundary point that is inserted while it lies strictly outside the
        // intermediate hull may remain a vertex of the triangulation even though it is
        // not an extreme point, so only the corners and the hull itself are checked.
        // TODO: this could be an issue! While the triangulations (and therefore
        // volumes) are correct, this is technically not a minimal hull. It's not clear
        // to me how one would solve this though, as the body is correct even if the
        // point set is not.
        let mut points = tesseract();
        points.extend(iproduct!(0..4, [-1.0, 1.0]).flat_map(|(axis, sign)| {
            // The facet center, then the midpoints of its edges that
            // meet the positive corner on `axis`.
            let midpoints = (0..4)
                .filter(move |&other| other != axis)
                .map(move |other| {
                    let mut midpoint = axis_point(axis, sign).coordinates;
                    midpoint[other] = 1.0;
                    Cartesian::from(midpoint)
                });
            once(axis_point(axis, sign)).chain(midpoints)
        }));
        points.extend([[0.0; 4], [0.25; 4], [-1.0; 4]].map(Cartesian::from));

        let mut rng = StdRng::seed_from_u64(44);
        for _ in 0..50 {
            let mut shuffled = points.clone();
            shuffled.shuffle(&mut rng);

            let (vertices, cells) = incremental_hull_4d(&shuffled)
                .expect("hard-coded points should form a convex body");
            validate_4d_hull(&shuffled, &vertices, &cells);

            // Every corner is a vertex, and the hull is still the tesseract.
            // The duplicated corner appears once among the hull vertices.
            let is_corner = |i: usize| shuffled[i].coordinates.iter().all(|&c| c.abs() == 1.0);
            let corners_on_hull = vertices.iter().filter(|&&i| is_corner(i)).count();
            check!(
                corners_on_hull == 16,
                "only {corners_on_hull} corners are hull vertices"
            );
            let volume = hull_volume(&shuffled, &vertices, &cells);
            assert_relative_eq!(volume, 16.0, epsilon = 1e-9);
        }
    }

    #[rstest]
    fn test_4d_public_hull_facets() {
        let points = tesseract();
        let (vertices, facets) = Cartesian::<4>::convex_hull(&points)
            .expect("hard-coded points should form a convex body");

        check!(vertices.len() == 16);
        // Each of the eight cubical facets triangulates into at least five
        // tetrahedra, and the exact count depends on the insertion order.
        check!(facets.len() >= 40);
        // Every facet references an existing vertex, and every vertex of the
        // hull is referenced by some facet.
        let referenced = || facets.iter().flat_map(Facet::indices);
        check!(referenced().all(|index| index < vertices.len()));
        check!(referenced().sorted().dedup().eq(0..16));
    }

    #[rstest]
    fn test_4d_random_point_clouds(#[values(0, 1, 2, 3, 4)] seed: u64) {
        // The points are drawn from the uniform distribution over the hypercube
        // [-1, 1]^4.
        let mut rng = StdRng::seed_from_u64(seed);
        let points: Vec<Cartesian<4>> = (0..30)
            .map(|_| std::array::from_fn(|_| rng.random::<f64>() * 2.0 - 1.0).into())
            .collect();

        let (vertices, cells) =
            incremental_hull_4d(&points).expect("random points should form a convex body");

        validate_4d_hull(&points, &vertices, &cells);
        check!(vertices.len() >= 5);
    }

    #[rstest]
    fn test_4d_input_types() {
        let points = hexadecachoron();

        let (from_slice, _) = Cartesian::<4>::convex_hull(&points[..])
            .expect("hard-coded points should form a convex body");
        let (from_vec, _) = Cartesian::<4>::convex_hull(points.clone())
            .expect("hard-coded points should form a convex body");
        let (from_iterator, _) = Cartesian::<4>::convex_hull(points.iter().copied())
            .expect("hard-coded points should form a convex body");

        assert_eq!(from_slice.len(), 8);
        itertools::assert_equal(&from_slice, &from_vec);
        itertools::assert_equal(&from_slice, &from_iterator);
    }
}
