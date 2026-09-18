// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Compute the convex hull of a set of points.

use std::{borrow::Borrow, cmp::Ordering};

use itertools::Itertools;

use crate::Error;
use hoomd_vector::{Cartesian, Cross, InnerProduct};

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
/// let hull_vertices = Cartesian::<2>::convex_hull(&points)?;
///
/// assert_eq!(hull_vertices.len(), 4);
/// # Ok(())
/// # }
/// ```
pub trait ConvexHull: Sized {
    /// Compute the convex hull of a set of points.
    ///
    /// The resulting vector contains a subset of the given points, including
    /// only the non-degenerate points on the convex hull. The output vertices
    /// are arranged in a deterministic order: counter-clockwise in two dimensions, and
    /// in the order of the input in three dimensions and higher.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if the input points do not form a convex body with 3 or more points.
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
    /// let hull_vertices = Cartesian::<2>::convex_hull((0..3).map(|i| {
    ///     let angle = 2.0 * std::f64::consts::PI * f64::from(i) / 3.0;
    ///     Cartesian::from([angle.cos(), angle.sin()])
    /// }))?;
    ///
    /// assert_eq!(hull_vertices.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    fn convex_hull<I>(points: I) -> Result<Vec<Self>, Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>;
}

impl ConvexHull for Cartesian<2> {
    /// Compute the convex hull of points in 2D with the Graham scan algorithm.
    ///
    /// The orientation tests use robust adaptive predicates that compute the
    /// exact sign of the orientation determinant, so the resulting hull does
    /// not depend on where the point set lies relative to the origin (up to
    /// the precision of the coordinates themselves).
    #[inline]
    fn convex_hull<I>(points: I) -> Result<Vec<Self>, Error>
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
        if points.len() >= 3 {
            Ok(points)
        } else {
            Err(Error::DegeneratePolytope)
        }
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

impl ConvexHull for Cartesian<3> {
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
    /// let hull_vertices = Cartesian::<3>::convex_hull(&cube)?;
    ///
    /// assert_eq!(hull_vertices.len(), 8);
    /// # Ok(())
    /// # }
    /// ```
    #[inline]
    fn convex_hull<I>(points: I) -> Result<Vec<Self>, Error>
    where
        I: IntoIterator,
        I::Item: Borrow<Self>,
    {
        let points: Vec<Self> = points.into_iter().map(|p| *p.borrow()).collect();

        // No convex body without at least 4 points.
        if points.len() < 4 {
            return Err(Error::DegeneratePolytope);
        }

        let (vertices, _faces) = incremental_hull(&points)?;

        Ok(vertices.into_iter().map(|i| points[i]).collect())
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
fn incremental_hull(points: &[Cartesian<3>]) -> Result<(Vec<usize>, Vec<[usize; 3]>), Error> {
    let (t0, t1, t2, t3) = initial_tetrahedron(points)?;

    // The four faces of the initial tetrahedron. Each face is oriented so
    // that the vertex opposite it is on its negative side, which places
    // points strictly outside a face on its positive side.
    let mut faces: Vec<[usize; 3]> = vec![[t0, t1, t2], [t0, t2, t3], [t0, t3, t1], [t1, t3, t2]];
    let opposite = [t3, t1, t2, t0];
    for (face, o) in faces.iter_mut().zip(opposite) {
        if orient3d_at(points, face[0], face[1], face[2], o) > 0.0 {
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
            if orient3d_at(points, face[0], face[1], face[2], p) > 0.0 {
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
    // let on_hull = (0..points.len())
    //     .map(|i| faces.iter().any(|f| f.contains(&i)))
    //     .collect::<Vec<bool>>();

    Ok(((0..points.len()).filter(|&i| on_hull[i]).collect(), faces))
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
    // Any nonzero value from the robust predicate guarantees our tetrahedron is valid.
    let (mut d, mut d_volume) = (None, 0.0);
    for i in 0..points.len() {
        if i == a || i == b || i == c {
            continue;
        }
        let volume = orient3d_at(points, a, b, c, i).abs();
        if volume > d_volume {
            d_volume = volume;
            d = Some(i);
        }
    }

    // Otherwise, all points are coplanar.
    let d = d.ok_or(Error::DegeneratePolytope)?;

    Ok((a, b, c, d))
}

/// The orientation determinant of four points given by index.
///
/// The sign of the returned value is positive when `d` lies strictly outside the face
/// `(a, b, c)` oriented as by [`incremental_hull`], negative when it lies strictly
/// inside, and zero when the four points are coplanar.
#[inline]
fn orient3d_at(points: &[Cartesian<3>], a: usize, b: usize, c: usize, d: usize) -> f64 {
    let coord = |p: &Cartesian<3>| robust::Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    };

    robust::orient3d(
        coord(&points[a]),
        coord(&points[b]),
        coord(&points[c]),
        coord(&points[d]),
    )
}

/// Whether three points given by index are exactly collinear.
///
/// The points are collinear when they are collinear in each of the three coordinate
/// plane projections, decided by the two-dimensional exact predicate.
#[inline]
fn collinear(points: &[Cartesian<3>], a: usize, b: usize, c: usize) -> bool {
    let coord = |p: &Cartesian<3>, i: usize, j: usize| robust::Coord { x: p[i], y: p[j] };
    [(0, 1), (0, 2), (1, 2)].iter().all(|&(i, j)| {
        robust::orient2d(
            coord(&points[a], i, j),
            coord(&points[b], i, j),
            coord(&points[c], i, j),
        ) == 0.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Volume,
        platonic::{cube, dodecahedron, icosahedron, octahedron, tetrahedron},
        shape::Simplex3,
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
            let vertices1 = Cartesian::<2>::convex_hull(&points1)
                .expect("hard-coded points should lie on a convex hull");

            let mut rng = StdRng::seed_from_u64(123);
            let points2: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let vertices2 = Cartesian::<2>::convex_hull(&points2)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let vertices = Cartesian::<2>::convex_hull(&points)
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
        let from_slice = Cartesian::<2>::convex_hull(&points[..])
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_slice, &hull);

        // An owned Vec of points.
        let from_vec = Cartesian::<2>::convex_hull(points.clone())
            .expect("hard-coded points should lie on a convex hull");
        itertools::assert_equal(&from_vec, &hull);

        // An iterator of points.
        let from_iterator = Cartesian::<2>::convex_hull(points.iter().copied())
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

            let vertices = Cartesian::<2>::convex_hull(&points)
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

            let vertices = Cartesian::<2>::convex_hull(&points)
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
    /// The vertex sets of the platonic solids, shared by the 3d hull tests.
    #[template]
    #[rstest]
    #[case::tetrahedron(tetrahedron())]
    #[case::cube(cube())]
    #[case::octahedron(octahedron())]
    #[case::dodecahedron(dodecahedron())]
    #[case::icosahedron(icosahedron())]
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
                check!(
                    orient3d_at(points, face[0], face[1], face[2], q) <= 0.0,
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
        itertools::assert_equal(hull, cube());
    }

    #[rstest]
    fn test_3d_random_point_clouds(#[values(0, 1, 2, 3, 4)] seed: u64) {
        // The points are drawn from the uniform distribution over the cube
        // [-1, 1]^3.
        let mut rng = StdRng::seed_from_u64(seed);
        let points: Vec<Cartesian<3>> = (0..30).map(|_| rng.random()).collect();

        let (vertices, faces) =
            incremental_hull(&points).expect("random points should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        check!(vertices.len() >= 4);
    }

    #[rstest]
    fn test_3d_fibonacci_sphere() {
        // The points of a Fibonacci lattice on the unit sphere are all
        // vertices of their hull.
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
        let points: Vec<Cartesian<3>> = dodecahedron()
            .into_iter()
            .map(|p| p * scale + offset)
            .collect();

        let (vertices, faces) =
            incremental_hull(&points).expect("hard-coded points should form a convex body");

        validate_3d_hull(&points, &vertices, &faces);
        check!(vertices == (0..20).collect::<Vec<usize>>());
    }

    #[rstest]
    fn test_3d_input_types() {
        let points = octahedron();

        let from_slice = Cartesian::<3>::convex_hull(&points[..])
            .expect("hard-coded points should form a convex body");
        let from_vec = Cartesian::<3>::convex_hull(points.clone())
            .expect("hard-coded points should form a convex body");
        let from_iterator = Cartesian::<3>::convex_hull(points.iter().copied())
            .expect("hard-coded points should form a convex body");

        assert_eq!(from_slice.len(), 6);
        itertools::assert_equal(&from_slice, &from_vec);
        itertools::assert_equal(&from_slice, &from_iterator);
    }

    #[rstest]
    fn test_3d_volume() {
        // The hull volume is the sum of the volumes of the tetrahedra formed
        // by the faces and any interior point. The origin, the centroid of
        // the cube, is interior.
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
}
