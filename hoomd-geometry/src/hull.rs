// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Compute the convex hull of a set of points.

use std::{borrow::Borrow, cmp::Ordering};

use itertools::Itertools;

use crate::Error;
use hoomd_vector::{Cartesian, InnerProduct};

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
    /// are arranged in a counter-clockwise order.
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
/// # Warning
///
/// This predicate is not robust: points very close to the line may be misclassified
/// due to floating-point precision limits. For all practical inputs, this will not
/// result in issues.
///
/// # Note
///
/// This formulation (often referred to as the shoelace formula) guarantees
/// **cyclic invariance**, or the property that the orientation sign is identical
/// for any ordering of the three points `(e0, e1, t)`, `(e1, t, e0)`, and
/// `(t, e0, e1)`. As a result, the result is antisymmetric about the edge `e`
/// such that `p == -p'` for any `p'` reflected over `e`.
///
/// These properties do *not* prevent misclassification of points near the line, they
/// only ensure consistent behavior for related inputs.
///
/// **Source:** [Robust Arithmetic in Computational Geometry](https://observablehq.com/@mourner/non-robust-arithmetic-as-art)
#[inline]
fn predicate_orient2d((p, q): (Cartesian<2>, Cartesian<2>), test: Cartesian<2>) -> i64 {
    let orientation = (p[0] * q[1] - p[1] * q[0])
        + (q[0] * test[1] - q[1] * test[0])
        + (test[0] * p[1] - test[1] * p[0]);

    match orientation.total_cmp(&0.0) {
        Ordering::Greater => 1,
        Ordering::Less => -1,
        Ordering::Equal => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert2::check;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;

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
}
