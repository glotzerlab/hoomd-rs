// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

use std::collections::VecDeque;

use crate::Error;
use hoomd_vector::{Cartesian, InnerProduct};
use itertools::Itertools;

/// Find the lowest, leftmost point from a slice of Cartesian vectors.
#[inline]
fn find_lowest_leftmost(vertices: &[Cartesian<2>]) -> (usize, Cartesian<2>) {
    vertices.iter().enumerate().fold(
        (usize::MAX, Cartesian::from([f64::INFINITY, f64::INFINITY])),
        |(i, p), (min_i, &min_p)| {
            if p[1] < min_p[1] || (p[1] == min_p[1] && p[0] < min_p[0]) {
                (i, p)
            } else {
                (min_i, min_p)
            }
        },
    )
}

/// Get the key for a lexographical sort of points with respect to an anchor.
#[inline]
pub fn get_graham_key(p: Cartesian<2>, anchor: Cartesian<2>) -> (f64, f64) {
    let diff = p - anchor;
    (f64::atan2(diff[1], diff[0]), diff.dot(&diff))
}
/// Determines whether a point `test` is to the left, right, or colinear with `edge`.
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
/// These properties do *not* prevent misclassification of points near the line—they
/// only ensure consistent behavior for related inputs.
///
/// **Source:** [Robust Arithmetic in Computational Geometry](https://observablehq.com/@mourner/non-robust-arithmetic-as-art)
fn predicate_orient2d(edge: (Cartesian<2>, Cartesian<2>), test: Cartesian<2>) -> i64 {
    let (p, q) = edge;
    let orientation = (p[0] * q[1] - p[1] * q[0])
        + (q[0] * test[1] - q[1] * test[0])
        + (test[0] * p[1] - test[1] * p[0]);
    if orientation > 0.0 {
        // clockwise
        return 1;
    } else if orientation < 0.0 {
        // counterclockwise
        return -1;
    }
    0
}

/// Compute the convex hull of points in two dimensions using a Graham scan.
/// # Errors
///
/// `[Error::PolytopeNotConvex]` when the provided vertices do not form a convex set.
#[inline]
pub fn hull_2d_grahamscan(vertices: &mut [Cartesian<2>]) -> Option<Vec<Cartesian<2>>> {
    // vertices.sort_by(|a, b| {});
    let (anchor_idx, anchor) = find_lowest_leftmost(&vertices[..]);

    // Move the anchor to the front of the list of vertices, as it is always in the hull
    vertices.swap(0, anchor_idx);
    let anchor = vertices[0];

    // Sort the remainder of the slice in-place
    vertices[1..].sort_unstable_by(|&a, &b| {
        let (a0, a1) = get_graham_key(a, anchor);
        let (b0, b1) = get_graham_key(b, anchor);
        a0.total_cmp(&b0).then(a1.total_cmp(&b1))
    });

    // TODO: no allocation - swap remove? Ideally move unwanted to end and then truncate
    let mut angle_order = vertices[1..].iter().copied().collect::<VecDeque<_>>();

    // The anchor and the first vertex in the sorted list are always on the hull
    let mut hull_vertices = vec![anchor, angle_order.pop_front()?];
    while !angle_order.is_empty() {
        let c = angle_order.pop_front()?;
        while hull_vertices.len() >= 2 {
            // Backtrack while hull_indices[-1] is inside hull
            let &[p, n] = hull_vertices.last_chunk::<2>()?;

            if predicate_orient2d((p, n), c) <= 0 {
                hull_vertices.pop(); // point n is inside the hull, so we remove it
            } else {
                break;
            }
        }
        hull_vertices.push(c);
    }
    Some(hull_vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_relative_eq;
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use rstest::*;

    // Helper to check if a point is approximately in the hull
    fn hull_contains(hull: &[Cartesian<2>], point: [f64; 2], tol: f64) -> bool {
        hull.iter()
            .any(|h| (h[0] - point[0]).abs() < tol && (h[1] - point[1]).abs() < tol)
    }

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
        let (idx, point) = find_lowest_leftmost(&vertices);
        assert_eq!(idx, expected_idx);
        assert_relative_eq!(point, vertices[expected_idx]);
    }

    #[rstest]
    fn test_single_point_various_coords(
        #[values([0.0, 0.0], [-1.0, -1.0], [100.0, -50.0], [f64::MIN_POSITIVE, f64::MIN_POSITIVE])]
        coords: [f64; 2],
    ) {
        let vertices = vec![Cartesian::from(coords)];
        let (idx, point) = find_lowest_leftmost(&vertices);
        assert_eq!(idx, 0);
        assert_relative_eq!(point, Cartesian::from(coords));
    }

    #[rstest]
    fn test_square_corners() {
        let mut vertices: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert_eq!(hull.len(), 4);
        // Check all corners are in hull
        for v in &vertices {
            assert!(hull.iter().any(|h| (*h - *v).norm() < 1e-10));
        }
    }

    #[rstest]
    fn test_square_with_edge_points() {
        let mut vertices: Vec<Cartesian<2>> = vec![
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
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Hull should have 4 corners (interior edge points excluded)
        assert_eq!(hull.len(), 4);
        let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        for corner in corners {
            assert!(hull_contains(&hull, corner, 1e-10));
        }
    }

    #[rstest]
    fn test_square_dense_boundary() {
        let mut pts: Vec<[f64; 2]> = Vec::new();
        // Bottom edge
        for i in 0..20 {
            pts.push([i as f64 / 19.0, 0.0]);
        }
        // Right edge
        for i in 0..20 {
            pts.push([1.0, i as f64 / 19.0]);
        }
        // Top edge
        for i in 0..20 {
            pts.push([i as f64 / 19.0, 1.0]);
        }
        // Left edge
        for i in 0..20 {
            pts.push([0.0, i as f64 / 19.0]);
        }
        let mut vertices: Vec<Cartesian<2>> = pts.into_iter().map(Cartesian::from).collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert_eq!(hull.len(), 4);
    }

    #[rstest]
    fn test_circle_uniform() {
        let n = 20;
        let mut vertices: Vec<Cartesian<2>> = (0..n)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // All points on circle should be in hull
        assert_eq!(hull.len(), n);
    }

    #[rstest]
    fn test_circle_with_interior_points() {
        let n_boundary = 12;
        let mut vertices: Vec<Cartesian<2>> = (0..n_boundary)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / n_boundary as f64;
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        // Add interior points
        vertices.extend([[0.0, 0.0], [0.3, 0.3], [-0.2, 0.1], [0.1, -0.4]].map(Cartesian::from));
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Only boundary points should be in hull
        assert_eq!(hull.len(), n_boundary);
    }

    #[rstest]
    fn test_circle_partial_arc() {
        let mut vertices: Vec<Cartesian<2>> = (0..10)
            .map(|i| {
                let angle =
                    -std::f64::consts::FRAC_PI_4 + (std::f64::consts::FRAC_PI_2 * i as f64 / 9.0);
                Cartesian::from([angle.cos(), angle.sin()])
            })
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // All points on partial arc should be in hull
        assert_eq!(hull.len(), 10);
    }

    #[rstest]
    fn test_random_unit_square(#[values(0, 42, 64, 100_000)] seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);
        let original: Vec<Cartesian<2>> = (0..50)
            .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
            .collect();
        let mut vertices = original.clone();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Hull should have at least 3 points
        assert!(hull.len() >= 3);
        // All hull points should be in original set
        for h in &hull {
            assert!(original.iter().any(|v| (*v - *h).norm() < 1e-10));
        }
    }

    #[rstest]
    fn test_random_gaussian() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut vertices: Vec<Cartesian<2>> = (0..100)
            .map(|_| {
                Cartesian::from([
                    rng.random::<f64>() * 2.0 - 1.0,
                    rng.random::<f64>() * 2.0 - 1.0,
                ])
            })
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_random_deterministic_output() {
        for _ in 0..3 {
            let mut rng = StdRng::seed_from_u64(123);
            let mut vertices1: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let hull1 = hull_2d_grahamscan(&mut vertices1).expect("hull should exist");

            let mut rng = StdRng::seed_from_u64(123);
            let mut vertices2: Vec<Cartesian<2>> = (0..30)
                .map(|_| Cartesian::from([rng.random::<f64>(), rng.random::<f64>()]))
                .collect();
            let hull2 = hull_2d_grahamscan(&mut vertices2).expect("hull should exist");

            assert_eq!(hull1.len(), hull2.len());
        }
    }

    #[rstest]
    fn test_duplicate_lowest_points() {
        let mut vertices: Vec<Cartesian<2>> = vec![
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
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_many_duplicates_few_unique() {
        let mut vertices: Vec<Cartesian<2>> = vec![
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
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Should handle duplicates gracefully
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_collinear_bottom_edge() {
        let mut vertices: Vec<Cartesian<2>> = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [1.5, 0.0],  // All at y=0
            [0.75, 1.0], // Apex
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Should pick leftmost of bottom points and include apex
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_leftmost_selected() {
        let mut vertices: Vec<Cartesian<2>> = vec![
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 0.0], // All y=0, but [0,0] is leftmost
            [0.5, 1.0],
        ]
        .into_iter()
        .map(Cartesian::from)
        .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // [0, 0] should be the anchor point
        assert!(hull_contains(&hull, [0.0, 0.0], 1e-10));
    }

    #[rstest]
    fn test_many_bottom_points() {
        let mut pts: Vec<[f64; 2]> = (0..10).map(|i| [i as f64 * 2.0 / 9.0, 0.0]).collect();
        pts.push([1.0, 1.0]); // Apex
        let mut vertices: Vec<Cartesian<2>> = pts.into_iter().map(Cartesian::from).collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Leftmost [0,0] and rightmost [2,0] should be in hull with apex
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_collinear_from_anchor() {
        let mut vertices: Vec<Cartesian<2>> = vec![
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
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Should only keep furthest point in each direction
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_radial_collinear_multiple_directions() {
        let mut vertices: Vec<Cartesian<2>> = vec![
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
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Should keep only outermost points
        assert!(hull.len() >= 3);
    }

    #[rstest]
    fn test_star_pattern() {
        let mut vertices: Vec<Cartesian<2>> = vec![Cartesian::from([0.0, 0.0])]; // Anchor
        // Create points along 8 rays
        for i in 0..8 {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / 8.0;
            for r in [0.5, 1.0, 1.5] {
                vertices.push(Cartesian::from([r * angle.cos(), r * angle.sin()]));
            }
        }
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        // Outer points should form the hull
        assert!(hull.len() >= 8); // At least 8 outer points
    }

    #[rstest]
    fn test_minimum_triangle() {
        let mut vertices: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert_eq!(hull.len(), 3);
    }

    #[rstest]
    fn test_negative_coordinates() {
        let mut vertices: Vec<Cartesian<2>> =
            vec![[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]]
                .into_iter()
                .map(Cartesian::from)
                .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert_eq!(hull.len(), 4);
    }

    #[rstest]
    fn test_large_coordinates() {
        let mut vertices: Vec<Cartesian<2>> = vec![[0.0, 0.0], [1e6, 0.0], [1e6, 1e6], [0.0, 1e6]]
            .into_iter()
            .map(Cartesian::from)
            .collect();
        let hull = hull_2d_grahamscan(&mut vertices).expect("hull should exist");
        assert_eq!(hull.len(), 4);
    }
}
